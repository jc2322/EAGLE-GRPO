from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rewards.online_component_reward import ComponentReward, ComponentRewardStore
from utils.advantage_protocol import AdvantageContext
from utils.completion_schema import CharacterSpan, FINAL_PROMPT_TAGS, char_spans_to_token_mask, parse_completion
from utils.modeling import decode_preserving_schema_tags


CONSTRAINT_ONLY_TAGS = {"preservation_constraints", "negative_constraints"}
DEFAULT_ELEMENT_TAGS = tuple(tag for tag in FINAL_PROMPT_TAGS if tag not in CONSTRAINT_ONLY_TAGS)
TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", flags=re.UNICODE)
HIDDEN_STATE_EMBEDDING_BACKENDS = {"hidden_states", "policy_hidden_mean", "policy_hidden_states"}
SUPPORTED_EMBEDDING_BACKENDS = {"hashed_ngram", *HIDDEN_STATE_EMBEDDING_BACKENDS}


@dataclass
class ElementCandidate:
    row_index: int
    reward: ComponentReward
    valid_length: int
    text: str
    elements: dict[str, CharacterSpan]
    masks: dict[str, list[int]]
    lengths: dict[str, int]
    hidden_states: torch.Tensor | None = None


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _configured_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_ELEMENT_TAGS
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    tags = [tag for tag in items if tag and tag in DEFAULT_ELEMENT_TAGS]
    return tuple(dict.fromkeys(tags)) or DEFAULT_ELEMENT_TAGS


def _block_span(text: str, tag: str) -> CharacterSpan | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        return None
    content = match.group(1)
    left_trim = len(content) - len(content.lstrip())
    right_trim = len(content) - len(content.rstrip())
    return CharacterSpan(
        name=tag,
        start=match.start(1) + left_trim,
        end=match.end(1) - right_trim,
        text=content.strip(),
    )


def _align_mask(mask: list[int], length: int) -> list[int]:
    if len(mask) >= length:
        return mask[:length]
    return mask + [0] * (length - len(mask))


def _span_token_mask(tokenizer: object, text: str, span: CharacterSpan, length: int) -> list[int]:
    try:
        return _align_mask(char_spans_to_token_mask(tokenizer, text, [span]), length)
    except Exception:  # pragma: no cover - tokenizer-specific fallback
        return [0] * length


def _apply_mask(
    tensor: torch.Tensor,
    loss_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    row_index: int,
    mask: list[int],
    value: float,
) -> None:
    for token_index, enabled in enumerate(mask):
        if enabled:
            tensor[row_index, token_index] = float(value)
            loss_mask[row_index, token_index] = completion_mask[row_index, token_index]


def _features(text: str, *, min_ngram: int, max_ngram: int) -> list[str]:
    tokens = TOKEN_RE.findall(str(text or "").lower())
    if not tokens:
        return []
    output: list[str] = []
    for ngram in range(max(1, min_ngram), max(min_ngram, max_ngram) + 1):
        if len(tokens) < ngram:
            continue
        output.extend(" ".join(tokens[index : index + ngram]) for index in range(len(tokens) - ngram + 1))
    return output


def _hashed_embedding(
    text: str,
    *,
    dim: int,
    min_ngram: int,
    max_ngram: int,
    normalize: bool,
) -> torch.Tensor:
    vector = torch.zeros(int(dim), dtype=torch.float64)
    for feature in _features(text, min_ngram=min_ngram, max_ngram=max_ngram):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        index = value % int(dim)
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[index] += sign
    norm = vector.norm()
    if normalize and float(norm.item()) > 0.0:
        vector = vector / norm
    return vector


def _kernel_alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = left.norm() * right.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float(((left * right).sum() / denom).item())


class ElementKernelAdvantageProvider:
    """Element-aware image credit assignment for final-prompt tokens.

    The default backend is ``hashed_ngram``. Hidden-state backends pool policy
    hidden states over the same element token spans used for credit assignment.
    """

    name = "element_kernel_image"

    def __init__(self, store: ComponentRewardStore, config: dict | None = None) -> None:
        self.store = store
        self.config = config or {}
        cfg = _section(self.config, "element_credit")
        self.element_tags = _configured_tags(cfg.get("element_tags"))
        self.embedding_backend = str(cfg.get("embedding_backend", "hashed_ngram") or "hashed_ngram").strip().lower()
        if self.embedding_backend not in SUPPORTED_EMBEDDING_BACKENDS:
            raise ValueError(
                "Unsupported element_credit.embedding_backend: "
                f"{self.embedding_backend!r}. Supported: {sorted(SUPPORTED_EMBEDDING_BACKENDS)}"
            )
        self.hidden_state_fallback = str(cfg.get("hidden_state_fallback", "hashed_ngram") or "hashed_ngram").strip().lower()
        if self.hidden_state_fallback not in {"hashed_ngram", "error"}:
            raise ValueError("element_credit.hidden_state_fallback must be 'hashed_ngram' or 'error'")
        self.token_delta_epsilon = float(cfg.get("token_delta_epsilon", 1e-6))
        self.lambda_reg = float(cfg.get("lambda", cfg.get("lambda_reg", 0.1)))
        self.eta = max(0.0, min(1.0, float(cfg.get("eta", 0.3))))
        self.embedding_dim = int(cfg.get("embedding_dim", 256))
        self.min_ngram = int(cfg.get("min_ngram", 1))
        self.max_ngram = int(cfg.get("max_ngram", 2))
        self.normalize_embeddings = bool(cfg.get("normalize_embeddings", True))
        self.reward_epsilon = float(cfg.get("reward_epsilon", 1e-6))
        self.min_element_energy = float(cfg.get("min_element_energy", 1e-8))
        self.condition_number_max = float(cfg.get("condition_number_max", 1_000.0))
        self.high_kernel_alignment = float(cfg.get("high_kernel_alignment", 0.98))
        self.alignment_downweight = max(0.0, min(1.0, float(cfg.get("alignment_downweight", 0.5))))
        self.include_thinking = bool(cfg.get("include_thinking", True))
        self.save_kernel_credit = bool(cfg.get("save_kernel_credit", True))
        self.kernel_credit_dir = self._resolve_kernel_credit_dir(cfg)
        self.last_loss_mask: torch.Tensor | None = None
        self.last_diagnostics: list[dict[str, Any]] = []
        self.last_credit_rows: list[dict[str, Any]] = []

    def _resolve_kernel_credit_dir(self, cfg: dict[str, Any]) -> Path | None:
        configured = cfg.get("kernel_credit_dir")
        if configured:
            path = Path(str(configured)).expanduser()
            if not path.is_absolute():
                path = Path(str(self.config.get("_project_dir", "."))).expanduser() / path
            return path
        reward_cfg = _section(self.config, "reward")
        work_dir = reward_cfg.get("work_dir")
        if not work_dir:
            return None
        path = Path(str(work_dir)).expanduser()
        if not path.is_absolute():
            path = Path(str(self.config.get("_project_dir", "."))).expanduser() / path
        return path / "kernel_credit"

    def token_advantages(self, context: AdvantageContext) -> torch.Tensor:
        batch = self.store.consume_latest()
        completion_ids = context.completion_ids
        completion_mask = context.completion_mask
        tokenizer = context.tokenizer
        device = completion_ids.device
        token_advantages = torch.zeros_like(completion_mask, dtype=torch.float32, device=device)
        loss_mask = torch.zeros_like(completion_mask, dtype=completion_mask.dtype, device=device)
        self.last_diagnostics = []
        self.last_credit_rows = []
        hidden_states = context.hidden_states
        if self.embedding_backend in HIDDEN_STATE_EMBEDDING_BACKENDS and not torch.is_tensor(hidden_states):
            if self.hidden_state_fallback == "error":
                raise ValueError(
                    f"element_credit.embedding_backend={self.embedding_backend!r} requires AdvantageContext.hidden_states"
                )
            self.last_diagnostics.append(
                {
                    "fallback": "missing_hidden_states",
                    "embedding_backend": self.embedding_backend,
                    "hidden_state_fallback": self.hidden_state_fallback,
                }
            )

        if len(batch.rewards) != int(completion_ids.shape[0]):
            raise ValueError(
                f"Component reward count {len(batch.rewards)} does not match completion batch {int(completion_ids.shape[0])}"
            )

        groups: dict[str, list[ElementCandidate]] = defaultdict(list)
        for row_index, reward in enumerate(batch.rewards):
            valid_length = int(completion_mask[row_index].sum().item())
            if valid_length <= 0 or not reward.active:
                continue
            ids = completion_ids[row_index, :valid_length].detach().cpu().tolist()
            text = decode_preserving_schema_tags(tokenizer, ids, skip_special_tokens=False)
            if reward.format_invalid:
                token_advantages[row_index, :valid_length] = float(reward.format_advantage)
                loss_mask[row_index, :valid_length] = completion_mask[row_index, :valid_length]
                continue
            try:
                parsed = parse_completion(text, require_structured_children=True)
            except ValueError:
                token_advantages[row_index, :valid_length] = float(reward.format_advantage)
                loss_mask[row_index, :valid_length] = completion_mask[row_index, :valid_length]
                continue

            if self.include_thinking:
                thinking = _block_span(text, "thinking") or _block_span(text, "reasoning") or _block_span(text, "think")
                if thinking is not None:
                    _apply_mask(
                        token_advantages,
                        loss_mask,
                        completion_mask,
                        row_index,
                        _span_token_mask(tokenizer, text, thinking, valid_length),
                        float(reward.cot_advantage),
                    )

            elements: dict[str, CharacterSpan] = {}
            masks: dict[str, list[int]] = {}
            lengths: dict[str, int] = {}
            for tag in self.element_tags:
                span = parsed.final_prompt_spans.get(tag)
                if span is None or not span.text.strip():
                    continue
                mask = _span_token_mask(tokenizer, text, span, valid_length)
                length = int(sum(mask))
                if length <= 0:
                    continue
                elements[tag] = span
                masks[tag] = mask
                lengths[tag] = length
            if elements:
                groups[str(reward.row_key)].append(
                    ElementCandidate(
                        row_index=row_index,
                        reward=reward,
                        valid_length=valid_length,
                        text=text,
                        elements=elements,
                        masks=masks,
                        lengths=lengths,
                        hidden_states=self._candidate_hidden_states(hidden_states, row_index, valid_length),
                    )
                )

        for row_key, candidates in groups.items():
            self._assign_group_image_advantages(row_key, candidates, token_advantages, loss_mask, completion_mask)

        self._write_kernel_credit_audit()
        self.last_loss_mask = loss_mask
        return token_advantages

    def _embed(self, text: str) -> torch.Tensor:
        return _hashed_embedding(
            text,
            dim=self.embedding_dim,
            min_ngram=self.min_ngram,
            max_ngram=self.max_ngram,
            normalize=self.normalize_embeddings,
        )

    def _candidate_hidden_states(
        self,
        hidden_states: torch.Tensor | None,
        row_index: int,
        valid_length: int,
    ) -> torch.Tensor | None:
        if self.embedding_backend not in HIDDEN_STATE_EMBEDDING_BACKENDS or not torch.is_tensor(hidden_states):
            return None
        if hidden_states.ndim != 3 or row_index >= int(hidden_states.shape[0]):
            return None
        length = min(valid_length, int(hidden_states.shape[1]))
        if length <= 0:
            return None
        return hidden_states[row_index, :length].detach().to(device="cpu", dtype=torch.float64)

    def _embed_candidate_element(self, candidate: ElementCandidate, tag: str) -> torch.Tensor:
        if self.embedding_backend in HIDDEN_STATE_EMBEDDING_BACKENDS:
            hidden = candidate.hidden_states
            mask = candidate.masks.get(tag)
            if hidden is not None and mask:
                enabled = torch.tensor(_align_mask(mask, int(hidden.shape[0])), dtype=torch.bool)
                if enabled.any():
                    vector = hidden[enabled].mean(dim=0)
                    norm = vector.norm()
                    if self.normalize_embeddings and float(norm.item()) > 0.0:
                        vector = vector / norm
                    return vector
            if self.hidden_state_fallback == "error":
                raise ValueError(f"Missing hidden states for row={candidate.row_index} tag={tag}")
        return self._embed(candidate.elements[tag].text)

    def _assign_standard_image(
        self,
        candidates: list[ElementCandidate],
        token_advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        fallback: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        for candidate in candidates:
            for mask in candidate.masks.values():
                _apply_mask(
                    token_advantages,
                    loss_mask,
                    completion_mask,
                    candidate.row_index,
                    mask,
                    float(candidate.reward.image_advantage),
                )
            self._append_standard_credit_row(candidate, fallback=fallback, diagnostics=diagnostics or {})

    def _assign_group_image_advantages(
        self,
        row_key: str,
        candidates: list[ElementCandidate],
        token_advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> None:
        group_size = len(candidates)
        if group_size == 0:
            return
        rewards = torch.tensor([float(candidate.reward.image_reward) for candidate in candidates], dtype=torch.float64)
        centered = rewards - rewards.mean()
        variance = torch.mean(centered.square())
        std = math.sqrt(float(variance.item()))
        if std <= self.reward_epsilon:
            diagnostics = {"row_key": row_key, "fallback": "zero_reward_variance", "group_size": group_size, "std": std}
            self._assign_standard_image(
                candidates,
                token_advantages,
                loss_mask,
                completion_mask,
                fallback="zero_reward_variance",
                diagnostics=diagnostics,
            )
            self.last_diagnostics.append(diagnostics)
            return

        kernels: dict[str, torch.Tensor] = {}
        confidence: dict[str, float] = {}
        energies: dict[str, float] = {}
        for tag in self.element_tags:
            present = [candidate for candidate in candidates if tag in candidate.elements]
            use_hash = False
            if self.embedding_backend in HIDDEN_STATE_EMBEDDING_BACKENDS:
                hidden_dims = set()
                for candidate in present:
                    hidden = candidate.hidden_states
                    mask = candidate.masks[tag]
                    if (hidden is None or hidden.ndim != 2 or
                            hidden.shape[0] < len(mask) or not torch.isfinite(hidden).all()):
                        use_hash = True
                        break
                    hidden_dims.add(int(hidden.shape[1]))
                use_hash = use_hash or len(hidden_dims) > 1
                if use_hash and self.hidden_state_fallback == "error":
                    raise ValueError("Incomplete or inconsistent hidden-state representations in kernel group")
                if use_hash:
                    self.last_diagnostics.append({"row_key": row_key, "field": tag,
                        "fallback": "group_field_hashed_representation"})
            vectors: list[torch.Tensor | None] = [
                (self._embed(candidate.elements[tag].text) if use_hash
                 else self._embed_candidate_element(candidate, tag))
                if tag in candidate.elements else None
                for candidate in candidates
            ]
            vector_dim = next((int(vector.numel()) for vector in vectors if vector is not None), self.embedding_dim)
            vectors = [
                vector if vector is not None else torch.zeros(vector_dim, dtype=torch.float64)
                for vector in vectors
            ]
            matrix = torch.stack(vectors, dim=0)
            centered_matrix = matrix - matrix.mean(dim=0, keepdim=True)
            kernel = centered_matrix @ centered_matrix.T
            energy = float(torch.trace(kernel).item())
            energies[tag] = energy
            if energy < self.min_element_energy:
                kernel = torch.zeros_like(kernel)
                confidence[tag] = 0.0
            else:
                confidence[tag] = 1.0
            kernels[tag] = kernel

        k_total = sum(kernels.values(), torch.zeros((group_size, group_size), dtype=torch.float64))
        system = k_total + max(self.lambda_reg, 1e-12) * torch.eye(group_size, dtype=torch.float64)
        try:
            condition_number = float(torch.linalg.cond(system).item())
        except RuntimeError:
            condition_number = float("inf")
        if not math.isfinite(condition_number) or condition_number > self.condition_number_max:
            diagnostics = {
                "row_key": row_key,
                "fallback": "ill_conditioned_kernel",
                "condition_number": condition_number,
                "group_size": group_size,
                "energies": energies,
                "std": std,
            }
            self._assign_standard_image(
                candidates,
                token_advantages,
                loss_mask,
                completion_mask,
                fallback="ill_conditioned_kernel",
                diagnostics=diagnostics,
            )
            self.last_diagnostics.append(diagnostics)
            return

        alignments: dict[str, float] = {}
        active_tags = [tag for tag in self.element_tags if confidence.get(tag, 0.0) > 0.0]
        for left_index, left_tag in enumerate(active_tags):
            for right_tag in active_tags[left_index + 1 :]:
                alignment = _kernel_alignment(kernels[left_tag], kernels[right_tag])
                alignments[f"{left_tag}:{right_tag}"] = alignment
                if alignment >= self.high_kernel_alignment:
                    confidence[left_tag] *= self.alignment_downweight
                    confidence[right_tag] *= self.alignment_downweight

        try:
            omega = torch.linalg.solve(system, centered)
        except RuntimeError:
            diagnostics = {"row_key": row_key, "fallback": "kernel_solve_failed", "group_size": group_size, "std": std}
            self._assign_standard_image(
                candidates,
                token_advantages,
                loss_mask,
                completion_mask,
                fallback="kernel_solve_failed",
                diagnostics=diagnostics,
            )
            self.last_diagnostics.append(diagnostics)
            return

        contributions = {tag: kernels[tag] @ omega * float(confidence.get(tag, 0.0)) for tag in self.element_tags}
        denom = std + self.reward_epsilon
        standard_advantages = centered / denom
        for candidate_index, candidate in enumerate(candidates):
            total_tokens = int(sum(candidate.lengths.values()))
            if total_tokens <= 0:
                continue
            mapped_contribution = sum(
                float(contributions[tag][candidate_index].item())
                for tag in self.element_tags
                if candidate.lengths.get(tag, 0) > 0
            )
            residual = float(centered[candidate_index].item()) - mapped_contribution
            residual_advantage = residual / denom
            standard_advantage = float(standard_advantages[candidate_index].item())
            element_rows: dict[str, dict[str, Any]] = {}
            for tag in self.element_tags:
                length = int(candidate.lengths.get(tag, 0))
                if length <= 0:
                    continue
                element_advantage = float(contributions[tag][candidate_index].item()) / denom
                token_value = residual_advantage + (total_tokens * element_advantage / max(length, 1))
                mixed_value = (1.0 - self.eta) * standard_advantage + self.eta * token_value
                element_rows[tag] = {
                    "token_count": length,
                    "raw_contribution": float(contributions[tag][candidate_index].item()),
                    "element_advantage": element_advantage,
                    "token_value": token_value,
                    "mixed_token_advantage": mixed_value,
                    "energy": float(energies.get(tag, 0.0)),
                    "confidence": float(confidence.get(tag, 0.0)),
                }
                _apply_mask(
                    token_advantages,
                    loss_mask,
                    completion_mask,
                    candidate.row_index,
                    candidate.masks[tag],
                    mixed_value,
                )
            self.last_credit_rows.append(
                self._credit_row(
                    candidate,
                    group_size=group_size,
                    group_candidate_index=candidate_index,
                    fallback="",
                    std=std,
                    centered_image_reward=float(centered[candidate_index].item()),
                    standard_advantage=standard_advantage,
                    mapped_contribution=mapped_contribution,
                    residual=residual,
                    residual_advantage=residual_advantage,
                    condition_number=condition_number,
                    diagnostics={
                        "eta": self.eta,
                        "lambda_reg": self.lambda_reg,
                        "embedding_backend": self.embedding_backend,
                        "alignments": alignments,
                    },
                    elements=element_rows,
                    token_delta=self._token_delta(element_rows, float(candidate.reward.image_advantage)),
                )
            )

        self.last_diagnostics.append(
            {
                "row_key": row_key,
                "fallback": "",
                "condition_number": condition_number,
                "group_size": group_size,
                "eta": self.eta,
                "embedding_backend": self.embedding_backend,
                "energies": energies,
                "confidence": confidence,
                "alignments": alignments,
            }
        )

    def _append_standard_credit_row(
        self,
        candidate: ElementCandidate,
        *,
        fallback: str,
        diagnostics: dict[str, Any],
    ) -> None:
        elements = {
            tag: {
                "token_count": int(length),
                "raw_contribution": 0.0,
                "element_advantage": float(candidate.reward.image_advantage),
                "token_value": float(candidate.reward.image_advantage),
                "mixed_token_advantage": float(candidate.reward.image_advantage),
                "energy": float(diagnostics.get("energies", {}).get(tag, 0.0))
                if isinstance(diagnostics.get("energies"), dict)
                else 0.0,
                "confidence": 0.0,
            }
            for tag, length in candidate.lengths.items()
            if int(length) > 0
        }
        self.last_credit_rows.append(
            self._credit_row(
                candidate,
                group_size=int(diagnostics.get("group_size", 1) or 1),
                group_candidate_index=-1,
                fallback=fallback,
                std=float(diagnostics.get("std", 0.0) or 0.0),
                centered_image_reward=0.0,
                standard_advantage=float(candidate.reward.image_advantage),
                mapped_contribution=0.0,
                residual=0.0,
                residual_advantage=0.0,
                condition_number=diagnostics.get("condition_number"),
                diagnostics={
                    "eta": self.eta,
                    "lambda_reg": self.lambda_reg,
                    "embedding_backend": self.embedding_backend,
                },
                elements=elements,
                token_delta=self._token_delta(elements, float(candidate.reward.image_advantage)),
            )
        )

    def _token_delta(self, elements: dict[str, dict[str, Any]], decoupled_value: float) -> dict[str, Any]:
        total_tokens = 0
        changed_tokens = 0
        weighted_abs = 0.0
        max_abs = 0.0
        signed_sum = 0.0
        by_tag: dict[str, dict[str, Any]] = {}
        for tag, row in elements.items():
            count = int(row.get("token_count", 0) or 0)
            if count <= 0:
                continue
            value = float(row.get("mixed_token_advantage", decoupled_value))
            delta = value - float(decoupled_value)
            abs_delta = abs(delta)
            changed = count if abs_delta > self.token_delta_epsilon else 0
            total_tokens += count
            changed_tokens += changed
            weighted_abs += abs_delta * count
            signed_sum += delta * count
            max_abs = max(max_abs, abs_delta)
            by_tag[tag] = {
                "token_count": count,
                "decoupled_token_advantage": float(decoupled_value),
                "kernel_token_advantage": value,
                "delta": delta,
                "abs_delta": abs_delta,
                "changed": bool(changed),
            }
        return {
            "eligible_tokens": total_tokens,
            "changed_tokens": changed_tokens,
            "fraction_changed": float(changed_tokens / total_tokens) if total_tokens else 0.0,
            "mean_abs_vs_decoupled": float(weighted_abs / total_tokens) if total_tokens else 0.0,
            "mean_signed_vs_decoupled": float(signed_sum / total_tokens) if total_tokens else 0.0,
            "max_abs_vs_decoupled": float(max_abs),
            "epsilon": float(self.token_delta_epsilon),
            "by_tag": by_tag,
        }

    def _credit_row(
        self,
        candidate: ElementCandidate,
        *,
        group_size: int,
        group_candidate_index: int,
        fallback: str,
        std: float,
        centered_image_reward: float,
        standard_advantage: float,
        mapped_contribution: float,
        residual: float,
        residual_advantage: float,
        condition_number: float | None,
        diagnostics: dict[str, Any],
        elements: dict[str, dict[str, Any]],
        token_delta: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "request_id": candidate.reward.request_id,
            "row_key": candidate.reward.row_key,
            "sample_index": candidate.reward.sample_index,
            "row_index": candidate.row_index,
            "group_size": group_size,
            "group_candidate_index": group_candidate_index,
            "fallback": fallback,
            "image_reward": float(candidate.reward.image_reward),
            "image_advantage": float(candidate.reward.image_advantage),
            "std": float(std),
            "centered_image_reward": float(centered_image_reward),
            "standard_advantage": float(standard_advantage),
            "mapped_contribution": float(mapped_contribution),
            "residual": float(residual),
            "residual_advantage": float(residual_advantage),
            "group_reward_collapse": {
                "image_reward_std": float(std),
                "image_reward_zero_std": bool(std <= self.reward_epsilon),
                "fallback": str(fallback or ""),
                "reward_epsilon": float(self.reward_epsilon),
            },
            "condition_number": condition_number,
            "diagnostics": diagnostics,
            "elements": elements,
            "token_delta": token_delta,
        }

    def _write_kernel_credit_audit(self) -> None:
        if not self.save_kernel_credit or self.kernel_credit_dir is None or not self.last_credit_rows:
            return
        first_request = str(self.last_credit_rows[0].get("request_id") or "")
        match = re.search(r"step-(\d+).*?rank-(\d+).*?call-(\d+)", first_request)
        if match:
            step, rank, call = match.groups()
            filename = f"kernel_credit_step-{step}_rank-{rank}_call-{call}.jsonl"
        else:
            digest = hashlib.sha256(first_request.encode("utf-8")).hexdigest()[:12]
            filename = f"kernel_credit_{digest}.jsonl"
        self.kernel_credit_dir.mkdir(parents=True, exist_ok=True)
        path = self.kernel_credit_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in self.last_credit_rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
