from __future__ import annotations

import logging
import hashlib
import time
from contextlib import contextmanager
from typing import Any

import torch

from utils.advantage_protocol import AdvantageContext, AdvantageProvider
from utils.gpu_memory import log_cuda_memory


LOG = logging.getLogger(__name__)


def _tensor_fingerprint(tensor: torch.Tensor, *, max_values: int = 4096) -> str:
    flat = tensor.detach().reshape(-1)
    if flat.numel() > max_values:
        flat = flat[:max_values]
    payload = flat.float().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _trainable_parameter_summary(model: Any, *, max_tensors: int = 32, max_values_per_tensor: int = 4096) -> dict[str, Any]:
    pieces: list[str] = []
    total_params = 0
    trainable_tensors = 0
    l1_sum = 0.0
    value_sum = 0.0
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        trainable_tensors += 1
        total_params += int(param.numel())
        sample = param.detach().reshape(-1)
        if sample.numel() > max_values_per_tensor:
            sample = sample[:max_values_per_tensor]
        sample = sample.float()
        l1_sum += float(sample.abs().sum().item())
        value_sum += float(sample.sum().item())
        if len(pieces) < max_tensors:
            pieces.append(f"{name}:{_tensor_fingerprint(sample, max_values=max_values_per_tensor)}")
    digest = hashlib.sha256("|".join(pieces).encode("utf-8")).hexdigest()[:16]
    return {
        "digest": digest,
        "trainable_tensors": trainable_tensors,
        "trainable_params": total_params,
        "sampled_value_sum": value_sum,
        "sampled_l1_sum": l1_sum,
    }


def _completion_fingerprint(output: dict[str, Any]) -> str:
    completion_ids = output.get("completion_ids")
    completion_mask = output.get("completion_mask")
    if not torch.is_tensor(completion_ids):
        return "missing"
    ids = completion_ids.detach().cpu()
    if torch.is_tensor(completion_mask):
        mask = completion_mask.detach().to(dtype=ids.dtype).cpu()
        ids = ids * mask
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:16]


def trl_major_minor() -> tuple[int, int]:
    import trl

    version = str(getattr(trl, "__version__", ""))
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return -1, -1


def require_supported_trl_024() -> None:
    import trl

    version = str(getattr(trl, "__version__", ""))
    if trl_major_minor() != (0, 24):
        raise RuntimeError(
            f"Token-advantage extension is pinned to trl==0.24.x, found {version!r}. "
            "Version 0 uniform GRPO may work, but token-level extensions must be reviewed before upgrading."
        )


def trl_supports_token_advantages_in_base_loss() -> bool:
    """TRL 1.6+ accepts `advantages` with shape (batch, completion_length)."""
    return trl_major_minor() >= (1, 6)


def make_extensible_grpo_trainer_class():
    from trl import GRPOTrainer

    class ExtensibleGRPOTrainer(GRPOTrainer):
        """Stock TRL GRPO plus an optional token-advantage transformation.

        When the provider returns `None`, Version 0 delegates loss computation
        entirely to TRL. TRL 1.6+ accepts token-level `advantages` directly in
        the base loss. TRL 0.24 still uses the older local clipped-GRPO loss.
        """

        def __init__(self, *args: Any, advantage_provider: AdvantageProvider, **kwargs: Any) -> None:
            self.advantage_provider = advantage_provider
            self._timing_phase = "init"
            super().__init__(*args, **kwargs)

        @contextmanager
        def _timing_phase_scope(self, phase: str):
            previous = getattr(self, "_timing_phase", "unknown")
            self._timing_phase = phase
            try:
                yield
            finally:
                self._timing_phase = previous

        def training_step(self, *args: Any, **kwargs: Any):
            start = time.monotonic()
            log_cuda_memory("before_training_step", logger=LOG)
            before_summary = _trainable_parameter_summary(self.model)
            LOG.info(
                "policy_trainable_checksum: phase=before_training_step step=%s digest=%s "
                "trainable_tensors=%d trainable_params=%d sampled_value_sum=%.6f sampled_l1_sum=%.6f",
                getattr(self.state, "global_step", "unknown"),
                before_summary["digest"],
                before_summary["trainable_tensors"],
                before_summary["trainable_params"],
                before_summary["sampled_value_sum"],
                before_summary["sampled_l1_sum"],
            )
            with self._timing_phase_scope("training_step"):
                result = super().training_step(*args, **kwargs)
            LOG.info("trl_grpo training_step timing: seconds=%.3f", time.monotonic() - start)
            after_summary = _trainable_parameter_summary(self.model)
            LOG.info(
                "policy_trainable_checksum: phase=after_training_step step=%s digest=%s "
                "trainable_tensors=%d trainable_params=%d sampled_value_sum=%.6f sampled_l1_sum=%.6f changed=%s",
                getattr(self.state, "global_step", "unknown"),
                after_summary["digest"],
                after_summary["trainable_tensors"],
                after_summary["trainable_params"],
                after_summary["sampled_value_sum"],
                after_summary["sampled_l1_sum"],
                after_summary["digest"] != before_summary["digest"],
            )
            log_cuda_memory("after_training_step", logger=LOG)
            return result

        def _prepare_inputs(self, inputs: Any):
            start = time.monotonic()
            with self._timing_phase_scope("prepare_inputs"):
                output = super()._prepare_inputs(inputs)
            LOG.info("trl_grpo prepare_inputs timing: seconds=%.3f", time.monotonic() - start)
            return output

        def compute_loss(self, *args: Any, **kwargs: Any):
            start = time.monotonic()
            log_cuda_memory("before_compute_loss", logger=LOG)
            inputs = args[1] if len(args) >= 2 and isinstance(args[1], dict) else kwargs.get("inputs")
            self._debug_sampling_per_token_logps = None
            self._debug_completion_mask = None
            if isinstance(inputs, dict):
                sampling_logps = inputs.get("sampling_per_token_logps")
                completion_mask = inputs.get("completion_mask")
                if torch.is_tensor(sampling_logps) and torch.is_tensor(completion_mask):
                    self._debug_sampling_per_token_logps = sampling_logps.detach()
                    self._debug_completion_mask = completion_mask.detach()
            with self._timing_phase_scope("compute_loss"):
                result = super().compute_loss(*args, **kwargs)
            self._debug_sampling_per_token_logps = None
            self._debug_completion_mask = None
            LOG.info("trl_grpo compute_loss timing: seconds=%.3f", time.monotonic() - start)
            log_cuda_memory("after_compute_loss", logger=LOG)
            return result

        def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any):
            start = time.monotonic()
            phase = str(getattr(self, "_timing_phase", "unknown"))
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) >= 2:
                input_ids = args[1]
            logits_to_keep = kwargs.get("logits_to_keep")
            if logits_to_keep is None and len(args) >= 4:
                logits_to_keep = args[3]
            compute_entropy = kwargs.get("compute_entropy", None)
            if hasattr(input_ids, "shape"):
                input_shape = tuple(int(dim) for dim in input_ids.shape)
            else:
                input_shape = None
            result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            self._log_vllm_sampling_logp_diff(result, phase)
            LOG.info(
                "trl_grpo logprob timing: phase=%s seconds=%.3f input_shape=%s logits_to_keep=%s compute_entropy=%s",
                phase,
                time.monotonic() - start,
                input_shape,
                logits_to_keep,
                compute_entropy,
            )
            return result

        def _log_vllm_sampling_logp_diff(self, result: Any, phase: str) -> None:
            if phase != "compute_loss":
                return
            sampling_logps = getattr(self, "_debug_sampling_per_token_logps", None)
            completion_mask = getattr(self, "_debug_completion_mask", None)
            if not torch.is_tensor(sampling_logps) or not torch.is_tensor(completion_mask):
                return
            current_logps = result[0] if isinstance(result, tuple) else result
            if not torch.is_tensor(current_logps) or current_logps.shape != sampling_logps.shape:
                LOG.info(
                    "vllm_policy_logp_diff: skipped shape_mismatch current_shape=%s sampling_shape=%s",
                    tuple(current_logps.shape) if torch.is_tensor(current_logps) else None,
                    tuple(sampling_logps.shape),
                )
                return
            valid = completion_mask.to(device=current_logps.device).bool()
            if not valid.any():
                return
            sampling = sampling_logps.to(device=current_logps.device, dtype=current_logps.dtype)
            diff = (current_logps.detach() - sampling).abs()[valid]
            LOG.info(
                "vllm_policy_logp_diff: step=%s mean=%.6f max=%.6f p95=%.6f tokens=%d",
                getattr(self.state, "global_step", "unknown"),
                float(diff.mean().item()),
                float(diff.max().item()),
                float(torch.quantile(diff.float(), 0.95).item()) if diff.numel() > 1 else float(diff.max().item()),
                int(diff.numel()),
            )

        def _generate(self, *args: Any, **kwargs: Any):
            start = time.monotonic()
            log_cuda_memory("before_trl_generate", logger=LOG)
            with self._timing_phase_scope("generate"):
                result = super()._generate(*args, **kwargs)
            log_cuda_memory("after_trl_generate", logger=LOG)
            LOG.info("trl_grpo generation timing: seconds=%.3f", time.monotonic() - start)
            return result

        def _generate_and_score_completions(self, inputs):
            total_start = time.monotonic()
            log_cuda_memory("before_generate_and_score", logger=LOG)
            summary = _trainable_parameter_summary(self.model)
            LOG.info(
                "policy_trainable_checksum: phase=before_generate step=%s digest=%s "
                "trainable_tensors=%d trainable_params=%d sampled_value_sum=%.6f sampled_l1_sum=%.6f",
                getattr(self.state, "global_step", "unknown"),
                summary["digest"],
                summary["trainable_tensors"],
                summary["trainable_params"],
                summary["sampled_value_sum"],
                summary["sampled_l1_sum"],
            )
            with self._timing_phase_scope("generate_score_super"):
                output = super()._generate_and_score_completions(inputs)
            super_seconds = time.monotonic() - total_start
            log_cuda_memory("after_generate_score_super", logger=LOG)
            log_cuda_memory("after_reward_score", logger=LOG)
            generate_score_seconds = time.monotonic() - total_start
            completion_ids = output.get("completion_ids")
            completion_mask = output.get("completion_mask")
            LOG.info(
                "trl_grpo generate_score_super timing: seconds=%.3f keys=%s samples=%s completion_tokens=%s "
                "has_old_logps=%s has_ref_logps=%s has_advantages=%s",
                super_seconds,
                sorted(str(key) for key in output.keys()),
                int(completion_ids.shape[0]) if hasattr(completion_ids, "shape") else "unknown",
                int(completion_mask.sum().item()) if hasattr(completion_mask, "sum") else "unknown",
                "old_per_token_logps" in output,
                "ref_per_token_logps" in output,
                "advantages" in output,
            )
            LOG.info(
                "rollout_completion_fingerprint: step=%s fingerprint=%s samples=%s completion_tokens=%s "
                "has_sampling_logps=%s",
                getattr(self.state, "global_step", "unknown"),
                _completion_fingerprint(output),
                int(completion_ids.shape[0]) if hasattr(completion_ids, "shape") else "unknown",
                int(completion_mask.sum().item()) if hasattr(completion_mask, "sum") else "unknown",
                "sampling_per_token_logps" in output,
            )
            advantage_start = time.monotonic()
            sequence_advantages = output["advantages"]
            hidden_state_start = time.monotonic()
            hidden_states = self._maybe_completion_hidden_states(output)
            if torch.is_tensor(hidden_states):
                LOG.info(
                    "token_credit_hidden_states: seconds=%.3f shape=%s dtype=%s backend=%s",
                    time.monotonic() - hidden_state_start,
                    tuple(int(dim) for dim in hidden_states.shape),
                    hidden_states.dtype,
                    getattr(self.advantage_provider, "embedding_backend", ""),
                )
            token_advantages = self.advantage_provider.token_advantages(
                AdvantageContext(
                    completion_ids=output["completion_ids"],
                    completion_mask=output["completion_mask"],
                    sequence_advantages=sequence_advantages,
                    tokenizer=self.processing_class.tokenizer
                    if hasattr(self.processing_class, "tokenizer")
                    else self.processing_class,
                    hidden_states=hidden_states,
                )
            )
            advantage_seconds = time.monotonic() - advantage_start
            if token_advantages is not None:
                if token_advantages.shape != output["completion_mask"].shape:
                    raise ValueError(
                        f"Token advantages shape {tuple(token_advantages.shape)} must match "
                        f"completion mask {tuple(output['completion_mask'].shape)}"
                    )
                completion_mask = output["completion_mask"]
                token_advantages = token_advantages * completion_mask
                token_loss_mask = getattr(self.advantage_provider, "last_loss_mask", None)
                if token_loss_mask is not None:
                    if token_loss_mask.shape != completion_mask.shape:
                        raise ValueError(
                            f"Token loss mask shape {tuple(token_loss_mask.shape)} must match "
                            f"completion mask {tuple(completion_mask.shape)}"
                        )
                    token_loss_mask = token_loss_mask.to(device=completion_mask.device) * completion_mask
                self._log_token_credit_debug(
                    sequence_advantages=sequence_advantages,
                    token_advantages=token_advantages,
                    completion_mask=completion_mask,
                    token_loss_mask=token_loss_mask,
                )

                if trl_supports_token_advantages_in_base_loss():
                    output["advantages"] = token_advantages
                    if token_loss_mask is not None:
                        existing_tool_mask = output.get("tool_mask")
                        if existing_tool_mask is not None:
                            output["tool_mask"] = existing_tool_mask.to(
                                device=token_loss_mask.device,
                                dtype=token_loss_mask.dtype,
                            ) * token_loss_mask
                        else:
                            output["tool_mask"] = token_loss_mask
                else:
                    require_supported_trl_024()
                    output["token_advantages"] = token_advantages
                    if token_loss_mask is not None:
                        output["token_loss_mask"] = token_loss_mask
            LOG.info(
                "trl_grpo generate_score timing: samples=%d completion_tokens=%d generate_score=%.3fs token_advantage=%.3fs",
                int(output["completion_ids"].shape[0]),
                int(output["completion_mask"].sum().item()),
                generate_score_seconds,
                advantage_seconds,
            )
            log_cuda_memory("after_token_advantage", logger=LOG)
            return output

        def _maybe_completion_hidden_states(self, output: dict[str, Any]) -> torch.Tensor | None:
            backend = str(getattr(self.advantage_provider, "embedding_backend", "") or "").lower()
            if backend not in {"hidden_states", "policy_hidden_mean", "policy_hidden_states"}:
                return None
            completion_ids = output.get("completion_ids")
            completion_mask = output.get("completion_mask")
            if not torch.is_tensor(completion_ids) or not torch.is_tensor(completion_mask):
                return None
            cfg = getattr(self.advantage_provider, "config", {}) or {}
            element_cfg = cfg.get("element_credit", {}) if isinstance(cfg, dict) else {}
            if not isinstance(element_cfg, dict):
                element_cfg = {}
            chunk_size = max(1, int(element_cfg.get("hidden_state_batch_size", 1) or 1))
            backbone = self._hidden_state_backbone()
            was_training = bool(getattr(backbone, "training", False))
            backbone.eval()
            rows: list[torch.Tensor] = []
            try:
                with torch.no_grad():
                    for start in range(0, int(completion_ids.shape[0]), chunk_size):
                        end = min(start + chunk_size, int(completion_ids.shape[0]))
                        chunk_ids = completion_ids[start:end]
                        chunk_mask = completion_mask[start:end]
                        try:
                            outputs = backbone(
                                input_ids=chunk_ids,
                                attention_mask=chunk_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                        except TypeError:
                            outputs = backbone(
                                input_ids=chunk_ids,
                                attention_mask=chunk_mask,
                                output_hidden_states=True,
                                use_cache=False,
                                return_dict=True,
                            )
                        hidden = getattr(outputs, "last_hidden_state", None)
                        if hidden is None:
                            hidden_states = getattr(outputs, "hidden_states", None)
                            if hidden_states:
                                hidden = hidden_states[-1]
                        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
                            hidden = outputs[0]
                        if not torch.is_tensor(hidden):
                            LOG.warning("token_credit_hidden_states: failed_no_hidden_output backend=%s", backend)
                            return None
                        rows.append(hidden.detach().to(device="cpu", dtype=torch.float32))
            finally:
                if was_training:
                    backbone.train()
            return torch.cat(rows, dim=0) if rows else None

        def _hidden_state_backbone(self):
            model = self.model
            if hasattr(model, "get_base_model"):
                try:
                    model = model.get_base_model()
                except Exception:
                    pass
            backbone = getattr(model, "model", None)
            return backbone if backbone is not None else model

        def _log_token_credit_debug(
            self,
            *,
            sequence_advantages: torch.Tensor,
            token_advantages: torch.Tensor,
            completion_mask: torch.Tensor,
            token_loss_mask: torch.Tensor | None,
        ) -> None:
            completion_mask_bool = completion_mask.bool()
            if token_loss_mask is None:
                loss_mask_bool = completion_mask_bool
            else:
                loss_mask_bool = token_loss_mask.bool()
            active_rows = loss_mask_bool.any(dim=1)
            valid_advantages = token_advantages[loss_mask_bool]
            completion_tokens_per_row = completion_mask.sum(dim=1).detach().float()
            loss_tokens_per_row = loss_mask_bool.sum(dim=1).detach().float()

            seq = sequence_advantages.detach().float()
            if seq.numel():
                seq_mean = float(seq.mean().item())
                seq_std = float(seq.std(unbiased=False).item()) if seq.numel() > 1 else 0.0
                seq_min = float(seq.min().item())
                seq_max = float(seq.max().item())
            else:
                seq_mean = seq_std = seq_min = seq_max = 0.0

            if valid_advantages.numel():
                adv = valid_advantages.detach().float()
                adv_mean = float(adv.mean().item())
                adv_abs_mean = float(adv.abs().mean().item())
                adv_min = float(adv.min().item())
                adv_max = float(adv.max().item())
            else:
                adv_mean = adv_abs_mean = adv_min = adv_max = 0.0

            LOG.info(
                "token_credit_debug: samples=%d active_rows=%d completion_tokens=%d "
                "loss_tokens=%d loss_token_ratio=%.6f sequence_adv_mean=%.6f "
                "sequence_adv_std=%.6f sequence_adv_min=%.6f sequence_adv_max=%.6f "
                "token_adv_mean=%.6f token_adv_abs_mean=%.6f token_adv_min=%.6f token_adv_max=%.6f "
                "completion_tokens_per_row_min=%.1f completion_tokens_per_row_max=%.1f "
                "loss_tokens_per_row_min=%.1f loss_tokens_per_row_max=%.1f",
                int(completion_mask.shape[0]),
                int(active_rows.sum().item()),
                int(completion_mask.sum().item()),
                int(loss_mask_bool.sum().item()),
                float(loss_mask_bool.float().mean().item()) if loss_mask_bool.numel() else 0.0,
                seq_mean,
                seq_std,
                seq_min,
                seq_max,
                adv_mean,
                adv_abs_mean,
                adv_min,
                adv_max,
                float(completion_tokens_per_row.min().item()) if completion_tokens_per_row.numel() else 0.0,
                float(completion_tokens_per_row.max().item()) if completion_tokens_per_row.numel() else 0.0,
                float(loss_tokens_per_row.min().item()) if loss_tokens_per_row.numel() else 0.0,
                float(loss_tokens_per_row.max().item()) if loss_tokens_per_row.numel() else 0.0,
            )

        def _compute_loss(self, model, inputs):
            if "token_advantages" not in inputs:
                return super()._compute_loss(model, inputs)
            if self.use_liger_loss:
                raise NotImplementedError("Token-level advantages are not compatible with Liger loss")

            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            loss_mask = inputs.get("token_loss_mask", completion_mask).to(completion_mask.dtype)
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )
            token_advantages = inputs["token_advantages"].to(per_token_logps.dtype)
            old_per_token_logps = inputs.get("old_per_token_logps")
            old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
            log_ratio = per_token_logps - old_per_token_logps
            if self.importance_sampling_level == "token":
                log_importance_weights = log_ratio
            elif self.importance_sampling_level == "sequence":
                log_importance_weights = (
                    (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                ).unsqueeze(-1)
            else:
                raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

            ratio = torch.exp(log_importance_weights)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if ratio.shape[1] == 1:
                ratio = ratio.expand_as(token_advantages)
                clipped_ratio = clipped_ratio.expand_as(token_advantages)
            per_token_loss = -torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
            pre_vllm_correction_loss = per_token_loss
            if self.beta != 0.0:
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
                    ref_per_token_logps - per_token_logps
                ) - 1
                per_token_loss = per_token_loss + self.beta * per_token_kl
            if self.use_vllm and self.vllm_importance_sampling_correction:
                per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

            masked_loss = per_token_loss * loss_mask
            if self.loss_type == "grpo":
                loss = (masked_loss.sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "bnpo":
                loss = masked_loss.sum() / loss_mask.sum().clamp(min=1.0)
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "dr_grpo":
                loss = masked_loss.sum() / (masked_loss.size(0) * self.max_completion_length)
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "dapo":
                normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
                loss = masked_loss.sum() / normalizer
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            mode = "train" if self.model.training else "eval"
            valid = loss_mask.bool()
            if valid.any():
                self._metrics[mode]["token_advantage/mean"].append(token_advantages[valid].mean().item())
                self._metrics[mode]["token_advantage/abs_mean"].append(token_advantages[valid].abs().mean().item())
                self._metrics[mode]["token_loss_mask/active_tokens"].append(valid.float().sum().item())
                self._metrics[mode]["token_loss_mask/active_ratio"].append(valid.float().mean().item())
                self._metrics[mode]["policy_ratio/mean"].append(ratio[valid].mean().item())
                self._metrics[mode]["policy_ratio/min"].append(ratio[valid].min().item())
                self._metrics[mode]["policy_ratio/max"].append(ratio[valid].max().item())
                if "importance_sampling_ratio" in inputs:
                    is_ratio = inputs["importance_sampling_ratio"]
                    if is_ratio.shape[1] == 1:
                        is_ratio = is_ratio.expand_as(loss_mask)
                    self._metrics[mode]["vllm_importance_sampling_ratio/loss_region_mean"].append(
                        is_ratio[valid].float().mean().item()
                    )
                self._metrics[mode]["pre_vllm_correction_loss/mean"].append(pre_vllm_correction_loss[valid].mean().item())
                self._metrics[mode]["entropy"].append(entropies[valid].mean().item())
                clipped = (ratio != clipped_ratio) & valid
                self._metrics[mode]["clip_ratio/region_mean"].append(clipped.float().sum().div(valid.sum()).item())
            return loss

    return ExtensibleGRPOTrainer
