from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class SchemaTokenRows:
    token_ids: list[int]
    tokens: list[str]
    input_rows: torch.Tensor
    output_rows: torch.Tensor | None = None


def schema_token_ids(tokenizer: Any, tokens: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Token {token!r} is not in the tokenizer vocabulary")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None and token_id == unk_id:
            raise ValueError(f"Token {token!r} resolved to unk_token_id; add it as a special token first")
        ids.append(int(token_id))
    if len(ids) != len(set(ids)):
        raise ValueError("Schema token ids contain duplicates")
    return ids


def schema_subtoken_ids(tokenizer: Any, tokens: Iterable[str]) -> dict[str, list[int]]:
    """Tokenize schema strings before they are added as special tokens.

    For a new tag like `<headline>`, this usually returns the old-tokenizer
    decomposition such as ids for `<`, `headline`, and `>`. Those rows are a
    better initialization source than random or global-vocab mean rows.
    """

    by_token: dict[str, list[int]] = {}
    for token in tokens:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(token, add_special_tokens=False)
        else:
            encoded = tokenizer(token, add_special_tokens=False)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
        by_token[token] = [int(token_id) for token_id in ids if int(token_id) >= 0]
    return by_token


def are_input_output_embeddings_tied(model: Any) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return False
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    return bool(input_weight is not None and output_weight is not None and input_weight.data_ptr() == output_weight.data_ptr())


def initialize_token_rows_from_mean(model: Any, token_ids: Iterable[int]) -> None:
    """Initialize newly-added rows to the mean old row instead of random noise.

    Call this immediately after `resize_token_embeddings`. The token rows are still
    trainable later; this only gives schema tags a sane starting point.
    """

    token_ids = sorted({int(token_id) for token_id in token_ids})
    if not token_ids:
        return
    first_new_id = min(token_ids)
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        input_weight = input_embeddings.weight
        old_input_rows = input_weight[:first_new_id]
        input_weight[token_ids] = old_input_rows.mean(dim=0, keepdim=True).to(input_weight.dtype)

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            return
        output_weight = output_embeddings.weight
        if output_weight.data_ptr() == input_weight.data_ptr():
            return
        old_output_rows = output_weight[:first_new_id]
        output_weight[token_ids] = old_output_rows.mean(dim=0, keepdim=True).to(output_weight.dtype)


def initialize_token_rows_from_subtokens(
    model: Any,
    token_ids: Iterable[int],
    token_to_subtoken_ids: dict[str, list[int]],
    tokens: list[str],
) -> None:
    """Initialize new token rows from their old-tokenizer subtoken rows.

    This is useful for XML-like tags because the base model has usually seen
    the pieces (`<`, `/`, words, `_`, `>`) even if the full tag is newly added.
    Untied `lm_head` rows can use the same averaging idea on output rows.
    """

    ids = [int(token_id) for token_id in token_ids]
    if len(ids) != len(tokens):
        raise ValueError("token_ids and tokens must have the same length")
    if not ids:
        return
    new_id_set = set(ids)

    def source_ids_for(weight: torch.Tensor, token: str) -> list[int]:
        return [
            int(source_id)
            for source_id in token_to_subtoken_ids.get(token, [])
            if 0 <= int(source_id) < int(weight.shape[0]) and int(source_id) not in new_id_set
        ]

    with torch.no_grad():
        input_weight = model.get_input_embeddings().weight
        input_fallback = input_weight[: min(ids)].mean(dim=0, keepdim=True).to(input_weight.dtype)
        for token_id, token in zip(ids, tokens):
            sources = source_ids_for(input_weight, token)
            if sources:
                input_weight[token_id] = input_weight[sources].mean(dim=0).to(input_weight.dtype)
            else:
                input_weight[token_id] = input_fallback

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            return
        output_weight = output_embeddings.weight
        if output_weight.data_ptr() == input_weight.data_ptr():
            return
        output_fallback = output_weight[: min(ids)].mean(dim=0, keepdim=True).to(output_weight.dtype)
        for token_id, token in zip(ids, tokens):
            sources = source_ids_for(output_weight, token)
            if sources:
                output_weight[token_id] = output_weight[sources].mean(dim=0).to(output_weight.dtype)
            else:
                output_weight[token_id] = output_fallback


class TrainableTokenRowEmbedding(nn.Module):
    """Freeze an existing embedding and train delta rows for selected token ids."""

    def __init__(self, base_embedding: nn.Module, token_ids: Iterable[int]) -> None:
        super().__init__()
        if not hasattr(base_embedding, "weight"):
            raise TypeError("base_embedding must expose a weight tensor")
        self.base_embedding = base_embedding
        self.base_embedding.weight.requires_grad_(False)
        ids = sorted({int(token_id) for token_id in token_ids})
        if not ids:
            raise ValueError("token_ids must not be empty")
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long), persistent=True)
        vocab_size = int(base_embedding.weight.shape[0])
        mapping = torch.full((vocab_size,), -1, dtype=torch.long)
        mapping[self.token_ids] = torch.arange(len(ids), dtype=torch.long)
        self.register_buffer("token_id_to_row", mapping, persistent=True)
        self.delta = nn.Embedding(len(ids), int(base_embedding.weight.shape[1]))
        nn.init.zeros_(self.delta.weight)
        self.delta.to(device=base_embedding.weight.device, dtype=base_embedding.weight.dtype)

    @property
    def weight(self) -> torch.Tensor:
        weight = self.base_embedding.weight
        if not torch.is_grad_enabled():
            return self.effective_weight()
        return weight

    @property
    def num_embeddings(self) -> int:
        return int(self.base_embedding.weight.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.base_embedding.weight.shape[1])

    @property
    def padding_idx(self) -> int | None:
        return getattr(self.base_embedding, "padding_idx", None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.base_embedding(input_ids)
        row_ids = self.token_id_to_row.to(input_ids.device)[input_ids]
        mask = row_ids.ge(0)
        if mask.any():
            safe_row_ids = row_ids.clamp_min(0)
            if self.delta.weight.device != input_ids.device:
                self.delta.to(device=input_ids.device)
            delta = self.delta(safe_row_ids)
            delta = delta.to(device=output.device, dtype=output.dtype)
            output = output + delta * mask.unsqueeze(-1).to(delta.dtype)
        return output

    def effective_rows(self) -> torch.Tensor:
        rows = self.base_embedding.weight.detach()[self.token_ids.to(self.base_embedding.weight.device)].clone()
        return rows + self.delta.weight.detach().to(rows.device, rows.dtype)

    def effective_weight(self) -> torch.Tensor:
        weight = self.base_embedding.weight.detach().clone()
        weight[self.token_ids.to(weight.device)] = self.effective_rows().to(weight.device, weight.dtype)
        return weight


class TrainableTokenRowLmHead(nn.Module):
    """Freeze an existing LM head and train additive output rows for selected tokens."""

    def __init__(self, base_lm_head: nn.Module, token_ids: Iterable[int]) -> None:
        super().__init__()
        if not hasattr(base_lm_head, "weight"):
            raise TypeError("base_lm_head must expose a weight tensor")
        self.base_lm_head = base_lm_head
        self.base_lm_head.weight.requires_grad_(False)
        if getattr(self.base_lm_head, "bias", None) is not None:
            self.base_lm_head.bias.requires_grad_(False)
        ids = sorted({int(token_id) for token_id in token_ids})
        if not ids:
            raise ValueError("token_ids must not be empty")
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long), persistent=True)
        self.delta_weight = nn.Parameter(
            torch.zeros(
                len(ids),
                int(base_lm_head.weight.shape[1]),
                device=base_lm_head.weight.device,
                dtype=base_lm_head.weight.dtype,
            )
        )
        if getattr(base_lm_head, "bias", None) is not None:
            self.delta_bias = nn.Parameter(torch.zeros(len(ids), device=base_lm_head.weight.device, dtype=base_lm_head.weight.dtype))
        else:
            self.delta_bias = None

    @property
    def weight(self) -> torch.Tensor:
        weight = self.base_lm_head.weight
        if not torch.is_grad_enabled():
            return self.effective_weight()
        return weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base_lm_head, "bias", None)

    @property
    def in_features(self) -> int:
        return int(self.base_lm_head.weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.base_lm_head.weight.shape[0])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base_lm_head(hidden_states)
        delta_weight = self.delta_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        delta_logits = torch.matmul(hidden_states, delta_weight.t())
        if self.delta_bias is not None:
            delta_logits = delta_logits + self.delta_bias.to(device=delta_logits.device, dtype=delta_logits.dtype)
        index = self.token_ids.to(logits.device).view(*([1] * (logits.ndim - 1)), -1)
        index = index.expand(*logits.shape[:-1], -1)
        return logits.scatter_add(-1, index, delta_logits.to(logits.dtype))

    def effective_rows(self) -> torch.Tensor:
        rows = self.base_lm_head.weight.detach()[self.token_ids.to(self.base_lm_head.weight.device)].clone()
        return rows + self.delta_weight.detach().to(rows.device, rows.dtype)

    def effective_weight(self) -> torch.Tensor:
        weight = self.base_lm_head.weight.detach().clone()
        weight[self.token_ids.to(weight.device)] = self.effective_rows().to(weight.device, weight.dtype)
        return weight


def enable_schema_token_row_training(model: Any, token_ids: Iterable[int], *, train_lm_head: bool = True) -> None:
    ids = sorted({int(token_id) for token_id in token_ids})
    if not ids:
        raise ValueError("token_ids must not be empty")
    input_embeddings = model.get_input_embeddings()
    if not isinstance(input_embeddings, TrainableTokenRowEmbedding):
        model.set_input_embeddings(TrainableTokenRowEmbedding(input_embeddings, ids))

    output_embeddings = model.get_output_embeddings()
    if train_lm_head and output_embeddings is not None and not are_input_output_embeddings_tied(model):
        if not isinstance(output_embeddings, TrainableTokenRowLmHead):
            model.set_output_embeddings(TrainableTokenRowLmHead(output_embeddings, ids))


def collect_schema_token_rows(model: Any, tokens: list[str]) -> SchemaTokenRows:
    input_embeddings = model.get_input_embeddings()
    if not isinstance(input_embeddings, TrainableTokenRowEmbedding):
        raise TypeError("Input embeddings are not wrapped for schema token row training")
    output_embeddings = model.get_output_embeddings()
    output_rows = output_embeddings.effective_rows().detach().cpu() if isinstance(output_embeddings, TrainableTokenRowLmHead) else None
    return SchemaTokenRows(
        token_ids=input_embeddings.token_ids.detach().cpu().tolist(),
        tokens=tokens,
        input_rows=input_embeddings.effective_rows().detach().cpu(),
        output_rows=output_rows,
    )


def save_schema_token_rows(model: Any, tokens: list[str], path: str | Path) -> None:
    rows = collect_schema_token_rows(model, tokens)
    payload = {
        "token_ids": rows.token_ids,
        "tokens": rows.tokens,
        "input_rows": rows.input_rows,
        "output_rows": rows.output_rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def apply_schema_token_rows(model: Any, path: str | Path) -> None:
    payload = torch.load(Path(path), map_location="cpu")
    token_ids = [int(token_id) for token_id in payload["token_ids"]]
    with torch.no_grad():
        input_weight = model.get_input_embeddings().weight
        input_weight[token_ids] = payload["input_rows"].to(input_weight.device, input_weight.dtype)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and payload.get("output_rows") is not None:
            output_weight = output_embeddings.weight
            output_weight[token_ids] = payload["output_rows"].to(output_weight.device, output_weight.dtype)
