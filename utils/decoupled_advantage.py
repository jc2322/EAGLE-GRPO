from __future__ import annotations

import re
from typing import Iterable

import torch

from rewards.online_component_reward import ComponentRewardStore
from utils.advantage_protocol import AdvantageContext
from utils.completion_schema import CharacterSpan, char_spans_to_token_mask, parse_completion
from utils.modeling import decode_preserving_schema_tags


CONSTRAINT_ONLY_TAGS = {"preservation_constraints", "negative_constraints"}


def block_span(text: str, tags: Iterable[str]) -> CharacterSpan | None:
    for tag in tags:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if match is None:
            continue
        content = match.group(1)
        left_trim = len(content) - len(content.lstrip())
        right_trim = len(content) - len(content.rstrip())
        return CharacterSpan(
            name=tag,
            start=match.start(1) + left_trim,
            end=match.end(1) - right_trim,
            text=content.strip(),
        )
    return None


def align_mask(mask: list[int], length: int) -> list[int]:
    if len(mask) >= length:
        return mask[:length]
    return mask + [0] * (length - len(mask))


def span_token_mask(tokenizer: object, text: str, spans: list[CharacterSpan], length: int) -> list[int]:
    if not spans:
        return [0] * length
    try:
        return align_mask(char_spans_to_token_mask(tokenizer, text, spans), length)
    except Exception:  # pragma: no cover - tokenizer-specific fallback
        return [0] * length


class DecoupledAdvantageProvider:
    name = "decoupled_cot_image"

    def __init__(self, store: ComponentRewardStore, config: dict | None = None) -> None:
        self.store = store
        self.config = config or {}
        self.last_loss_mask: torch.Tensor | None = None

    def token_advantages(self, context: AdvantageContext) -> torch.Tensor:
        batch = self.store.consume_latest()
        completion_ids = context.completion_ids
        completion_mask = context.completion_mask
        tokenizer = context.tokenizer
        device = completion_ids.device
        token_advantages = torch.zeros_like(completion_mask, dtype=torch.float32, device=device)
        loss_mask = torch.zeros_like(completion_mask, dtype=completion_mask.dtype, device=device)

        if len(batch.rewards) != int(completion_ids.shape[0]):
            raise ValueError(
                f"Component reward count {len(batch.rewards)} does not match completion batch {int(completion_ids.shape[0])}"
            )

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

            thinking = block_span(text, ("thinking", "reasoning", "think"))
            thinking_mask = span_token_mask(tokenizer, text, [thinking] if thinking else [], valid_length)
            prompt_spans = [
                span
                for tag, span in parsed.final_prompt_spans.items()
                if tag not in CONSTRAINT_ONLY_TAGS and span.text.strip()
            ]
            prompt_mask = span_token_mask(tokenizer, text, prompt_spans, valid_length)
            for token_index, enabled in enumerate(thinking_mask):
                if enabled:
                    token_advantages[row_index, token_index] = float(reward.cot_advantage)
                    loss_mask[row_index, token_index] = completion_mask[row_index, token_index]
            for token_index, enabled in enumerate(prompt_mask):
                if enabled:
                    token_advantages[row_index, token_index] = float(reward.image_advantage)
                    loss_mask[row_index, token_index] = completion_mask[row_index, token_index]

        self.last_loss_mask = loss_mask
        return token_advantages
