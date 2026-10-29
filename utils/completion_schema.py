from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


REASONING_TAGS = (
    "source_analysis",
    "selling_point_selection",
    "layout_rationale",
    "style_rationale",
    "risk_control",
)

FINAL_PROMPT_TAGS = (
    "preservation_constraints",
    "product_presentation",
    "headline",
    "subtitle",
    "feature_callouts",
    "specification_information",
    "quantity_bundle_information",
    "variant_information",
    "price_promotion_information",
    "seller_logistics_information",
    "trust_information",
    "usage_information",
    "comparison_visual",
    "supporting_visuals",
    "composition_layout",
    "scene_style",
    "lighting_rendering",
    "negative_constraints",
)


@dataclass(frozen=True)
class CharacterSpan:
    name: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ParsedCompletion:
    raw_text: str
    reasoning: str
    final_prompt: str
    reasoning_spans: dict[str, CharacterSpan]
    final_prompt_spans: dict[str, CharacterSpan]


def _extract_block(text: str, tag: str, *, required: bool) -> CharacterSpan | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        if required:
            raise ValueError(f"Missing required <{tag}>...</{tag}> block")
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


def _child_spans(parent: CharacterSpan, tags: Iterable[str]) -> dict[str, CharacterSpan]:
    spans: dict[str, CharacterSpan] = {}
    for tag in tags:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", parent.text, flags=re.DOTALL | re.IGNORECASE)
        if match is None:
            continue
        spans[tag] = CharacterSpan(
            name=tag,
            start=parent.start + match.start(1),
            end=parent.start + match.end(1),
            text=match.group(1).strip(),
        )
    return spans


def parse_completion(text: str, *, require_structured_children: bool = False) -> ParsedCompletion:
    raw = str(text or "").strip()
    reasoning = _extract_block(raw, "thinking", required=False)
    if reasoning is None:
        reasoning = _extract_block(raw, "reasoning", required=False)
    if reasoning is None:
        # Backward-compatible parsing for the current SFT format.
        reasoning = _extract_block(raw, "think", required=True)
    final_prompt = _extract_block(raw, "final_prompt", required=True)

    reasoning_spans = _child_spans(reasoning, REASONING_TAGS)
    final_prompt_spans = _child_spans(final_prompt, FINAL_PROMPT_TAGS)
    if require_structured_children:
        missing_final = sorted(set(FINAL_PROMPT_TAGS) - set(final_prompt_spans))
        if missing_final:
            raise ValueError(f"Missing structured final_prompt children: final_prompt={missing_final}")
    return ParsedCompletion(
        raw_text=raw,
        reasoning=reasoning.text,
        final_prompt=final_prompt.text,
        reasoning_spans=reasoning_spans,
        final_prompt_spans=final_prompt_spans,
    )


def char_spans_to_token_mask(
    tokenizer: object,
    text: str,
    spans: Iterable[CharacterSpan],
    *,
    max_length: int | None = None,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    offsets = encoded["offset_mapping"]
    span_list = list(spans)
    return [
        int(any(token_end > span.start and token_start < span.end for span in span_list))
        for token_start, token_end in offsets
    ]
