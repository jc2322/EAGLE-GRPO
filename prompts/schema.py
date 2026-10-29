from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


THINKING_TAG = "thinking"

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

OPTIONAL_FINAL_PROMPT_TAGS = frozenset(
    {
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
    }
)

REQUIRED_FINAL_PROMPT_TAGS = frozenset(set(FINAL_PROMPT_TAGS) - set(OPTIONAL_FINAL_PROMPT_TAGS))

CONSTRAINT_ONLY_TAGS = frozenset({"preservation_constraints", "negative_constraints"})

POSITIVE_CREDIT_TAGS = tuple(tag for tag in FINAL_PROMPT_TAGS if tag not in CONSTRAINT_ONLY_TAGS)

SOURCE_INFO_DENSITIES = ("low_info_source", "medium_info_source", "high_info_source", "visual_demo_source")

DESIGN_STRATEGIES = (
    "balanced",
    "commercial_rich",
    "feature_rich",
    "visual_support_rich",
    "comparison_focused",
    "trust_logistics_focused",
    "title_focused_clean",
    "title_plus_one_fact_clean",
    "balanced_clean",
    "light_expression_clean",
    "commercial_dense_clean",
)

SPECIAL_TOKENS = tuple(
    token
    for tag in (THINKING_TAG, "final_prompt", *FINAL_PROMPT_TAGS)
    for token in (f"<{tag}>", f"</{tag}>")
)


@dataclass(frozen=True)
class Block:
    tag: str
    start: int
    end: int
    content: str


@dataclass(frozen=True)
class SchemaValidation:
    thinking: str
    final_prompt: str
    final_prompt_blocks: dict[str, Block]


def open_tag(tag: str) -> str:
    return f"<{tag}>"


def close_tag(tag: str) -> str:
    return f"</{tag}>"


def render_empty_final_prompt() -> str:
    lines = ["<final_prompt>"]
    for tag in FINAL_PROMPT_TAGS:
        lines.append(f"  <{tag}></{tag}>")
    lines.append("</final_prompt>")
    return "\n".join(lines)


def render_sft_prompt(thinking: str, final_prompt: str) -> str:
    return f"<thinking>\n{thinking.strip()}\n</thinking>\n\n{final_prompt.strip()}"


def _find_blocks(text: str, tag: str) -> list[Block]:
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", flags=re.DOTALL)
    blocks: list[Block] = []
    for match in pattern.finditer(text):
        raw_content = match.group(1)
        left_trim = len(raw_content) - len(raw_content.lstrip())
        right_trim = len(raw_content) - len(raw_content.rstrip())
        blocks.append(
            Block(
                tag=tag,
                start=match.start(1) + left_trim,
                end=match.end(1) - right_trim,
                content=raw_content.strip(),
            )
        )
    return blocks


def _single_block(text: str, tag: str) -> Block:
    blocks = _find_blocks(text, tag)
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one <{tag}>...</{tag}> block, found {len(blocks)}")
    return blocks[0]


def _validate_final_prompt_children(final_block_text: str) -> dict[str, Block]:
    cursor = 0
    children: dict[str, Block] = {}
    for tag in FINAL_PROMPT_TAGS:
        open_index = final_block_text.find(open_tag(tag), cursor)
        if open_index < 0:
            raise ValueError(f"Missing <{tag}> tag in <final_prompt>")
        close_index = final_block_text.find(close_tag(tag), open_index + len(open_tag(tag)))
        if close_index < 0:
            raise ValueError(f"Missing </{tag}> tag in <final_prompt>")
        next_other_open = min(
            [
                index
                for other in FINAL_PROMPT_TAGS
                if other != tag
                for index in [final_block_text.find(open_tag(other), cursor)]
                if index >= 0
            ]
            or [open_index]
        )
        if next_other_open < open_index:
            raise ValueError(f"Tag <{tag}> is out of order in <final_prompt>")
        duplicate_open = final_block_text.find(open_tag(tag), close_index + len(close_tag(tag)))
        if duplicate_open >= 0:
            raise ValueError(f"Duplicate <{tag}> tag in <final_prompt>")
        content_start = open_index + len(open_tag(tag))
        raw_content = final_block_text[content_start:close_index]
        left_trim = len(raw_content) - len(raw_content.lstrip())
        right_trim = len(raw_content) - len(raw_content.rstrip())
        content = raw_content.strip()
        if tag in REQUIRED_FINAL_PROMPT_TAGS and not content:
            raise ValueError(f"Required tag <{tag}> must not be empty")
        children[tag] = Block(
            tag=tag,
            start=content_start + left_trim,
            end=close_index - right_trim,
            content=content,
        )
        cursor = close_index + len(close_tag(tag))

    trailing_open = re.search(r"<([a-z_]+)>", final_block_text[cursor:])
    if trailing_open:
        raise ValueError(f"Unexpected tag after schema fields: <{trailing_open.group(1)}>")
    return children


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _first_overlap(a: Iterable[str], b: Iterable[str], ngram: int) -> str:
    a_words = list(a)
    b_words = list(b)
    if len(a_words) < ngram or len(b_words) < ngram:
        return ""
    a_ngrams = {tuple(a_words[index : index + ngram]) for index in range(len(a_words) - ngram + 1)}
    for index in range(len(b_words) - ngram + 1):
        candidate = tuple(b_words[index : index + ngram])
        if candidate in a_ngrams:
            return " ".join(candidate)
    return ""


def validate_sft_prompt(text: str, *, overlap_ngram: int = 18) -> SchemaValidation:
    raw = str(text or "").strip()
    thinking = _single_block(raw, THINKING_TAG)
    final_prompt = _single_block(raw, "final_prompt")
    if raw.find(open_tag(THINKING_TAG)) > raw.find(open_tag("final_prompt")):
        raise ValueError("<thinking> must appear before <final_prompt>")
    final_blocks = _validate_final_prompt_children(final_prompt.content)
    overlap = _first_overlap(
        _normalized_words(thinking.content),
        _normalized_words(final_prompt.content),
        overlap_ngram,
    )
    if overlap:
        raise ValueError(f"<thinking> and <final_prompt> share a long phrase: {overlap!r}")
    return SchemaValidation(
        thinking=thinking.content,
        final_prompt=final_prompt.content,
        final_prompt_blocks=final_blocks,
    )


def compile_final_prompt_for_image(text: str) -> str:
    validation = validate_sft_prompt(text, overlap_ngram=10_000)
    sections = [block.content for tag, block in validation.final_prompt_blocks.items() if block.content]
    return "\n\n".join(sections).strip()
