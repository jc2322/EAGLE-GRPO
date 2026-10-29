from __future__ import annotations

from prompts.schema import SPECIAL_TOKENS


def schema_special_tokens() -> list[str]:
    return list(SPECIAL_TOKENS)


def add_schema_special_tokens(tokenizer: object) -> int:
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    tokens = [token for token in schema_special_tokens() if token not in existing]
    if not tokens:
        return 0
    return int(tokenizer.add_special_tokens({"additional_special_tokens": tokens}))
