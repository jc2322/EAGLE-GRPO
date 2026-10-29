from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Protocol

from utils.completion_schema import parse_completion
from utils.modeling import decode_preserving_schema_tags


class RewardProvider(Protocol):
    def __call__(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        ...


def _completion_text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "content" in value:
            text = _completion_text_from_value(value.get("content", ""))
            if text:
                return text
        if "text" in value:
            return _completion_text_from_value(value.get("text", ""))
        return ""
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _completion_text_from_value(item))).strip()
    return str(value or "").strip()


def _token_ids_to_list(token_ids: Any) -> list[int]:
    if token_ids is None:
        return []
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, (list, tuple)):
        return []
    ids: list[int] = []
    for token_id in token_ids:
        try:
            ids.append(int(token_id))
        except (TypeError, ValueError):
            return []
    return ids


def completion_text_from_token_ids(token_ids: Any, processing_class: Any) -> str:
    ids = _token_ids_to_list(token_ids)
    if not ids or processing_class is None:
        return ""
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return ""
    try:
        return decode_preserving_schema_tags(tokenizer, ids, skip_special_tokens=True).strip()
    except Exception:
        try:
            return str(tokenizer.decode(ids, skip_special_tokens=False) or "").strip()
        except Exception:
            return ""


def completion_text(value: Any, *, token_ids: Any = None, processing_class: Any = None) -> str:
    text = _completion_text_from_value(value)
    if text:
        return text
    return completion_text_from_token_ids(token_ids, processing_class)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    filename = str(getattr(value, "filename", "") or "").strip()
    if filename:
        return filename
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {
            "type": value.__class__.__name__,
            "mode": str(getattr(value, "mode", "")),
            "size": list(getattr(value, "size", ()) or ()),
        }
    return str(value)
