from __future__ import annotations

import logging
from typing import Any


LOG = logging.getLogger(__name__)


def cuda_memory_snapshot() -> str:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        return f"cuda_memory unavailable: torch import failed: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return "cuda_memory unavailable: cuda is not available"
    try:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        max_allocated = torch.cuda.max_memory_allocated(device_index)
        max_reserved = torch.cuda.max_memory_reserved(device_index)
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"cuda_memory unavailable: {type(exc).__name__}: {exc}"

    def gb(value: int | float) -> float:
        return float(value) / (1024**3)

    used_bytes = total_bytes - free_bytes
    used_pct = 100.0 * used_bytes / max(float(total_bytes), 1.0)
    return (
        f"device={device_index} name={device_name!r} "
        f"used={gb(used_bytes):.2f}GiB/{gb(total_bytes):.2f}GiB ({used_pct:.1f}%) "
        f"allocated={gb(allocated):.2f}GiB reserved={gb(reserved):.2f}GiB "
        f"max_allocated={gb(max_allocated):.2f}GiB max_reserved={gb(max_reserved):.2f}GiB"
    )


def log_cuda_memory(label: str, *, logger: logging.Logger | None = None, level: int = logging.INFO) -> None:
    target = logger or LOG
    target.log(level, "cuda_memory %s: %s", label, cuda_memory_snapshot())


def maybe_log_cuda_memory(enabled: Any, label: str, *, logger: logging.Logger | None = None) -> None:
    if bool(enabled):
        log_cuda_memory(label, logger=logger)
