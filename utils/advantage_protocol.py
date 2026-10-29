from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class AdvantageContext:
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    sequence_advantages: torch.Tensor
    tokenizer: Any
    hidden_states: torch.Tensor | None = None


class AdvantageProvider(Protocol):
    name: str

    def token_advantages(self, context: AdvantageContext) -> torch.Tensor | None:
        """Return `(batch, completion_length)` token advantages or `None`.

        Returning `None` preserves TRL's canonical sequence-advantage GRPO loss.
        """
        ...
