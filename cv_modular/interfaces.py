from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class ProcessorResult:
    """Standard output from each CV processor."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)


class FrameProcessor(Protocol):
    """A pluggable processor that reads/writes frame state."""

    name: str

    def process(self, frame: np.ndarray) -> ProcessorResult:
        ...

    def close(self) -> None:
        ...
