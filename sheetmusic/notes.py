from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Note:
    """A note in seconds (before quantisation)."""
    start: float
    end: float
    pitch: int
    velocity: int = 80


@dataclass
class QNote:
    """A quantised note in ticks (480 per quarter note)."""
    start: int
    end: int
    pitch: int
    velocity: int = 80
