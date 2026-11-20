from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Note:
    """A note in seconds (before quantisation)."""
    start: float
    end: float
    pitch: int
    velocity: int = 80
    struck: bool = False   # freely decaying (piano, guitar): where it fades says little
    decay: float = float("nan")   # dB/s lost by the note's own partials after the attack


@dataclass
class QNote:
    """A quantised note in ticks (480 per quarter note)."""
    start: int
    end: int
    pitch: int
    velocity: int = 80
    struck: bool = False
    hand: int | None = None    # 0 = left, 1 = right (piano part), None = not assigned
