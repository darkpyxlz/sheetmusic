"""Key estimation (Krumhansl-Kessler) and pitch spelling (MuseScore TPC)."""

from __future__ import annotations

import numpy as np

from .notes import QNote

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# fifths of the major key whose tonic has this pitch class (prefer <= 5 sharps / flats)
MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: -6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_key(groups: list[list[QNote]], weights: list[float] | None = None):
    """Return (fifths, mode, name) using duration-weighted pitch classes."""
    hist = np.zeros(12)
    weights = weights or [1.0] * len(groups)
    for notes, w in zip(groups, weights):
        for q in notes:
            hist[q.pitch % 12] += w * (q.end - q.start)
    if hist.sum() == 0:
        return 0, "major", "C major"
    best = (-np.inf, 0, "major")
    for tonic in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            if r > best[0]:
                best = (r, tonic, mode)
    _, tonic, mode = best
    rel_major = tonic if mode == "major" else (tonic + 3) % 12
    fifths = MAJOR_FIFTHS[rel_major]
    name = _tonic_name(tonic, fifths) + " " + mode
    return fifths, mode, name


def _tonic_name(pc: int, fifths: int) -> str:
    tpc = spell(pc, fifths)
    return tpc_name(tpc)


def spell(pitch: int, key_fifths: int) -> int:
    """MuseScore tonal pitch class (C=14, G=15, F=13, ...) for a MIDI pitch.

    Chooses the spelling on the line of fifths within the window
    [key-1, key+10] relative to C=0, i.e. diatonic notes of the key plus the
    usual chromatic alterations (raised 4th/5th/1st/2nd, flat 7th...).
    """
    pc = pitch % 12
    lo = key_fifths - 4
    for f in range(lo, lo + 12):
        if (f * 7) % 12 == pc:
            return f + 14
    raise AssertionError


def tpc_name(tpc: int) -> str:
    f = tpc - 14
    letters = "FCGDAEB"
    letter = letters[(f + 1) % 7]
    acc = (f + 1) // 7
    return letter + ("#" * acc if acc > 0 else "b" * (-acc))
