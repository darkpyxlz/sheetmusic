"""Key estimation (Krumhansl-Kessler) and pitch spelling (MuseScore TPC)."""

from __future__ import annotations

import numpy as np

from .notes import QNote

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# fifths of the major key whose tonic has this pitch class (prefer <= 5 sharps / flats)
MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: -6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
CHROMA_WEIGHT = 0.0   # raw-spectrum chroma carries every third partial as a "fifth": it pulled 3 of 24 chorales to the dominant
TONIC_WEIGHT = 0.12
NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_key(groups: list[list[QNote]], weights: list[float] | None = None,
                 chroma: np.ndarray | None = None):
    """Return (fifths, mode, name).

    * Krumhansl-Kessler correlation of the duration-weighted pitch-class
      histogram (bass notes count double; a spectral chroma average can be
      blended in, which is robust to transcription errors);
    * relative major/minor share their notes, so the *tonic* decides: the
      final bass note, the final melody note, the first bass note and the
      bass notes on strong positions get a bonus when they are the key's tonic.
    """
    hist = np.zeros(12)
    weights = weights or [1.0] * len(groups)
    allq = []
    for notes, w in zip(groups, weights):
        for q in notes:
            d = q.end - q.start
            hist[q.pitch % 12] += w * d * (2.0 if q.pitch < 55 else 1.0)
            allq.append(q)
    if chroma is not None and chroma.sum() > 0 and hist.sum() > 0:
        hist = hist / hist.sum() + CHROMA_WEIGHT * chroma / chroma.sum()
    if hist.sum() == 0:
        return 0, "major", "C major"
    tonic_ev = np.zeros(12)
    if allq:
        end = max(q.end for q in allq)
        start = min(q.start for q in allq)
        span = max(1, end - start)
        tail = [q for q in allq if q.end >= end - 0.06 * span]
        head = [q for q in allq if q.start <= start + 0.04 * span]
        if tail:
            tonic_ev[min(tail, key=lambda q: q.pitch).pitch % 12] += 1.0       # final bass
            tonic_ev[max(tail, key=lambda q: q.end).pitch % 12] += 0.4          # last note
        if head:
            tonic_ev[min(head, key=lambda q: q.pitch).pitch % 12] += 0.4        # first bass
        bass = np.zeros(12)
        for q in allq:
            if q.pitch < 55:
                bass[q.pitch % 12] += q.end - q.start
        if bass.sum() > 0:
            tonic_ev += 0.6 * bass / bass.max()
    best = (-np.inf, 0, "major")
    for tonic in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            r += TONIC_WEIGHT * tonic_ev[tonic]
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


def spell(pitch: int, key_fifths: int, minor: bool = False) -> int:
    """MuseScore tonal pitch class (C=14, G=15, F=13, ...) for a MIDI pitch.

    Chooses the spelling on the line of fifths within a window around the key:
    major keys [key-4, key+7] (diatonic notes plus the usual chromatic
    alterations: raised 4th/5th/1st/2nd, flat 3rd/6th/7th); minor keys are
    shifted up by one so the raised 6th and 7th (leading note, e.g. G# in
    A minor) come out sharp while the flat 2nd/6th stay flat.
    """
    pc = pitch % 12
    lo = key_fifths - (3 if minor else 4)
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
