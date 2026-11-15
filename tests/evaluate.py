"""Compare a transcription to the synthetic ground truth (note-level F-measure)."""

from __future__ import annotations

import numpy as np


def _match(ref, est, onset_tol=0.26):
    """ref/est: lists of (onset_beats, dur_beats, pitch). Greedy one-to-one match."""
    used = set()
    hits = 0
    for o, d, p in ref:
        best = None
        for j, (oe, de, pe) in enumerate(est):
            if j in used or pe != p:
                continue
            err = abs(oe - o)
            if err <= onset_tol and (best is None or err < best[0]):
                best = (err, j)
        if best is not None:
            used.add(best[1])
            hits += 1
    prec = hits / len(est) if est else 0.0
    rec = hits / len(ref) if ref else 0.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f


def evaluate(qnotes, truth_notes, ticks_per_beat=480, max_shift_beats=8):
    est = [(q.start / ticks_per_beat, (q.end - q.start) / ticks_per_beat, q.pitch) for q in qnotes]
    best = (-1, 0, None)
    for shift in np.arange(-max_shift_beats, max_shift_beats + 0.01, 0.25):
        sh = [(o - shift, d, p) for o, d, p in est]
        r = _match(truth_notes, sh)
        if r[2] > best[0]:
            best = (r[2], shift, r)
    return {"precision": best[2][0], "recall": best[2][1], "f": best[2][2], "shift": best[1]}


def rhythm_exactness(qnotes, truth_notes, shift, ticks_per_beat=480):
    """Fraction of reference notes written with exactly the right pitch, onset
    and duration (best alignment within half a beat of ``shift``)."""
    ref = [(round(o, 4), round(d, 4), p) for o, d, p in truth_notes]
    best = {"onset_exact": 0.0, "onset_and_duration_exact": 0.0}
    for sh in np.arange(shift - 0.5, shift + 0.51, 0.125):
        est = {(round(q.start / ticks_per_beat - sh, 4),
                round((q.end - q.start) / ticks_per_beat, 4), q.pitch) for q in qnotes}
        onsets = {(e[0], e[2]) for e in est}
        onset_ok = sum((o, p) in onsets for o, d, p in ref) / len(ref)
        full_ok = sum((o, d, p) in est for o, d, p in ref) / len(ref)
        if onset_ok > best["onset_exact"]:
            best = {"onset_exact": onset_ok, "onset_and_duration_exact": full_ok}
    return best
