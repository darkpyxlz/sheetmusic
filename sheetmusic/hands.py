"""Left hand / right hand.

A piano score is two streams, one per hand, and almost everything about how
it is written follows from that: 82-98 % of the notes in the benchmark piano
scores end exactly at the next onset *of their own staff*, a hand holds at
most what it can span, and "playable" means each hand can reach its notes.
A fixed split at middle C gets the hands wrong whenever the tune dips below
it or the accompaniment rises above it.

``assign_hands`` walks through the chords (notes that start together) in time
order and gives the lowest k notes of each to the left hand, choosing k by
cost:

* reach: a hand spans an octave comfortably, a ninth or tenth with effort,
  more not at all;
* travel: a hand stays where it is unless it has to move (its position
  relaxes back towards its home register while it rests);
* register: without other evidence, notes above the split are the right
  hand's and notes below it the left hand's;
* the hands do not cross.
"""

from __future__ import annotations

import numpy as np

LEFT, RIGHT = 0, 1


def _span_cost(lo: int, hi: int, comfortable: int = 12, limit: int = 16) -> float:
    span = hi - lo
    if span <= comfortable:
        return 0.0
    return 1.0 * (span - comfortable) + 6.0 * max(0, span - limit)


def assign_hands(notes, ticks_per_beat: int = 480, split: int = 60, move_cost: float = 0.3,
                 register_cost: float = 0.05, home=(46.0, 72.0), relax_beats: float = 8.0,
                 free_move: float = 2.0) -> list[int]:
    """Hand (0 = left, 1 = right) for every note (anything with ``start`` in
    ticks and ``pitch``), in the order given."""
    n = len(notes)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (notes[i].start, notes[i].pitch))
    hands = [RIGHT] * n
    pos = [float(home[0]), float(home[1])]       # where each hand is
    last = [None, None]                          # when it last played (ticks)
    i = 0
    while i < n:
        j = i
        t = notes[order[i]].start
        while j + 1 < n and notes[order[j + 1]].start == t:
            j += 1
        idx = order[i:j + 1]
        ps = [notes[k].pitch for k in idx]
        # hands drift back to their home register while idle
        cur = []
        for h in (LEFT, RIGHT):
            if last[h] is None:
                cur.append(float(home[h]))
            else:
                idle = (t - last[h]) / ticks_per_beat
                cur.append(home[h] + (pos[h] - home[h]) * float(np.exp(-idle / relax_beats)))
        best_k, best_c = 0, np.inf
        m = len(ps)
        for k in range(m + 1):                   # lowest k notes -> left hand
            c = 0.0
            if k > 0:
                lo, hi = ps[0], ps[k - 1]
                centre = 0.5 * (lo + hi)
                c += _span_cost(lo, hi) + move_cost * max(0.0, abs(centre - cur[LEFT]) - free_move)
                c += register_cost * sum(max(0, p - split) for p in ps[:k])
                if hi > cur[RIGHT] + 2 and k < m + 1 and last[RIGHT] is not None \
                        and (t - last[RIGHT]) < ticks_per_beat:
                    c += 0.5 * (hi - cur[RIGHT])                 # reaching over the right hand
            if k < m:
                lo, hi = ps[k], ps[m - 1]
                centre = 0.5 * (lo + hi)
                c += _span_cost(lo, hi) + move_cost * max(0.0, abs(centre - cur[RIGHT]) - free_move)
                c += register_cost * sum(max(0, split - 1 - p) for p in ps[k:])
                if lo < cur[LEFT] - 2 and last[LEFT] is not None and (t - last[LEFT]) < ticks_per_beat:
                    c += 0.5 * (cur[LEFT] - lo)                  # reaching under the left hand
            if c < best_c:
                best_k, best_c = k, c
        for r, k in enumerate(idx):
            hands[k] = LEFT if r < best_k else RIGHT
        if best_k > 0:
            pos[LEFT] = 0.5 * (ps[0] + ps[best_k - 1])
            last[LEFT] = t
        if best_k < m:
            pos[RIGHT] = 0.5 * (ps[best_k] + ps[m - 1])
            last[RIGHT] = t
        i = j + 1
    return hands
