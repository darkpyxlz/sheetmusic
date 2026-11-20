"""Playable piano arrangements.

A transcription writes down what sounds: every note of a chord, every
passing note, every repeated strum.  Two hands cannot play all of that, and
a reader does not want to.  ``arrange_piano`` keeps the main notes, the tune
on top, the bass underneath and enough of the harmony between them, within
what one pair of hands can play:

* **chord size**: at most ``max_notes`` notes held at once in each hand;
* **reach**: the notes one hand holds lie within ``span`` semitones (an
  octave at "playable", a sixth at "easy");
* **note values**: no note shorter than ``shortest``, onsets on a grid of
  that value (a quaver at "easy": no semiquavers on the page);
* **changes**: two onsets in the same hand are at least ``right_gap`` /
  ``left_gap`` apart, and there are no more than ``max_rate`` onsets per
  second in a hand at the piece's tempo, so a fast tremolo or strum becomes
  one chord;
* the hands keep to their side: left below right;
* a lone weak blip (shortest value, nothing else in that hand within a beat)
  is left out: it is an artefact of the analysis, not a note anyone played.

What stays is decided by importance: the top note of the right hand (the
tune) and the bottom note of the left hand (the bass) are kept first, then
notes by strength, length and metrical position (downbeat > beat > off-beat).
Notes are only ever removed, shortened or lengthened to the next kept note
of the same hand, never moved to another pitch, so every note on the page is
one the transcription heard.

``playability`` measures the result: notes per second, chord size and span per
hand, and how much of the page is semiquavers.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from .notes import QNote

LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class Level:
    name: str
    max_notes: tuple[int, int]        # (left, right): notes held at once
    span: tuple[int, int]             # (left, right): widest interval one hand holds
    shortest: int                     # shortest written value and onset grid, ticks
    right_gap: float                  # closest onsets in the right hand, in beats
    left_gap: float                   # closest onsets in the left hand, in beats
    max_rate: tuple[float, float]     # (left, right): onsets per second
    legato: float                     # bridge gaps up to this many beats in a hand


LEVELS = {
    "easy": Level("easy", (2, 2), (9, 9), 240, 0.5, 1.0, (2.5, 4.0), 2.0),
    "playable": Level("playable", (3, 3), (12, 12), 120, 0.25, 0.5, (4.0, 7.0), 1.0),
}
ARRANGEMENTS = ("playable", "easy", "full")


_OVERTONES = (12, 19, 24, 28, 31)
# The bass (left) and tune (right) must be at least this share as strong as the
# strongest note the hand strikes with them.  Overtones of inner voices that
# survive the analysis (winds especially) sit above the tune but are weaker;
# measured on the chorales, 0.75 keeps 0.84 of the tune at "playable" instead
# of 0.75, with no change for piano or strings.
OUTER_STRENGTH = (0.5, 0.75)


def _outer(qs: list[QNote], hand: int, ghost: set) -> QNote:
    """The tune (right hand: highest) or bass (left hand: lowest) note of an
    onset: the outermost note that is neither a likely overtone nor much
    weaker than the strongest note of the onset."""
    vmax = max(q.velocity for q in qs)
    cand = [q for q in qs if id(q) not in ghost and q.velocity >= OUTER_STRENGTH[hand] * vmax] or qs
    return max(cand, key=lambda q: q.pitch) if hand == RIGHT else min(cand, key=lambda q: q.pitch)


def _metric_weight(t: int, beat: int, measure: int) -> float:
    if t % measure == 0:
        return 1.0
    if t % beat == 0:
        return 0.6
    if (2 * t) % beat == 0 or (3 * t) % beat == 0:
        return 0.3
    return 0.1


def arrange_piano(notes: list[QNote], level: str | Level = "playable", beat_ticks: int = 480,
                  measure_ticks: int = 1920, crotchets_per_second: float = 2.0,
                  compound: bool = False, split: int = 60) -> list[QNote]:
    """Reduce a piano part to what one player's hands can manage (see the
    module text).  ``notes`` are quantised notes; ``hand`` is used when set,
    otherwise notes at or above ``split`` go to the right hand."""
    if isinstance(level, str):
        if level == "full":
            return notes
        level = LEVELS[level]
    if not notes:
        return []
    grid = level.shortest
    if compound and beat_ticks % grid:
        grid = beat_ticks // 3
    # ---- 1. snap to the level's grid, merge what lands together.  Triplets
    # whose notes are at least the shortest value stay as they are (quaver
    # triplets at "playable"); shorter ones keep only their on-beat notes.
    def tunit(t):
        r = t % beat_ticks
        if compound or r % 30 == 0:
            return None
        return next((beat_ticks // m for m in (3, 6) if (r * m) % beat_ticks == 0), grid)

    def place(t):
        u = tunit(t)
        if u is None or u < grid:
            return int(round(t / grid)) * grid, None
        return t, u

    work: dict[tuple[int, int, int], QNote] = {}
    for q in notes:
        hand = q.hand if q.hand is not None else int(q.pitch >= split)
        if tunit(q.start) is not None and tunit(q.start) < grid:
            continue                       # a fast triplet note: the group's on-beat note carries it
        s, us = place(q.start)
        e, _ue = place(q.end)
        e = max(s + (us or grid), e)
        key = (hand, s, q.pitch)
        old = work.get(key)
        if old is None or q.velocity > old.velocity:
            work[key] = QNote(s, max(e, old.end if old else e), q.pitch, q.velocity, q.struck, hand)
        else:
            old.end = max(old.end, e)
    # ---- 1b. a lone blip is an artefact, not music: a note that would be
    # written at the shortest value, with nothing else in its hand within a
    # beat before or after it, and weaker than the hand's typical note (real
    # fast notes come in runs, figures or chords)
    for hand in (LEFT, RIGHT):
        hn = sorted((q for q in work.values() if q.hand == hand), key=lambda q: q.start)
        if len(hn) < 4:
            continue
        med = float(np.median([q.velocity for q in hn]))
        starts = np.array([q.start for q in hn])
        for q in hn:
            if q.end - q.start > grid or q.velocity >= med:
                continue
            lo = np.searchsorted(starts, q.start - beat_ticks)
            hi = np.searchsorted(starts, q.end + beat_ticks, side="right")
            if hi - lo <= 1:
                del work[(hand, q.start, q.pitch)]
    groups: dict[tuple[int, int], list[QNote]] = {}
    for q in work.values():
        groups.setdefault((q.hand, q.start), []).append(q)

    # ---- 2. importance of notes and of onsets.  A note an octave, twelfth or
    # double octave above a stronger note struck with it is most likely that
    # note's overtone left behind by the analysis (common with winds and
    # strings): it is never taken for the tune or the bass and goes first.
    at_onset: dict[int, list[QNote]] = {}
    for q in work.values():
        at_onset.setdefault(q.start, []).append(q)
    ghost = {id(q) for qs in at_onset.values() for q in qs
             if any(q.pitch - o.pitch in _OVERTONES and q.velocity < 0.95 * o.velocity for o in qs)}

    def importance(q: QNote, top: bool, bottom: bool) -> float:
        dur = max(q.end - q.start, grid)
        v = q.velocity / 127.0 + 0.25 * np.log2(1 + dur / 120.0) \
            + 0.6 * _metric_weight(q.start, beat_ticks, measure_ticks)
        if q.hand == RIGHT and top:
            v += 1.0                    # the tune
        if q.hand == LEFT and bottom:
            v += 0.8                    # the bass
        if id(q) in ghost:
            v -= 0.6
        return float(v)

    imp: dict[int, float] = {}
    group_imp: dict[tuple[int, int], float] = {}
    outer_of: dict[tuple[int, int], QNote] = {}
    for key, qs in groups.items():
        outer = _outer(qs, key[0], ghost)
        outer_of[key] = outer
        for q in qs:
            imp[id(q)] = importance(q, q is outer, q is outer)
        group_imp[key] = max(imp[id(q)] for q in qs)

    # ---- 3. which onsets each hand plays: most important first, keeping the
    # minimum gap and the rate limit (weighted interval selection, greedy)
    gap_ticks = {RIGHT: max(grid, int(round(level.right_gap * beat_ticks))),
                 LEFT: max(grid, int(round(level.left_gap * beat_ticks)))}
    window = 480.0 * crotchets_per_second                     # ticks per second
    accepted: dict[int, list[int]] = {LEFT: [], RIGHT: []}
    for key in sorted(groups, key=lambda k: -group_imp[k]):
        hand, s = key
        acc = accepted[hand]                        # kept sorted
        i = bisect.bisect_left(acc, s)
        if (i > 0 and s - acc[i - 1] < gap_ticks[hand]) or \
                (i < len(acc) and acc[i] - s < gap_ticks[hand]):
            continue
        # no second may hold more than max_rate onsets: any R + 1 consecutive
        # onsets (with this one) must span at least a second
        R = int(level.max_rate[hand])
        trial = acc[:i] + [s] + acc[i:]
        if any(trial[j + R] - trial[j] < window
               for j in range(max(0, i - R), min(i, len(trial) - 1 - R) + 1)):
            continue
        acc.insert(i, s)

    # ---- 4. which notes of each kept onset: the outer note first, then by
    # importance within the hand's limits (count and reach)
    kept: list[QNote] = []
    for hand in (LEFT, RIGHT):
        for s in sorted(accepted[hand]):
            qs = sorted(groups[(hand, s)], key=lambda q: -imp[id(q)])
            outer = outer_of[(hand, s)]
            chosen = [outer]
            for q in qs:
                if q is outer or len(chosen) >= level.max_notes[hand]:
                    continue
                ps = [c.pitch for c in chosen] + [q.pitch]
                if max(ps) - min(ps) <= level.span[hand]:
                    chosen.append(q)
            kept += chosen

    # ---- 5. durations: a note lasts until the next onset of its hand when the
    # gap is short (a removed note leaves its predecessor sounding), and held
    # notes give way when the hand's limits would be exceeded
    out: list[QNote] = []
    legato = int(round(level.legato * beat_ticks))
    for hand in (LEFT, RIGHT):
        hn = sorted((q for q in kept if q.hand == hand), key=lambda q: (q.start, q.pitch))
        starts = sorted({q.start for q in hn})
        nxt = {s: (starts[i + 1] if i + 1 < len(starts) else None) for i, s in enumerate(starts)}
        for q in hn:
            n = nxt[q.start]
            if n is not None and q.end < n and n - q.start <= max(legato, q.end - q.start + grid):
                q.end = n
            q.end = max(q.end, q.start + (tunit(q.start) or grid))
        # when a new onset would exceed the hand's limits together with what is
        # still held, the least important note gives way: a held note is cut
        # short there, a new note is left out (an overtone ghost arriving with
        # a chord must not cut the tune short)
        active: list[QNote] = []
        dropped: set[int] = set()
        for s in starts:
            new = [q for q in hn if q.start == s]
            active = [a for a in active if a.end > s]
            protected = {id(outer_of[(hand, s)])} if (hand, s) in outer_of else set()
            pool = active + new
            while pool:
                ps = [x.pitch for x in pool]
                if len(pool) <= level.max_notes[hand] and max(ps) - min(ps) <= level.span[hand]:
                    break
                victims = [x for x in pool if id(x) not in protected] or pool
                v = min(victims, key=lambda x: imp.get(id(x), 0.0))
                pool.remove(v)
                if v.start == s:
                    dropped.add(id(v))
                else:
                    v.end = s
            active = [x for x in pool]
        out += [q for q in hn if q.end > q.start and id(q) not in dropped]

    # ---- 6. left hand stays below the right hand
    rh = sorted((q for q in out if q.hand == RIGHT), key=lambda q: q.start)
    res = []
    for q in out:
        if q.hand == LEFT:
            above = [r.pitch for r in rh if r.start < q.end and r.end > q.start]
            if above and q.pitch > min(above):
                continue            # would cross into the right hand: leave it out
        res.append(q)
    res.sort(key=lambda q: (q.start, q.pitch))
    return res


def playability(notes: list[QNote], crotchets_per_second: float = 2.0) -> dict:
    """How hard a piano part is to play, per hand: onsets per second (mean and
    the busiest second), notes held at once (95th percentile and maximum), the
    widest interval held at once, and the share of semiquavers or shorter."""
    out = {}
    tps = 480.0 * crotchets_per_second               # ticks per second
    for hand, name in ((LEFT, "left"), (RIGHT, "right")):
        hn = [q for q in notes if (q.hand if q.hand is not None else int(q.pitch >= 60)) == hand]
        if not hn:
            out[name] = dict(notes=0)
            continue
        starts = np.array(sorted({q.start for q in hn}), float)
        dur_s = (max(q.end for q in hn) - min(q.start for q in hn)) / tps
        busiest = 0
        j = 0
        for i in range(starts.size):
            while starts[i] - starts[j] >= tps:
                j += 1
            busiest = max(busiest, i - j + 1)
        ev = sorted([(q.start, 1, q.pitch) for q in hn] + [(q.end, 0, q.pitch) for q in hn])
        held: dict[int, int] = {}
        sizes, spans = [], []
        for t, kind, p in ev:
            if kind:
                held[p] = held.get(p, 0) + 1
                sizes.append(len(held))
                spans.append(max(held) - min(held))
            else:
                held[p] -= 1
                if held[p] <= 0:
                    del held[p]
        out[name] = dict(notes=len(hn), onsets_per_s=round(starts.size / max(dur_s, 1e-9), 2),
                         busiest_second=int(busiest),
                         held_p95=float(np.percentile(sizes, 95)), held_max=int(max(sizes)),
                         span_max=int(max(spans)), span_p95=float(np.percentile(spans, 95)),
                         short_share=round(float(np.mean([q.end - q.start <= 120 for q in hn])), 3))
    return out
