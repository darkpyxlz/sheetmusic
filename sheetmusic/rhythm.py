"""Onset envelope, tempo estimation, dynamic-programming beat tracking
(Ellis 2007), downbeat phase and quantisation onto a beat grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from .notes import Note, QNote


def onset_envelope(flux: np.ndarray, fps: float) -> np.ndarray:
    o = flux.astype(float)
    o = o - uniform_filter1d(o, size=max(3, int(round(0.5 * fps))), mode="nearest")
    o = np.maximum(o, 0.0)
    sd = o.std()
    return o / sd if sd > 0 else o


def estimate_tempo(o: np.ndarray, fps: float, bpm_min: float = 55.0, bpm_max: float = 200.0,
                   prior_bpm: float = 110.0, prior_octaves: float = 0.9) -> float:
    if o.size < 8 or not np.any(o > 0):
        return 120.0
    x = o - o.mean()
    n = 1 << int(np.ceil(np.log2(2 * x.size)))
    F = np.fft.rfft(x, n)
    ac = np.fft.irfft(F * np.conj(F), n)[: x.size]
    lags = np.arange(ac.size, dtype=float)
    lo = int(np.floor(60.0 * fps / bpm_max))
    hi = min(ac.size - 2, int(np.ceil(60.0 * fps / bpm_min)))
    if hi <= lo + 2:
        return 120.0
    L = lags[lo:hi + 1]
    bpm = 60.0 * fps / L
    w = np.exp(-0.5 * (np.log2(bpm / prior_bpm) / prior_octaves) ** 2)
    # reinforce with the half-lag and double-lag (metrical structure)
    a = ac[lo:hi + 1].copy()
    dbl = np.interp(2 * L, lags, ac, right=0.0)
    a = a + 0.5 * dbl
    score = a * w
    k = int(np.argmax(score))
    lag = L[k]
    if 0 < k < L.size - 1:
        y0, y1, y2 = score[k - 1], score[k], score[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            lag += 0.5 * (y0 - y2) / den
    return float(60.0 * fps / lag)


def track_beats(o: np.ndarray, fps: float, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Return beat times (seconds)."""
    period = 60.0 * fps / bpm
    T = o.size
    if T < 2 or not np.any(o > 0):
        return np.arange(0, T / fps, 60.0 / bpm)
    oo = o / (o.std() + 1e-9)
    # local score smoothed with a Gaussian of width ~ period/32 (as librosa)
    win = np.exp(-0.5 * (np.arange(-period, period + 1) * 32.0 / period) ** 2)
    local = np.convolve(oo, win, mode="same")
    offs = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    txwt = -tightness * (np.log(-offs / period)) ** 2
    score = np.zeros(T)
    back = np.full(T, -1, dtype=int)
    for t in range(T):
        prev = t + offs
        ok = prev >= 0
        if np.any(ok):
            cand = score[prev[ok]] + txwt[ok]
            j = int(np.argmax(cand))
            if cand[j] > 0:
                score[t] = local[t] + cand[j]
                back[t] = prev[ok][j]
                continue
        score[t] = local[t]
    # last beat: the strongest local maximum within the final period
    tail = max(0, T - int(round(2 * period)))
    seg = score[tail:]
    t = tail + int(np.argmax(seg))
    beats = [t]
    while back[t] >= 0:
        t = back[t]
        beats.append(t)
    beats = np.array(beats[::-1], dtype=float)
    # trim weak beats at the edges (silence / fade-out)
    strength = uniform_filter1d(local, size=max(1, int(period)))[beats.astype(int)]
    thr = 0.15 * np.sqrt(np.mean(local ** 2))
    good = np.nonzero(strength > thr)[0]
    if good.size >= 2:
        beats = beats[good[0]:good[-1] + 1]
    return beats / fps


@dataclass
class BeatMap:
    beat_times: np.ndarray
    period: float  # seconds per beat (median)

    @classmethod
    def from_beats(cls, beat_times: np.ndarray, fallback_bpm: float) -> "BeatMap":
        bt = np.asarray(beat_times, float)
        if bt.size >= 2:
            period = float(np.median(np.diff(bt)))
        else:
            period = 60.0 / fallback_bpm
            bt = np.array([bt[0] if bt.size else 0.0])
        return cls(bt, period)

    @classmethod
    def constant(cls, bpm: float, first_beat: float = 0.0, duration: float = 0.0) -> "BeatMap":
        p = 60.0 / bpm
        n = int(np.ceil(max(duration - first_beat, 0) / p)) + 2
        return cls(first_beat + p * np.arange(n), p)

    @property
    def bpm(self) -> float:
        return 60.0 / self.period

    def to_beats(self, t):
        t = np.asarray(t, float)
        bt = self.beat_times
        if bt.size == 1:
            return (t - bt[0]) / self.period
        out = np.interp(t, bt, np.arange(bt.size, dtype=float))
        out = np.where(t < bt[0], (t - bt[0]) / self.period, out)
        out = np.where(t > bt[-1], bt.size - 1 + (t - bt[-1]) / self.period, out)
        return out


def downbeat_phase(bm: BeatMap, o: np.ndarray, fps: float, beats_per_bar: int,
                   notes: list[Note] | None = None) -> int:
    """Which beat index (mod beats_per_bar) is most likely the downbeat."""
    if beats_per_bar <= 1:
        return 0
    n = bm.beat_times.size
    strength = np.zeros(n)
    idx = np.clip(np.round(bm.beat_times * fps).astype(int), 0, max(0, o.size - 1))
    if o.size:
        strength += o[idx] / (o.max() + 1e-9)
    if notes:
        # note onsets, low notes weighted more (bass tends to change on the 1)
        for nt in notes:
            b = float(bm.to_beats(nt.start))
            k = int(round(b))
            if 0 <= k < n and abs(b - k) < 0.15:
                strength[k] += 0.5 + max(0, (60 - nt.pitch)) / 24.0
    scores = [strength[p::beats_per_bar].mean() if strength[p::beats_per_bar].size else 0
              for p in range(beats_per_bar)]
    return int(np.argmax(scores))


def snap(b: float, grid: int, tol_coarse: float = 0.12) -> float:
    """Snap a beat position to the *coarsest* grid that is close enough.

    Tries whole beats, then halves, quarters, eighths ... up to ``grid``
    subdivisions per beat.  A position is snapped to a coarse grid when it is
    within ``tol`` of it, where ``tol`` shrinks with the subdivision; the
    finest grid always matches.  So a note sung 50 ms late still lands on the
    beat, while genuinely fast or off-beat notes get 16ths / 32nds."""
    # grids tried coarsest first: 1, 2, 4, 8 ... or, for compound time
    # (grid divisible by 3: beat = dotted crotchet), 1, 3, 6, 12 ...
    g = 1
    step = 3 if grid % 3 == 0 else 2
    while g < grid:
        q = round(b * g) / g
        tol = tol_coarse if g <= 3 else min(tol_coarse, 0.36 / g)
        if abs(b - q) <= tol:
            return q
        g *= step
        step = 2
    return round(b * grid) / grid


UNIT = 20      # internal tick unit: divides every binary value and every triplet (160, 80 ...)


def _near(f, targets, tol):
    return min(abs(f - t) for t in targets) < tol


def detect_swing(positions, compound: bool, min_hits: int = 8) -> bool:
    """Swung quavers: the off-beat quavers of the piece land near 2/3 of the
    beat (long-short), not at 1/2, and the middle third of the beat is
    rarely used (that would be real triplets, or a jig).  ``positions`` are onsets in
    notated beats."""
    if compound or len(positions) < min_hits:
        return False
    f = np.mod(np.asarray(positions, float), 1.0)
    n23 = int(np.sum(np.abs(f - 2 / 3) < 0.06))
    n12 = int(np.sum(np.abs(f - 0.5) < 0.06))
    n13 = int(np.sum(np.abs(f - 1 / 3) < 0.06))
    # jigs and slip jigs use the middle third 0.28-1.6 times as often as the
    # last; a swung tune almost never (0.06 for the synthetic one)
    return n23 >= min_hits and n23 >= 2 * n12 and n13 <= 0.2 * n23


def detect_triplet_beats(positions, compound: bool, swing: bool = False,
                         tol: float = 0.06, margin: float = 0.03, min_hits: int = 2,
                         sixths: bool = True) -> dict[int, int]:
    """Beats (notated beat numbers) divided in three: {beat: 3} for quaver
    triplets (thirds of the beat), {beat: 6} for semiquaver triplets (sixths:
    a triplet on each half of the beat).

    An onset is a third-hit when it is within ``tol`` of 1/3 or 2/3 of the
    beat and clearly closer to it than to any semiquaver position (1/4, 1/2,
    3/4); a sixth-hit likewise for 1/6 or 5/6; a binary hit is clearly on
    1/4 or 3/4 (1/2 belongs to both binary and sixths).  A beat with a binary
    hit stays binary.  Otherwise it is in sixths with a sixth-hit and at
    least ``min_hits`` hits on sixths, in thirds with ``min_hits`` third-hits,
    or with one hit when a neighbouring beat is a triplet beat (triplets come
    in runs).  Under swing a beat needs a hit on its first third (swung
    quavers alone sit on 2/3).  Compound metres divide in three anyway."""
    if compound or not len(positions):
        return {}
    pos = np.asarray(positions, float)
    k = np.floor(pos + 0.04).astype(int)
    f = pos - k
    inner = (f > 0.08) & (f < 0.92)
    d16 = np.min(np.abs(f[:, None] - np.array([0.25, 0.5, 0.75])[None, :]), axis=1)
    d3 = np.minimum(np.abs(f - 1 / 3), np.abs(f - 2 / 3))
    d6 = np.minimum(np.abs(f - 1 / 6), np.abs(f - 5 / 6))
    q4 = np.minimum(np.abs(f - 0.25), np.abs(f - 0.75))
    third = inner & (d3 < tol) & (d3 < d16 - margin)
    sixth = inner & (d6 < tol) & (d6 < d16 - margin) & sixths
    binh = inner & (q4 < tol) & (q4 < np.minimum(d3, d6) - margin)
    half = inner & (np.abs(f - 0.5) < tol)
    first = inner & (np.abs(f - 1 / 3) < tol)
    beats: dict[int, list[int]] = {}
    for kk, a, b6, bh, hh, fi in zip(k, third, sixth, binh, half, first):
        c = beats.setdefault(int(kk), [0, 0, 0, 0, 0])
        c[0] += int(a)
        c[1] += int(b6)
        c[2] += int(bh)
        c[3] += int(hh)
        c[4] += int(fi)
    out: dict[int, int] = {}
    for kk, (n3, n6, nb, nh, nf) in beats.items():
        if nb or (swing and not nf):
            continue
        if n6 and n6 + n3 + nh >= min_hits:
            out[kk] = 6
        elif n3 >= min_hits:
            out[kk] = 3
    for kk, (n3, n6, nb, nh, nf) in beats.items():
        if kk in out or nb or (swing and not nf) or not (kk - 1 in out or kk + 1 in out):
            continue
        if n6:
            out[kk] = 6
        elif n3:
            out[kk] = 3
    return out


class Grid:
    """Where notes may be written: per beat the binary adaptive grid
    (``snap``), thirds (quaver triplets) or sixths (semiquaver triplets);
    under swing, off-beat quavers played long-short are written straight
    (the score says "Swing")."""

    def __init__(self, ticks_per_beat: int, grid: int, triplets: dict[int, int] | None = None,
                 swing: bool = False):
        self.tpb = ticks_per_beat
        self.grid = grid
        self.triplets = triplets or {}
        self.swing = swing
        self.ok = ticks_per_beat % 6 == 0

    def snap_units(self, b: float, tol: float) -> int:
        k = int(np.floor(b + 0.04))
        f = b - k
        m = self.triplets.get(k) if self.ok else None
        if m:
            return int(round((k + round(f * m) / m) * self.tpb / UNIT))
        if self.swing and abs(f - 2 / 3) < 0.12:
            return int(round((k + 0.5) * self.tpb / UNIT))
        return int(round(snap(b, self.grid, tol) * self.tpb / UNIT))

    def allows(self, units: int) -> bool:
        """Can a note of this stream start or end here?"""
        t = units * UNIT
        r = t % self.tpb
        if r == 0:
            return True
        m = self.triplets.get(t // self.tpb)
        if m:
            return (r * m) % self.tpb == 0
        return r % 30 == 0


def quantize(notes: list[Note], bm: BeatMap, beat_offset: float, ticks_per_beat: int,
             grid: int, monophonic: bool = False, fill_gaps: float = 0.0,
             min_units: int = 1, snap_tol: float = 0.12, swing: bool = False,
             triplets: bool = True) -> list[QNote]:
    """Quantise notes adaptively, down to ``grid`` subdivisions per beat
    (8 = demisemiquavers in x/4).

    ``beat_offset`` is added to the beat position so that the first downbeat
    lands on a barline.  ``fill_gaps`` (in beats) extends a note to the next
    one if the rest in between would be at most that long (legato).
    """
    unit = UNIT
    min_units = min_units * (ticks_per_beat // grid) // UNIT
    pos = [float(bm.to_beats(n.start)) + beat_offset for n in notes]
    trip = detect_triplet_beats(pos, ticks_per_beat % 3 != 0 or ticks_per_beat in (720, 360, 1440),
                                swing) if triplets else {}
    G = Grid(ticks_per_beat, grid, trip, swing)
    out: list[QNote] = []
    for n in notes:
        b0 = float(bm.to_beats(n.start)) + beat_offset
        b1 = float(bm.to_beats(n.end)) + beat_offset
        s = G.snap_units(b0, snap_tol)
        e = G.snap_units(b1, snap_tol * 1.5)
        if e - s < min_units:
            # keep short notes only if they were at least ~half a unit long
            if (b1 - b0) * ticks_per_beat / UNIT >= 0.5 * min_units:
                e = s + min_units
            else:
                continue
        if s < 0:
            continue
        out.append(QNote(s * unit, e * unit, n.pitch, n.velocity))
    gap_ticks = int(round(fill_gaps * ticks_per_beat))
    out.sort(key=lambda q: (q.start, q.pitch))
    if monophonic:
        mono: list[QNote] = []
        for q in out:
            if mono and q.start < mono[-1].end:
                prev = mono[-1]
                if q.start <= prev.start:
                    # same onset: keep the longer one
                    if q.end - q.start > prev.end - prev.start:
                        mono[-1] = q
                    continue
                prev.end = q.start
            mono.append(q)
        out = mono
        if fill_gaps > 0:
            for a, b in zip(out[:-1], out[1:]):
                if 0 < b.start - a.end <= gap_ticks:
                    a.end = b.start
    else:
        # merge overlapping same-pitch notes
        merged: dict[int, QNote] = {}
        res: list[QNote] = []
        for q in out:
            last = merged.get(q.pitch)
            if last is not None and q.start < last.end:
                last.end = max(last.end, q.end)
                continue
            if last is not None and 0 < q.start - last.end <= gap_ticks:
                last.end = q.start
            merged[q.pitch] = q
            res.append(q)
        out = sorted(res, key=lambda q: (q.start, q.pitch))
    return [q for q in out if q.end > q.start]


# Written length against sounding length, r = log2(written / sounding), as
# cost per octave.  Measured on the benchmark scores: a sustained note is
# written within +-0.2 octaves of how long it sounds (3 % are written more than
# 0.3 octaves longer), so both directions are expensive; a struck note is
# written over an octave longer than it sounds in 12-21 % of cases (it fades or
# is damped), so lengthening it is cheap.
VALUE_SLOPES = {False: (9.0, 4.0), True: (1.1, 4.0)}     # struck -> (longer, shorter)


def quantize_poly(notes: list[Note], bm: BeatMap, beat_offset: float, ticks_per_beat: int,
                  grid: int, snap_tol: float = 0.18, min_units: int = 1,
                  chord_window: float = 0.035, rest_cost: float = 3.2,
                  near_cost: float = 0.4, far_cost: float = 1.4, later_cost: float = 0.7,
                  hand_cost: float = 0.3, hand_far_cost: float = 0.9, other_hand_cost: float = 2.2,
                  max_extend: float = 4.0, voice_span: int = 5, split: int = 60,
                  slopes: dict | None = None, use_hands: bool = True, swing: bool = False,
                  triplets: bool = True, compound: bool = False) -> list[QNote]:
    """Quantise polyphonic notes and choose their written values.

    Starts: notes struck together (within ``chord_window`` s) get one
    position, snapped adaptively like a single note.

    Ends: in real scores a note almost always ends exactly where another
    note begins (96 % of the notes in the benchmark scores; 4 % before a
    rest).  So the candidates for a note's end are the *starts of later
    notes*, plus the grid position where the sound stopped (a rest follows).
    Each candidate has a cost in nats,

        cost = prior + slope * |log2(written / sounding)|

    * slope: see ``VALUE_SLOPES``; it depends on whether the note decays
      freely (``struck``) and on the direction;
    * prior for a rest: ``rest_cost`` (-ln 0.04);
    * prior for an onset, sustained notes: ``near_cost`` where a note within
      ``voice_span`` semitones starts (this voice moving on), else
      ``far_cost``;
    * prior for an onset, struck notes: these are piano-like, and in piano
      scores about 90 % of notes end at the very next onset *of their own
      hand* (see ``hands.assign_hands``): ``hand_cost`` for an onset of the
      same hand near this pitch, ``hand_far_cost`` for one of the same hand
      elsewhere, ``other_hand_cost`` when only the other hand plays;
    * ``later_cost`` more for every such "own" onset passed over.
    """
    from .hands import assign_hands
    slopes = slopes or VALUE_SLOPES
    old_unit = ticks_per_beat // grid
    unit = UNIT
    min_units = max(1, min_units * old_unit // UNIT)
    if not notes:
        return []
    order = sorted(notes, key=lambda n: (n.start, n.pitch))
    # ---- chords: notes struck together share one position
    pos = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1].start - order[i].start <= chord_window:
            j += 1
        b0 = float(np.mean([float(bm.to_beats(n.start)) for n in order[i:j + 1]])) + beat_offset
        for k in range(i, j + 1):
            pos[k] = b0
        i = j + 1
    # ---- hands from a provisional binary snap, then each hand's own grid:
    # triplets are decided per hand, so two against three stays possible
    binary = Grid(ticks_per_beat, grid, {}, swing)
    prov = [binary.snap_units(b, snap_tol) for b in pos]
    hands = assign_hands([QNote(s * unit, s * unit + old_unit, n.pitch) for s, n in zip(prov, order)],
                         ticks_per_beat, split=split)
    grids = {}
    for h in set(hands):
        hp = [b for b, hh in zip(pos, hands) if hh == h]
        grids[h] = Grid(ticks_per_beat, grid,
                        detect_triplet_beats(hp, compound, swing) if triplets else {}, swing)
    starts = [grids[h].snap_units(b, snap_tol) for b, h in zip(pos, hands)]
    onset_units = np.array(sorted(set(starts)))
    by_onset: dict[int, list[tuple[int, int]]] = {}
    for k, s in enumerate(starts):
        by_onset.setdefault(s, []).append((order[k].pitch, hands[k]))
    out: list[QNote] = []
    for k, n in enumerate(order):
        s = starts[k]
        if s < 0:
            continue
        b0 = float(bm.to_beats(n.start)) + beat_offset
        b1 = float(bm.to_beats(n.end)) + beat_offset
        G = grids[hands[k]]
        sound = max((b1 - b0) * ticks_per_beat / UNIT, 0.25)  # sounding length in units
        if sound < 0.5 * min_units:
            continue                                           # a blip, not a note
        up, down = slopes[bool(n.struck)]

        def fit(length):
            r = np.log2(length / sound)
            return up * r if r > 0 else -down * r
        # a rest follows: the end is where the sound stopped, snapped like a start
        g = G.snap_units((s + sound) * UNIT / ticks_per_beat, snap_tol * 1.5)
        if g < s + min_units:
            g = s + min_units
            while not G.allows(g):
                g += 1
        best_e, best_c = g, rest_cost + fit(g - s)
        passed = 0
        lo = np.searchsorted(onset_units, s + min_units)
        for o in onset_units[lo:]:
            length = o - s
            if length > max(sound * 8.0, max_extend * ticks_per_beat / UNIT):
                break
            if not G.allows(int(o)):
                continue                                       # not on this hand's grid
            there = by_onset[int(o)]
            near = any(abs(q - n.pitch) <= voice_span for q, _h in there)
            if n.struck and use_hands:
                same = [q for q, h in there if h == hands[k]]
                own = bool(same)
                prior = other_hand_cost if not same else (
                    hand_cost if any(abs(q - n.pitch) <= voice_span for q in same) else hand_far_cost)
            else:
                own = near
                prior = near_cost if near else far_cost
            c = prior + later_cost * passed + fit(length)
            if c < best_c:
                best_e, best_c = int(o), c
            passed += own
        out.append(QNote(s * unit, best_e * unit, n.pitch, n.velocity, n.struck, hands[k]))
    # ---- same pitch: a note cannot overlap its own repetition
    out.sort(key=lambda q: (q.start, q.pitch))
    last: dict[int, QNote] = {}
    res: list[QNote] = []
    for q in out:
        prev = last.get(q.pitch)
        if prev is not None and q.start < prev.end:
            if q.start <= prev.start:
                prev.end = max(prev.end, q.end)
                continue
            prev.end = q.start
        last[q.pitch] = q
        res.append(q)
    return [q for q in res if q.end > q.start]


def fix_tempo_octave(bm: BeatMap, notes: list[Note], o: np.ndarray, fps: float,
                     slow_bpm: float = 50.0, fast_bpm: float = 200.0) -> BeatMap:
    """Resolve double/half tempo using the note rhythm.

    Beat trackers often lock onto the 8th-note pulse (e.g. hi-hats) of a slow
    song, which writes every note twice as long as it should be (or the
    reverse).  A sung melody's median inter-onset interval is normally between
    half a beat and a beat, so we halve/double the beat grid when it is not.
    """
    if len(notes) < 8 or bm.beat_times.size < 8:
        return bm
    on = np.sort(np.array([n.start for n in notes]))
    ioi = np.diff(bm.to_beats(on))
    ioi = ioi[ioi > 0.12]
    if ioi.size < 6:
        return bm
    med = float(np.median(ioi))
    bt = bm.beat_times
    if med >= 1.8 and bm.bpm / 2 >= slow_bpm:
        idx = np.clip(np.round(bt * fps).astype(int), 0, o.size - 1)
        ph = int(np.argmax([o[idx[p::2]].sum() for p in (0, 1)]))
        return BeatMap.from_beats(bt[ph::2], bm.bpm / 2)
    if med <= 0.3 and bm.bpm * 2 <= fast_bpm:
        mid = 0.5 * (bt[:-1] + bt[1:])
        nb = np.empty(bt.size + mid.size)
        nb[0::2] = bt
        nb[1::2] = mid
        return BeatMap.from_beats(nb, bm.bpm * 2)
    return bm
