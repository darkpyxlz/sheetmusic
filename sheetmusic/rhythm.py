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


def quantize(notes: list[Note], bm: BeatMap, beat_offset: float, ticks_per_beat: int,
             grid: int, monophonic: bool = False, fill_gaps: int = 0,
             min_units: int = 1) -> list[QNote]:
    """Quantise notes to ``grid`` subdivisions per beat.

    ``beat_offset`` is added to the beat position so that the first downbeat
    lands on a barline.  ``fill_gaps`` (in grid units) extends a note to the
    next one if the rest in between would be at most that long (legato).
    """
    unit = ticks_per_beat // grid
    out: list[QNote] = []
    for n in notes:
        b0 = float(bm.to_beats(n.start)) + beat_offset
        b1 = float(bm.to_beats(n.end)) + beat_offset
        s = int(round(b0 * grid))
        e = int(round(b1 * grid))
        if e - s < min_units:
            # keep short notes only if they were at least ~half a unit long
            if (b1 - b0) * grid >= 0.5 * min_units:
                e = s + min_units
            else:
                continue
        if s < 0:
            continue
        out.append(QNote(s * unit, e * unit, n.pitch, n.velocity))
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
                if 0 < b.start - a.end <= fill_gaps * unit:
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
            if last is not None and fill_gaps > 0 and 0 < q.start - last.end <= fill_gaps * unit:
                last.end = q.start
            merged[q.pitch] = q
            res.append(q)
        out = sorted(res, key=lambda q: (q.start, q.pitch))
    return [q for q in out if q.end > q.start]


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
