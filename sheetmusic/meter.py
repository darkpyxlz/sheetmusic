"""Time-signature (meter) and downbeat detection.

The beat tracker gives a pulse P but not how beats group into bars, nor
whether a beat divides in two (simple time: 2/4, 3/4, 4/4) or three
(compound time: 6/8, 9/8, 12/8).  A tracker can also lock onto *pairs of
eighth notes* in 6/8 (a "quarter-note" pulse that is not a real beat).

We place a grid of six slots per tracked beat (so both halves and thirds are
on it) and measure an *accent* in every slot from several cues:

* onset strength (any attack),
* note onsets, weighted by note length and extra for bass notes (long notes
  and bass notes tend to start on strong beats),
* harmonic change (chroma novelty from the spectrum; chords tend to change on
  downbeats).

Every candidate meter is then scored by how strongly its downbeats, beats and
beat subdivisions stand out from the other positions, over all bars, for
every possible downbeat position.  A mild prior favours the common meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notes import Note
from .rhythm import BeatMap
from .spectral import hz_to_midi


@dataclass
class Meter:
    num: int
    den: int
    compound: bool
    beat_ticks: int           # ticks of the notated beat (480 = crotchet, 720 = dotted crotchet)
    beat_map: BeatMap         # notated beats
    downbeat_phase: int       # which notated beat index (mod beats_per_bar) is a downbeat
    beats_per_bar: int
    scores: dict

    @property
    def name(self) -> str:
        return f"{self.num}/{self.den}"


# name -> (bar length in slots, notated beat length in slots, subdivision,
#          num, den, compound, prior)
_HYP = {
    "4/4": (24, 6, 2, 4, 4, False, 0.10),
    "3/4": (18, 6, 2, 3, 4, False, 0.0),
    "2/4": (12, 6, 2, 2, 4, False, -0.12),
    "6/8": (18, 9, 2, 6, 8, True, 0.0),       # tracker locked on pairs of eighths
    "6/8t": (12, 6, 3, 6, 8, True, 0.02),     # tracked beat = dotted crotchet
    "12/8t": (24, 6, 3, 12, 8, True, 0.0),
    "9/8t": (18, 6, 3, 9, 8, True, -0.08),
    # tracker locked on the eighth-note pulse: real beat = 2 tracked beats
    "4/4x2": (48, 12, 2, 4, 4, False, 0.10),
    "3/4x2": (36, 12, 2, 3, 4, False, 0.0),
    "6/8x2": (36, 18, 2, 6, 8, True, 0.0),   # beat = 3 tracked eighths
    "12/8x2": (72, 18, 2, 12, 8, True, -0.02),
}


def chroma(spec: np.ndarray, df: float, ref_hz: float, fmin: float = 60.0,
           fmax: float = 2000.0) -> np.ndarray:
    """(T, 12) pitch-class energy from a magnitude spectrum."""
    K = spec.shape[1]
    f = np.arange(K) * df
    sel = (f >= fmin) & (f <= fmax)
    pc = np.round(hz_to_midi(f[sel], ref_hz)).astype(int) % 12
    M = np.zeros((sel.sum(), 12), np.float32)
    M[np.arange(sel.sum()), pc] = 1.0
    C = np.sqrt(spec[:, sel].astype(np.float32)) @ M
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)


def detect_meter(bm: BeatMap, onset: np.ndarray, fps: float, notes: list[Note],
                 spec: np.ndarray | None = None, df: float = 0.0, ref_hz: float = 440.0,
                 only: list[str] | None = None, debug: bool = False,
                 lead: list[Note] | None = None) -> Meter:
    bt = bm.beat_times
    nb = bt.size
    if nb < 8:
        return _make("4/4", bm, 0, {})
    N = 6 * (nb - 1) + 1
    u = np.arange(N) / 6.0
    slot_t = np.interp(u, np.arange(nb), bt)
    P = bm.period

    # --- cue 1: onset strength
    o = onset / (np.percentile(onset, 95) + 1e-9)
    fr = np.clip(np.round(slot_t * fps).astype(int), 0, o.size - 1)
    a_on = np.max(np.stack([o[np.clip(fr + d, 0, o.size - 1)] for d in (-2, -1, 0, 1, 2)]), axis=0)

    # --- cue 2: note onsets (length- and bass-weighted)
    a_note = np.zeros(N)
    a_bass = np.zeros(N)
    for n in notes:
        k = float(bm.to_beats(n.start)) * 6.0
        kk = int(round(k))
        if 0 <= kk < N and abs(k - kk) <= 0.5:
            dur = min((n.end - n.start) / P, 2.0)
            a_note[kk] += 0.3 + dur
            if n.pitch < 52:
                a_bass[kk] += 0.5 + dur
    # --- cue 3: harmonic change (chroma novelty over one tracked beat)
    a_chord = np.zeros(N)
    if spec is not None and df > 0:
        C = chroma(spec, df, ref_hz)
        cs = np.vstack([np.zeros((1, 12)), np.cumsum(C, axis=0)])
        w = max(2, int(round(max(P, 0.7) * fps)))   # harmony moves slower than notes
        T = C.shape[0]
        for k in range(0, N, 3):
            t = int(round(slot_t[k] * fps))
            if t - w < 0 or t + w > T:
                continue
            before = cs[t] - cs[t - w]
            after = cs[t + w] - cs[t]
            cos = before @ after / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-9)
            a_chord[k] = 1.0 - cos

    def norm(x):
        m = x[x > 0].mean() if np.any(x > 0) else 1.0
        return x / (m + 1e-9)

    a_on, a_note, a_bass, a_chord = norm(a_on), norm(a_note), norm(a_bass), norm(a_chord)
    A_bar = 2.0 * a_chord + a_bass + 0.5 * a_note + 0.3 * a_on      # downbeat evidence
    A_beat = a_on + a_note + 0.5 * a_chord + 0.5 * a_bass            # beat evidence

    # --- subdivision of the tracked beat: halves vs thirds
    k6 = np.arange(N) % 6
    halves = A_beat[k6 == 3].mean()
    thirds = A_beat[(k6 == 2) | (k6 == 4)].mean()
    sub_tri = (thirds - halves) / (thirds + halves + 1e-9)

    # rhythm of the lead line (vocal melody, or the top line of the music):
    # tunes move mostly in crotchets, which settles double/half-tempo
    # ambiguities that the accent pattern alone cannot.
    lead_ioi = None
    src = lead if lead and len(lead) >= 8 else skyline(notes)
    if len(src) >= 8:
        on = np.sort(np.array([n.start for n in src]))
        ioi = np.diff(on)
        ioi = ioi[ioi > 0.08]
        if ioi.size >= 6:
            lead_ioi = float(np.median(ioi))

    scores = {}
    best = None
    for name, (bar, beat, sub, num, den, comp, prior) in _HYP.items():
        if only and name not in only:
            continue
        if N < 2 * bar:
            continue
        best_ph, best_s = 0, -np.inf
        for ph in range(0, bar, 3 if beat % 6 else 6):
            down = A_bar[ph::bar]
            # downbeats against every other half-beat position in the bar (the
            # same reference for all meters with this bar length)
            others_pos = [(ph + j) % bar for j in range(3, bar, 3)]
            others = np.concatenate([A_bar[p::bar] for p in others_pos])
            bar_c = (down.mean() - others.mean()) / (down.mean() + others.mean() + 1e-9)
            # beats should stand out from the off-beat half-positions in the bar
            on_b = np.concatenate([A_beat[(ph + j * beat) % bar::bar] for j in range(bar // beat)])
            off_pos = [p for p in range(0, bar, 3) if (p - ph) % beat]
            off_b = np.concatenate([A_beat[(p) % bar::bar] for p in off_pos]) if off_pos else np.zeros(1)
            beat_c = (on_b.mean() - off_b.mean()) / (on_b.mean() + off_b.mean() + 1e-9)
            sub_c = sub_tri if sub == 3 else -sub_tri
            bpm = 60.0 / (P * beat / 6.0)
            # typical tempi: ~100 crotchets/min, ~70 dotted crotchets/min
            tempo_prior = -0.35 * np.log2(bpm / (70.0 if comp else 100.0)) ** 2
            rhythm_c = 0.0
            if lead_ioi is not None:
                # no penalty while the typical gap is 3/4 .. 1 1/2 beats
                x = lead_ioi / (P * beat / 6.0) / (0.67 if comp else 1.0)
                dev = max(0.0, np.log2(0.75 / x)) + max(0.0, np.log2(x / 1.5))
                rhythm_c = -0.6 * dev ** 2
            s = 1.0 * bar_c + 0.6 * beat_c + 0.8 * sub_c + prior + tempo_prior + rhythm_c
            if debug:
                comps = (round(bar_c, 3), round(beat_c, 3), round(sub_c, 3), round(tempo_prior, 3),
                         round(rhythm_c, 3))
            if s > best_s:
                best_s, best_ph = s, ph
                best_comp = comps if debug else None
        scores[name] = round(float(best_s), 3)
        if debug:
            print("   meter", name, round(float(best_s), 3), "bar/beat/sub/tempo", best_comp)
        if best is None or best_s > best[1]:
            best = (name, best_s, best_ph)
    if best is None:
        return _make("4/4", bm, 0, scores)
    return _make(best[0], bm, best[2], scores)


def skyline(notes: list[Note], min_dur: float = 0.1, merge: float = 0.06) -> list[Note]:
    """Top line of a polyphonic texture: an onset belongs to it only if its
    highest note is at least as high as everything still sounding then (so an
    accompaniment figure under a held melody note is not taken for melody)."""
    ns = sorted((n for n in notes if n.end - n.start >= min_dur), key=lambda n: n.start)
    out: list[Note] = []
    i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1].start - ns[i].start <= merge:
            j += 1
        top = max(ns[i:j + 1], key=lambda g: g.pitch)
        t = ns[i].start
        sounding = [m.pitch for m in ns[:i] if m.end > t + 0.05]
        if not sounding or top.pitch >= max(sounding):
            out.append(top)
        i = j + 1
    return out


def _make(name: str, bm: BeatMap, ph_slots: int, scores: dict) -> Meter:
    bar, beat, sub, num, den, comp, _ = _HYP[name]
    bt = bm.beat_times
    nb = bt.size
    # notated beat grid: every `beat` slots, aligned so slot ph is a beat
    s0 = ph_slots % beat
    N = 6 * (nb - 1) + 1
    slots = np.arange(s0, N, beat)
    if slots.size >= 2:
        times = np.interp(slots / 6.0, np.arange(nb), bt)
        # extend one beat backwards so notes just before the first tracked beat fit
        nbm = BeatMap.from_beats(times, 60.0 / (bm.period * beat / 6.0))
    else:
        nbm = bm
    beats_per_bar = bar // beat
    down = ((ph_slots - s0) // beat) % beats_per_bar
    beat_ticks = 720 if comp else 4 * 480 // den
    return Meter(num, den, comp, beat_ticks, nbm, down, beats_per_bar,
                 {k.replace("x2", "").replace("t", ""): v for k, v in scores.items()} | scores)
