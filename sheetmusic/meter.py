"""Time signature, beat level and downbeat.

The beat tracker delivers a steady pulse, but not which metrical level that
pulse is, how pulses group into bars, or whether the beat divides in two
(simple time) or three (compound time).  On real recordings it is often half
the beat, a third of it (the quavers of 6/8), or even three semiquavers of a
syncopated tune.

Music is periodic at every level of its metre: the same kinds of events come
back every subdivision, every beat and every bar.  So the metre is found as
the *hierarchy of periods* that the music's periodicities support best:

    subdivision  --x2 or x3-->  beat  --x2, x3 or x4-->  bar

The pulse is divided into twelve slots (so halves, thirds, quarters and
two-thirds of it are all on the grid) and three accent signals are sampled
there: attacks (onset strength and note starts, long notes counting more),
harmony (chord change and bass notes) and melodic shape (pitch height, which
reveals repeating figures in an even stream of notes).  Their
autocorrelations R(lag) say how strongly the music repeats at each lag.  A
hierarchy with beat period B and bar n*B scores

* R(B): there is a pulse at the beat;
* R(B/2) against R(B/3): how the beat divides;
* R at multiples of the bar against R at the rival groupings (in triple time
  3B and 6B repeat, 2B and 4B do not);
* a tempo preference (a beat near 110 per minute, a dotted beat near 75), a
  bar-length preference, and a readability preference: the usual gap between
  notes should come out as a quaver or a crotchet, not as a semiquaver;
* a small prior for the common signatures.

The downbeat is then the phase of the bar at which accents, harmony changes
and long notes fall on the barline and the fewest notes are tied across it.

Any time signature can be forced (5/4, 7/8, 2/2, 3/8, 12/16 ...): the search
is then only over beat level and downbeat.  Automatic detection chooses among
2/4, 3/4, 4/4, 3/8, 6/8, 9/8 and 12/8, plus groupings in five and seven
(5/4, 5/8, 7/4, 7/8) when ``odd_meters`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .notes import Note
from .rhythm import BeatMap
from .spectral import hz_to_midi

SLOTS = 12                                    # slots per tracked pulse
BEAT_SLOTS = (6, 8, 9, 12, 16, 18, 24, 36)    # beat = pulse x 1/2, 2/3, 3/4, 1, 4/3, 3/2, 2, 3

# Weights of the hierarchy score.  They were fitted by random search on the
# development pieces of the benchmark (real piano recordings with annotated
# beats, strict-tempo scores in eight metres), under the constraint that every
# synthetic test piece keeps its metre, and checked on held-out pieces.  The
# best fifteen settings found agree to within about 10 % on every weight.
WEIGHTS = dict(
    harm=0.3, pitch=0.1,          # share of harmony / melodic shape in bar-level periodicity
    pitch_beat=0.2,               # share of melodic shape in beat-level periodicity
    beat=1.0, sub=0.8, bar=1.5, bar_lag=0.5, four=0.32,
    tempo=0.8, beat_simple=0.54, beat_compound=0.85,      # preferred beat periods (s)
    bar_length=0.1, bar_seconds=2.0,                      # preferred bar length (s)
    readable=0.8,                 # the usual note gap should be 0.4 .. 1.25 beats
    min_pulse=0.1, rel_pulse=0.3, sub_default=0.25,
    cross=1.5, long_on_one=0.3, bar_accent=0.5, down_contrast=3.0,   # downbeat phase
    beat_contrast=2.0, bar_contrast=0.34, sub_contrast=1.0,  # phase-aware accent contrasts
)
# (subdivision, beats per bar) -> prior
PRIOR = {(2, 4): 0.10, (2, 3): 0.0, (2, 2): -0.10, (3, 2): 0.0, (3, 3): -0.08, (3, 4): -0.05,
         (2, 5): -0.35, (2, 7): -0.45}


@dataclass
class Meter:
    num: int
    den: int
    compound: bool
    beat_ticks: int           # ticks of the notated beat (480 = crotchet, 720 = dotted crotchet)
    beat_map: BeatMap         # notated beats
    downbeat_phase: int       # which notated beat index (mod beats_per_bar) is a downbeat
    beats_per_bar: int
    scores: dict = field(default_factory=dict)
    confidence: float = 0.0   # margin of the chosen metre over the best different one

    @property
    def name(self) -> str:
        return f"{self.num}/{self.den}"


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


def signature_spec(num: int, den: int):
    """(subdivision, beats per bar, beat ticks, compound) of a time signature.

    Compound signatures (6/8, 9/8, 12/8, 6/4, 6/16 ...) have a dotted beat that
    divides in three; in every other signature the beat is the denominator's
    note value (a quaver in 3/8, 5/8 and 7/8, a minim in 2/2)."""
    if den not in (1, 2, 4, 8, 16, 32) or num < 1:
        raise ValueError(f"unsupported time signature {num}/{den}")
    unit = 4 * 480 // den
    if den >= 4 and num % 3 == 0 and num > 3:          # 6/8, 9/8, 12/8, 6/4, 12/16 ...
        return 3, num // 3, 3 * unit, True
    return 2, num, unit, False


def _slot_accents(bm: BeatMap, onset: np.ndarray, fps: float, notes: list[Note],
                  C: np.ndarray | None) -> dict:
    S = SLOTS
    bt = bm.beat_times
    nb = bt.size
    N = S * (nb - 1) + 1
    slot_t = np.interp(np.arange(N) / S, np.arange(nb), bt)
    P = bm.period
    o = onset / (np.percentile(onset, 95) + 1e-9)
    fr = np.clip(np.round(slot_t * fps).astype(int), 0, o.size - 1)
    a_on = np.max(np.stack([o[np.clip(fr + k, 0, o.size - 1)] for k in (-1, 0, 1)]), axis=0)
    a_note = np.zeros(N)
    a_bass = np.zeros(N)
    psum = np.zeros(N)
    pcnt = np.zeros(N)
    spans = []
    for n in notes:
        a = float(bm.to_beats(n.start)) * S
        kk = int(round(a))
        if 0 <= kk < N:
            dur = min((n.end - n.start) / P, 2.0)
            a_note[kk] += 0.3 + dur                   # long notes start on strong positions
            if n.pitch < 52:
                a_bass[kk] += 0.5 + dur
            psum[kk] += n.pitch
            pcnt[kk] += 1
            spans.append((a, float(bm.to_beats(n.end)) * S))
    a_pitch = np.zeros(N)
    has = pcnt > 0
    if has.sum() > 4:
        mp = psum[has] / pcnt[has]
        a_pitch[has] = (mp - mp.mean()) / (mp.std() + 1e-9)
    a_chord = np.zeros(N)
    if C is not None and C.shape[0] > 4:
        cs = np.vstack([np.zeros((1, 12)), np.cumsum(C, axis=0)])
        w = max(2, int(round(max(P, 0.7) * fps)))     # harmony moves slower than notes
        T = C.shape[0]
        t = np.round(slot_t * fps).astype(int)
        ok = (t - w >= 0) & (t + w <= T)
        before = cs[t[ok]] - cs[t[ok] - w]
        after = cs[t[ok] + w] - cs[t[ok]]
        cos = (before * after).sum(1) / (np.linalg.norm(before, axis=1)
                                         * np.linalg.norm(after, axis=1) + 1e-9)
        a_chord[ok] = 1.0 - cos

    def norm(x):
        m = x[x > 0].mean() if np.any(x > 0) else 1.0
        return x / (m + 1e-9)

    ks = np.nonzero(a_note > 0)[0]
    ioi = np.diff(ks)
    ioi = ioi[ioi >= 2]
    tatum = float(np.median(ioi)) if ioi.size >= 6 else float(S)   # usual gap between notes
    return dict(N=N, slot_t=slot_t, on=norm(a_on), note=norm(a_note), bass=norm(a_bass),
                chord=norm(a_chord), pitch=a_pitch, tatum=tatum, spans=spans)


def _acf(x: np.ndarray, lmax: int) -> np.ndarray:
    """Unbiased, normalised autocorrelation for lags 0..lmax."""
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, 1 << int(np.ceil(np.log2(2 * n))))
    r = np.fft.irfft(f * np.conj(f))[:lmax + 1]
    r = r / (r[0] + 1e-12)
    return r * n / np.maximum(n - np.arange(lmax + 1), 1)


def _peak(r: np.ndarray, lag: int, w: int = 1):
    """R near ``lag`` (one slot of jitter allowed); None if out of range."""
    lag = int(lag)
    if lag - w < 1 or lag + w >= r.size:
        return None
    return float(r[lag - w:lag + w + 1].max())


def _bar_evidence(Rc: np.ndarray, B: int, n: int, four: float):
    """How much better the music repeats at multiples of n beats than at the
    rival groupings.  None if the lags needed are not available."""
    r = {k: _peak(Rc, k * B) for k in range(2, 15)}
    if n in (2, 3, 4):
        trip = [v for v in (r[3], r[6]) if v is not None]
        dup = [v for v in (r[2], r[4], r[8]) if v is not None]
        if not trip or not dup:
            return None
        t = float(np.mean(trip) - np.mean(dup))
        e = t if n == 3 else -t
        if n != 3 and r[4] is not None and r[2] is not None:
            e += four * ((r[4] - r[2]) if n == 4 else (r[2] - r[4]))
        return e
    own = [r[k] for k in (n, 2 * n) if k in r and r[k] is not None]
    rivals = [r[k] for k in range(2, 9) if k % n and r[k] is not None]
    if not own or not rivals:
        return None
    return float(np.mean(own) - np.mean(rivals))


def _contrasts(prof_beat: np.ndarray, prof_bar: np.ndarray, B: int, m: int, n: int):
    """Best (beat, bar, subdivision) accent contrasts over the phases of a
    bar of ``n`` beats of ``B`` slots divided in ``m``, from accent profiles
    folded over the bar.  Contrasts are (a - b) / (a + b): do the beats stand
    out from the subdivisions between them, does the downbeat stand out from
    the other beats, and do the subdivisions stand out from the positions of
    the rival division (thirds against halves)?  Unlike periodicity, these
    depend on *where* the accents fall, so an even stream of quavers does not
    support every multiple of a quaver as a beat."""
    M = n * B
    u = B // m
    k = np.arange(n * m)
    best = None
    other = 5 - m
    for ph in range(0, M, max(1, u // 2)):
        pos = (ph + k * u) % M
        beats = pos[k % m == 0]
        subs = pos[k % m != 0]
        bt = prof_beat[beats].mean()
        sb = prof_beat[subs].mean() if subs.size else 0.0
        beat_c = (bt - sb) / (bt + sb + 1e-9)
        down = prof_bar[ph % M]
        rest = prof_bar[beats[1:]].mean() if beats.size > 1 else 0.0
        bar_c = (down - rest) / (down + rest + 1e-9)
        sub_c = 0.0
        if B % other == 0:
            alt = (ph + np.arange(n * other) * (B // other)) % M
            alt = np.setdiff1d(alt, pos)
            if alt.size and subs.size:
                a1, a2 = prof_beat[subs].mean(), prof_beat[alt].mean()
                sub_c = (a1 - a2) / (a1 + a2 + 1e-9)
        tot = beat_c + bar_c
        if best is None or tot > best[0]:
            best = (tot, beat_c, bar_c, sub_c, ph)
    return best[1], best[2], best[3], best[4]


def _phase(acc: dict, B: int, n: int, W: dict) -> int:
    """Slot (mod bar) of the downbeat.  Accent profiles are folded over the
    bar; the downbeat is the phase whose beats carry accents, whose first beat
    stands out from the other beats (harmony changes and bass notes on the
    one), at which long notes start, and across whose barlines the fewest
    notes are held."""
    M = n * B
    N = acc["N"]
    A_beat = acc["on"] + acc["note"] + 0.5 * acc["chord"] + 0.5 * acc["bass"]
    A_bar = 2.0 * acc["chord"] + acc["bass"] + 0.5 * acc["note"] + 0.3 * acc["on"]
    L = max(M, (N // M) * M)
    pb = np.resize(A_beat, L)[:L].reshape(-1, M).mean(axis=0)
    pr = np.resize(A_bar, L)[:L].reshape(-1, M).mean(axis=0)
    cross = np.zeros(M)
    longs = np.zeros(M)
    for a, b in acc["spans"]:
        if b - a < 0.25 * B:
            continue
        lo, hi = a + 0.15 * B, b - 0.15 * B
        if hi > lo:
            for k in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
                cross[k % M] += min(k - a, b - k) / B
        longs[int(round(a)) % M] += min((b - a) / B, n)
    cross /= cross.mean() + 1e-9
    longs /= longs.mean() + 1e-9
    best, best_s = 0, -np.inf
    for ph in range(M):
        beats = (ph + np.arange(n) * B) % M
        down = pr[ph]
        rest = pr[beats[1:]].mean() if n > 1 else 0.0
        s = (pb[beats].mean() + W["bar_accent"] * down - W["cross"] * cross[ph]
             + W["long_on_one"] * longs[ph] + W["down_contrast"] * (down - rest) / (down + rest + 1e-9))
        if s > best_s:
            best, best_s = ph, s
    return best


def _name(m: int, n: int, T: float) -> tuple[int, int]:
    if m == 3:
        return 3 * n, 8
    if n == 3 and T < 0.36:
        return 3, 8                     # a very fast three: written in quavers
    if n in (5, 7) and T < 0.36:
        return n, 8
    return n, 4


def detect_meter(bm: BeatMap, onset: np.ndarray, fps: float, notes: list[Note],
                 spec: np.ndarray | None = None, df: float = 0.0, ref_hz: float = 440.0,
                 time_signature: tuple[int, int] | None = None, debug: bool = False,
                 lead: list[Note] | None = None, chroma_frames: np.ndarray | None = None,
                 odd_meters: bool = False, weights: dict | None = None) -> Meter:
    """Choose time signature, beat level and downbeat (see the module text).

    ``time_signature`` forces the signature; ``odd_meters`` adds bars of five
    and seven beats to the automatic search; ``chroma_frames`` (T, 12) can be
    given instead of ``spec``."""
    W = dict(WEIGHTS)
    if weights:
        W.update(weights)
    S = SLOTS
    forced = signature_spec(*time_signature) if time_signature else None
    if bm.beat_times.size < 8:
        return _fallback(bm, time_signature)
    C = chroma_frames
    if C is None and spec is not None and df > 0:
        C = chroma(spec, df, ref_hz)
    acc = _slot_accents(bm, onset, fps, notes, C)
    N, P = acc["N"], bm.period
    lmax = min(N // 2, S * 24)
    Rr = _acf(acc["on"] + acc["note"], lmax)
    Rh = _acf(acc["chord"] + 0.5 * acc["bass"], lmax)
    Rp = _acf(acc["pitch"], lmax)
    Rc = (1 - W["harm"] - W["pitch"]) * Rr + W["harm"] * Rh + W["pitch"] * Rp
    Rb = (1 - W["pitch_beat"]) * Rr + W["pitch_beat"] * Rp

    if forced:
        groupings = [(forced[0], forced[1])]
    else:
        groupings = [(m, n) for m in (2, 3) for n in (2, 3, 4)]
        if odd_meters:
            groupings += [(2, 5), (2, 7)]
    cands = []
    for B in BEAT_SLOTS:
        rb = _peak(Rb, B)
        if rb is None:
            continue
        for m, n in groupings:
            if B % m == 0 and (2 * n * B <= lmax or (forced and n * B <= lmax)):
                cands.append((B, m, n, rb))
    if not cands:
        return _fallback(bm, time_signature)
    A_beat = acc["on"] + acc["note"] + 0.5 * acc["chord"] + 0.5 * acc["bass"]
    A_bar = 2.0 * acc["chord"] + acc["bass"] + 0.5 * acc["note"] + 0.3 * acc["on"]
    folded = {}

    def fold(M):
        if M not in folded:
            L = (N // M) * M
            folded[M] = (A_beat[:L].reshape(-1, M).mean(axis=0), A_bar[:L].reshape(-1, M).mean(axis=0))
        return folded[M]
    rb_max = max(c[3] for c in cands)
    floor = max(W["min_pulse"], W["rel_pulse"] * rb_max)
    if forced and not any(c[3] >= floor for c in cands):
        floor = -np.inf                           # forced metre: take the best there is
    scored = []
    for B, m, n, rb in cands:
        if rb < floor:
            continue                              # nothing repeats at this period: not a beat
        T = P * B / S
        sub = _peak(Rr, B // m, 0) or 0.0
        other = 5 - m
        alt = _peak(Rr, B // other, 0) if B % other == 0 else None
        sub_e = sub - (alt if alt is not None else W["sub_default"])
        bar_e = _bar_evidence(Rc, B, n, W["four"])
        if bar_e is None:
            if not forced:
                continue
            bar_e = 0.0
        rM = _peak(Rc, n * B) or 0.0
        if forced:
            den = time_signature[1]
            T0 = ({4: 1.2, 8: 0.8, 16: 0.45}.get(den, 0.8) if forced[3]
                  else {1: 1.2, 2: 0.8, 4: 0.55, 8: 0.3, 16: 0.2, 32: 0.15}[den])
            bar0 = max(1.0, min(4.0, n * T0))
        else:
            T0 = W["beat_compound"] if m == 3 else W["beat_simple"]
            bar0 = W["bar_seconds"]
        tempo = -W["tempo"] * np.log2(T / T0) ** 2
        barp = -W["bar_length"] * np.log2(n * T / bar0) ** 2
        # readability: the usual gap between notes, in beats (a third of a
        # dotted beat counts like half a plain one: both are quavers)
        x = acc["tatum"] / B * (1.5 if m == 3 else 1.0)
        read = 0.0 if forced else \
            -W["readable"] * (max(0.0, np.log2(0.4 / x)) + max(0.0, np.log2(x / 1.25))) ** 2
        con = 0.0
        if W["beat_contrast"] or W["bar_contrast"] or W["sub_contrast"]:
            pb, pr = fold(n * B)
            beat_c, bar_c, sub_c, _ph = _contrasts(pb, pr, B, m, n)
            con = W["beat_contrast"] * beat_c + W["bar_contrast"] * bar_c + W["sub_contrast"] * sub_c
        s = (W["beat"] * rb + W["sub"] * sub_e + W["bar"] * bar_e + W["bar_lag"] * rM
             + con + tempo + barp + read + PRIOR.get((m, n), -0.2))
        scored.append((float(s), B, m, n, T))
    if not scored:
        return _fallback(bm, time_signature)
    scored.sort(key=lambda z: -z[0])
    s0, B, m, n, T = scored[0]
    if forced:
        num, den = time_signature
        beat_ticks, compound = forced[2], forced[3]
    else:
        num, den = _name(m, n, T)
        _m, _n, beat_ticks, compound = signature_spec(num, den)
    ph = _phase(acc, B, n, W)
    # notated beats: every B slots, aligned so that slot ph is a beat
    s_first = ph % B
    slots = np.arange(s_first, N, B)
    if slots.size < 2:
        return _fallback(bm, time_signature)
    times = acc["slot_t"][slots]
    nbm = BeatMap.from_beats(times, 60.0 / T)
    down = ((ph - s_first) // B) % n
    table = {}
    for sc, B2, m2, n2, T2 in scored:
        nm = "%d/%d" % (_name(m2, n2, T2) if not forced else time_signature)
        key = f"{nm} @ {60.0 / T2:.0f}"
        if key not in table:
            table[key] = round(sc, 3)
        if len(table) >= 8:
            break
    rival = next((sc for sc, _B, m2, n2, _T in scored if (m2, n2) != (m, n)), None)
    conf = float(s0 - rival) if rival is not None else 1.0
    if debug:
        for k, v in table.items():
            print("   meter", k, v)
    return Meter(num, den, compound, beat_ticks, nbm, int(down), n, table, round(conf, 3))


def _fallback(bm: BeatMap, time_signature) -> Meter:
    """Too little music to analyse: keep the tracked pulse as the beat."""
    if time_signature:
        _m, n, beat_ticks, compound = signature_spec(*time_signature)
        return Meter(time_signature[0], time_signature[1], compound, beat_ticks, bm, 0, n, {}, 0.0)
    return Meter(4, 4, False, 480, bm, 0, 4, {}, 0.0)


def skyline(notes: list[Note], min_dur: float = 0.1, merge: float = 0.06) -> list[Note]:
    """Top line of a polyphonic texture: an onset belongs to it only if its
    highest note is at least as high as everything still sounding then (so an
    accompaniment figure under a held melody note is not taken for melody)."""
    ns = sorted((n for n in notes if n.end - n.start >= min_dur), key=lambda n: n.start)
    out: list[Note] = []
    active: list[Note] = []
    i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1].start - ns[i].start <= merge:
            j += 1
        top = max(ns[i:j + 1], key=lambda g: g.pitch)
        t = ns[i].start
        active = [m for m in active if m.end > t + 0.05]
        if not active or top.pitch >= max(m.pitch for m in active):
            out.append(top)
        active.extend(ns[i:j + 1])
        i = j + 1
    return out
