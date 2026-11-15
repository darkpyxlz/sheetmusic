"""Polyphonic (accompaniment) transcription by iterative harmonic estimation
and cancellation (after Klapuri, 2006), followed by per-pitch note tracking."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d

from .melody import _runs
from .notes import Note
from .spectral import HarmonicSalience, midi_to_hz, peak_spectrum, whiten


def multipitch_activations(spec: np.ndarray, df: float, ref_hz: float = 440.0,
                           midi_lo: int = 28, midi_hi: int = 96, max_poly: int = 6,
                           sub: int = 3, n_harm: int = 10, alpha: float = 0.85,
                           tol_cents: float = 20.0, chunk: int = 2048):
    """Return per-iteration (pitch, salience) arrays of shape (T, max_poly).

    Each iteration picks the strongest harmonic series in every frame, records
    it and subtracts its (spectrally smoothed) partials, so overtones of a
    detected note are not reported as extra notes.
    """
    T, K = spec.shape
    semis = np.arange(midi_lo, midi_hi + 1)
    offs = (np.arange(sub) - (sub - 1) / 2) / sub  # in semitones
    grid = (semis[:, None] + offs[None, :]).ravel()
    f0s = midi_to_hz(grid, ref_hz)
    sal_fn = HarmonicSalience(f0s, df, K, n_harm=n_harm, alpha=alpha,
                              tol_cents=tol_cents, max_halfwidth=6, fmax=5000.0)
    ratio = 2 ** (tol_cents / 1200.0) - 1.0
    W = whiten(spec, df, nu=0.33)
    # per-semitone harmonic energy (first 3 partials) of the *uncancelled*
    # spectrum: used to detect re-attacks of an already sounding pitch
    en_fn = HarmonicSalience(midi_to_hz(semis, ref_hz), df, K, n_harm=3, alpha=0.8,
                             tol_cents=tol_cents, max_halfwidth=6, fmax=5000.0)
    energy = np.zeros((T, 128), np.float32)
    pitches = np.full((T, max_poly), -1, np.int16)
    sals = np.zeros((T, max_poly), np.float32)
    wmax = 6
    for s in range(0, T, chunk):
        P = peak_spectrum(W[s:s + chunk])
        n = P.shape[0]
        energy[s:s + n, semis] = en_fn(P)
        rows = np.arange(n)
        for it in range(max_poly):
            S = sal_fn(P)
            j = np.argmax(S, axis=1)
            val = S[rows, j]
            pitches[s:s + n, it] = semis[j // sub]
            sals[s:s + n, it] = val
            # cancel the detected harmonic series
            f0 = f0s[j]
            H = np.arange(1, n_harm + 1)
            target = H[None, :] * f0[:, None] / df            # (n, H)
            valid = target < K - 2
            w = np.clip(np.ceil(target * ratio), 1, wmax)
            center = np.round(target).astype(int)
            A = np.zeros_like(target, dtype=np.float32)
            for o in range(-wmax, wmax + 1):
                b = np.clip(center + o, 0, K - 1)
                ok = valid & (np.abs(o) <= w)
                A = np.maximum(A, np.where(ok, P[rows[:, None], b], 0.0))
            # spectral smoothness: a partial much louder than its neighbours is
            # probably shared with another note -> only remove the smooth part
            Ap = np.pad(A, ((0, 0), (1, 1)), mode="edge")
            smooth = (Ap[:, :-2] + Ap[:, 1:-1] + Ap[:, 2:]) / 3.0
            est = np.minimum(A, smooth)
            for o in range(-wmax, wmax + 1):
                b = np.clip(center + o, 0, K - 1)
                ok = valid & (np.abs(o) <= w)
                cur = P[rows[:, None], b]
                newv = np.where(ok, np.maximum(cur - est, 0.0), cur)
                P[rows[:, None], b] = newv
    return pitches, sals, energy


def harmonic_templates(df: float, n_bins: int, ref_hz: float, midi_lo: int, midi_hi: int,
                       tilts=(0.6, 1.1, 1.8), n_harm: int = 20, inharm: float = 2e-4,
                       fmax: float = 5000.0):
    """Fixed harmonic dictionary: for every semitone several spectral tilts
    (bright -> dull timbres).  Partials are Gaussian bumps whose width grows
    with frequency (window main lobe + vibrato / inharmonicity spread)."""
    k = np.arange(n_bins, dtype=np.float64)
    cols, owner = [], []
    for p in range(midi_lo, midi_hi + 1):
        f0 = float(midi_to_hz(p, ref_hz))
        for a in tilts:
            t = np.zeros(n_bins)
            for h in range(1, n_harm + 1):
                fh = h * f0 * np.sqrt(1 + inharm * h * h)
                if fh > fmax:
                    break
                c = fh / df
                sig = max(1.0, 0.8 + c * 0.006)
                t += h ** (-a) * np.exp(-0.5 * ((k - c) / sig) ** 2)
            t /= t.sum() + 1e-12
            cols.append(t)
            owner.append(p)
    return np.array(cols, np.float32).T, np.array(owner)   # (K, R), (R,)


def nmf_activations(spec: np.ndarray, df: float, ref_hz: float = 440.0, midi_lo: int = 28,
                    midi_hi: int = 96, n_iter: int = 60, sparsity: float = 0.08,
                    power: float = 0.6, chunk: int = 4096) -> np.ndarray:
    """Pitch activations (T, 128) by NMF with a fixed harmonic dictionary
    (KL divergence, multiplicative updates, L1 sparsity).

    All notes explain the spectrum *jointly*, so a partial shared by two
    notes (bass + its fifth / octave, very common in chords) is split between
    them instead of being taken by whichever note was found first."""
    T, K = spec.shape
    W, owner = harmonic_templates(df, K, ref_hz, midi_lo, midi_hi)
    R = W.shape[1]
    Wt = W.T.copy()
    colsum = W.sum(axis=0)[:, None]                       # (R, 1)
    act = np.zeros((T, 128), np.float32)
    for s in range(0, T, chunk):
        V = (spec[s:s + chunk].astype(np.float32).T) ** power   # (K, n)
        scale = V.sum(axis=0, keepdims=True) + 1e-9
        H = (Wt @ V) / (colsum + 1e-9) + 1e-6                   # (R, n)
        lam = sparsity * scale / R
        for _ in range(n_iter):
            WH = W @ H + 1e-9
            H *= (Wt @ (V / WH)) / (colsum + lam)
        Hp = np.zeros((128, H.shape[1]), np.float32)
        np.add.at(Hp, owner, H)
        act[s:s + chunk] = Hp.T
    return suppress_overtone_activations(act)


def suppress_overtone_activations(act: np.ndarray, ratio: float = 0.6,
                                  intervals=(12, 19, 24, 28)) -> np.ndarray:
    """Frame-wise: an activation an octave / twelfth / two octaves / two
    octaves + third above a clearly stronger active note is most likely that
    note's overtone left over by a template-timbre mismatch: attenuate it."""
    out = act.copy()
    for d in intervals:
        lower = act[:, :-d]
        upper = act[:, d:]
        ghost = upper < ratio * lower
        out[:, d:] = np.where(ghost, np.minimum(out[:, d:], 0.25 * upper), out[:, d:])
    return out


def voice_exclusion(melody_midi: np.ndarray, n_frames: int, strength: float = 1.0,
                    harmonics=(0, 12, 19, 24, 28), spread: int = 10) -> np.ndarray:
    """(T, 128) attenuation factors for accompaniment pitches that coincide
    with the sung pitch or its first overtones (+12, +19, +24 semitones).

    The voice's harmonic comb is already notched out of the spectrum; this
    removes what leaks through (vibrato excursions, scoops, onsets)."""
    w = np.ones((n_frames, 128), np.float32)
    m = melody_midi[:n_frames]
    voiced = ~np.isnan(m)
    if not np.any(voiced):
        return w
    pv = np.where(voiced, np.round(m), -100).astype(int)
    # extend a few frames around the voiced frames (onsets / releases)
    if spread > 0:
        from scipy.ndimage import maximum_filter1d as _mf
        pv = np.maximum(pv, _mf(pv, 2 * spread + 1))
    t = np.nonzero(pv >= 0)[0]
    for h in harmonics:
        att = strength if h == 0 else strength * 0.85
        # +-1 semitone: scoops / vibrato excursions of the voice
        for d, a in ((0, att), (-1, 0.7 * att), (1, 0.7 * att)):
            p = pv[t] + h + d
            ok = (p >= 0) & (p < 128)
            w[t[ok], p[ok]] = np.minimum(w[t[ok], p[ok]], 1 - a)
    return w


def activations_to_notes(pitches: np.ndarray | None, sals: np.ndarray | None, fps: float,
                         energy: np.ndarray | None = None,
                         abs_threshold: float = 0.12, rel_threshold: float = 0.3,
                         min_note: float = 0.08, fill_gap: float = 0.08,
                         reattack_db: float = 5.0, max_bridge: float = 0.5,
                         attack_skip: int = 8, release_db: float = 6.0,
                         release_win: int = 12,
                         weights: np.ndarray | None = None,
                         spec: np.ndarray | None = None, df: float = 0.0,
                         ref_hz: float = 440.0, partial_rise_db: float = 3.5,
                         partial_frac: float = 0.6,
                         short_spec: np.ndarray | None = None, short_df: float = 0.0,
                         onset_env: np.ndarray | None = None,
                         act_in: np.ndarray | None = None,
                         refine_onsets: bool = True) -> list[Note]:
    if act_in is not None:
        act = act_in.astype(np.float32).copy()
        T = act.shape[0]
        first = act.max(axis=1)
        ref = np.percentile(first[first > 0], 95) if np.any(first > 0) else 1.0
        act[act < rel_threshold * first[:, None]] = 0.0
    else:
        T, Kp = pitches.shape
        first = sals[:, 0]
        ref = np.percentile(first[first > 0], 95) if np.any(first > 0) else 1.0
        # candidates: everything that is not negligible next to the frame's strongest
        keep = (sals >= rel_threshold * first[:, None]) & (sals >= 0.25 * abs_threshold * ref) \
            & (pitches >= 0)
        act = np.zeros((T, 128), np.float32)
        rows = np.repeat(np.arange(T)[:, None], Kp, axis=1)
        pk = np.where(keep, pitches, 0).astype(int)
        np.maximum.at(act, (rows[keep], pk[keep]), sals[keep])
    if weights is not None:
        act *= weights
    thr_on = abs_threshold * ref
    thr_off = 0.5 * thr_on
    min_len = max(2, int(round(min_note * fps)))
    gap = int(round(fill_gap * fps))
    bridge = int(round(max_bridge * fps))
    notes: list[Note] = []
    if energy is not None:
        le = np.log(uniform_filter1d(energy, 3, axis=0) + 1e-6)
        back = 6
        past_min = np.full_like(le, np.inf)
        for k in range(1, back + 1):
            past_min[k:] = np.minimum(past_min[k:], le[:-k])
        fut = np.vstack([le[2:], np.repeat(le[-1:], 2, axis=0)])
        rise = fut - past_min
        thr = reattack_db / 20.0 * np.log(10.0)
    onset_peaks = None
    if onset_env is not None:
        o = onset_env
        pk = np.zeros(o.size, bool)
        pk[1:-1] = (o[1:-1] > o[:-2]) & (o[1:-1] >= o[2:]) & (o[1:-1] > 0.7)
        onset_peaks = maximum_filter1d(pk.astype(np.uint8), 5).astype(bool)  # +-2 frames
    # ---- pass 1: per-pitch segments (hysteresis + bridging)
    segs = []   # (a, b, p)
    for p in np.nonzero(act.max(axis=0) >= thr_on)[0]:
        x = act[:, p]
        if x.max() < thr_on:
            continue
        # hysteresis: a note starts above thr_on and lasts while above thr_off
        # (decaying piano/guitar notes would otherwise flicker on and off)
        xs = median_filter(x, size=3)
        d = np.zeros(T, bool)
        for a, b in _runs(xs >= thr_off):
            if xs[a:b].max() >= thr_on:
                d[a:b] = True
        # bridge short dropouts, and longer ones (a sustained note briefly
        # masked by a louder chord) when the note comes back without a new attack
        for a, b in _runs(~d):
            if a > 0 and b < T and (b - a <= gap or (
                    energy is not None and b - a <= bridge
                    and rise[max(0, b - 4):b + 4, p].max() < thr)):
                d[a:b] = True
        for a, b in _runs(d):
            if b - a >= min_len:
                segs.append((a, b, int(p)))

    # ---- pass 2: re-attacks and releases, judged on each note's *own*
    # partials (partials shared with other notes sounding at the same time
    # are ignored, otherwise e.g. a bass note "lasts" as long as the melody
    # note that sits on its 4th harmonic)
    for a, b, p in segs:
        x = act[:, p]
        others = [q for (a2, b2, q) in segs if q != p and a2 < b and b2 > a]
        excl = _shared_partials(p, others, ref_hz)
        cuts = [a]
        L = None
        if spec is not None and df > 0:
            L = _partial_logmag(spec, p, a, b, df, ref_hz, exclude=excl)
        if short_spec is not None and b - a > 2 * min_len:
            Ls = _partial_logmag(short_spec, p, a, b, short_df, ref_hz, smooth=1, exclude=excl)
            if Ls is not None:
                for t in _partial_reattacks(Ls, min_len, 3, partial_rise_db + 1.5,
                                            partial_frac, back=3, ahead=1,
                                            onset_ok=None if onset_peaks is None
                                            else onset_peaks[a:b]):
                    if a + t - cuts[-1] >= min_len and b - (a + t) >= min_len:
                        cuts.append(a + t)
        elif L is not None and b - a > 2 * min_len:
            for t in _partial_reattacks(L, min_len, attack_skip, partial_rise_db,
                                        partial_frac):
                if a + t - cuts[-1] >= min_len and b - (a + t) >= min_len:
                    cuts.append(a + t)
        elif energy is not None and b - a > 2 * min_len:
            r = rise[:, p]
            lead = min_len + attack_skip  # STFT smearing of the note's own attack
            seg = r[a + lead:b - min_len]
            cand = np.nonzero((seg > thr) & (seg >= maximum_filter1d(seg, 2 * min_len + 1)))[0]
            for t in cand + a + lead:
                if t - cuts[-1] >= min_len and b - t >= min_len:
                    cuts.append(int(t))
        cuts.append(b)
        if short_spec is not None and refine_onsets:
            cuts[0] = _attack_frame(short_spec, p, a, min(b, cuts[1]), short_df, ref_hz, excl,
                                    min_len)
        for i in range(len(cuts) - 1):
            s0, e = cuts[i], cuts[i + 1]
            if p < 40 and (e - s0) / fps < 0.25:
                continue  # short sub-bass blips are almost always kick-drum tails
            if x[s0:e].max() < thr_on:
                continue
            if L is not None and release_db > 0:
                e = s0 + _partial_release(L[s0 - a:e - a], min_len, release_db, release_win,
                                          partial_frac)
            elif energy is not None and release_db > 0:
                e = _release_point(le[:, p], s0, e, min_len, release_db, release_win)
            v = float(np.median(x[s0:e][x[s0:e] > 0])) if np.any(x[s0:e] > 0) else 0.0
            vel = int(np.clip(30 + 80 * v / (ref + 1e-9), 25, 115))
            notes.append(Note(start=s0 / fps, end=e / fps, pitch=p, velocity=vel))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _attack_frame(short_spec, p, a, b, df, ref_hz, exclude, min_len, before: int = 5,
                  after: int = 12, min_rise_db: float = 12.0) -> int:
    """Start of a note from the attack of its own partials in the short
    (46 ms) window.

    The activation that finds a note comes from a 186 ms window, which smears
    a struck note's energy up to 93 ms *ahead* of the attack: a loud piano
    note crosses the detection threshold several frames early, a slowly
    bowed one late.  The steepest rise of the partials' energy in the short
    window marks the attack itself.  Only sharp attacks (struck or plucked:
    12 dB within 35 ms) are moved: for a slow bowed or blown attack the
    steepest rise comes well after the note has begun, and the original
    start is the better one (measured: string-quartet F fell from 0.84 to
    0.75 when every note was moved)."""
    lo = max(0, a - before)
    hi = min(short_spec.shape[0], max(a + after, a + 2), b)
    if hi - lo < 5:
        return a
    L = _partial_logmag(short_spec, p, lo, hi, df, ref_hz, smooth=1, exclude=exclude)
    if L is None:
        return a
    E = np.log(np.exp(2 * L).sum(axis=1) + 1e-12)        # log energy of the partials
    rise = E[3:] - E[:-3]                                  # over 3 frames (35 ms)
    k = int(np.argmax(rise))
    if rise[k] < min_rise_db / 10.0 * np.log(10.0):
        return a
    t = lo + k + 2                                         # middle of the rise
    return int(min(max(t, a), b - min_len)) if b - min_len > a else a


def _shared_partials(p: int, others: list[int], ref_hz: float, n_harm: int = 6,
                     tol: float = 0.03) -> set[int]:
    """Harmonic numbers of ``p`` that coincide with a partial of another note."""
    f0 = float(midi_to_hz(p, ref_hz))
    out = set()
    for q in set(others):
        g0 = float(midi_to_hz(q, ref_hz))
        for h in range(1, n_harm + 1):
            k = round(h * f0 / g0)
            if 1 <= k <= 16 and abs(h * f0 - k * g0) <= tol * h * f0:
                out.add(h)
    return out


def _partial_logmag(spec: np.ndarray, p: int, a: int, b: int, df: float, ref_hz: float,
                    n_harm: int = 6, tol_cents: float = 30.0, smooth: int = 3,
                    exclude: set[int] | None = None) -> np.ndarray | None:
    """(frames, partials) log-magnitude of pitch ``p``'s partials over [a, b),
    leaving out the harmonic numbers in ``exclude`` (if at least two remain)."""
    f0 = float(midi_to_hz(p, ref_hz))
    K = spec.shape[1]
    cols = []
    usable = [h for h in range(1, n_harm + 1) if h * f0 / df < K - 2]
    if exclude and len([h for h in usable if h not in exclude]) >= 2:
        usable = [h for h in usable if h not in exclude]
    for h in usable:
        c = h * f0 / df
        if c >= K - 2:
            break
        w = max(1, int(np.ceil(c * (2 ** (tol_cents / 1200) - 1))))
        lo, hi = max(0, int(round(c - w))), min(K, int(round(c + w)) + 1)
        cols.append(spec[a:b, lo:hi].max(axis=1))
    if not cols:
        return None
    L = np.log(np.stack(cols, axis=1).astype(np.float64) + 1e-7)
    return uniform_filter1d(L, smooth, axis=0, mode="nearest") if smooth > 1 else L


def _significant(L: np.ndarray, floor_db: float = 30.0) -> np.ndarray:
    peak = L.max(axis=0)
    return peak > peak.max() - floor_db / 20.0 * np.log(10.0)


def _partial_reattacks(L, min_len, skip, rise_db, frac_needed, back=6, ahead=2,
                       onset_ok=None):
    """Frames where most of the note's *own* partials jump up together = the
    same pitch struck again.  (A different note sharing a few partials only
    moves some of them.)"""
    n, H = L.shape
    sig = _significant(L)
    if sig.sum() == 0:
        return []
    thr = rise_db / 20.0 * np.log(10.0)
    past_min = np.full_like(L, np.inf)
    for k in range(1, back + 1):
        past_min[k:] = np.minimum(past_min[k:], L[:-k])
    fut = np.vstack([L[ahead:], np.repeat(L[-1:], ahead, axis=0)]) if ahead else L
    rise = (fut - past_min)[:, sig]
    frac = (rise > thr).mean(axis=1)
    score = frac * np.clip(rise, 0, None).mean(axis=1)
    out = []
    lead = min_len + skip
    for t in range(lead, n - min_len):
        if onset_ok is not None and not onset_ok[t]:
            continue
        if frac[t] >= frac_needed and score[t] >= score[max(0, t - min_len):t + min_len + 1].max():
            if not out or t - out[-1] >= min_len:
                out.append(t)
    return out


def _partial_release(L, min_len, drop_db, win, frac_needed) -> int:
    """First frame where most of the note's own partials have fallen steeply."""
    n = L.shape[0]
    sig = _significant(L)
    if sig.sum() == 0:
        return n
    Ls = L[:, sig]
    drop = drop_db / 20.0 * np.log(10.0)
    for t in range(min_len, n):
        lo = max(0, t - win)
        fallen = (Ls[lo:t].max(axis=0) - Ls[t]) > drop
        if fallen.mean() >= frac_needed:
            # STFT smearing: place the release where the fall began
            m = Ls[lo:t + 1].mean(axis=1)
            pk = int(np.argmax(m))
            k = t - lo
            while k > pk + 1 and m[k - 1] < m[pk] - 0.35 * drop:
                k -= 1
            return max(min_len, lo + k)
    return n


def _release_point(le: np.ndarray, s: int, e: int, min_len: int, drop_db: float,
                   win: int) -> int:
    """End of a note = where its partials' energy falls steeply (key/string
    released or damped), not where it finally fades below the detection
    threshold (ringing, sustain pedal-like decay and room reverb make notes
    look far too long otherwise).  A natural slow decay does not count."""
    drop = drop_db / 20.0 * np.log(10.0)
    seg = le[s:e]
    for t in range(min_len, seg.size):
        lo = max(0, t - win)
        if seg[t] < seg[lo:t].max() - drop:
            # the STFT window smears the release: place it at the start of the fall
            peak = lo + int(np.argmax(seg[lo:t]))
            k = t
            while k > peak + 1 and seg[k - 1] < seg[peak] - 0.35 * drop:
                k -= 1
            return s + max(k, min_len)
    return e


def prune_overtone_ghosts(notes: list[Note], intervals=(12, 19, 24, 28, 31),
                          overlap: float = 0.9, ratio: float = 0.75) -> list[Note]:
    """Drop notes that sit on an overtone of a louder, co-sounding lower note
    (residue the harmonic cancellation left behind)."""
    keep = []
    for n in notes:
        dur = n.end - n.start
        ghost = False
        for q in notes:
            if q is n or (n.pitch - q.pitch) not in intervals:
                continue
            ov = min(n.end, q.end) - max(n.start, q.start)
            if ov >= overlap * dur and n.velocity < ratio * q.velocity + 5 and q.start <= n.start + 0.05:
                ghost = True
                break
        if not ghost:
            keep.append(n)
    return keep
