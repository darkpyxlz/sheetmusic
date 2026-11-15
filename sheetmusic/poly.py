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


def activations_to_notes(pitches: np.ndarray, sals: np.ndarray, fps: float,
                         energy: np.ndarray | None = None,
                         abs_threshold: float = 0.12, rel_threshold: float = 0.3,
                         min_note: float = 0.08, fill_gap: float = 0.08,
                         reattack_db: float = 5.0, max_bridge: float = 0.5,
                         attack_skip: int = 8,
                         weights: np.ndarray | None = None) -> list[Note]:
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
            if b - a < min_len:
                continue
            cuts = [a]
            if energy is not None and b - a > 2 * min_len:
                r = rise[:, p]
                lead = min_len + attack_skip  # STFT smearing of the note's own attack
                seg = r[a + lead:b - min_len]
                cand = np.nonzero((seg > thr) & (seg >= maximum_filter1d(seg, 2 * min_len + 1)))[0]
                for t in cand + a + lead:
                    if t - cuts[-1] >= min_len and b - t >= min_len:
                        cuts.append(int(t))
            cuts.append(b)
            for i in range(len(cuts) - 1):
                s, e = cuts[i], cuts[i + 1]
                if p < 40 and (e - s) / fps < 0.25:
                    continue  # short sub-bass blips are almost always kick-drum tails
                if x[s:e].max() < thr_on:
                    continue
                v = float(np.median(x[s:e][x[s:e] > 0])) if np.any(x[s:e] > 0) else 0.0
                vel = int(np.clip(30 + 80 * v / (ref + 1e-9), 25, 115))
                notes.append(Note(start=s / fps, end=e / fps, pitch=int(p), velocity=vel))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


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
