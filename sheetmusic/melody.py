"""Predominant melody (lead vocal) tracking and vocal-aware note segmentation.

Why a voice "garbles" naive transcriptions
------------------------------------------
Picking the loudest frequency of each frame fails on singing because

* the loudest partial is usually not the fundamental but whichever harmonic
  sits under a formant (vocal-tract resonance), and formants move with every
  vowel -> the "melody" jumps between octaves/fifths;
* vibrato, scoops and portamento make the pitch wander continuously, so a
  frame-by-frame quantiser produces strings of short chromatic notes;
* consonants are broadband noise and produce random notes;
* the voice's many harmonics also show up as phantom notes in any
  polyphonic analysis of the accompaniment.

What we do instead
------------------
1. Analyse the centre-panned, harmonic (HPSS) part of the mix, where lead
   vocals live, and *whiten* it to remove the formant envelope.
2. Harmonic summation: every candidate f0 collects the energy of its whole
   harmonic series (the voice's resonant partials are *added into* the
   fundamental), so the sung pitch wins even if f0 itself is weak.
3. Viterbi tracking on a 20-cent grid with a jump penalty gives a smooth
   contour that follows vibrato/glides without octave flips.
4. Voicing detection (salience + spectral flatness) drops consonants/breaths.
5. Notes are segmented with an HMM over semitones on a vibrato-averaged
   contour; scoops at note starts are trimmed and each note's pitch is the
   (tuning-corrected) mean of its stable part.
6. The returned f0 track is later used to notch the voice's harmonic comb out
   of the spectrum before transcribing the accompaniment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .notes import Note
from .spectral import HarmonicSalience, hz_to_midi, midi_to_hz, peak_spectrum, whiten


@dataclass
class MelodyTrack:
    f0: np.ndarray        # (T,) Hz, 0 where unvoiced
    midi: np.ndarray      # (T,) continuous MIDI (tuning corrected), nan unvoiced
    salience: np.ndarray  # (T,) harmonic salience along the path
    voiced: np.ndarray    # (T,) bool
    fluct_salience: np.ndarray | None = None  # (T,) salience of the fluctuating part


def _viterbi_pitch(logp: np.ndarray, step_semitones: float, jump_cost: float,
                   octave_extra: float = 1.0) -> np.ndarray:
    T, N = logp.shape
    d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) * step_semitones
    trans = -(jump_cost * d + octave_extra * (d > 11.5))
    trans = trans.astype(np.float32)
    back = np.empty((T, N), np.int32)
    score = logp[0].astype(np.float32).copy()
    for t in range(1, T):
        cand = score[:, None] + trans
        j = np.argmax(cand, axis=0)
        back[t] = j
        score = cand[j, np.arange(N)] + logp[t]
    path = np.empty(T, np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def track_melody(spec: np.ndarray, df: float, fps: float, flatness: np.ndarray,
                 ref_hz: float = 440.0, midi_lo: float = 38, midi_hi: float = 86,
                 step_cents: float = 20.0, jump_cost: float = 0.35,
                 voicing_threshold: float = 0.3, flatness_max: float = 0.35,
                 fluct_spec: np.ndarray | None = None, fluct_weight: float = 0.5,
                 chunk: int = 2048) -> MelodyTrack:
    """Track the predominant (vocal) f0 in ``spec`` (T, K) magnitude.

    If ``fluct_spec`` (the non-stationary part of ``spec``) is given, the
    tracking emission is a geometric blend of both saliences, which favours
    the harmonic series that carries vibrato / pitch movement (the voice) over
    a steady instrument doubling it an octave lower or higher.  Voicing is
    always decided on the plain salience so straight-tone singing survives.
    """
    T, K = spec.shape
    grid = np.arange(midi_lo, midi_hi + 1e-9, step_cents / 100.0)
    f0s = midi_to_hz(grid, ref_hz)
    sal_fn = HarmonicSalience(f0s, df, K, n_harm=16, alpha=0.82, tol_cents=18.0,
                              max_halfwidth=10, fmax=5000.0)
    def salience(mag):
        W = whiten(mag, df, nu=0.33)
        out = np.empty((T, grid.size), np.float32)
        for s in range(0, T, chunk):
            out[s:s + chunk] = sal_fn(peak_spectrum(W[s:s + chunk]))
        return out

    S = salience(spec)
    logp = np.log(S / (S.max(axis=1, keepdims=True) + 1e-9) + 0.02)
    if fluct_spec is not None and fluct_weight > 0:
        Sf = salience(fluct_spec)
        logf = np.log(Sf / (Sf.max(axis=1, keepdims=True) + 1e-9) + 0.02)
        logp = (1 - fluct_weight) * logp + fluct_weight * logf
        del logf
    else:
        Sf = None
    path = _viterbi_pitch(logp, step_cents / 100.0, jump_cost)

    # parabolic refinement of the path on the salience grid
    idx = np.clip(path, 1, grid.size - 2)
    a = S[np.arange(T), idx - 1]
    b = S[np.arange(T), idx]
    c = S[np.arange(T), idx + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    off = np.clip(off, -0.5, 0.5)
    off[(path == 0) | (path == grid.size - 1)] = 0.0
    midi = grid[path] + off * step_cents / 100.0
    sal = S[np.arange(T), path]
    fsal = Sf[np.arange(T), path] if Sf is not None else None

    # Voicing: salience relative to the loud (sung) parts + not noise-like.
    active = sal[sal > 1e-9]
    ref = np.percentile(active, 90) if active.size else 1.0
    s_norm = sal / (ref + 1e-12)
    if fsal is not None and fluct_weight > 0:
        # Frames whose harmonic series is perfectly steady (an instrument
        # filling a gap between vocal phrases) get a reduced voicing score.
        fr = uniform_filter1d(fsal / (sal + 1e-9), size=max(1, int(0.2 * fps)), mode="nearest")
        s_norm = s_norm * (1 - 0.4 + 0.4 * np.clip(fr / 0.5, 0, 1))
    on = s_norm > voicing_threshold
    off_thr = 0.7 * voicing_threshold
    voiced = np.zeros(T, bool)
    state = False
    for t in range(T):  # hysteresis
        if state:
            state = s_norm[t] > off_thr
        else:
            state = bool(on[t])
        voiced[t] = state
    voiced &= flatness < flatness_max
    voiced = median_filter(voiced.astype(np.uint8), size=5).astype(bool)
    # remove very short voiced islands (< 50 ms)
    voiced = _remove_short_runs(voiced, int(round(0.05 * fps)))

    f0 = np.where(voiced, midi_to_hz(midi, ref_hz), 0.0)
    midi_out = np.where(voiced, midi, np.nan)
    return MelodyTrack(f0=f0, midi=midi_out, salience=sal, voiced=voiced, fluct_salience=fsal)


def _runs(mask: np.ndarray):
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return list(zip(starts, ends))


def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask):
        if e - s < min_len:
            out[s:e] = False
    return out


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(~mask):
        if s > 0 and e < mask.size and e - s <= max_gap:
            out[s:e] = True
    return out


def segment_melody(track: MelodyTrack, fps: float, min_note: float = 0.09,
                   vibrato_window: float = 0.16, change_cost: float = 9.0,
                   sigma: float = 0.33, gap_fill: float = 0.05,
                   reattack_ratio: float = 0.35) -> list[Note]:
    """Turn the continuous vocal contour into discrete notes.

    * gaps < ``gap_fill`` s (e.g. short consonants) are bridged;
    * the contour is low-passed over ``vibrato_window`` (~ one vibrato cycle)
      so vibrato averages to its centre pitch;
    * an HMM over integer semitones with switching cost ``change_cost``
      yields piecewise-constant notes (glides/scoops are not turned into
      chromatic runs);
    * segments shorter than ``min_note`` are merged into the neighbour with
      the closest pitch;
    * repeated notes on the same pitch are split where the vocal energy dips
      (re-articulation).
    """
    voiced = _fill_short_gaps(track.voiced, int(round(gap_fill * fps)))
    midi = track.midi.copy()
    # interpolate pitch across the bridged gaps
    for s, e in _runs(voiced):
        seg = midi[s:e]
        ok = ~np.isnan(seg)
        if ok.sum() == 0:
            continue
        if not ok.all():
            xi = np.arange(seg.size)
            seg[~ok] = np.interp(xi[~ok], xi[ok], seg[ok])
        midi[s:e] = seg
    notes: list[Note] = []
    act = track.salience[track.salience > 1e-9]
    sal_ref = float(np.percentile(act, 90)) if act.size else 1.0
    min_len = max(2, int(round(min_note * fps)))
    win = max(1, int(round(vibrato_window * fps)))
    for s, e in _runs(voiced):
        if e - s < min_len:
            continue
        raw = midi[s:e]
        sal = track.salience[s:e]
        smooth = uniform_filter1d(raw, size=win, mode="nearest")
        states = _segment_semitones(smooth, change_cost, sigma)
        segs = _merge_short(_state_runs(states), raw, min_len)
        # final pitch from the stable part; merge neighbours that agree (e.g. a
        # scoop that the HMM briefly put on the semitone below)
        pitched = []
        for a, b in segs:
            p = _stable_pitch(raw[a:b])
            if pitched and pitched[-1][2] == p:
                pitched[-1][1] = b
            else:
                pitched.append([a, b, p])
        for a0, b0, p in pitched:
            for a, b in _split_reattacks([(a0, b0)], sal, min_len, reattack_ratio):
                if not _plausible_voice(s + a, s + b, track, sal_ref):
                    continue
                notes.append(Note(start=(s + a) / fps, end=(s + b) / fps, pitch=p,
                                  velocity=_velocity(sal[a:b], track.salience)))
    return notes


def _plausible_voice(a: int, b: int, track: MelodyTrack, sal_ref: float) -> bool:
    """Contour-level check: reject weak *and* perfectly steady segments.

    Those are typically an accompaniment instrument that the tracker latched
    onto between vocal phrases.  Loud steady segments are kept (straight-tone
    or auto-tuned singing)."""
    if track.fluct_salience is None:
        return True
    sal = track.salience[a:b]
    fr = float(np.mean(track.fluct_salience[a:b] / (sal + 1e-9)))
    level = float(np.mean(sal)) / (sal_ref + 1e-12)
    return not (fr < 0.4 and level < 0.55)


def fix_octave_errors(notes: list[Note], max_dev: int = 9, ok_dev: int = 5) -> list[Note]:
    """Fold isolated octave jumps back (a typical f0-tracking error).

    A note that is ``>= max_dev`` semitones away from *both* neighbours while
    moving it by an octave puts it within ``ok_dev`` of both is moved.  Real
    octave leaps are rarely followed immediately by a leap back, so this
    almost only touches tracking errors.
    """
    out = [Note(n.start, n.end, n.pitch, n.velocity) for n in notes]
    for i in range(len(out)):
        nb = [out[j].pitch for j in (i - 1, i + 1) if 0 <= j < len(out)
              and abs(out[j].start - out[i].start) < 3.0]
        if not nb:
            continue
        p = out[i].pitch
        if all(abs(p - q) >= max_dev for q in nb):
            for shift in (12, -12):
                if all(abs(p + shift - q) <= ok_dev for q in nb):
                    out[i].pitch = p + shift
                    break
    return out


def snap_onsets(notes: list[Note], onset_env: np.ndarray, fps: float,
                before: float = 0.15, after: float = 0.03, min_strength: float = 0.8) -> list[Note]:
    """Move note starts back to the attack (consonant / onset) preceding the
    voiced part, since voicing detection only fires once the vowel is stable."""
    o = onset_env
    peaks = np.nonzero((o[1:-1] > o[:-2]) & (o[1:-1] >= o[2:]) & (o[1:-1] > min_strength))[0] + 1
    tp = peaks / fps
    out = []
    prev_end = -1.0
    for n in notes:
        cand = tp[(tp >= n.start - before) & (tp <= n.start + after) & (tp > prev_end - 0.02)]
        s = n.start
        if cand.size:
            s = float(cand[np.argmin(np.abs(cand - (n.start - 0.05)))])
        s = min(s, n.end - 0.03)
        out.append(Note(s, n.end, n.pitch, n.velocity))
        prev_end = n.end
    return out


def _segment_semitones(contour: np.ndarray, change_cost: float, sigma: float) -> np.ndarray:
    lo = int(np.floor(np.nanmin(contour))) - 1
    hi = int(np.ceil(np.nanmax(contour))) + 1
    states = np.arange(lo, hi + 1)
    em = ((contour[:, None] - states[None, :]) ** 2) / (2 * sigma ** 2)
    T, N = em.shape
    cost = em[0].copy()
    back = np.zeros((T, N), np.int32)
    for t in range(1, T):
        best = int(np.argmin(cost))
        stay = cost
        switch = cost[best] + change_cost
        use_switch = switch < stay
        back[t] = np.where(use_switch, best, np.arange(N))
        cost = np.where(use_switch, switch, stay) + em[t]
    path = np.empty(T, np.int32)
    path[-1] = int(np.argmin(cost))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return states[path]


def _state_runs(states: np.ndarray):
    out = []
    s = 0
    for t in range(1, states.size + 1):
        if t == states.size or states[t] != states[s]:
            out.append([s, t, int(states[s])])
            s = t
    return out


def _merge_short(runs, raw, min_len):
    runs = [list(r) for r in runs]
    changed = True
    while changed and len(runs) > 1:
        changed = False
        # shortest first
        order = sorted(range(len(runs)), key=lambda i: runs[i][1] - runs[i][0])
        for i in order:
            a, b, p = runs[i]
            if b - a >= min_len:
                break
            cands = []
            if i > 0:
                cands.append((abs(runs[i - 1][2] - p), i - 1))
            if i < len(runs) - 1:
                cands.append((abs(runs[i + 1][2] - p), i + 1))
            _, j = min(cands)
            if j < i:
                runs[j][1] = b
            else:
                runs[j][0] = a
            del runs[i]
            changed = True
            break
    # merge neighbours that ended up on the same pitch
    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2]:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b, _ in merged]


def _split_reattacks(segs, sal, min_len, ratio):
    out = []
    # smooth over ~ one vibrato period: vibrato sweeping the harmonics through
    # the formants causes amplitude modulation that must not look like a re-attack
    sm = uniform_filter1d(sal.astype(float), size=max(5, int(round(min_len * 1.8))), mode="nearest")
    for a, b in segs:
        cuts = [a]
        if b - a >= 2 * min_len:
            x = sm[a:b]
            for t in range(min_len, b - a - min_len):
                if x[t] <= x[t - 1] and x[t] < x[t + 1]:
                    left = x[max(0, t - 3 * min_len):t].max()
                    right = x[t + 1:t + 1 + 3 * min_len].max()
                    if x[t] < ratio * min(left, right) and (a + t - cuts[-1]) >= min_len:
                        cuts.append(a + t)
        cuts.append(b)
        for i in range(len(cuts) - 1):
            out.append((cuts[i], cuts[i + 1]))
    return out


def _stable_pitch(raw: np.ndarray) -> int:
    n = raw.size
    trim = int(0.2 * n) if n >= 8 else 0
    core = raw[trim:n - trim] if n - 2 * trim > 0 else raw
    core = core[~np.isnan(core)]
    if core.size == 0:
        core = raw[~np.isnan(raw)]
    # mean of the stable part = centre of vibrato; median guards against outliers
    m = 0.5 * (np.mean(core) + np.median(core))
    return int(np.round(m))


def _velocity(sal_seg: np.ndarray, sal_all: np.ndarray) -> int:
    ref = np.percentile(sal_all[sal_all > 0], 95) if np.any(sal_all > 0) else 1.0
    v = float(np.median(sal_seg)) / (ref + 1e-12)
    return int(np.clip(40 + 70 * v, 30, 120))


def vocal_harmonic_mask(n_frames: int, n_bins: int, df: float, f0: np.ndarray,
                        width_cents: float = 50.0, min_bins: float = 2.0,
                        depth: float = 0.8, hold_frames: int = 8,
                        chunk: int = 2048) -> np.ndarray:
    """Multiplicative mask (T, K) that notches out the voice's harmonic comb."""
    f = f0.copy()
    # STFT windows are long: hold the f0 a few frames around voiced regions
    voiced = f > 0
    if hold_frames > 0 and np.any(voiced):
        idx = np.where(voiced, np.arange(n_frames), -1)
        last = np.maximum.accumulate(idx)
        idx_r = np.where(voiced, np.arange(n_frames), n_frames * 2)
        nxt = np.minimum.accumulate(idx_r[::-1])[::-1]
        t = np.arange(n_frames)
        use_last = (~voiced) & (last >= 0) & (t - last <= hold_frames)
        use_next = (~voiced) & (~use_last) & (nxt < n_frames) & (nxt - t <= hold_frames)
        f = f.copy()
        f[use_last] = f0[last[use_last]]
        f[use_next] = f0[nxt[use_next]]
    mask = np.ones((n_frames, n_bins), np.float32)
    freqs = np.arange(n_bins) * df
    ratio = 2 ** (width_cents / 1200.0) - 1.0
    for s in range(0, n_frames, chunk):
        ff = f[s:s + chunk]
        rows = np.nonzero(ff > 0)[0]
        if rows.size == 0:
            continue
        fr = ff[rows][:, None]
        r = freqs[None, :] / fr
        h = np.maximum(np.round(r), 1.0)
        dev = np.abs(freqs[None, :] - h * fr)
        tol = np.maximum(min_bins * df, h * fr * ratio)
        hit = (dev <= tol) & (r > 0.5)
        mask[s + rows] = np.where(hit, 1.0 - depth, 1.0)
    return mask


__all__ = ["MelodyTrack", "track_melody", "segment_melody", "vocal_harmonic_mask",
           "hz_to_midi"]
