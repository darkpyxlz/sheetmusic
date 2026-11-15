"""Spectral front end: STFT, stereo centre extraction, HPSS, whitening,
harmonic-summation salience and tuning estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import get_window

A4_MIDI = 69


def midi_to_hz(m, ref_hz: float = 440.0):
    return ref_hz * 2.0 ** ((np.asarray(m, dtype=float) - A4_MIDI) / 12.0)


def hz_to_midi(f, ref_hz: float = 440.0):
    return A4_MIDI + 12.0 * np.log2(np.maximum(np.asarray(f, dtype=float), 1e-9) / ref_hz)


@dataclass
class Features:
    sr: int
    n_fft: int
    hop: int
    mono: np.ndarray       # (T, K) magnitude of the mid signal, K bins up to fmax
    center: np.ndarray     # (T, K) magnitude weighted by L/R similarity (centre-panned)
    flux: np.ndarray       # (T,) full-band spectral flux (onset strength)
    flatness: np.ndarray   # (T,) spectral flatness of the mid signal (noise-likeness)

    @property
    def fps(self) -> float:
        return self.sr / self.hop

    @property
    def df(self) -> float:
        return self.sr / self.n_fft

    @property
    def n_frames(self) -> int:
        return self.mono.shape[0]

    def frame_times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.hop / self.sr


def analyze(x: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 256,
            fmax: float = 5500.0, center_power: float = 2.0,
            chunk: int = 1024) -> Features:
    """Chunked STFT of a stereo signal (2, n)."""
    if x.ndim == 1:
        x = np.stack([x, x])
    pad = n_fft // 2
    xl = np.pad(x[0].astype(np.float32), pad)
    xr = np.pad(x[1].astype(np.float32), pad)
    if xl.size < n_fft:
        xl = np.pad(xl, (0, n_fft - xl.size))
        xr = np.pad(xr, (0, n_fft - xr.size))
    n_frames = 1 + (xl.size - n_fft) // hop
    win = get_window("hann", n_fft).astype(np.float32)
    kmax = min(n_fft // 2 + 1, int(np.ceil(fmax * n_fft / sr)) + 2)
    fl = np.lib.stride_tricks.sliding_window_view(xl, n_fft)[::hop]
    fr = np.lib.stride_tricks.sliding_window_view(xr, n_fft)[::hop]

    # short window (1024) for onset timing, centred on the same instants
    ns = 1024
    xm = 0.5 * (xl + xr)
    if xm.size < pad - ns // 2 + ns:
        xm = np.pad(xm, (0, pad + ns))
    fs = np.lib.stride_tricks.sliding_window_view(xm, ns)[pad - ns // 2::hop]
    wins = get_window("hann", ns).astype(np.float32)

    mono = np.empty((n_frames, kmax), np.float32)
    center = np.empty((n_frames, kmax), np.float32)
    flux = np.zeros(n_frames, np.float32)
    flat = np.zeros(n_frames, np.float32)
    prev = None
    for s in range(0, n_frames, chunk):
        e = min(s + chunk, n_frames)
        XL = np.fft.rfft(fl[s:e] * win, axis=1)
        XR = np.fft.rfft(fr[s:e] * win, axis=1)
        M = 0.5 * (XL + XR)
        aM = np.abs(M).astype(np.float32)
        mono[s:e] = aM[:, :kmax]
        num = 2.0 * np.real(XL[:, :kmax] * np.conj(XR[:, :kmax]))
        den = np.abs(XL[:, :kmax]) ** 2 + np.abs(XR[:, :kmax]) ** 2 + 1e-12
        sim = np.clip(num / den, 0.0, 1.0) ** center_power
        center[s:e] = aM[:, :kmax] * sim.astype(np.float32)
        # full band log-magnitude spectral flux (short window = sharp timing)
        blk = fs[s:e]
        if blk.shape[0] < e - s:
            blk = np.vstack([blk, np.zeros((e - s - blk.shape[0], ns), np.float32)])
        lg = np.log1p(100.0 * np.abs(np.fft.rfft(blk * wins, axis=1))).astype(np.float32)
        if prev is None:
            d = np.zeros_like(lg)
            d[1:] = np.diff(lg, axis=0)
        else:
            d = np.diff(np.vstack([prev[None, :], lg]), axis=0)
        flux[s:e] = np.maximum(d, 0.0).sum(axis=1)
        prev = lg[-1]
        # flatness over 300 Hz .. fmax
        lo = int(300 * n_fft / sr)
        band = aM[:, lo:kmax] + 1e-10
        flat[s:e] = np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1)
    return Features(sr, n_fft, hop, mono, center, flux, flat)


def hpss_mask(mag: np.ndarray, time_size: int = 17, freq_size: int = 17,
              power: float = 2.0) -> np.ndarray:
    """Soft harmonic mask from median-filtering HPSS (Fitzgerald 2010)."""
    H = median_filter(mag, size=(time_size, 1), mode="nearest")
    P = median_filter(mag, size=(1, freq_size), mode="nearest")
    Hp = H.astype(np.float32) ** power
    Pp = P.astype(np.float32) ** power
    return Hp / (Hp + Pp + 1e-12)


def fluctuating_part(mag: np.ndarray, time_size: int = 41) -> np.ndarray:
    """Magnitude minus its long-term (~0.5 s) per-bin median.

    Rock-steady partials (sustained piano/organ/pad notes, held chords) are
    removed, while partials whose frequency keeps moving -- vibrato, scoops,
    expressive intonation, i.e. typically the singing voice -- survive.  This
    is used as an additional cue for which harmonic series is the voice.
    """
    med = median_filter(mag, size=(time_size, 1), mode="nearest")
    return np.maximum(mag - med, 0.0).astype(np.float32)


def whitening_gain(n_bins: int, df: float):
    """Interpolation matrix + band weights for Klapuri-style spectral whitening.

    Whitening divides out the smooth spectral envelope (for singing this is
    mostly the formant / vocal-tract resonance structure) so that the
    harmonic-summation salience depends on *where* the harmonics are, not on
    which harmonic happens to be boosted by the current vowel.
    """
    freqs = np.arange(n_bins) * df
    fmax = freqs[-1]
    centers = []
    b = 0
    while True:
        c = 229.0 * (10 ** ((b + 1) / 21.4) - 1.0)
        if c > fmax:
            break
        centers.append(c)
        b += 1
    centers = np.array([0.0] + centers + [fmax + 1.0])
    nb = len(centers) - 2
    W = np.zeros((nb, n_bins), np.float32)
    for i in range(1, nb + 1):
        lo, c, hi = centers[i - 1], centers[i], centers[i + 1]
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        W[i - 1] = np.clip(np.minimum(up, dn), 0, None)
    W /= W.sum(axis=1, keepdims=True) + 1e-12
    # linear interpolation of per-band gains back to bins
    cc = centers[1:-1]
    I = np.zeros((nb, n_bins), np.float32)
    for k, f in enumerate(freqs):
        if f <= cc[0]:
            I[0, k] = 1.0
        elif f >= cc[-1]:
            I[-1, k] = 1.0
        else:
            j = np.searchsorted(cc, f) - 1
            t = (f - cc[j]) / (cc[j + 1] - cc[j])
            I[j, k] = 1 - t
            I[j + 1, k] = t
    return W, I


def whiten(mag: np.ndarray, df: float, nu: float = 0.33, chunk: int = 4096) -> np.ndarray:
    W, I = whitening_gain(mag.shape[1], df)
    out = np.empty_like(mag, dtype=np.float32)
    floor = float(np.mean(mag)) * 1e-3 + 1e-9
    for s in range(0, mag.shape[0], chunk):
        m = mag[s:s + chunk].astype(np.float32)
        sigma = np.sqrt((m ** 2) @ W.T) + floor
        gamma = sigma ** (nu - 1.0)
        out[s:s + chunk] = m * (gamma @ I)
    return out


def peak_spectrum(mag: np.ndarray) -> np.ndarray:
    """Keep only local maxima along frequency (everything else zero)."""
    P = np.zeros_like(mag)
    c = mag[:, 1:-1]
    is_peak = (c > mag[:, :-2]) & (c >= mag[:, 2:])
    P[:, 1:-1] = np.where(is_peak, c, 0.0)
    return P


class HarmonicSalience:
    """Harmonic summation: S(f0) = sum_h alpha^(h-1) * max_{|k - h f0| <= w} P(k).

    The search half-width ``w`` grows with the harmonic number (``tol_cents``)
    so that vibrato-smeared upper partials of a voice are still collected.
    """

    def __init__(self, f0s: np.ndarray, df: float, n_bins: int, n_harm: int = 12,
                 alpha: float = 0.8, tol_cents: float = 15.0, max_halfwidth: int = 8,
                 fmax: float | None = None):
        self.f0s = np.asarray(f0s, float)
        self.df = df
        self.n_bins = n_bins
        ratio = 2 ** (tol_cents / 1200.0) - 1.0
        lim = n_bins - 2 if fmax is None else min(n_bins - 2, fmax / df)
        self.terms = []   # (weight, width, column indices, bin indices)
        widths = set()
        for h in range(1, n_harm + 1):
            target = h * self.f0s / df
            valid = target < lim
            if not np.any(valid):
                continue
            w = np.clip(np.ceil(target * ratio), 1, max_halfwidth).astype(int)
            for ww in np.unique(w[valid]):
                cols = np.nonzero(valid & (w == ww))[0]
                bins = np.round(target[cols]).astype(int)
                self.terms.append((alpha ** (h - 1), int(ww), cols, bins))
                widths.add(int(ww))
        self.widths = sorted(widths)

    def __call__(self, P: np.ndarray) -> np.ndarray:
        D = {w: maximum_filter1d(P, 2 * w + 1, axis=1, mode="constant") for w in self.widths}
        S = np.zeros((P.shape[0], self.f0s.size), np.float32)
        for weight, w, cols, bins in self.terms:
            S[:, cols] += weight * D[w][:, bins]
        return S


def estimate_tuning(mag: np.ndarray, df: float, fmin: float = 80.0, fmax: float = 2000.0,
                    step: int = 4) -> float:
    """Global tuning deviation from A440 in cents (-50, 50], from spectral peaks."""
    m = mag[::step].astype(np.float64)
    if m.shape[0] == 0:
        return 0.0
    lo = max(1, int(fmin / df))
    hi = min(m.shape[1] - 2, int(fmax / df))
    a = np.log(m[:, lo - 1:hi - 1] + 1e-9)
    b = np.log(m[:, lo:hi] + 1e-9)
    c = np.log(m[:, lo + 1:hi + 1] + 1e-9)
    raw = m[:, lo:hi]
    is_peak = (raw > m[:, lo - 1:hi - 1]) & (raw >= m[:, lo + 1:hi + 1])
    strong = raw > 0.2 * raw.max(axis=1, keepdims=True)
    sel = is_peak & strong & (raw > 1e-6)
    if not np.any(sel):
        return 0.0
    denom = a - 2 * b + c
    p = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0.0)
    p = np.clip(p, -0.5, 0.5)
    k = np.arange(lo, hi)[None, :] + p
    f = k[sel] * df
    w = raw[sel]
    cents = 1200 * np.log2(f / 440.0)
    ang = 2 * np.pi * cents / 100.0
    z = np.sum(w * np.exp(1j * ang))
    if np.abs(z) < 1e-12:
        return 0.0
    return float(np.angle(z) / (2 * np.pi) * 100.0)
