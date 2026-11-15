"""Synthesise a test 'song' with known notes.

The voice is deliberately hard for naive pitch picking: weak fundamental,
vowel formants that change every note, 5.5 Hz vibrato, onset scoops and
consonant noise bursts.  It is accompanied by piano-like chords, a bass line
and drums, mixed in stereo with the voice in the centre.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, butter

SR = 22050

VOWELS = {  # (F1, F2, F3) Hz
    "a": (800, 1200, 2500), "e": (500, 1900, 2600), "i": (300, 2300, 3000),
    "o": (500, 900, 2400), "u": (320, 800, 2300),
}


A4 = 440.0


def mtof(m):
    return A4 * 2 ** ((np.asarray(m, float) - 69) / 12)


def _formant_filter(x, freqs, sr, bw=90.0):
    y = np.zeros_like(x)
    for i, f in enumerate(freqs):
        r = np.exp(-np.pi * bw * (1 + 0.5 * i) / sr)
        th = 2 * np.pi * f / sr
        b = [1 - r]
        a = [1, -2 * r * np.cos(th), r * r]
        y += lfilter(b, a, x) * (1.0 / (1 + i))
    return y


def voice_note(midi, dur, vowel, sr=SR, vibrato=True, scoop=True, consonant=True, rng=None):
    rng = rng or np.random.default_rng(0)
    n = int(dur * sr)
    t = np.arange(n) / sr
    cents = np.zeros(n)
    if vibrato:
        depth = 40.0 * np.clip((t - 0.15) / 0.2, 0, 1)  # vibrato develops after onset
        cents += depth * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 6.28))
    if scoop:
        cents += -120.0 * np.exp(-t / 0.035)
    cents += 8 * np.cumsum(rng.standard_normal(n)) / np.sqrt(sr * 50)  # jitter / drift
    f0 = mtof(midi) * 2 ** (cents / 1200)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = np.zeros(n)
    for h in range(1, 40):
        amp = 1.0 / h ** 1.2
        if h == 1:
            amp *= 0.25   # weak fundamental: the classic octave-error trap
        ok = h * f0 < sr / 2 - 200
        src += np.where(ok, amp * np.sin(h * phase), 0)
    y = _formant_filter(src, VOWELS[vowel], sr)
    env = np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.06).clip(0, 1)
    y *= env
    if consonant:
        k = int(0.05 * sr)
        b, a = butter(2, [3000 / (sr / 2), 8000 / (sr / 2)], btype="band")
        noise = lfilter(b, a, rng.standard_normal(k)) * np.hanning(k)
        y[:k] += 0.6 * noise * np.abs(y).max()
    return y


def piano_note(midi, dur, sr=SR, vel=0.5):
    n = int((dur + 0.3) * sr)
    t = np.arange(n) / sr
    f = mtof(midi)
    y = np.zeros(n)
    for h in range(1, 16):
        fh = f * h * np.sqrt(1 + 0.0004 * h * h)
        if fh > sr / 2 - 100:
            break
        y += (0.8 ** h) * np.exp(-t * (1.2 + 0.6 * h)) * np.sin(2 * np.pi * fh * t)
    rel = np.where(t > dur, np.exp(-(t - dur) / 0.08), 1.0)
    return vel * y * rel * np.minimum(1, t / 0.004)


def kick(sr=SR):
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    f = 50 + 100 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.08)


def hat(sr=SR, rng=None):
    rng = rng or np.random.default_rng(1)
    n = int(0.05 * sr)
    b, a = butter(2, 7000 / (sr / 2), btype="high")
    return lfilter(b, a, rng.standard_normal(n)) * np.exp(-np.arange(n) / (0.01 * sr))


# Melody in G major, 100 BPM, 4/4. (midi, beats); None = rest.
BPM = 100.0
MELODY = [
    (67, 1), (69, 1), (71, 1), (67, 1),
    (72, 2), (71, 1), (69, 1),
    (67, 1.5), (66, 0.5), (64, 1), (62, 1),
    (67, 3), (None, 1),
    (71, 1), (72, 1), (74, 2),
    (76, 1), (74, 1), (72, 1), (71, 1),
    (69, 2), (71, 1), (69, 1),
    (67, 4),
]
CHORDS = [  # one chord per bar (pitches), played as half notes
    [55, 59, 62], [52, 55, 60], [50, 54, 57], [55, 59, 62],
    [55, 59, 62], [48, 52, 55], [50, 54, 57], [55, 59, 62],
]
BASS = [43, 48, 38, 43, 43, 36, 38, 43]
VOWEL_SEQ = "aeiouaeoiuaeioua"


def make_song(lead_in_beats: float = 1.0, with_drums: bool = True, seed: int = 0,
              bpm: float = BPM, transpose_voice: int = 0, detune_cents: float = 0.0,
              repeats: int = 1, stereo: bool = True):
    """Return (stereo (2, n) float32, ground_truth dict)."""
    global A4
    A4 = 440.0 * 2 ** (detune_cents / 1200.0)
    try:
        return _make_song(lead_in_beats, with_drums, seed, bpm, transpose_voice, repeats, stereo)
    finally:
        A4 = 440.0


def _make_song(lead_in_beats, with_drums, seed, bpm, transpose_voice, repeats, stereo):
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    melody = [(None if p is None else p + transpose_voice, d) for p, d in MELODY] * repeats
    chords = CHORDS * repeats
    bass = BASS * repeats
    total_beats = sum(d for _, d in melody) + lead_in_beats + 2
    n = int(total_beats * beat * SR)
    voice = np.zeros(n)
    left = np.zeros(n)
    right = np.zeros(n)
    truth_mel = []
    b = lead_in_beats
    vi = 0
    for p, d in melody:
        if p is not None:
            s = int(b * beat * SR)
            y = voice_note(p, d * beat * 0.97, VOWEL_SEQ[vi % len(VOWEL_SEQ)], rng=rng)
            vi += 1
            voice[s:s + y.size] += y[: max(0, n - s)]
            truth_mel.append((b - lead_in_beats, d, p))
        b += d
    truth_acc = []
    for bar, ch in enumerate(chords):
        for half in range(2):
            bb = lead_in_beats + bar * 4 + half * 2
            s = int(bb * beat * SR)
            for k, p in enumerate(ch):
                y = piano_note(p, 2 * beat * 0.95, vel=0.35)
                pan = 0.3 + 0.4 * (k / 2)
                e = min(n, s + y.size)
                left[s:e] += (1 - pan) * y[: e - s]
                right[s:e] += pan * y[: e - s]
                truth_acc.append((bb - lead_in_beats, 2, p))
        # bass: whole note
        bb = lead_in_beats + bar * 4
        s = int(bb * beat * SR)
        y = piano_note(bass[bar], 4 * beat * 0.95, vel=0.5)
        e = min(n, s + y.size)
        left[s:e] += 0.5 * y[: e - s]
        right[s:e] += 0.5 * y[: e - s]
        truth_acc.append((bb - lead_in_beats, 4, bass[bar]))
    if with_drums:
        k, h = kick(), hat(rng=rng)
        nb = int(total_beats)
        for i in range(nb):
            s = int((lead_in_beats + i) * beat * SR)
            if s >= n:
                break
            if i % 2 == 0:
                e = min(n, s + k.size)
                left[s:e] += 0.5 * k[: e - s]
                right[s:e] += 0.5 * k[: e - s]
            for off in (0, 0.5):
                ss = int((lead_in_beats + i + off) * beat * SR)
                e = min(n, ss + h.size)
                if ss < n:
                    left[ss:e] += 0.12 * h[: e - ss]
                    right[ss:e] += 0.06 * h[: e - ss]
    voice *= 0.5 / (np.abs(voice).max() + 1e-9)
    acc_peak = max(np.abs(left).max(), np.abs(right).max()) + 1e-9
    left *= 0.35 / acc_peak
    right *= 0.35 / acc_peak
    mix = np.stack([left + voice, right + voice])
    if not stereo:
        mix = np.repeat(mix.mean(axis=0, keepdims=True), 2, axis=0)
    mix *= 0.9 / np.abs(mix).max()
    return mix.astype(np.float32), {"melody": truth_mel, "accomp": truth_acc, "bpm": bpm,
                                    "lead_in": lead_in_beats}


def write_mp3(path: str, x: np.ndarray, sr: int = SR):
    import subprocess
    pcm = (np.clip(x.T, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "2",
                    "-i", "-", "-b:a", "192k", path], input=pcm, check=True)


if __name__ == "__main__":
    import sys
    mix, truth = make_song()
    write_mp3(sys.argv[1] if len(sys.argv) > 1 else "synth_song.mp3", mix)
