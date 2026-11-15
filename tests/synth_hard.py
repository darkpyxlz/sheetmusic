"""A harder, more 'real-recording-like' synthetic song.

Compared to synth.py it adds what usually breaks transcription of real songs:
an instrumental intro with a melodic guitar riff (no voice), reverb on the
voice, short syllabic notes, syncopation, legato slides between notes, rests
between phrases, a quieter backing-vocal harmony, strummed guitar chords,
a moving bass line, kick/snare/hi-hat, slight tempo drift and breath noise.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter

from synth import SR, VOWELS, _formant_filter, kick, hat, mtof, piano_note

# (midi, beats) ; None = rest.  Key of D major, 4/4.
VERSE = [
    (None, 1), (66, 0.5), (66, 0.5), (69, 0.5), (71, 0.5), (69, 1),
    (66, 0.5), (64, 0.5), (62, 1), (None, 1), (62, 0.5), (64, 0.5),
    (66, 0.75), (66, 0.25), (66, 0.5), (69, 1), (71, 0.5), (69, 0.5), (66, 0.5),
    (64, 3), (None, 1),
    (None, 1), (66, 0.5), (69, 0.5), (74, 1), (73, 0.5), (71, 0.5),
    (69, 1.5), (66, 0.5), (64, 1), (62, 1),
    (64, 0.5), (66, 1), (64, 0.5), (62, 1), (59, 1),
    (62, 4),
]
PROG = [[50, 54, 57], [47, 50, 54], [43, 47, 50], [45, 49, 52]]  # D Bm G A
BASS_PAT = [(0, 1.5), (1.5, 0.5), (2, 1), (3, 1)]


def sung_note(f0_curve: np.ndarray, vowel: str, rng, breath=0.03):
    n = f0_curve.size
    phase = 2 * np.pi * np.cumsum(f0_curve) / SR
    src = np.zeros(n)
    for h in range(1, 40):
        amp = 1.0 / h ** 1.1 * (0.3 if h == 1 else 1.0)
        src += np.where(h * f0_curve < SR / 2 - 200, amp * np.sin(h * phase), 0)
    src += breath * rng.standard_normal(n) * np.abs(src).max()
    return _formant_filter(src, VOWELS[vowel], SR)


def make_hard_song(bpm: float = 96.0, seed: int = 3, intro_bars: int = 2,
                   drift: float = 0.03):
    rng = np.random.default_rng(seed)
    melody = VERSE
    n_bars = intro_bars + int(np.ceil(sum(d for _, d in melody) / 4)) + 1
    total_beats = n_bars * 4 + 1
    # tempo map with slow drift: beat -> seconds
    beats = np.arange(total_beats + 1, dtype=float)
    local = 60.0 / bpm * (1 + drift * np.sin(2 * np.pi * beats / (total_beats * 0.7)))
    btime = np.concatenate([[0.0], np.cumsum(local)[:-1]]) + 0.4

    def t_of(b):
        return float(np.interp(b, beats, btime))

    n = int((t_of(total_beats) + 2.0) * SR)
    voice = np.zeros(n)
    back = np.zeros(n)
    L = np.zeros(n)
    R = np.zeros(n)

    # ---- lead vocal: continuous f0 with legato slides inside phrases ----
    truth = []
    b = intro_bars * 4.0
    vowels = list("aeoiuaoeia")
    events = []
    for p, d in melody:
        if p is not None:
            events.append((b, d, p))
            truth.append((b - intro_bars * 4.0, d, p))
        b += d
    vi = 0
    for i, (b0, d, p) in enumerate(events):
        s, e = t_of(b0), t_of(b0 + d)
        ns = int(s * SR)
        ln = int((e - s) * 0.95 * SR)
        t = np.arange(ln) / SR
        cents = np.zeros(ln)
        prev = events[i - 1][2] if i > 0 else p
        # slide (no new consonant) between different contiguous pitches; a
        # repeated pitch is re-articulated with a consonant like real singing
        legato = i > 0 and abs(events[i - 1][0] + events[i - 1][1] - b0) < 1e-6 and prev != p
        if legato and prev != p:  # slide from the previous pitch
            cents += (prev - p) * 100 * np.exp(-t / 0.04)
        elif not legato:
            cents += -80 * np.exp(-t / 0.03)
        vib = 35 * np.clip((t - 0.25) / 0.2, 0, 1)
        cents += vib * np.sin(2 * np.pi * 5.8 * t)
        f0 = mtof(p) * 2 ** (cents / 1200)
        y = sung_note(f0, vowels[vi % len(vowels)], rng)
        vi += 1
        env = np.minimum(1, t / 0.02) * np.clip((t[-1] - t) / 0.05, 0, 1)
        y *= env * (0.8 + 0.2 * rng.random())
        if not legato:  # consonant
            k = int(0.04 * SR)
            bb, aa = butter(2, [2500 / (SR / 2), 8000 / (SR / 2)], btype="band")
            y[:k] += 0.5 * lfilter(bb, aa, rng.standard_normal(k)) * np.hanning(k) * np.abs(y).max()
        voice[ns:ns + ln] += y[: max(0, n - ns)]
        # backing vocal a third/sixth below on long notes, quieter, wider
        if d >= 1:
            dmaj = [1, 2, 4, 6, 7, 9, 11]
            q = max(x for x in range(p - 5, p - 2) if x % 12 in dmaj)  # diatonic third below
            yb = sung_note(mtof(q) * 2 ** (cents / 1200 * 0.5), "o", rng) * env
            back[ns:ns + ln] += 0.25 * yb[: max(0, n - ns)]

    # reverb (exponential-decay noise IR)
    ir_len = int(1.2 * SR)
    ir = rng.standard_normal(ir_len) * np.exp(-np.arange(ir_len) / (0.35 * SR))
    ir[0] = 25.0
    ir /= np.abs(ir).sum() / 8
    wet = fftconvolve(voice, ir)[:n]
    voice = 0.75 * voice / np.abs(voice).max() + 0.25 * wet / np.abs(wet).max()

    # ---- guitar: strummed chords (8th strums) + intro riff ----
    acc_truth = []
    vb = intro_bars * 4.0
    for bar in range(n_bars):
        ch = PROG[bar % 4]
        for k8 in range(8):
            if k8 in (1, 5):
                continue
            bb0 = bar * 4 + k8 * 0.5
            s = int(t_of(bb0) * SR)
            for j, p in enumerate(ch + [ch[0] + 12]):
                acc_truth.append((bb0 - vb, 0.5, p))
                y = piano_note(p, 0.45, vel=0.18 if k8 % 2 else 0.25)
                off = s + int(j * 0.008 * SR)
                e = min(n, off + y.size)
                L[off:e] += 0.7 * y[: e - off]
                R[off:e] += 0.3 * y[: e - off]
    riff = [(74, 0.5), (73, 0.5), (71, 0.5), (69, 0.5), (71, 1), (66, 1)] * 2
    b = 0.0
    for p, d in riff:  # intro lead guitar riff, panned right, no vibrato
        acc_truth.append((b - vb, d, p))
        s = int(t_of(b) * SR)
        y = piano_note(p, d * 60 / bpm * 0.9, vel=0.35)
        e = min(n, s + y.size)
        R[s:e] += 0.8 * y[: e - s]
        L[s:e] += 0.2 * y[: e - s]
        b += d
    # ---- bass ----
    for bar in range(n_bars):
        root = PROG[bar % 4][0] - 12
        for o, d in BASS_PAT:
            acc_truth.append((bar * 4 + o - vb, d, root + (7 if o == 1.5 else 0)))
            s = int(t_of(bar * 4 + o) * SR)
            y = piano_note(root + (7 if o == 1.5 else 0), d * 60 / bpm * 0.9, vel=0.5)
            e = min(n, s + y.size)
            L[s:e] += 0.5 * y[: e - s]
            R[s:e] += 0.5 * y[: e - s]
    # ---- drums ----
    kk, hh = kick(), hat(rng=rng)
    snare_len = int(0.15 * SR)
    for bt in range(n_bars * 4):
        for off, kind in ((0, "h"), (0.5, "h")):
            s = int(t_of(bt + off) * SR)
            e = min(n, s + hh.size)
            L[s:e] += 0.08 * hh[: e - s]
            R[s:e] += 0.12 * hh[: e - s]
        s = int(t_of(bt) * SR)
        if bt % 4 in (0, 2):
            e = min(n, s + kk.size)
            L[s:e] += 0.6 * kk[: e - s]
            R[s:e] += 0.6 * kk[: e - s]
        else:
            sn = rng.standard_normal(snare_len) * np.exp(-np.arange(snare_len) / (0.04 * SR))
            sn += 0.5 * np.sin(2 * np.pi * 190 * np.arange(snare_len) / SR) * np.exp(-np.arange(snare_len) / (0.05 * SR))
            e = min(n, s + snare_len)
            L[s:e] += 0.25 * sn[: e - s]
            R[s:e] += 0.25 * sn[: e - s]

    acc = np.stack([L, R])
    acc *= 0.45 / np.abs(acc).max()
    back = back / (np.abs(back).max() + 1e-9) * 0.12
    mix = acc + np.stack([0.5 * voice + back * 1.2, 0.5 * voice + back * 0.8])
    mix *= 0.9 / np.abs(mix).max()
    return mix.astype(np.float32), {"melody": truth, "accomp": acc_truth, "bpm": bpm,
                                    "first_vocal_beat": intro_bars * 4.0}
