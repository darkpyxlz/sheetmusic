"""Instrumental (no vocals) test pieces with known notes, key, tempo and
time signature, rendered with simple additive instrument models."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from synth import SR, hat, kick, mtof, voice_note


# ---------------------------------------------------------------- instruments
def _env(n, attack, release, sustain_decay=None):
    t = np.arange(n) / SR
    e = np.minimum(1.0, t / max(attack, 1e-4))
    if sustain_decay is not None:
        e = e * np.exp(-t / sustain_decay)
    rel = int(release * SR)
    if rel > 0 and n > 0:
        e[-min(rel, n):] *= np.linspace(1, 0, min(rel, n))
    return e


def instrument(name: str, midi: int, dur: float, vel: float, rng) -> np.ndarray:
    f0 = mtof(midi)
    if name == "piano":
        n = int((dur + 0.25) * SR)
        t = np.arange(n) / SR
        y = np.zeros(n)
        for h in range(1, 16):
            fh = f0 * h * np.sqrt(1 + 0.0004 * h * h)
            if fh > SR / 2 - 100:
                break
            y += 0.8 ** h * np.exp(-t * (0.8 + 0.5 * h)) * np.sin(2 * np.pi * fh * t)
        y *= np.where(t > dur, np.exp(-(t - dur) / 0.07), 1.0) * np.minimum(1, t / 0.004)
        return vel * y
    if name == "guitar":  # plucked: bright attack, fast high-partial decay
        n = int((dur + 0.15) * SR)
        t = np.arange(n) / SR
        y = np.zeros(n)
        for h in range(1, 20):
            fh = f0 * h
            if fh > SR / 2 - 100:
                break
            y += (1.0 / h) * np.sin(np.pi * h * 0.2) * np.exp(-t * (1.5 + 0.9 * h)) * np.sin(2 * np.pi * fh * t)
        y *= np.where(t > dur, np.exp(-(t - dur) / 0.04), 1.0) * np.minimum(1, t / 0.002)
        return vel * y
    if name in ("flute", "strings", "organ"):
        n = int(dur * SR)
        t = np.arange(n) / SR
        vib = {"flute": 18, "strings": 25, "organ": 0}[name]
        cents = vib * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6)) * np.clip((t - 0.15) / 0.2, 0, 1)
        f = f0 * 2 ** (cents / 1200)
        ph = 2 * np.pi * np.cumsum(f) / SR
        amps = {"flute": [1, 0.35, 0.12, 0.05, 0.02],
                "strings": [1 / h ** 0.9 for h in range(1, 18)],
                "organ": [1, 0.6, 0.0, 0.4, 0.0, 0.2, 0.0, 0.15]}[name]
        y = sum(a * np.sin((h + 1) * ph) for h, a in enumerate(amps) if (h + 1) * f0 < SR / 2 - 200)
        if name == "flute":
            y = y + 0.03 * rng.standard_normal(n)
        att = {"flute": 0.04, "strings": 0.08, "organ": 0.01}[name]
        return vel * y * _env(n, att, 0.05)
    if name == "voice":
        return vel * 4.5 * voice_note(midi, dur, "aeiou"[int(rng.integers(5))], rng=rng)
    if name == "bass":
        n = int((dur + 0.1) * SR)
        t = np.arange(n) / SR
        y = sum((0.9 ** h) * np.exp(-t * (0.6 + 0.4 * h)) * np.sin(2 * np.pi * f0 * h * t)
                for h in range(1, 8))
        y *= np.where(t > dur, np.exp(-(t - dur) / 0.03), 1.0) * np.minimum(1, t / 0.003)
        return vel * y
    raise ValueError(name)


# ---------------------------------------------------------------- pieces
# events: (beat, dur_beats, midi, instrument, pan 0..1, velocity)
def _waltz():
    """F major, 3/4, 132 BPM: piano melody + oom-pah-pah."""
    mel = [(72, 2), (69, 1), (70, 1), (72, 1), (74, 1), (72, 3),
           (77, 2), (76, 1), (74, 1), (72, 1), (70, 1), (69, 3),
           (70, 2), (67, 1), (69, 1), (70, 1), (72, 1), (74, 2), (72, 1),
           (70, 1), (69, 1), (67, 1), (65, 3)]
    chords = [(53, [57, 60]), (53, [57, 60]), (46, [58, 62]), (48, [57, 60]),
              (46, [58, 62]), (48, [55, 58]), (48, [55, 58]), (53, [57, 60])]
    ev = []
    b = 0.0
    for p, d in mel:
        ev.append((b, d, p, "piano", 0.5, 0.45))
        b += d
    for bar, (bass, ch) in enumerate(chords):
        ev.append((bar * 3, 1, bass, "piano", 0.5, 0.4))
        for k in (1, 2):
            for p in ch:
                ev.append((bar * 3 + k, 1, p, "piano", 0.5, 0.22))
    return dict(events=ev, bpm=132, meter=(3, 4), key=(-1, "major"), drums=False)


def _six_eight():
    """A minor, 6/8 at 66 dotted-quarter BPM: flute melody over guitar
    arpeggios (8ths) and bass.  Beats are dotted quarters."""
    # melody in eighth-note units (1 = eighth = 1/3 beat)
    mel = [(69, 3), (72, 2), (71, 1), (69, 3), (64, 3),
           (65, 2), (67, 1), (69, 2), (71, 1), (72, 6),
           (74, 3), (72, 2), (71, 1), (69, 2), (68, 1), (69, 3),
           (71, 2), (69, 1), (68, 2), (64, 1), (69, 6)]
    arps = [[45, 52, 57, 60, 57, 52], [41, 48, 53, 57, 53, 48], [40, 47, 52, 56, 52, 47],
            [45, 52, 57, 60, 57, 52]]
    ev = []
    b = 0.0
    for p, d in mel:
        ev.append((b, d / 3, p, "flute", 0.45, 0.5))
        b += d / 3
    nbars = int(round(b / 2))
    for bar in range(nbars):
        a = arps[(bar // 2) % 4]
        for k, p in enumerate(a):
            ev.append((bar * 2 + k / 3, 1 / 3, p, "guitar", 0.7, 0.35))
        ev.append((bar * 2, 2, a[0] - 12, "bass", 0.5, 0.45))
    return dict(events=ev, bpm=66, meter=(6, 8), key=(0, "minor"), drums=False)


def _four_four_band():
    """Eb major, 4/4, 112 BPM: strings lead, piano chords, bass, drums."""
    mel = [(70, 1), (72, 0.5), (70, 0.5), (67, 1), (63, 1),
           (65, 1.5), (67, 0.5), (68, 2),
           (67, 1), (65, 0.5), (63, 0.5), (65, 1), (70, 1),
           (67, 4),
           (75, 1), (74, 0.5), (72, 0.5), (70, 2),
           (68, 1), (67, 1), (65, 1), (63, 1),
           (65, 1.5), (67, 0.5), (65, 1), (62, 1),
           (63, 4)]
    prog = [[51, 55, 58], [56, 60, 63], [51, 55, 58], [51, 55, 58],
            [56, 60, 63], [48, 51, 55], [46, 50, 53], [51, 55, 58]]
    ev = []
    b = 0.0
    for p, d in mel:
        ev.append((b, d, p, "strings", 0.5, 0.45))
        b += d
    for bar, ch in enumerate(prog):
        for k in (0, 2):
            for p in ch:
                ev.append((bar * 4 + k, 2, p, "piano", 0.35, 0.22))
        ev.append((bar * 4, 2, ch[0] - 12, "bass", 0.5, 0.5))
        ev.append((bar * 4 + 2, 2, ch[0] - 12 + 7 if ch[0] - 5 > 40 else ch[0] - 12, "bass", 0.5, 0.5))
    return dict(events=ev, bpm=112, meter=(4, 4), key=(-3, "major"), drums=True)


def _minor_piano():
    """D minor, 4/4, 84 BPM: solo piano, melody + left-hand chords in 8ths."""
    mel = [(74, 1.5), (72, 0.5), (69, 1), (65, 1),
           (67, 1), (69, 1), (70, 2),
           (69, 1), (67, 0.5), (65, 0.5), (64, 1), (61, 1),
           (62, 4)]
    lh = [[50, 57, 62], [46, 53, 58], [45, 52, 57], [50, 57, 62]]
    ev = []
    b = 0.0
    for p, d in mel:
        ev.append((b, d, p, "piano", 0.5, 0.45))
        b += d
    for bar, ch in enumerate(lh):
        for k in range(8):
            ev.append((bar * 4 + k * 0.5, 0.5, ch[k % 3] if k % 2 == 0 else ch[1 + (k // 2) % 2],
                       "piano", 0.5, 0.22))
    return dict(events=ev, bpm=84, meter=(4, 4), key=(-1, "minor"), drums=False)


def _sung_waltz():
    """The waltz with its tune sung (vibrato, formants, consonants) - G major."""
    spec = _waltz()
    spec["events"] = [(b, d, p + 2, "voice" if v == 0.45 else inst, pan, v)
                      for b, d, p, inst, pan, v in spec["events"]]
    spec["key"] = (1, "major")
    spec["has_vocals"] = True
    return spec


def _triplets():
    """C major, 4/4, 92 BPM, piano: a tune mixing crotchets, straight quavers
    and triplet quavers, over left-hand minim chords."""
    t = 1 / 3
    mel = [(72, 1), (74, t), (76, t), (77, t), (79, 1), (77, 0.5), (76, 0.5),
           (74, 1), (72, t), (74, t), (76, t), (74, 2),
           (76, 0.5), (77, 0.5), (79, t), (77, t), (76, t), (74, 1), (72, 1),
           (71, t), (72, t), (74, t), (72, 1), (67, 2)]
    lh = [[48, 55, 64], [53, 57, 60], [55, 59, 62], [48, 55, 64]]
    ev, b = [], 0.0
    for p, d in mel:
        ev.append((b, d, p, "piano", 0.5, 0.45))
        b += d
    for bar, ch in enumerate(lh):
        for k in (0, 2):
            for p in ch:
                ev.append((bar * 4 + k, 2, p, "piano", 0.5, 0.22))
    return dict(events=ev, bpm=92, meter=(4, 4), key=(0, "major"), drums=False)


def _swing():
    """F major, 4/4, 132 BPM: a swung quaver tune (played long-short, written
    straight) over crotchet bass and off-beat chords."""
    mel = [72, 70, 69, 67, 65, 67, 69, 72, 74, 72, 70, 69, 67, 69, 70, 67,
           65, 67, 69, 70, 72, 74, 72, 70, 69, 67, 65, 64, 65, 67, 65, 60]
    ev = [(i * 0.5, 0.5, p, "piano", 0.5, 0.45) for i, p in enumerate(mel)]
    roots = [41, 46, 48, 41]
    for bar, r in enumerate(roots):
        for beat in range(4):
            ev.append((bar * 4 + beat, 1, r if beat % 2 == 0 else r + 7, "bass", 0.5, 0.45))
    return dict(events=ev, bpm=132, meter=(4, 4), key=(-1, "major"), drums=False, swing=True)


PIECES = {"waltz_F_3-4": _waltz, "flute_Am_6-8": _six_eight, "band_Eb_4-4": _four_four_band,
          "piano_Dm_4-4": _minor_piano, "sung_waltz_G_3-4": _sung_waltz,
          "triplets_C_4-4": _triplets, "swing_F_4-4": _swing}


def _swung(b: float) -> float:
    """Where a straight-written position is played under swing (2:1)."""
    k = np.floor(b + 1e-9)
    f = b - k
    return k + (2 / 3 if abs(f - 0.5) < 1e-6 else f)


def make_piece(name: str, seed: int = 0, lead_in: float = 0.5, repeats: int = 2,
               transpose: int = 0):
    rng = np.random.default_rng(seed)
    spec = PIECES[name]()
    if transpose:
        spec["events"] = [(b, d, p + transpose, *rest) for b, d, p, *rest in spec["events"]]
        f, mode = spec["key"]
        tonic = ((f * 7) % 12 + (9 if mode == "minor" else 0) + transpose) % 12
        rel = tonic if mode == "major" else (tonic + 3) % 12
        spec["key"] = ({0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: -6, 1: -5, 8: -4, 3: -3,
                        10: -2, 5: -1}[rel], mode)
    if repeats > 1:  # play the piece several times, like verses of a song
        num, den = spec["meter"]
        bar = num if den == 4 else num / 3
        length = np.ceil(max(b + d for b, d, *_ in spec["events"]) / bar) * bar
        spec["events"] = [(b + r * length, *rest) for r in range(repeats)
                          for b, *rest in spec["events"]]
    bpm = spec["bpm"]
    num, den = spec["meter"]
    beat_s = 60.0 / bpm
    last = max(b + d for b, d, *_ in spec["events"])
    n = int((lead_in + last * beat_s + 1.5) * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    for b, d, p, inst, pan, vel in spec["events"]:
        if spec.get("swing"):
            b, d = _swung(b), _swung(b + d) - _swung(b)
        s = int((lead_in + b * beat_s) * SR)
        y = instrument(inst, p, d * beat_s * 0.97, vel, rng)
        e = min(n, s + y.size)
        L[s:e] += (1 - pan) * y[: e - s]
        R[s:e] += pan * y[: e - s]
    if spec["drums"]:
        k, h = kick(), hat(rng=rng)
        b2 = int(SR * 0.15)
        for i in range(int(last)):
            s = int((lead_in + i * beat_s) * SR)
            if i % 2 == 0:
                e = min(n, s + k.size)
                L[s:e] += 0.5 * k[: e - s]
                R[s:e] += 0.5 * k[: e - s]
            else:
                sn = rng.standard_normal(b2) * np.exp(-np.arange(b2) / (0.04 * SR))
                e = min(n, s + b2)
                L[s:e] += 0.2 * sn[: e - s]
                R[s:e] += 0.2 * sn[: e - s]
            for off in (0, 0.5):
                ss = int((lead_in + (i + off) * beat_s) * SR)
                e = min(n, ss + h.size)
                L[ss:e] += 0.08 * h[: e - ss]
                R[ss:e] += 0.08 * h[: e - ss]
    # a little room reverb
    ir = rng.standard_normal(int(0.6 * SR)) * np.exp(-np.arange(int(0.6 * SR)) / (0.15 * SR))
    ir[0] = 60  # direct sound ~3 dB above the reverb tail (a normal room)
    ir /= np.abs(ir).sum() / 6
    from scipy.signal import fftconvolve
    L = fftconvolve(L, ir)[:n]
    R = fftconvolve(R, ir)[:n]
    b_, a_ = butter(1, 30 / (SR / 2), btype="high")
    mix = lfilter(b_, a_, np.stack([L, R]), axis=1)
    mix *= 0.9 / np.abs(mix).max()
    truth = [(b, d, p) for b, d, p, *_ in spec["events"]]
    truth.sort()
    return mix.astype(np.float32), dict(notes=truth, bpm=bpm, meter=spec["meter"],
                                        key=spec["key"],
                                        has_vocals=spec.get("has_vocals", False), swing=spec.get("swing", False),
                                        melody=[(b, d, p) for b, d, p, inst, *_ in spec["events"]
                                                if inst == "voice"])
