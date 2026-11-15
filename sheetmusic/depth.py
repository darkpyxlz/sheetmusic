"""Transcription "depth": how many notes may sound at once.

A multi-pitch analysis reports everything that looks like a harmonic series:
real notes, but also overtones, sub-harmonics and reverberation of notes that
have already ended.  Most music has two to four real notes sounding at a
time, and in the bass register far fewer: close intervals down there sound
muddy, so composers avoid them, while the analysis is at its least reliable
(semitones are closer together than the frequency resolution).  Limiting the
depth therefore removes mostly errors.

``limit_depth`` applies two rules to the notes (in seconds, before
quantisation):

* **low-interval limit**: two notes below ``bass_limit`` may not sound
  together closer than ``bass_interval`` semitones (octaves and fifths are
  fine); the weaker one gives way;
* **depth**: at most ``depth`` notes at once.  When one more starts, the
  least important note gives way.  Importance is the note's strength, with a
  bonus for the top line and the bass line, which carry the music.

* **stricter bass**: below ``weak_bass_limit`` (A3) a note must be clearly
  present next to the piece's other low notes: at least ``bass_strictness``
  times the strength of a strong low note (their 80th percentile).  On real
  piano recordings two thirds of the notes reported below E3 were false
  (sub-harmonics of higher notes, rumble, reverberation), and they are
  weaker than the real ones.  The reference is the low register itself, not
  the whole mix, so a soft accompaniment under a loud tune is kept.

"Gives way" means: a note that is already sounding is cut short at that
moment (it was most likely ringing on), a note that is just starting is
dropped.
"""

from __future__ import annotations

import numpy as np

from .notes import Note

DEPTH_NAMES = {"melody": 1, "sparse": 2, "normal": 4, "rich": 6}


def parse_depth(value) -> int | None:
    """'auto' -> 0, 'full' -> None (no limit), names or a number -> int."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(1, int(value))
    v = str(value).strip().lower()
    if v == "auto":
        return 0
    if v in ("full", "all", "none", "off"):
        return None
    if v in DEPTH_NAMES:
        return DEPTH_NAMES[v]
    return max(1, int(v))


def auto_depth(notes: list[Note], lo: int = 2, hi: int = 6) -> int:
    """Depth the music itself suggests: how many *clearly present* notes
    sound together most of the time.  A note counts while it is among the
    strong ones of its moment (at least 60 % of the strongest sounding
    note); the 80th percentile over time, plus one for a weaker inner voice."""
    if len(notes) < 4:
        return hi
    t0 = min(n.start for n in notes)
    t1 = max(n.end for n in notes)
    step = 0.05
    T = int((t1 - t0) / step) + 1
    top = np.zeros(T)
    spans = []
    for n in notes:
        a, b = int((n.start - t0) / step), max(int((n.start - t0) / step) + 1, int((n.end - t0) / step))
        spans.append((a, b, n.velocity))
        top[a:b] = np.maximum(top[a:b], n.velocity)
    cnt = np.zeros(T)
    for a, b, v in spans:
        cnt[a:b] += v >= 0.6 * top[a:b]
    active = cnt[cnt > 0]
    if active.size == 0:
        return hi
    return int(np.clip(round(float(np.percentile(active, 80))) + 1, lo, hi))


def bass_strictness(depth: int | None) -> float:
    """Required strength of low notes for a depth: shallower = stricter."""
    if depth is None:
        return 0.0
    return 0.75 if depth <= 2 else 0.6 if depth <= 5 else 0.5


def limit_depth(notes: list[Note], depth: int | None, bass_limit: int = 52,
                bass_interval: int = 7, min_keep: float = 0.08,
                edge_bonus: float = 0.25, weak_bass_limit: int = 57,
                strictness: float | None = None) -> list[Note]:
    if depth is None or not notes:
        return notes
    if strictness is None:
        strictness = bass_strictness(depth)
    # velocity = 30 + 80 * strength (see poly.activations_to_notes)
    low = np.array([n.velocity - 30 for n in notes if n.pitch < weak_bass_limit], float)
    if strictness > 0 and low.size >= 4:
        thr = 30 + strictness * np.percentile(low, 80)
        notes = [n for n in notes if n.pitch >= weak_bass_limit or n.velocity >= thr]
    order = sorted(notes, key=lambda n: (n.start, -n.velocity))
    out = [Note(n.start, n.end, n.pitch, n.velocity) for n in order]
    alive = [True] * len(out)
    active: list[int] = []

    def give_way(i, now, starting):
        if starting or now - out[i].start < min_keep:
            alive[i] = False
        else:
            out[i].end = now

    for i, n in enumerate(out):
        now = n.start
        active = [j for j in active if alive[j] and out[j].end > now + 1e-6]
        # ---- low-interval limit
        ok = True
        if n.pitch < bass_limit:
            for j in list(active):
                m = out[j]
                if m.pitch < bass_limit and 0 < abs(m.pitch - n.pitch) < bass_interval:
                    if m.velocity >= n.velocity and m.start > now - 0.05:
                        ok = False          # struck together: the stronger stays
                        break
                    give_way(j, now, m.start > now - 0.05)
                    active.remove(j)
        if not ok:
            alive[i] = False
            continue
        active.append(i)
        # ---- depth
        while len(active) > depth:
            ps = [out[j].pitch for j in active]
            hi_p, lo_p = max(ps), min(ps)

            def importance(j):
                m = out[j]
                w = m.velocity * (1.0 + edge_bonus * ((m.pitch == hi_p) + (m.pitch == lo_p)))
                return (w, m.start)        # ties: the older note goes first
            j = min(active, key=importance)
            give_way(j, now, j == i or out[j].start > now - 0.05)
            active.remove(j)
    return [n for n, a in zip(out, alive) if a and n.end - n.start >= min(min_keep, 0.03)]
