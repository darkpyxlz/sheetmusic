"""Benchmark data: real piano recordings (MAESTRO v3) and real scores rendered
through a General MIDI soundfont.  Nothing here is needed at runtime.

    python -m benchmarks.datasets            # fetch / render dev and test sets

Needs: pretty_midi (ships the soundfont), music21 (score corpus), tinysoundfont.
MAESTRO pieces are pulled out of the 100 GB zip with HTTP range requests.

dev  = pieces used while developing (look at them as much as you like)
test = held out: run only to confirm a finished change
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np

DATA = os.environ.get("SHEETMUSIC_BENCH_DATA", os.path.join(os.path.dirname(__file__), "data"))
MAESTRO = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0"
SR = 44100


# ------------------------------------------------------------------ MAESTRO
class _HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = r.read()
                break
            except OSError:
                if attempt == 3:
                    raise
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def maestro_manifest():
    """Short pieces, one per composer.  dev: the 8 shortest of the first
    composers; test: 8 pieces by *other* composers (no composer overlap)."""
    d = os.path.join(DATA, "maestro")
    os.makedirs(d, exist_ok=True)
    mp = os.path.join(d, "maestro-v3.0.0.json")
    if not os.path.exists(mp):
        urllib.request.urlretrieve(MAESTRO + ".json", mp)
    meta = json.load(open(mp))
    rows = [{k: meta[k][str(i)] for k in meta} for i in range(len(meta["duration"]))]
    rows = [r for r in rows if 60 <= r["duration"] <= 150]
    rows.sort(key=lambda r: (r["split"] != "test", r["duration"]))
    seen, pick = set(), []
    for r in rows:
        if r["canonical_composer"] not in seen:
            seen.add(r["canonical_composer"])
            pick.append(r)
    return {"dev": [(f"m{i}", r) for i, r in enumerate(pick[:8])],
            "test": [(f"t{i}", r) for i, r in enumerate(pick[8:16])]}


def fetch_maestro(which=("dev", "test")):
    d = os.path.join(DATA, "maestro")
    man = maestro_manifest()
    zf = None
    for split in which:
        index = []
        for pid, r in man[split]:
            for key, ext in (("midi_filename", ".mid"), ("audio_filename", ".wav")):
                out = os.path.join(d, pid + ext)
                if not os.path.exists(out):
                    zf = zf or zipfile.ZipFile(io.BufferedReader(_HttpFile(MAESTRO + ".zip"), 1 << 20))
                    with zf.open("maestro-v3.0.0/" + r[key]) as src, open(out + ".part", "wb") as dst:
                        while blk := src.read(1 << 22):
                            dst.write(blk)
                    os.replace(out + ".part", out)
            index.append(dict(id=pid, composer=r["canonical_composer"], title=r["canonical_title"],
                              duration=r["duration"]))
            print("maestro", split, pid, r["canonical_composer"], "|", r["canonical_title"], file=sys.stderr)
        json.dump(index, open(os.path.join(d, split + ".json"), "w"), indent=1)


def maestro_reference(pid):
    """[(onset, key_release, pitch, velocity, pedal_extended_offset)]"""
    import pretty_midi
    inst = pretty_midi.PrettyMIDI(os.path.join(DATA, "maestro", pid + ".mid")).instruments[0]
    cc = sorted((c for c in inst.control_changes if c.number == 64), key=lambda c: c.time)
    down, iv = None, []
    for c in cc:
        if c.value >= 64 and down is None:
            down = c.time
        elif c.value < 64 and down is not None:
            iv.append((down, c.time))
            down = None
    out, nxt = [], {}
    for n in sorted(inst.notes, key=lambda n: -n.start):
        end = next((b for a, b in iv if a <= n.end < b), n.end)
        if n.pitch in nxt:
            end = min(end, max(n.end, nxt[n.pitch]))
        nxt[n.pitch] = n.start
        out.append((n.start, n.end, n.pitch, n.velocity, end))
    return sorted(out)


# ------------------------------------------------------------------ soundfont corpus
def _render(events, programs, pans, path, tail=1.5):
    import pretty_midi
    import tinysoundfont
    from scipy.io import wavfile
    syn = tinysoundfont.Synth(samplerate=SR, gain=-6.0)
    sf = syn.sfload(os.path.join(os.path.dirname(pretty_midi.__file__), "TimGM6mb.sf2"))
    for ch, prog in enumerate(programs):
        syn.program_select(ch, sf, 0, prog)
        syn.control_change(ch, 10, pans[ch])
    msgs = sorted([(s, 1, ch, p, v) for s, e, p, v, ch in events]
                  + [(e, 0, ch, p, 0) for s, e, p, v, ch in events])
    out, t = [], 0.0
    for tm, on, ch, p, v in msgs:
        n = int(round((tm - t) * SR))
        if n > 0:
            out.append(np.frombuffer(syn.generate(n), np.float32).copy())
            t += n / SR
        syn.noteon(ch, p, v) if on else syn.noteoff(ch, p)
    out.append(np.frombuffer(syn.generate(int(tail * SR)), np.float32).copy())
    x = np.concatenate(out).reshape(-1, 2)
    wavfile.write(path, SR, (x / (np.abs(x).max() + 1e-9) * 0.9 * 32767).astype(np.int16))


STYLES = {"piano": [0, 0, 0, 0], "strings": [40, 40, 41, 42], "winds": [73, 68, 71, 70]}


def render_corpus(which=("dev", "test"), n=24, beats=40.0):
    """Bach chorales (music21 corpus), 40 beats each, as piano / string
    quartet / wind quartet.  dev and test draw disjoint chorales."""
    import warnings
    warnings.filterwarnings("ignore")
    from music21 import corpus
    d = os.path.join(DATA, "corpus")
    os.makedirs(d, exist_ok=True)
    chor = sorted((p for p in corpus.getComposer("bach") if "bwv" in str(p)), key=str)
    used = set()
    for split, seed, prefix in (("dev", 7, "c"), ("test", 11, "x")):
        rng = np.random.default_rng(seed)
        order = [chor[i] for i in rng.choice(len(chor), 60 if split == "dev" else 200, replace=False)]
        index = []
        for path in order:
            if len(index) >= n:
                break
            name = str(path).split("/")[-1]
            if split == "dev":
                pass
            elif name in used:
                continue
            try:
                sc = corpus.parse(path)
                tss = sc.recurse().getElementsByClass("TimeSignature")
                kss = sc.recurse().getElementsByClass("KeySignature")
                if len({k.sharps for k in kss}) != 1 or len(sc.parts) != 4 \
                        or len({t.ratioString for t in tss}) != 1:
                    continue
                key = sc.analyze("key")
            except Exception:
                continue
            ts = tss[0]
            m0 = sc.parts[0].getElementsByClass("Measure")[0]
            pickup = float(m0.duration.quarterLength)
            pickup = pickup if pickup < ts.barDuration.quarterLength else 0.0
            bpm = float(rng.choice([66, 76, 84, 92, 100, 112]))
            ev = []
            for pi, part in enumerate(sc.parts):
                for nt in part.flatten().stripTies().notes:
                    b, dur = float(nt.offset), float(nt.quarterLength)
                    if dur > 0 and b < beats:
                        ev += [(b, min(dur, beats - b), p.midi, pi) for p in nt.pitches]
            ev.sort()
            if len(ev) < 40:
                continue
            if split == "dev":
                used.add(name)
            style = list(STYLES)[len(index) % 3]
            pid = f"{prefix}{len(index):02d}"
            spb, lead = 60.0 / bpm, 0.4
            wav = os.path.join(d, pid + ".wav")
            if not os.path.exists(wav):
                _render([(lead + b * spb, lead + (b + du * 0.95) * spb, p, 85 if style == "piano" else 80, ch)
                         for b, du, p, ch in ev], STYLES[style], [64, 50, 78, 64], wav)
            index.append(dict(id=pid, src=name, style=style, bpm=bpm, ts=ts.ratioString,
                              fifths=kss[0].sharps, key=f"{key.tonic.name} {key.mode}".replace("-", "b"),
                              pickup=pickup, lead=lead, notes=[(b, du, p) for b, du, p, _ in ev]))
            print("corpus", split, pid, name, style, bpm, ts.ratioString, file=sys.stderr)
        json.dump(index, open(os.path.join(d, split + ".json"), "w"))


if __name__ == "__main__":
    which = tuple(sys.argv[1:]) or ("dev", "test")
    fetch_maestro(which)
    render_corpus(which)
