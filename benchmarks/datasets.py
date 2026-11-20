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


# ------------------------------------------------------------------ ASAP (rhythm ground truth)
ASAP = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
# pieces per time signature (split evenly between dev and test)
ASAP_QUOTA = {"4/4": 16, "3/4": 16, "2/4": 8, "6/8": 10, "12/8": 6, "2/2": 4, "3/8": 4, "9/8": 3}


def asap_manifest():
    """MAESTRO performances with ASAP beat / downbeat / time-signature / key
    annotations: one performance per piece, a single time signature, starting
    at the beginning of the audio file.  dev and test hold different pieces."""
    import csv
    import hashlib
    d = os.path.join(DATA, "asap")
    os.makedirs(d, exist_ok=True)
    for name in ("metadata.csv", "asap_annotations.json"):
        if not os.path.exists(os.path.join(d, name)):
            urllib.request.urlretrieve(ASAP + name, os.path.join(d, name))
    rows = list(csv.DictReader(open(os.path.join(d, "metadata.csv"))))
    ann = json.load(open(os.path.join(d, "asap_annotations.json")))
    by = {}
    for r in rows:
        a = ann.get(r["midi_performance"])
        if not a or not r["maestro_audio_performance"] or r["start"] \
                or not a.get("score_and_performance_aligned", True):
            continue
        sigs = {v[0] for v in a["perf_time_signatures"].values()}
        if len(sigs) == 1 and r["folder"] not in by:
            by[r["folder"]] = (r, a, sigs.pop())
    out = {"dev": [], "test": []}
    for ts, quota in ASAP_QUOTA.items():
        pieces = sorted((v for v in by.values() if v[2] == ts),
                        key=lambda v: hashlib.md5(v[0]["folder"].encode()).hexdigest())[:quota]
        # alternate within each composer so both splits get the same mix of styles
        pieces.sort(key=lambda v: (v[0]["composer"], hashlib.md5(v[0]["folder"].encode()).hexdigest()))
        for i, (r, a, _ts) in enumerate(pieces):
            split = "dev" if i % 2 == 0 else "test"
            out[split].append((f"{'a' if split == 'dev' else 'b'}{len(out[split]):02d}", r, a))
    return out


def fetch_asap(which=("dev", "test"), seconds=60.0):
    """The first ``seconds`` of each piece, stored as 22.05 kHz stereo WAV."""
    import subprocess
    d = os.path.join(DATA, "asap")
    man = asap_manifest()
    zf = None
    for split in which:
        index = []
        for pid, r, a in man[split]:
            out = os.path.join(d, pid + ".wav")
            if not os.path.exists(out):
                zf = zf or zipfile.ZipFile(io.BufferedReader(_HttpFile(MAESTRO + ".zip"), 1 << 20))
                member = "maestro-v3.0.0/" + r["maestro_audio_performance"].replace("{maestro}/", "")
                with zf.open(member) as src:
                    head = src.read(44)
                    rate = int.from_bytes(head[24:28], "little")
                    block = int.from_bytes(head[32:34], "little")
                    pcm = src.read(int(seconds * rate) * block)
                cmd = ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(rate), "-ac",
                       str(int.from_bytes(head[22:24], "little")), "-i", "-", "-ar", "22050", out]
                subprocess.run(cmd, input=pcm, check=True)
            beats = [b for b in a["performance_beats"] if b < seconds]
            downs = [b for b in a["performance_downbeats"] if b < seconds]
            ts, n = next(iter(a["perf_time_signatures"].values()))
            keys = [(float(t), v[1]) for t, v in a["perf_key_signatures"].items() if float(t) < seconds]
            index.append(dict(id=pid, composer=r["composer"], title=r["title"], folder=r["folder"],
                              ts=ts, beats_per_bar=n, beats=beats, downbeats=downs,
                              fifths=keys[0][1] if keys else None, key_changes=len(keys) > 1))
            print("asap", split, pid, ts, r["folder"], file=sys.stderr)
        json.dump(index, open(os.path.join(d, split + ".json"), "w"))


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


# ------------------------------------------------------------------ mixed corpus (meters, textures)
# (corpus path, split, instrument style, tempo of the notated beat)
# beat = crotchet in x/4, dotted crotchet in 6/8 9/8 12/8, minim in 2/2, quaver in 3/8
MIXED = [
    ("joplin/maple_leaf_rag", "dev", "piano", 88), ("chopin/mazurka06-2", "test", "piano", 132),
    ("mozart/k545/movement1_exposition", "dev", "piano", 120),
    ("schumann_clara/polonaise_op1n1", "dev", "piano", 96), ("schumann_clara/polonaise_op1n2", "test", "piano", 96),
    ("schumann_clara/polonaise_op1n3", "dev", "piano", 100), ("schumann_clara/polonaise_op1n4", "test", "piano", 92),
    ("verdi/laDonnaEMobile", "dev", "piano", 176), ("cpebach/h186", "test", "piano", 76),
    ("mozart/k155/movement1", "dev", "strings", 126), ("mozart/k155/movement2", "test", "strings", 72),
    ("mozart/k155/movement3", "dev", "strings", 120), ("mozart/k156/movement1", "dev", "strings", 200),
    ("mozart/k156/movement2", "test", "strings", 60), ("mozart/k156/movement3", "dev", "strings", 120),
    ("mozart/k458/movement1", "dev", "strings", 92), ("mozart/k458/movement2", "test", "strings", 112),
    ("mozart/k458/movement3", "dev", "strings", 56), ("mozart/k458/movement4", "test", "strings", 132),
    ("mozart/k80/movement1", "dev", "strings", 66), ("mozart/k80/movement2", "test", "strings", 126),
    ("mozart/k80/movement3", "test", "strings", 120), ("mozart/k80/movement4", "dev", "strings", 80),
    ("haydn/opus1no1/movement1", "test", "strings", 100), ("haydn/opus1no1/movement2", "dev", "strings", 120),
    ("haydn/opus1no1/movement3", "test", "strings", 58), ("haydn/opus1no1/movement4", "test", "strings", 126),
    ("haydn/opus1no1/movement5", "dev", "strings", 132), ("haydn/opus74no1/movement1", "dev", "strings", 132),
    ("haydn/opus74no1/movement2", "test", "strings", 132), ("haydn/opus74no1/movement3", "dev", "strings", 150),
    ("haydn/opus74no1/movement4", "test", "strings", 126), ("beethoven/opus18no1/movement1", "test", "strings", 144),
    ("beethoven/opus18no1/movement2", "dev", "strings", 46), ("beethoven/opus18no1/movement4", "dev", "strings", 120),
    ("schumann_robert/opus41no1/movement2", "test", "strings", 108),
    ("schumann_robert/opus41no1/movement4", "dev", "strings", 60),
    ("schumann_clara/opus17/movement3", "dev", "trio", 56), ("johnson_j_r/lift_every_voice", "test", "piano", 60),
    ("corelli/opus3no1/1grave", "test", "strings", 60),
    # single-line folk tunes: jigs (6/8), slip jigs (9/8), reels and hornpipes
    ("ryansMammoth/BanishMisfortune.abc", "dev", "violin", 108), ("ryansMammoth/BagOfMealThe.abc", "test", "flute", 112),
    ("ryansMammoth/BattleTheCashJig.abc", "dev", "accordion", 116), ("ryansMammoth/BillyORourkesJig.abc", "test", "violin", 104),
    ("ryansMammoth/AndrewCareysSlipjig.abc", "dev", "flute", 112), ("ryansMammoth/BarneysGoatJig.abc", "test", "violin", 108),
    ("ryansMammoth/DrinkOfBrandySlipjig.abc", "test", "accordion", 104), ("ryansMammoth/OffSheGoesJig.abc", "test", "flute", 100),
    ("ryansMammoth/7thRegimentReel.abc", "dev", "violin", 104), ("ryansMammoth/AcaciaReel.abc", "test", "accordion", 100),
    ("ryansMammoth/AcrobatsHornpipe.abc", "dev", "flute", 84), ("ryansMammoth/AdmiralsHornpipe.abc", "test", "violin", 80),
]
MIXED_PROGRAMS = {"piano": [0] * 8, "strings": [40, 40, 41, 42, 42], "trio": [40, 42, 0, 0],
                  "violin": [40], "flute": [73], "accordion": [21]}


def render_mixed(which=("dev", "test"), seconds=36.0):
    """Real scores with varied meters and textures (piano pieces, quartet
    movements, folk tunes), rendered through the soundfont in strict tempo."""
    import warnings
    warnings.filterwarnings("ignore")
    from music21 import corpus
    d = os.path.join(DATA, "mixed")
    os.makedirs(d, exist_ok=True)
    for split in which:
        index = []
        for path, sp, style, beat_bpm in MIXED:
            if sp != split:
                continue
            sc = corpus.parse(path)
            if not sc.parts:                       # an ABC "opus": take its first tune
                sc = sc.scores[0]
            ts = sc.recurse().getElementsByClass("TimeSignature")[0]
            ks = sc.recurse().getElementsByClass("KeySignature")[0]
            num, den = ts.numerator, ts.denominator
            compound = den == 8 and num % 3 == 0 and num > 3
            beat_q = 1.5 if compound else 4.0 / den           # notated beat in crotchets
            qbpm = beat_bpm * beat_q
            spb = 60.0 / qbpm                                  # seconds per crotchet
            beats = seconds / spb
            m0 = sc.parts[0].getElementsByClass("Measure")[0]
            pickup = float(m0.duration.quarterLength)
            pickup = pickup if pickup < ts.barDuration.quarterLength - 1e-6 else 0.0
            # stop at the first time-signature change
            for t in sc.parts[0].recurse().getElementsByClass("TimeSignature")[1:]:
                off = float(t.getOffsetInHierarchy(sc.parts[0]))
                if t.ratioString != ts.ratioString and off < beats:
                    beats = off
            ev = []
            for pi, part in enumerate(sc.parts):
                for nt in part.flatten().stripTies().notes:
                    b, dur = float(nt.offset), float(nt.quarterLength)
                    if dur > 0 and b < beats - 1e-6:
                        ev += [(b, min(dur, beats - b), p.midi, pi) for p in nt.pitches]
            ev.sort()
            pid = f"{'p' if split == 'dev' else 'q'}{len(index):02d}"
            lead = 0.4
            progs = MIXED_PROGRAMS[style]
            wav = os.path.join(d, pid + ".wav")
            if not os.path.exists(wav):
                nparts = max(e[3] for e in ev) + 1
                pans = [64, 50, 78, 64, 64, 64, 64, 64][:max(nparts, 1)]
                if style == "piano":
                    pans = [64] * 8
                vel = lambda ch: 88 if ch == 0 else 74     # top part a little louder
                _render([(lead + b * spb, lead + (b + du * 0.95) * spb, p, vel(ch), min(ch, len(progs) - 1))
                         for b, du, p, ch in ev], progs[:max(nparts, 1)], pans + [64] * 8, wav)
            index.append(dict(id=pid, src=path, style=style, bpm=qbpm, beat_bpm=beat_bpm,
                              ts=ts.ratioString, fifths=ks.sharps, pickup=pickup, lead=lead,
                              notes=[(b, du, p) for b, du, p, _ in ev],
                              parts=[pi for *_x, pi in ev]))
            print("mixed", split, pid, path, style, ts.ratioString, f"q={qbpm:.0f}", len(ev), file=sys.stderr)
        json.dump(index, open(os.path.join(d, split + ".json"), "w"))


if __name__ == "__main__":
    which = tuple(sys.argv[1:]) or ("dev", "test")
    fetch_maestro(which)
    render_corpus(which)
    render_mixed(which)
    fetch_asap(which)
