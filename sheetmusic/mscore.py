"""Native MuseScore writer (.mscx / .mscz, format 3.01 — opens in MuseScore 3.x and 4.x)."""

from __future__ import annotations

import zipfile

import numpy as np
from dataclasses import dataclass, field
from fractions import Fraction
from xml.sax.saxutils import escape

from .keys import spell
from .notes import QNote

DIVISION = 480          # ticks per quarter
WHOLE = 4 * DIVISION

# (ticks, durationType, dots) -- longest first
_DURS = [
    (1920, "whole", 0), (1440, "half", 1), (960, "half", 0), (720, "quarter", 1),
    (480, "quarter", 0), (360, "eighth", 1), (240, "eighth", 0), (180, "16th", 1),
    (120, "16th", 0), (90, "32nd", 1), (60, "32nd", 0), (30, "64th", 0),
]
_DUR_BY_TICKS = {d: (n, k) for d, n, k in _DURS}


@dataclass
class StaffSpec:
    notes: list[QNote]
    clef: str = "G"           # G, F, G8vb, ...


@dataclass
class PartSpec:
    name: str
    short: str
    program: int              # GM program number (0-based)
    staves: list[StaffSpec]
    instrument_id: str = "keyboard.piano"
    min_pitch: int = 21
    max_pitch: int = 108


@dataclass
class ScoreSpec:
    parts: list[PartSpec]
    title: str = ""
    subtitle: str = ""
    composer: str = ""
    key_fifths: int = 0
    key_minor: bool = False
    time_num: int = 4
    time_den: int = 4
    bpm: float = 120.0
    beat_ticks_override: int | None = None   # 720 for compound time (dotted crotchet beat)
    extra_meta: dict = field(default_factory=dict)

    @property
    def measure_ticks(self) -> int:
        return self.time_num * WHOLE // self.time_den

    @property
    def beat_ticks(self) -> int:
        return self.beat_ticks_override or WHOLE // self.time_den


def split_duration(pos: int, length: int, measure: int, beat: int):
    """Split [pos, pos+length) (positions within a measure) into as few
    notatable values as possible (fewest ties).

    A value may start on any multiple of half its undotted base value (so a
    syncopated crotchet on the "and", a dotted crotchet after a quaver, a
    semiquaver on a demisemiquaver ... are written as single notes); values of
    two beats or longer must start on a beat so the bar stays readable.
    """
    pieces = []
    end = pos + length
    while pos < end:
        rem = end - pos
        for d, name, dots in _DURS:
            if d > rem or d > measure:
                continue
            base = d * 2 // 3 if dots else d
            step = max(base // 2, 30)
            if d >= 2 * beat:
                step = max(step, beat)
            if pos % step == 0:
                pieces.append((d, name, dots))
                pos += d
                break
        else:
            pieces.append((30, "64th", 0))
            pos += 30
    return pieces


def assign_voices(notes: list[QNote], max_voices: int = 4) -> list[list[QNote]]:
    """Distribute a staff's notes over MuseScore voices (1-4).

    Notes that start and end together form a chord.  Each chord goes to the
    first voice that is free at its start, higher chords first, so a long bass
    note and moving chords above it are written with their real durations
    instead of being chopped into tied pieces.  If all voices are busy the
    chord is added to the voice that frees up first (it is then sliced and
    tied there, losing nothing)."""
    groups: dict[tuple[int, int], list[QNote]] = {}
    for q in notes:
        groups.setdefault((q.start, q.end), []).append(q)
    chords = sorted(groups.items(), key=lambda kv: (kv[0][0], -max(n.pitch for n in kv[1])))
    voices: list[list[QNote]] = [[] for _ in range(max_voices)]
    ends = [0] * max_voices
    for (st, en), ns in chords:
        free = [v for v in range(max_voices) if ends[v] <= st]
        v = free[0] if free else int(np.argmin(ends))
        voices[v].extend(ns)
        ends[v] = max(ends[v], en)
    while len(voices) > 1 and not voices[-1]:
        voices.pop()
    return voices


def _frac(ticks: int) -> str:
    f = Fraction(ticks, WHOLE)
    return f"{f.numerator}/{f.denominator}"


def _location(delta_ticks: int, measure_ticks: int, from_tick: int, note_delta: int = 0) -> str:
    """Relative MuseScore location: measure difference, in-measure fraction
    difference and difference of the note's index inside its chord."""
    to_tick = from_tick + delta_ticks
    m0, p0 = divmod(from_tick, measure_ticks)
    m1, p1 = divmod(to_tick, measure_ticks)
    parts = []
    if m1 != m0:
        parts.append(f"<measures>{m1 - m0}</measures>")
    if p1 != p0:
        parts.append(f"<fractions>{_frac(p1 - p0)}</fractions>")
    if note_delta:
        parts.append(f"<notes>{note_delta}</notes>")
    return "<location>" + "".join(parts) + "</location>"


def _staff_events(notes: list[QNote], n_measures: int, mticks: int, bticks: int):
    """Yield per-measure lists of events: ('rest'|'chord', tick, dur, name, dots, notes)."""
    notes = sorted(notes, key=lambda q: (q.start, q.pitch))
    measures = []
    for m in range(n_measures):
        ms, me = m * mticks, (m + 1) * mticks
        inside = [q for q in notes if q.start < me and q.end > ms]
        bounds = {ms, me}
        for q in inside:
            if ms < q.start < me:
                bounds.add(q.start)
            if ms < q.end < me:
                bounds.add(q.end)
        bounds = sorted(bounds)
        events = []
        if not inside:
            events.append(("mrest", ms, mticks, None, 0, []))
            measures.append(events)
            continue
        for a, b in zip(bounds[:-1], bounds[1:]):
            sounding = [q for q in inside if q.start <= a and q.end >= b]
            pieces = split_duration(a - ms, b - a, mticks, bticks)
            t = a
            for i, (d, name, dots) in enumerate(pieces):
                if not sounding:
                    events.append(("rest", t, d, name, dots, []))
                else:
                    nl = []
                    for q in sorted(sounding, key=lambda q: q.pitch):
                        tie_back = q.start < t
                        tie_fwd = q.end > t + d
                        nl.append((q.pitch, tie_back, tie_fwd, q.velocity))
                    events.append(("chord", t, d, name, dots, nl))
                t += d
        measures.append(events)
    return measures


def _note_xml(pitch: int, key: int | tuple, tie_back: bool, tie_fwd: bool,
              prev_tick: int, this_tick: int, next_tick: int, mticks: int,
              prev_dn: int = 0, next_dn: int = 0) -> str:
    x = ["<Note>"]
    # same order as MuseScore itself writes: tie start, then tie end
    if tie_fwd:
        x.append('<Spanner type="Tie"><Tie/><next>'
                 + _location(next_tick - this_tick, mticks, this_tick, next_dn)
                 + "</next></Spanner>")
    if tie_back:
        x.append('<Spanner type="Tie"><prev>'
                 + _location(prev_tick - this_tick, mticks, this_tick, prev_dn)
                 + "</prev></Spanner>")
    k, minor = key if isinstance(key, tuple) else (key, False)
    x.append(f"<pitch>{pitch}</pitch><tpc>{spell(pitch, k, minor)}</tpc>")
    x.append("</Note>")
    return "".join(x)


def _voice_measures_xml(measures, score: ScoreSpec, hidden_rests: bool) -> list[str]:
    """XML content of one voice, one string per measure ('' = nothing in it)."""
    mt = score.measure_ticks
    flat = [ev for events in measures for ev in events]
    pitch_index = [{p: i for i, (p, *_r) in enumerate(ev[5])} for ev in flat]
    vis = "<visible>0</visible>" if hidden_rests else ""
    out = []
    prev_tick = 0
    k = -1
    for events in measures:
        parts = []
        has_notes = any(ev[0] == "chord" for ev in events)
        for kind, tick, dur, name, dots, nl in events:
            k += 1
            if kind == "mrest":
                if not hidden_rests:
                    parts.append(f"<Rest><durationType>measure</durationType>"
                                 f"<duration>{score.time_num}/{score.time_den}</duration></Rest>")
                prev_tick = tick
                continue
            dots_xml = f"<dots>{dots}</dots>" if dots else ""
            if kind == "rest":
                parts.append(f"<Rest>{vis}{dots_xml}<durationType>{name}</durationType></Rest>")
            else:
                here = pitch_index[k]
                before = pitch_index[k - 1] if k > 0 else {}
                after = pitch_index[k + 1] if k + 1 < len(flat) else {}
                notes_xml = "".join(
                    _note_xml(p, (score.key_fifths, score.key_minor), tb, tf, prev_tick, tick,
                              tick + dur, mt,
                              prev_dn=before.get(p, here[p]) - here[p],
                              next_dn=after.get(p, here[p]) - here[p])
                    for p, tb, tf, _v in nl)
                parts.append(f"<Chord>{dots_xml}<durationType>{name}</durationType>{notes_xml}</Chord>")
            prev_tick = tick
        out.append("".join(parts) if (has_notes or not hidden_rests) else "")
    return out


def _staff_xml(staff_id: int, voices_measures, score: ScoreSpec, first_staff: bool) -> str:
    out = [f'<Staff id="{staff_id}">']
    if first_staff and (score.title or score.subtitle or score.composer):
        out.append("<VBox><height>10</height>")
        if score.title:
            out.append(f"<Text><style>Title</style><text>{escape(score.title)}</text></Text>")
        if score.subtitle:
            out.append(f"<Text><style>Subtitle</style><text>{escape(score.subtitle)}</text></Text>")
        if score.composer:
            out.append(f"<Text><style>Composer</style><text>{escape(score.composer)}</text></Text>")
        out.append("</VBox>")
    contents = [_voice_measures_xml(vm, score, hidden_rests=(v > 0))
                for v, vm in enumerate(voices_measures)]
    n = len(voices_measures[0])
    for mi in range(n):
        out.append("<Measure><voice>")
        if mi == 0:
            out.append(f"<KeySig><accidental>{score.key_fifths}</accidental>"
                       + ("<mode>minor</mode>" if score.key_minor else "<mode>major</mode>")
                       + "</KeySig>")
            out.append(f"<TimeSig><sigN>{score.time_num}</sigN><sigD>{score.time_den}</sigD></TimeSig>")
            if first_staff:
                beat_q = score.beat_ticks / DIVISION        # beat length in crotchets
                qps = score.bpm * beat_q / 60.0             # crotchets per second
                dot = "<sym>space</sym><sym>metAugmentationDot</sym>"
                sym = {240: "<sym>metNote8thUp</sym>", 360: "<sym>metNote8thUp</sym>" + dot,
                       480: "<sym>metNoteQuarterUp</sym>",
                       720: "<sym>metNoteQuarterUp</sym>" + dot,
                       960: "<sym>metNoteHalfUp</sym>", 1440: "<sym>metNoteHalfUp</sym>" + dot,
                       1920: "<sym>metNoteWhole</sym>"}.get(score.beat_ticks,
                                                            "<sym>metNoteQuarterUp</sym>")
                out.append(f"<Tempo><tempo>{qps:.6f}</tempo><followText>1</followText>"
                           f"<text>{sym} = {int(round(score.bpm))}</text></Tempo>")
        out.append(contents[0][mi])
        if mi == n - 1:
            out.append("<BarLine><subtype>end</subtype><span>1</span></BarLine>")
        out.append("</voice>")
        # further voices; empty ones are only written as placeholders before a used one
        last_used = max([v for v in range(1, len(contents)) if contents[v][mi]] or [0])
        for v in range(1, last_used + 1):
            out.append("<voice>" + contents[v][mi] + "</voice>")
        out.append("</Measure>")
    out.append("</Staff>")
    return "".join(out)


def _part_xml(part: PartSpec, first_staff_id: int) -> str:
    out = ["<Part>"]
    ns = len(part.staves)
    for i, st in enumerate(part.staves):
        sid = first_staff_id + i
        out.append(f'<Staff id="{sid}"><StaffType group="pitched"><name>stdNormal</name></StaffType>')
        if i == 0 and ns > 1:
            out.append(f'<bracket type="1" span="{ns}" col="0"/><barLineSpan>1</barLineSpan>')
        if st.clef != "G":
            out.append(f"<defaultClef>{st.clef}</defaultClef>")
        out.append("</Staff>")
    out.append(f"<trackName>{escape(part.name)}</trackName>")
    out.append("<Instrument>")
    out.append(f"<longName>{escape(part.name)}</longName><shortName>{escape(part.short)}</shortName>")
    out.append(f"<trackName>{escape(part.name)}</trackName>")
    out.append(f"<minPitchP>{part.min_pitch}</minPitchP><maxPitchP>{part.max_pitch}</maxPitchP>")
    out.append(f"<minPitchA>{part.min_pitch}</minPitchA><maxPitchA>{part.max_pitch}</maxPitchA>")
    out.append(f"<instrumentId>{part.instrument_id}</instrumentId>")
    for i, st in enumerate(part.staves):
        if st.clef != "G":
            attr = f' staff="{i + 1}"' if i > 0 else ""
            out.append(f"<clef{attr}>{st.clef}</clef>")
    out.append(f'<Channel><program value="{part.program}"/></Channel>')
    out.append("</Instrument></Part>")
    return "".join(out)


def score_to_mscx(score: ScoreSpec) -> str:
    end = 0
    for p in score.parts:
        for st in p.staves:
            for q in st.notes:
                end = max(end, q.end)
    mt = score.measure_ticks
    n_measures = max(1, -(-end // mt))
    head = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<museScore version="3.01">',
            "<programVersion>3.2.3</programVersion>",
            "<programRevision>3224f34</programRevision>",
            "<Score>",
            '<LayerTag id="0" tag="default"></LayerTag>',
            "<currentLayer>0</currentLayer>",
            f"<Division>{DIVISION}</Division>",
            "<Style><Spatium>1.74978</Spatium></Style>",
            "<showInvisible>1</showInvisible><showUnprintable>1</showUnprintable>"
            "<showFrames>1</showFrames><showMargins>0</showMargins>"]
    meta = {"workTitle": score.title, "composer": score.composer, "arranger": "",
            "copyright": "", "creationDate": "", "lyricist": "", "movementNumber": "",
            "movementTitle": "", "poet": "", "source": "", "translator": "",
            "platform": "", "workNumber": ""}
    meta.update(score.extra_meta)
    for k, v in meta.items():
        head.append(f'<metaTag name="{k}">{escape(str(v))}</metaTag>')
    body = []
    sid = 1
    for p in score.parts:
        body.append(_part_xml(p, sid))
        sid += len(p.staves)
    sid = 1
    for p in score.parts:
        for st in p.staves:
            voices = assign_voices(st.notes) if st.notes else [[]]
            vm = [_staff_events(v, n_measures, mt, score.beat_ticks) for v in voices]
            body.append(_staff_xml(sid, vm, score, first_staff=(sid == 1)))
            sid += 1
    tail = ["</Score>", "</museScore>"]
    return "\n".join(head + body + tail) + "\n"


CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container>
  <rootfiles>
    <rootfile full-path="{name}"/>
  </rootfiles>
</container>
"""


def write_mscz(score: ScoreSpec, path: str) -> None:
    import os
    xml = score_to_mscx(score)
    if path.lower().endswith(".mscx"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)
        return
    base = os.path.splitext(os.path.basename(path))[0] or "score"
    inner = f"{base}.mscx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("META-INF/container.xml", CONTAINER.format(name=inner))
        z.writestr(inner, xml)
