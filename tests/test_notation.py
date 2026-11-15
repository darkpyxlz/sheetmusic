import zipfile
import xml.etree.ElementTree as ET

from sheetmusic.keys import estimate_key, spell, tpc_name
from sheetmusic.midi import write_midi
from sheetmusic.mscore import (PartSpec, ScoreSpec, StaffSpec, score_to_mscx,
                               split_duration, write_mscz)
from sheetmusic.notes import QNote
from sheetmusic.pipeline import _parse_key


def test_split_duration_simple_values():
    assert split_duration(0, 480, 1920, 480) == [(480, "quarter", 0)]
    assert split_duration(0, 720, 1920, 480) == [(720, "quarter", 1)]       # dotted quarter
    assert split_duration(0, 1920, 1920, 480) == [(1920, "whole", 0)]
    assert split_duration(0, 1440, 1920, 480) == [(1440, "half", 1)]
    # syncopation across a beat is split at the beat
    assert split_duration(240, 480, 1920, 480) == [(240, "eighth", 0), (240, "eighth", 0)]


def test_split_duration_sums():
    for pos in range(0, 1920, 120):
        for ln in range(120, 1920 - pos + 1, 120):
            pieces = split_duration(pos, ln, 1920, 480)
            assert sum(p[0] for p in pieces) == ln


def _score(notes_upper, notes_lower=(), title="T"):
    return ScoreSpec(parts=[PartSpec("Voice", "Vo.", 52, [StaffSpec(list(notes_upper))]),
                            PartSpec("Piano", "Pno.", 0, [StaffSpec([]), StaffSpec(list(notes_lower), "F")])],
                     title=title, key_fifths=1, bpm=100)


def test_mscx_is_well_formed_and_measures_are_full():
    notes = [QNote(0, 720, 67), QNote(720, 960, 66), QNote(1440, 2400, 64), QNote(1440, 2400, 60)]
    xml = score_to_mscx(_score(notes, [QNote(0, 3840, 43)]))
    root = ET.fromstring(xml)
    staves = root.find("Score").findall("Staff")
    assert len(staves) == 3
    durs = {"whole": 1920, "half": 960, "quarter": 480, "eighth": 240, "16th": 120, "32nd": 60, "64th": 30}
    for st in staves:
        for m in st.findall("Measure"):
            total = 0
            for el in m.find("voice"):
                if el.tag in ("Chord", "Rest"):
                    dt = el.findtext("durationType")
                    if dt == "measure":
                        total += 1920
                        continue
                    d = durs[dt]
                    dots = int(el.findtext("dots") or 0)
                    total += d + (d // 2 if dots else 0)
            assert total == 1920


def test_ties_carry_chord_note_index():
    # 59 is the 2nd note of a chord, then the only note of the next chord
    notes = [QNote(960, 1200, 55), QNote(960, 1920, 59)]
    xml = score_to_mscx(_score(notes))
    assert "<notes>-1</notes>" in xml and "<notes>1</notes>" in xml


def test_tie_across_barline():
    xml = score_to_mscx(_score([QNote(1440, 2400, 60)]))
    assert "<measures>1</measures>" in xml and "<measures>-1</measures>" in xml


def test_mscz_container(tmp_path):
    p = tmp_path / "song.mscz"
    write_mscz(_score([QNote(0, 480, 60)]), str(p))
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        assert "META-INF/container.xml" in names and "song.mscx" in names
        ET.fromstring(z.read("song.mscx"))


def test_spelling_and_keys():
    assert tpc_name(spell(66, 1)) == "F#"     # G major
    assert tpc_name(spell(70, -1)) == "Bb"    # F major
    assert tpc_name(spell(63, -3)) == "Eb"
    assert _parse_key("Eb") == (-3, "major")
    assert _parse_key("F#m") == (3, "minor")
    g_major = [QNote(i * 480, i * 480 + 480, p) for i, p in enumerate([67, 71, 74, 66, 69, 72, 67, 62, 59, 67])]
    fifths, mode, name = estimate_key([g_major])
    assert (fifths, mode) == (1, "major"), name


def test_midi(tmp_path):
    p = tmp_path / "x.mid"
    write_midi(str(p), [("Voice", 52, [QNote(0, 480, 60)])], 120)
    data = p.read_bytes()
    assert data[:4] == b"MThd" and data.count(b"MTrk") == 2
