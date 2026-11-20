"""Instrumental pieces (no vocals) and a sung waltz: automatic detection of
vocals, time signature, tempo and key, plus note accuracy."""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import evaluate  # noqa: E402
from synth import write_mp3  # noqa: E402
from synth_instr import make_piece  # noqa: E402

from sheetmusic import TranscribeOptions, transcribe  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# piece -> minimum note F-measure (all notes, both staves)
CASES = {
    "waltz_F_3-4": 0.9,
    "piano_Dm_4-4": 0.85,
    "band_Eb_4-4": 0.65,
    "flute_Am_6-8": 0.6,
}


def _run(tmp_path, name, **kw):
    mix, truth = make_piece(name, **kw)
    path = str(tmp_path / f"{name}.mp3")
    write_mp3(path, mix)
    return transcribe(path, TranscribeOptions(verbose=False, arrangement="full")), truth


@pytest.mark.parametrize("name", list(CASES))
def test_instrumental_detection_and_accuracy(tmp_path, name):
    r, truth = _run(tmp_path, name)
    sc = r.score
    assert r.info["has_vocals"] is False, r.info["vocal_evidence"]
    assert (sc.time_num, sc.time_den) == tuple(truth["meter"]), r.info["meter_scores"]
    assert (sc.key_fifths, "minor" if sc.key_minor else "major") == tuple(truth["key"]), r.info["key"]
    assert abs(r.info["bpm"] / truth["bpm"] - 1) < 0.04, r.info["bpm"]
    tpb = sc.beat_ticks
    ev = evaluate(r.melody_notes + r.accomp_notes, truth["notes"], ticks_per_beat=tpb)
    assert ev["f"] >= CASES[name], ev


@pytest.mark.parametrize("name,shift", [("waltz_F_3-4", -4), ("flute_Am_6-8", 5),
                                        ("piano_Dm_4-4", 3)])
def test_detection_in_other_keys(tmp_path, name, shift):
    r, truth = _run(tmp_path, name, transpose=shift, seed=shift + 10)
    sc = r.score
    assert (sc.key_fifths, "minor" if sc.key_minor else "major") == tuple(truth["key"]), r.info["key"]
    assert (sc.time_num, sc.time_den) == tuple(truth["meter"])


def test_sung_waltz(tmp_path):
    r, truth = _run(tmp_path, "sung_waltz_G_3-4")
    assert r.info["has_vocals"] is True, r.info["vocal_evidence"]
    assert (r.score.time_num, r.score.time_den) == (3, 4)
    assert r.info["key"] == "G major"
    mel = evaluate(r.melody_notes, truth["melody"])
    assert mel["f"] >= 0.85, mel
