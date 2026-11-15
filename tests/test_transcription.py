"""End-to-end checks on a synthetic song with a hard-to-transcribe 'voice'
(weak fundamental, moving formants, vibrato, scoops, consonants) over piano
chords, bass and drums."""

import os
import shutil
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import evaluate  # noqa: E402
from synth import make_song, write_mp3  # noqa: E402

from sheetmusic import TranscribeOptions, transcribe  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    mix, truth = make_song()
    path = str(tmp_path_factory.mktemp("audio") / "song.mp3")
    write_mp3(path, mix)
    return path, truth


@needs_ffmpeg
def test_vocal_melody_and_accompaniment(song):
    path, truth = song
    r = transcribe(path, TranscribeOptions(verbose=False))
    mel = evaluate(r.melody_notes, truth["melody"])
    acc = evaluate(r.accomp_notes, truth["accomp"])
    assert mel["f"] >= 0.9, mel
    assert acc["f"] >= 0.6, acc
    assert abs(r.info["bpm"] - truth["bpm"]) < 3
    assert r.info["key"] == "G major"


@needs_ffmpeg
def test_voice_removal_reduces_accompaniment_garbage(song):
    path, truth = song
    on = transcribe(path, TranscribeOptions(verbose=False, melody=False))
    off = transcribe(path, TranscribeOptions(verbose=False, melody=False,
                                             remove_voice_from_accompaniment=False))
    p_on = evaluate(on.accomp_notes, truth["accomp"])["precision"]
    p_off = evaluate(off.accomp_notes, truth["accomp"])["precision"]
    assert p_on > p_off + 0.1, (p_on, p_off)


@needs_ffmpeg
@pytest.mark.parametrize("kw", [dict(bpm=72), dict(detune_cents=35), dict(transpose_voice=-12),
                                dict(stereo=False)])
def test_robustness(tmp_path, kw):
    mix, truth = make_song(**kw)
    path = str(tmp_path / "s.mp3")
    write_mp3(path, mix)
    r = transcribe(path, TranscribeOptions(verbose=False, accompaniment="stereo" in kw))
    assert evaluate(r.melody_notes, truth["melody"])["f"] >= 0.85


@needs_ffmpeg
def test_cli_writes_mscz_and_midi(song, tmp_path):
    path, _ = song
    out = tmp_path / "out.mscz"
    mid = tmp_path / "out.mid"
    subprocess.run([sys.executable, "-m", "sheetmusic", path, "-o", str(out), "--midi", str(mid),
                    "-q"], check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert zipfile.is_zipfile(out) and mid.stat().st_size > 100


@pytest.mark.skipif(shutil.which("mscore3") is None and shutil.which("mscore") is None,
                    reason="MuseScore not installed")
@needs_ffmpeg
def test_musescore_opens_file_without_repairs(song, tmp_path):
    path, _ = song
    out = tmp_path / "out.mscz"
    r = transcribe(path, TranscribeOptions(verbose=False))
    from sheetmusic.mscore import write_mscz
    write_mscz(r.score, str(out))
    exe = shutil.which("mscore3") or shutil.which("mscore")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    p = subprocess.run([exe, "-o", str(tmp_path / "out.musicxml"), str(out)],
                       capture_output=True, text=True, env=env, timeout=180)
    log = p.stdout + p.stderr
    assert (tmp_path / "out.musicxml").exists(), log
    assert "broken connectors" not in log, log
