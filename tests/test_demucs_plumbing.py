"""The Demucs path is optional; check its plumbing with a stand-in module that
'separates' by copying the input into both stems."""

import os
import shutil
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import make_song, write_mp3  # noqa: E402

from sheetmusic import TranscribeOptions, transcribe  # noqa: E402


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_demucs_path(tmp_path, monkeypatch):
    fake = tmp_path / "fake" / "demucs"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    (fake / "separate.py").write_text(textwrap.dedent('''
        import os, shutil, subprocess, sys
        a = sys.argv[1:]
        out = a[a.index("-o") + 1]; model = a[a.index("-n") + 1]; src = a[-1]
        d = os.path.join(out, model, os.path.splitext(os.path.basename(src))[0])
        os.makedirs(d, exist_ok=True)
        for stem in ("vocals", "no_vocals"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, os.path.join(d, stem + ".wav")], check=True)
    '''))
    monkeypatch.syspath_prepend(str(tmp_path / "fake"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    mix, truth = make_song()
    path = str(tmp_path / "s.mp3")
    write_mp3(path, mix)
    r = transcribe(path, TranscribeOptions(verbose=False, separate="demucs", duration=12))
    assert len(r.melody_notes) > 5 and len(r.accomp_notes) > 5
