"""Audio decoding (via ffmpeg, with a WAV fallback) and optional source separation."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def load_audio(path: str, sr: int = 22050, start: float | None = None,
               duration: float | None = None) -> np.ndarray:
    """Decode ``path`` to a float32 array of shape (2, n) at ``sr`` Hz.

    Mono files are duplicated to two identical channels (the stereo
    "centre" analysis then degrades gracefully to plain mono analysis).
    """
    ffmpeg = _ffmpeg()
    if ffmpeg is not None:
        cmd = [ffmpeg, "-nostdin", "-v", "error"]
        if start:
            cmd += ["-ss", str(float(start))]
        if duration:
            cmd += ["-t", str(float(duration))]
        cmd += ["-i", path, "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "2", "-ar", str(sr), "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode {path!r}:\n"
                               f"{proc.stderr.decode(errors='replace')}")
        x = np.frombuffer(proc.stdout, dtype="<f4")
        x = x[: x.size - x.size % 2].reshape(-1, 2).T.copy()
        if x.shape[1] == 0:
            raise RuntimeError(f"no audio decoded from {path!r}")
        return x

    # Fallback: WAV only, via scipy.
    if not path.lower().endswith(".wav"):
        raise RuntimeError("ffmpeg was not found on PATH; it is required to decode "
                           "MP3 and other compressed formats (WAV works without it).")
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    file_sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128.0) / 128.0
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    data = data[:, :2] if data.shape[1] >= 2 else np.repeat(data, 2, axis=1)
    if start:
        data = data[int(start * file_sr):]
    if duration:
        data = data[: int(duration * file_sr)]
    if file_sr != sr:
        g = np.gcd(int(file_sr), int(sr))
        data = resample_poly(data, sr // g, file_sr // g, axis=0)
    return np.ascontiguousarray(data.T, dtype=np.float32)


def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def separate_vocals_demucs(path: str, sr: int, start: float | None,
                           duration: float | None, model: str = "htdemucs",
                           log=print) -> tuple[np.ndarray, np.ndarray]:
    """Split ``path`` into (vocals, accompaniment) with Demucs, if installed.

    Demucs is an optional neural source separator (``pip install demucs``).
    Returns two (2, n) float32 arrays at ``sr``.
    """
    if not demucs_available():
        raise RuntimeError("--separate demucs requested but the 'demucs' package is "
                           "not installed (pip install demucs).")
    with tempfile.TemporaryDirectory() as tmp:
        src = path
        if start or duration:
            # Trim first so Demucs does not process the whole file.
            src = os.path.join(tmp, "clip.wav")
            ffmpeg = _ffmpeg()
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required for --start/--duration with demucs")
            cmd = [ffmpeg, "-nostdin", "-v", "error", "-y"]
            if start:
                cmd += ["-ss", str(start)]
            if duration:
                cmd += ["-t", str(duration)]
            cmd += ["-i", path, src]
            subprocess.run(cmd, check=True)
        log(f"  running demucs ({model}) source separation, this can take a while...")
        cmd = [sys.executable, "-m", "demucs.separate", "--two-stems", "vocals",
               "-n", model, "-o", tmp, src]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("demucs failed:\n" + proc.stderr.decode(errors="replace"))
        stem = os.path.splitext(os.path.basename(src))[0]
        out_dir = os.path.join(tmp, model, stem)
        voc = load_audio(_find_stem(out_dir, "vocals"), sr)
        acc = load_audio(_find_stem(out_dir, "no_vocals"), sr)
    n = min(voc.shape[1], acc.shape[1])
    return voc[:, :n], acc[:, :n]


def _find_stem(out_dir: str, name: str) -> str:
    for ext in (".wav", ".mp3", ".flac"):
        p = os.path.join(out_dir, name + ext)
        if os.path.exists(p):
            return p
    raise RuntimeError(f"demucs output {name!r} not found in {out_dir}")
