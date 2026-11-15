"""Audio (MP3/WAV/...) to MuseScore (.mscz) transcription.

The pipeline is pure DSP (numpy + scipy, ffmpeg for decoding):

* predominant-melody (vocal) tracking via harmonic summation over a
  formant-whitened spectrum, so the singer's harmonics/resonances are folded
  back into the sung fundamental instead of showing up as stray notes;
* vibrato/scoop-tolerant note segmentation of the vocal pitch contour;
* removal of the voice's harmonic comb before polyphonic transcription of the
  accompaniment (iterative harmonic estimation & cancellation);
* beat tracking, downbeat/key estimation, quantisation, and a native
  MuseScore writer.
"""

__version__ = "0.1.0"

from .pipeline import TranscribeOptions, Transcription, transcribe  # noqa: E402

__all__ = ["TranscribeOptions", "Transcription", "transcribe", "__version__"]
