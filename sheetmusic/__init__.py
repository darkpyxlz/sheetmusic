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
from .arrange import LEVELS as ARRANGEMENT_LEVELS, arrange_piano, playability  # noqa: E402
from .depth import auto_depth, limit_depth  # noqa: E402
from .hands import assign_hands  # noqa: E402
from .keys import estimate_key, spell, tpc_name  # noqa: E402
from .meter import detect_meter, signature_spec  # noqa: E402
from .midi import write_midi  # noqa: E402
from .mscore import PartSpec, ScoreSpec, StaffSpec, score_to_mscx, write_mscz  # noqa: E402
from .notes import Note, QNote  # noqa: E402

__all__ = ["TranscribeOptions", "Transcription", "transcribe", "__version__",
           "ARRANGEMENT_LEVELS", "arrange_piano", "playability", "auto_depth", "limit_depth",
           "assign_hands", "estimate_key", "spell", "tpc_name", "detect_meter", "signature_spec",
           "write_midi", "PartSpec", "ScoreSpec", "StaffSpec", "score_to_mscx", "write_mscz",
           "Note", "QNote"]
