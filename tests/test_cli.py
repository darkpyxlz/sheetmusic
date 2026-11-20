import dataclasses

import pytest

from sheetmusic.cli import FIELD_FLAGS, SHORTEST, build_parser, options_from_args
from sheetmusic.pipeline import TranscribeOptions


def test_every_option_has_a_documented_flag():
    fields = {f.name for f in dataclasses.fields(TranscribeOptions)}
    assert fields == set(FIELD_FLAGS), fields ^ set(FIELD_FLAGS)
    help_text = build_parser().format_help()
    for flag in set(FIELD_FLAGS.values()):
        assert flag in help_text, flag


def test_defaults_match_the_library():
    opts = options_from_args(build_parser().parse_args(["x.mp3"]))
    ref = TranscribeOptions()
    for f in dataclasses.fields(TranscribeOptions):
        a, b = getattr(opts, f.name), getattr(ref, f.name)
        assert (tuple(a) if isinstance(a, (list, tuple)) else a) == \
            (tuple(b) if isinstance(b, (list, tuple)) else b), f.name


def test_flags_reach_the_options():
    p = build_parser()
    o = options_from_args(p.parse_args([
        "x.mp3", "--arrangement", "easy", "--time-signature", "7/8", "--odd-meters",
        "--voice-range", "D2-D6", "--accomp-range", "E1-C7", "--shortest", "16th",
        "--key", "A minor", "--mode", "instrumental", "--no-key-filter", "--sample-rate", "16000",
        "--no-octave-correction", "--keep-voice-in-accompaniment", "-q"]))
    assert o.arrangement == "easy" and o.time_signature == (7, 8) and o.odd_meters
    assert o.melody_range == (38.0, 86.0) and o.accomp_range == (28, 96)
    assert o.melody_grid == o.accomp_grid == SHORTEST["16th"] == 4
    assert o.key == "A minor" and o.vocals is False and not o.key_filter and o.sr == 16000
    assert not o.octave_correction and not o.remove_voice_from_accompaniment and not o.verbose


@pytest.mark.parametrize("bad", [["--time-signature", "4/5"], ["--key", "H"],
                                 ["--voice-range", "D6-D2"], ["--depth", "lots"]])
def test_bad_values_are_rejected(bad):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x.mp3"] + bad)
