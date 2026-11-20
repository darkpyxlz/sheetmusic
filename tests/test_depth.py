from sheetmusic.depth import auto_depth, limit_depth, parse_depth
from sheetmusic.notes import Note


def test_parse_depth():
    assert parse_depth("auto") == 0 and parse_depth("full") is None
    assert parse_depth("sparse") == 2 and parse_depth("3") == 3 and parse_depth(5) == 5


def test_depth_caps_simultaneous_notes_and_keeps_outer_voices():
    chord = [Note(0.0, 1.0, p, v) for p, v in ((48, 70), (60, 60), (64, 50), (67, 55), (72, 65))]
    out = limit_depth(chord, 3)
    assert len(out) == 3
    assert {48, 72} <= {n.pitch for n in out}          # bass and top line survive
    assert limit_depth(chord, None) == chord           # "full" changes nothing


def test_new_note_cuts_a_ringing_one_short():
    notes = [Note(0.0, 2.0, 60, 50), Note(0.0, 2.0, 64, 80), Note(1.0, 2.0, 67, 90)]
    out = limit_depth(notes, 2)
    assert len(out) == 3
    assert next(n for n in out if n.pitch == 60).end == 1.0


def test_low_interval_limit():
    # a third deep in the bass: the weaker note goes; an octave is fine
    notes = [Note(0, 1, 36, 80), Note(0, 1, 40, 60), Note(0, 1, 72, 70)]
    assert len(limit_depth(notes, 6)) == 3                      # off by default
    out = limit_depth(notes, 6, bass_interval=7)
    assert sorted(n.pitch for n in out) == [36, 72]
    out = limit_depth([Note(0, 1, 36, 80), Note(0, 1, 48, 75), Note(0, 1, 72, 70)], 6, bass_interval=7)
    assert len(out) == 3


def test_weak_bass_notes_are_dropped_relative_to_the_low_register():
    strong = [Note(i, i + 0.9, 40 + (i % 3) * 7, 80) for i in range(8)]
    ghosts = [Note(i + 0.2, i + 0.5, 33, 40) for i in range(4)]
    high = [Note(i, i + 0.9, 76, 40) for i in range(8)]     # soft high notes are untouched
    out = limit_depth(strong + ghosts + high, 4)
    assert not [n for n in out if n.pitch == 33]
    assert len([n for n in out if n.pitch == 76]) == 8 and len([n for n in out if 40 <= n.pitch < 57]) == 8


def test_auto_depth_follows_the_texture():
    two = [Note(i, i + 1, 48, 80) for i in range(10)] + [Note(i, i + 1, 72, 80) for i in range(10)]
    four = two + [Note(i, i + 1, 60, 75) for i in range(10)] + [Note(i, i + 1, 64, 75) for i in range(10)]
    assert auto_depth(two) < auto_depth(four) <= 6
