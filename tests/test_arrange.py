from sheetmusic.arrange import LEVELS, arrange_piano, playability
from sheetmusic.notes import QNote


def _held(notes, hand):
    """(max notes held at once, max span held at once) for one hand."""
    ev = sorted([(q.start, 1, q.pitch) for q in notes if q.hand == hand]
                + [(q.end, 0, q.pitch) for q in notes if q.hand == hand])
    held, size, span = {}, 0, 0
    for _t, on, p in ev:
        if on:
            held[p] = held.get(p, 0) + 1
            size = max(size, len(held))
            span = max(span, max(held) - min(held))
        else:
            held[p] -= 1
            if not held[p]:
                del held[p]
    return size, span


def _busy_texture():
    """Two bars of 4/4: a tune in crotchets over thick semiquaver chords."""
    notes = []
    tune = [72, 74, 76, 77, 79, 77, 76, 74]
    for i, p in enumerate(tune):
        notes.append(QNote(i * 480, i * 480 + 480, p, 100, True, 1))
        for k in range(4):                                 # repeated semiquaver chords
            t = i * 480 + k * 120
            for q in (64, 67, 69):
                notes.append(QNote(t, t + 120, q, 60, True, 1))
            for q in (36, 43, 48, 52):
                notes.append(QNote(t, t + 120, q, 70 if q == 36 else 55, True, 0))
    return notes, tune


def test_full_changes_nothing():
    notes, _ = _busy_texture()
    assert arrange_piano(notes, "full") is notes


def test_playable_limits_per_hand_and_keeps_tune_and_bass():
    notes, tune = _busy_texture()
    for name in ("playable", "easy"):
        lv = LEVELS[name]
        out = arrange_piano(notes, name, crotchets_per_second=2.0)
        for hand in (0, 1):
            size, span = _held(out, hand)
            assert size <= lv.max_notes[hand], (name, hand, size)
            assert span <= lv.span[hand], (name, hand, span)
            starts = sorted({q.start for q in out if q.hand == hand})
            gaps = [b - a for a, b in zip(starts, starts[1:])]
            min_gap = (lv.right_gap if hand == 1 else lv.left_gap) * 480
            assert all(g >= min_gap for g in gaps), (name, hand, gaps)
        # every note value at least the level's shortest
        assert all(q.end - q.start >= lv.shortest for q in out), name
        # the tune is all there, on its beats
        top = {(q.start, q.pitch) for q in out if q.hand == 1}
        assert all((i * 480, p) in top for i, p in enumerate(tune)), name
        # the bass note C2 is kept on every beat it was struck
        assert all(any(q.pitch == 36 and q.start == i * 480 for q in out) for i in range(8)), name
        # only notes that were in the transcription
        assert {q.pitch for q in out} <= {q.pitch for q in notes}


def test_easy_is_sparser_than_playable():
    notes, _ = _busy_texture()
    pl = arrange_piano(notes, "playable", crotchets_per_second=2.0)
    ez = arrange_piano(notes, "easy", crotchets_per_second=2.0)
    assert len(ez) < len(pl) < len(notes)
    assert playability(ez, 2.0)["left"]["onsets_per_s"] < playability(pl, 2.0)["left"]["onsets_per_s"]


def test_rate_limit_follows_tempo():
    # quavers at a fast tempo are thinned by the notes-per-second limit
    notes = [QNote(i * 240, i * 240 + 240, 72 + (i % 5), 90, True, 1) for i in range(32)]
    slow = arrange_piano(notes, "playable", crotchets_per_second=1.5)    # 3 quavers / s
    fast = arrange_piano(notes, "playable", crotchets_per_second=5.0)    # 10 quavers / s
    assert len(slow) == 32
    assert len(fast) < 32
    assert playability(fast, 5.0)["right"]["busiest_second"] <= LEVELS["playable"].max_rate[1]


def test_left_hand_stays_below_right_hand():
    notes = [QNote(0, 960, 60, 90, True, 1), QNote(0, 960, 64, 80, True, 0),
             QNote(0, 960, 48, 80, True, 0)]
    out = arrange_piano(notes, "playable")
    assert not any(q.hand == 0 and q.pitch > 60 for q in out)


def test_song_as_playable_piano_solo(tmp_path):
    """--mode accompaniment --melody-in-piano --arrangement playable: the sung
    tune is the right hand's top line, and both hands stay within the limits."""
    import os
    import shutil
    import sys

    import pytest
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluate import evaluate
    from synth import make_song, write_mp3

    from sheetmusic import TranscribeOptions, transcribe
    mix, truth = make_song()
    path = str(tmp_path / "song.mp3")
    write_mp3(path, mix)
    r = transcribe(path, TranscribeOptions(verbose=False, melody=False, melody_in_piano=True,
                                           arrangement="playable"))
    assert len(r.score.parts) == 1                      # piano only
    rh = [q for q in r.accomp_notes if q.hand == 1]
    top = {}
    for q in rh:
        if q.start not in top or q.pitch > top[q.start].pitch:
            top[q.start] = q
    ev = evaluate(list(top.values()), truth["melody"])
    assert ev["f"] >= 0.85, ev
    lv = LEVELS["playable"]
    for hand in (0, 1):
        size, span = _held(r.accomp_notes, hand)
        assert size <= lv.max_notes[hand] and span <= lv.span[hand], (hand, size, span)
