"""Command line interface:  python -m sheetmusic song.mp3 [-o song.mscz]

Every setting of ``TranscribeOptions`` has a flag here (``FIELD_FLAGS`` lists
which); ``python -m sheetmusic -h`` describes them all.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from . import __version__
from .arrange import ARRANGEMENTS
from .meter import signature_spec
from .midi import write_midi
from .mscore import write_mscz
from .pipeline import TranscribeOptions, _parse_key, transcribe

# note value -> subdivisions of a crotchet
SHORTEST = {"64th": 16, "32nd": 8, "16th": 4, "8th": 2, "quarter": 1}

# every TranscribeOptions field and the flag(s) that set it
FIELD_FLAGS = {
    "sr": "--sample-rate", "n_fft": "--n-fft", "hop": "--hop",
    "start": "--start", "duration": "--duration",
    "vocals": "--mode", "melody": "--mode", "accompaniment": "--mode",
    "melody_name": "--melody-name", "melody_program": "--melody-program",
    "separate": "--separate",
    "remove_voice_from_accompaniment": "--keep-voice-in-accompaniment",
    "voicing_threshold": "--voicing", "voice_strictness": "--voice-strictness",
    "fluctuation_weight": "--fluctuation-weight", "octave_correction": "--no-octave-correction",
    "melody_range": "--voice-range", "min_melody_note": "--min-note",
    "arrangement": "--arrangement", "melody_in_piano": "--melody-in-piano", "depth": "--depth",
    "max_polyphony": "--polyphony", "poly_method": "--poly-method",
    "poly_threshold": "--poly-threshold", "poly_rel_threshold": "--poly-rel-threshold",
    "accomp_range": "--accomp-range", "min_accomp_note": "--min-accomp-note",
    "bpm": "--bpm", "time_signature": "--time-signature", "odd_meters": "--odd-meters",
    "triplets": "--no-triplets", "swing": "--swing",
    "melody_grid": "--grid", "accomp_grid": "--accomp-grid",
    "tuning": "--tuning", "key": "--key", "split_point": "--split",
    "key_filter": "--no-key-filter", "title": "--title", "verbose": "--quiet",
}

_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _pitch(s: str) -> float:
    """MIDI number from '45' or a note name such as 'A2', 'F#3', 'Bb4'."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    if not s or s[0].upper() not in _NOTE:
        raise argparse.ArgumentTypeError(f"not a MIDI number or note name: {s!r}")
    i, alt = 1, 0
    while i < len(s) and s[i] in "#b":
        alt += 1 if s[i] == "#" else -1
        i += 1
    try:
        octave = int(s[i:])
    except ValueError:
        raise argparse.ArgumentTypeError(f"note name needs an octave, e.g. C4: {s!r}")
    return float(12 * (octave + 1) + _NOTE[s[0].upper()] + alt)


def _range(s: str):
    parts = s.replace("..", "-").split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must look like 45-86 or A2-D6")
    lo, hi = _pitch(parts[0]), _pitch(parts[1])
    if hi <= lo:
        raise argparse.ArgumentTypeError("range must go from low to high")
    return lo, hi


def _timesig(s: str):
    try:
        a, b = (int(x) for x in s.split("/"))
        signature_spec(a, b)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "time signature must look like 4/4, 3/4, 2/2, 6/8, 5/4, 7/8 ... "
            "(denominator 1, 2, 4, 8, 16 or 32)")
    return a, b


def _key(s: str):
    try:
        _parse_key(s)
    except (KeyError, IndexError):
        raise argparse.ArgumentTypeError("key must look like C, G, Eb, F#m, A minor, Bb major")
    return s


def _depth(s: str):
    from .depth import parse_depth
    try:
        parse_depth(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "depth must be auto, full, melody, sparse, normal, rich or a number")
    return s


class _Formatter(argparse.RawDescriptionHelpFormatter):
    """Keeps the examples' line breaks and shows each option's default when
    it is a real value (not None / off)."""

    def _get_help_string(self, action):
        h = action.help or ""
        if action.option_strings and action.default not in (None, False, argparse.SUPPRESS) \
                and "%(default)" not in h and "default:" not in h.lower():
            h += " (default: %(default)s)"
        return h


def build_parser() -> argparse.ArgumentParser:
    d = TranscribeOptions()
    p = argparse.ArgumentParser(
        prog="sheetmusic",
        formatter_class=_Formatter,
        description="Transcribe an MP3 (or any audio ffmpeg can read) into MuseScore sheet "
                    "music (.mscz): a voice staff and a piano grand staff for songs, a piano "
                    "score for instrumental music. Vocals, tempo, time signature, key and "
                    "tuning are detected automatically; every detection can be overridden.",
        epilog="examples:\n"
               "  python -m sheetmusic song.mp3                      # song.mscz, all automatic\n"
               "  python -m sheetmusic song.mp3 --arrangement easy   # simplest playable piano part\n"
               "  python -m sheetmusic song.mp3 --arrangement full   # every note that was heard\n"
               "  python -m sheetmusic song.mp3 --time-signature 7/8 --bpm 140 --key F#m\n"
               "  python -m sheetmusic song.mp3 --midi out.mid --pdf out.pdf --info-json out.json")
    p.add_argument("input", help="audio file (mp3, wav, flac, ogg, m4a, ...)")
    p.add_argument("--version", action="version", version=f"sheetmusic {__version__}")

    g = p.add_argument_group("output")
    g.add_argument("-o", "--output", help="output .mscz or .mscx path; <input>.mscz if not given")
    g.add_argument("--midi", metavar="PATH", help="also write a MIDI file to listen to")
    g.add_argument("--pdf", metavar="PATH",
                   help="also render a PDF (needs MuseScore installed: mscore/musescore)")
    g.add_argument("--info-json", metavar="PATH",
                   help="also write what was detected (tempo, time signature, key, tuning, "
                        "vocals, playability per hand ...) as JSON")
    g.add_argument("--title", default=d.title, help="score title; the file name if not given")
    g.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")

    g = p.add_argument_group("what to transcribe")
    g.add_argument("--mode", choices=["auto", "vocal", "instrumental", "melody", "accompaniment"],
                   default="auto",
                   help="auto: detect whether the song has vocals; vocal: voice staff + piano "
                        "accompaniment with the voice removed; instrumental: no vocals, "
                        "everything into a piano score; melody: only the vocal line; "
                        "accompaniment: only the piano part")
    g.add_argument("--separate", choices=["none", "demucs"], default=d.separate,
                   help="source separation first; 'demucs' uses the optional Demucs neural "
                        "separator (pip install demucs) for cleaner vocals in dense mixes")
    g.add_argument("--start", type=float, default=d.start, help="start time in seconds")
    g.add_argument("--duration", type=float, default=d.duration,
                   help="only transcribe this many seconds")

    g = p.add_argument_group("piano part: how much to write")
    g.add_argument("--arrangement", choices=list(ARRANGEMENTS), default=d.arrangement,
                   help="playable: at most 3 notes per hand within an octave, no faster than "
                        "semiquavers, the tune and bass kept first; easy: at most 2 notes per "
                        "hand within a sixth, quavers at the shortest, the left hand changing at "
                        "most once a beat; full: every note that was heard")
    g.add_argument("--melody-in-piano", action="store_true",
                   help="songs: also put the sung tune into the piano's right hand (a piano "
                        "solo version; combine with --mode accompaniment to leave out the voice "
                        "staff). A low tune is played an octave up")
    g.add_argument("--depth", default=d.depth, type=_depth,
                   metavar="auto|full|melody|sparse|normal|rich|N",
                   help="how many notes may sound at once before arranging. auto: chosen from "
                        "the music, stricter in the bass; melody/sparse/normal/rich = 1/2/4/6; "
                        "full: keep everything the analysis finds")
    g.add_argument("--split", type=int, default=d.split_point, metavar="MIDI",
                   help="where the hands meet when nothing else decides it (60 = middle C); "
                        "hands are otherwise assigned by reach and movement")
    g.add_argument("--no-key-filter", action="store_true",
                   help="keep weak, short notes outside the key (normally dropped as artefacts)")

    g = p.add_argument_group("rhythm and notation")
    g.add_argument("--bpm", type=float, default=d.bpm, help="fixed tempo instead of beat tracking")
    g.add_argument("--time-signature", type=_timesig, default=d.time_signature, metavar="N/D",
                   help="force any time signature (4/4, 3/4, 2/2, 6/8, 12/8, 5/4, 7/8 ...); "
                        "beat level and downbeat are still found from the audio. Default: "
                        "detected (2/4, 3/4, 4/4, 3/8, 6/8, 9/8, 12/8)")
    g.add_argument("--odd-meters", action="store_true",
                   help="let detection also choose bars of 5 or 7 beats (5/4, 7/8 ...)")
    g.add_argument("--swing", choices=["auto", "on", "off"], default=d.swing,
                   help="swung quavers (played long-short) are written as straight quavers "
                        "under a 'Swing' marking; auto detects it, on forces it, off writes "
                        "what is played")
    g.add_argument("--no-triplets", action="store_true",
                   help="never write triplets: everything on the binary grid")
    g.add_argument("--shortest", choices=list(SHORTEST), default=None,
                   help="shortest note value written, for both staves (sets --grid and "
                        "--accomp-grid); values snap to the longest value that fits")
    g.add_argument("--grid", type=int, choices=[1, 2, 4, 8, 16], default=d.melody_grid,
                   help="melody: finest note value as subdivisions of a crotchet "
                        "(8 = demisemiquavers, 4 = semiquavers, 2 = quavers)")
    g.add_argument("--accomp-grid", type=int, choices=[1, 2, 4, 8, 16], default=d.accomp_grid,
                   help="piano part: finest note value as subdivisions of a crotchet")
    g.add_argument("--key", type=_key, default=d.key,
                   help="force the key, e.g. G, Eb, F#m, A minor (default: estimated)")
    g.add_argument("--tuning", type=float, default=d.tuning, metavar="CENTS",
                   help="tuning offset from A440 in cents (default: estimated)")

    g = p.add_argument_group("voice / melody")
    g.add_argument("--voice-range", type=_range, default=d.melody_range, metavar="LO-HI",
                   help="pitch range searched for the melody, MIDI numbers or note names "
                        "(45-86 = A2-D6; use e.g. D2-D6 for a bass singer)")
    g.add_argument("--voicing", type=float, default=d.voicing_threshold,
                   help="voicing threshold 0..1: lower = more (quieter) melody notes")
    g.add_argument("--voice-strictness", type=float, default=d.voice_strictness,
                   help="how strictly instrument notes (intros, riffs, fills) are kept out of "
                        "the voice staff: 0 = keep everything, ~0.9 = very strict")
    g.add_argument("--fluctuation-weight", type=float, default=d.fluctuation_weight,
                   help="weight 0..1 of the pitch-movement (vibrato) cue that tells the voice "
                        "from steady instruments; 0 = off")
    g.add_argument("--no-octave-correction", action="store_true",
                   help="do not fold isolated octave jumps in the melody back")
    g.add_argument("--min-note", type=float, default=d.min_melody_note, metavar="SECONDS",
                   help="shortest melody note in seconds")
    g.add_argument("--keep-voice-in-accompaniment", action="store_true",
                   help="do not remove the voice's harmonics from the piano part")
    g.add_argument("--melody-name", default=d.melody_name,
                   help="name of the melody staff ('Voice'/'Vocals' make it a voice part)")
    g.add_argument("--melody-program", type=int, default=d.melody_program, metavar="GM",
                   help="General MIDI program of the melody in the MIDI file "
                        "(52 = choir aahs, 73 = flute, 0 = piano)")

    g = p.add_argument_group("piano part: analysis")
    g.add_argument("--poly-method", choices=["auto", "greedy", "nmf"], default=d.poly_method,
                   help="multi-pitch method: nmf (joint harmonic decomposition, more accurate, "
                        "slower), greedy (iterative cancellation), auto = nmf for instrumental "
                        "music, greedy under vocals")
    g.add_argument("--poly-threshold", type=float, default=d.poly_threshold,
                   help="note detection threshold 0..1: lower = more notes")
    g.add_argument("--poly-rel-threshold", type=float, default=d.poly_rel_threshold,
                   help="ignore notes weaker than this share of the strongest at the moment")
    g.add_argument("--polyphony", type=int, default=d.max_polyphony,
                   help="harmonic series searched per frame by the greedy method")
    g.add_argument("--accomp-range", type=_range, default=d.accomp_range, metavar="LO-HI",
                   help="pitch range of the piano part, MIDI numbers or note names (28-96 = E1-C7)")
    g.add_argument("--min-accomp-note", type=float, default=d.min_accomp_note, metavar="SECONDS",
                   help="shortest piano-part note in seconds")

    g = p.add_argument_group("signal analysis (advanced)")
    g.add_argument("--sample-rate", type=int, default=d.sr, help="analysis sample rate (Hz)")
    g.add_argument("--n-fft", type=int, default=d.n_fft, help="STFT window length (samples)")
    g.add_argument("--hop", type=int, default=d.hop, help="STFT hop (samples)")
    return p


def options_from_args(args) -> TranscribeOptions:
    opts = TranscribeOptions(
        sr=args.sample_rate, n_fft=args.n_fft, hop=args.hop,
        start=args.start, duration=args.duration, separate=args.separate,
        melody_name=args.melody_name, melody_program=args.melody_program,
        remove_voice_from_accompaniment=not args.keep_voice_in_accompaniment,
        voicing_threshold=args.voicing, voice_strictness=args.voice_strictness,
        fluctuation_weight=args.fluctuation_weight, octave_correction=not args.no_octave_correction,
        melody_range=args.voice_range, min_melody_note=args.min_note,
        arrangement=args.arrangement, melody_in_piano=args.melody_in_piano, depth=args.depth,
        max_polyphony=args.polyphony, poly_method=args.poly_method,
        poly_threshold=args.poly_threshold, poly_rel_threshold=args.poly_rel_threshold,
        accomp_range=(int(args.accomp_range[0]), int(args.accomp_range[1])),
        min_accomp_note=args.min_accomp_note,
        bpm=args.bpm, time_signature=args.time_signature, odd_meters=args.odd_meters,
        triplets=not args.no_triplets, swing=args.swing,
        melody_grid=args.grid, accomp_grid=args.accomp_grid,
        tuning=args.tuning, key=args.key, split_point=args.split,
        key_filter=not args.no_key_filter, title=args.title, verbose=not args.quiet)
    if args.shortest:
        opts.melody_grid = opts.accomp_grid = SHORTEST[args.shortest]
    if args.mode == "instrumental":
        opts.vocals = False
    elif args.mode == "vocal":
        opts.vocals = True
    elif args.mode == "melody":
        opts.accompaniment = False
    elif args.mode == "accompaniment":
        opts.melody = False
    return opts


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.output or os.path.splitext(args.input)[0] + ".mscz"
    opts = options_from_args(args)
    try:
        res = transcribe(args.input, opts)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    write_mscz(res.score, out)
    if not args.quiet:
        print(f"Wrote {out}")
    if args.midi:
        tracks = []
        if opts.melody and res.melody_notes:
            tracks.append((opts.melody_name, opts.melody_program, res.melody_notes))
        if opts.accompaniment:
            tracks.append(("Piano", 0, res.accomp_notes))
        write_midi(args.midi, tracks, res.score.bpm * res.score.beat_ticks / 480,
                   time_sig=(res.score.time_num, res.score.time_den))
        if not args.quiet:
            print(f"Wrote {args.midi}")
    if args.info_json:
        with open(args.info_json, "w") as f:
            json.dump(res.info, f, indent=1, default=_jsonable)
        if not args.quiet:
            print(f"Wrote {args.info_json}")
    if args.pdf:
        exe = next((shutil.which(n) for n in ("mscore", "musescore", "mscore4", "mscore3",
                                              "musescore4", "musescore3", "MuseScore4")
                    if shutil.which(n)), None)
        if exe is None:
            print("warning: MuseScore not found, cannot render PDF", file=sys.stderr)
        else:
            env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
            subprocess.run([exe, "-o", args.pdf, out], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(args.pdf) and not args.quiet:
                print(f"Wrote {args.pdf}")
    return 0


def _jsonable(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)
