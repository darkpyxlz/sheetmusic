"""Command line interface:  python -m sheetmusic song.mp3 [-o song.mscz]"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from .midi import write_midi
from .mscore import write_mscz
from .pipeline import TranscribeOptions, transcribe


def _timesig(s: str):
    a, b = s.split("/")
    a, b = int(a), int(b)
    if b not in (2, 4, 8, 16) or a < 1:
        raise argparse.ArgumentTypeError("time signature must look like 4/4, 3/4, 6/8 ...")
    return a, b


def _range(s: str):
    a, b = s.split("-")
    return float(a), float(b)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sheetmusic",
        description="Transcribe an MP3 (or any ffmpeg-readable audio) into MuseScore "
                    "sheet music (.mscz): a lead/vocal melody staff plus a piano "
                    "accompaniment grand staff.")
    p.add_argument("input", help="audio file (mp3, wav, flac, ogg, m4a, ...)")
    p.add_argument("-o", "--output", help="output .mscz (or .mscx) path "
                                          "(default: <input>.mscz)")
    p.add_argument("--midi", metavar="PATH", help="also write a MIDI file to listen to")
    p.add_argument("--pdf", metavar="PATH",
                   help="also render a PDF (needs MuseScore installed: mscore/musescore)")
    g = p.add_argument_group("what to transcribe")
    g.add_argument("--mode", choices=["auto", "vocal", "instrumental", "melody",
                                      "accompaniment"],
                   default="auto",
                   help="auto (default): detect whether the song has vocals; vocal: "
                        "voice staff + piano accompaniment with the voice removed; "
                        "instrumental: no vocals, everything transcribed into a piano "
                        "score (slower, more accurate for instrumental music); melody: "
                        "only the vocal line; accompaniment: only the piano part")
    g.add_argument("--separate", choices=["none", "demucs"], default="none",
                   help="source separation before transcription; 'demucs' uses the "
                        "optional Demucs neural separator (pip install demucs) for much "
                        "cleaner vocals in dense mixes")
    g.add_argument("--keep-voice-in-accompaniment", action="store_true",
                   help="do not notch the voice's harmonics out of the accompaniment")
    g.add_argument("--start", type=float, help="start time in seconds")
    g.add_argument("--duration", type=float, help="only transcribe this many seconds")
    g = p.add_argument_group("vocal / melody")
    g.add_argument("--voice-range", type=_range, default=(45, 86), metavar="LO-HI",
                   help="MIDI range of the lead melody (default 45-86 = A2..D6; use "
                        "e.g. 38-86 for a bass singer)")
    g.add_argument("--voicing", type=float, default=0.3,
                   help="voicing threshold (0..1, lower = more notes, default 0.3)")
    g.add_argument("--voice-strictness", type=float, default=0.72,
                   help="how strictly instrument notes (intros, riffs, fills between "
                        "vocal lines) are kept out of the voice staff; 0 = keep "
                        "everything, ~0.9 = very strict (default 0.72)")
    g.add_argument("--min-note", type=float, default=0.09,
                   help="shortest melody note in seconds (default 0.09)")
    g = p.add_argument_group("accompaniment")
    g.add_argument("--polyphony", type=int, default=5, help="max simultaneous notes (5)")
    g.add_argument("--poly-threshold", type=float, default=0.15,
                   help="note detection threshold (0..1, lower = more notes, default 0.15)")
    g.add_argument("--split", type=int, default=60,
                   help="MIDI pitch splitting piano treble/bass staves (default 60 = C4)")
    g = p.add_argument_group("notation")
    g.add_argument("--bpm", type=float, help="fixed tempo instead of beat tracking")
    g.add_argument("--time-signature", type=_timesig, default=None, metavar="N/D",
                   help="force the time signature (2/4, 3/4, 4/4, 6/8, 9/8, 12/8); "
                        "default: detected automatically")
    g.add_argument("--grid", type=int, choices=[1, 2, 4, 8, 16], default=8,
                   help="finest melody note value as subdivisions per beat (8 = "
                        "demisemiquavers/32nds, 4 = 16ths, 2 = quavers). Quantisation "
                        "is adaptive: notes snap to longer values whenever they fit")
    g.add_argument("--accomp-grid", type=int, choices=[1, 2, 4, 8, 16], default=8,
                   help="finest accompaniment note value (default 8 = 32nds)")
    g.add_argument("--key", help="force key, e.g. G, Eb, F#m (default: estimated)")
    g.add_argument("--tuning", type=float, help="tuning offset in cents (default: estimated)")
    g.add_argument("--title", help="score title (default: file name)")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.output or os.path.splitext(args.input)[0] + ".mscz"
    opts = TranscribeOptions(
        start=args.start, duration=args.duration, separate=args.separate,
        remove_voice_from_accompaniment=not args.keep_voice_in_accompaniment,
        voicing_threshold=args.voicing, melody_range=args.voice_range,
        voice_strictness=args.voice_strictness,
        min_melody_note=args.min_note, max_polyphony=args.polyphony,
        poly_threshold=args.poly_threshold, split_point=args.split, bpm=args.bpm,
        time_signature=args.time_signature, melody_grid=args.grid,
        accomp_grid=args.accomp_grid, key=args.key, tuning=args.tuning,
        title=args.title, verbose=not args.quiet)
    if args.mode == "instrumental":
        opts.vocals = False
    elif args.mode == "vocal":
        opts.vocals = True
    elif args.mode == "melody":
        opts.accompaniment = False
    elif args.mode == "accompaniment":
        opts.melody = False
    try:
        res = transcribe(args.input, opts)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    write_mscz(res.score, out)
    if not args.quiet:
        print(f"Wrote {out}")
    if args.midi:
        tracks = []
        if opts.melody:
            tracks.append((opts.melody_name, opts.melody_program, res.melody_notes))
        if opts.accompaniment:
            tracks.append(("Piano", 0, res.accomp_notes))
        write_midi(args.midi, tracks, res.score.bpm * res.score.beat_ticks / 480,
                   time_sig=(res.score.time_num, res.score.time_den))
        if not args.quiet:
            print(f"Wrote {args.midi}")
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
