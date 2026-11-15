"""End-to-end transcription: audio file -> ScoreSpec (-> .mscz / .mid)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import audio, melody, poly, rhythm, spectral
from .keys import estimate_key, tpc_name, spell
from .mscore import PartSpec, ScoreSpec, StaffSpec
from .notes import Note, QNote


@dataclass
class TranscribeOptions:
    sr: int = 22050
    n_fft: int = 4096
    hop: int = 256
    start: float | None = None
    duration: float | None = None
    # what to output
    melody: bool = True              # lead line / vocal staff
    accompaniment: bool = True       # piano grand staff
    melody_name: str = "Voice"
    melody_program: int = 52         # GM "Choir Aahs"; 73 = flute, 0 = piano
    separate: str = "none"           # "none" | "demucs"
    # vocal handling
    remove_voice_from_accompaniment: bool = True
    voicing_threshold: float = 0.3
    fluctuation_weight: float = 0.5  # 0 = off; weight of the vibrato/fluctuation cue
    octave_correction: bool = True
    melody_range: tuple[float, float] = (38, 86)
    min_melody_note: float = 0.09
    # accompaniment
    max_polyphony: int = 5
    poly_threshold: float = 0.15
    poly_rel_threshold: float = 0.3
    accomp_range: tuple[int, int] = (28, 96)
    min_accomp_note: float = 0.09
    # rhythm / notation
    bpm: float | None = None
    time_signature: tuple[int, int] = (4, 4)
    melody_grid: int = 4             # subdivisions per beat (4 = 16ths in x/4)
    accomp_grid: int = 2             # 2 = 8ths in x/4
    tuning: float | None = None      # cents; None = auto
    key: str | None = None           # e.g. "G", "Eb", "F#m"; None = auto
    split_point: int = 60            # piano: >= split -> treble staff
    title: str | None = None
    verbose: bool = True


@dataclass
class Transcription:
    score: ScoreSpec
    melody_notes: list[QNote] = field(default_factory=list)
    accomp_notes: list[QNote] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    melody_track: melody.MelodyTrack | None = None
    melody_raw: list[Note] = field(default_factory=list)
    accomp_raw: list[Note] = field(default_factory=list)


def _log(opts, *a):
    if opts.verbose:
        print(*a, flush=True)


def _parse_key(s: str):
    s = s.strip()
    minor = s.endswith("m") and not s.lower().endswith("maj")
    root = s[:-1] if minor else s
    # line of fifths position of the written tonic
    f = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}[root[0].upper()]
    f += 7 * (root.count("#") - root[1:].count("b"))
    fifths = f - 3 if minor else f
    return fifths, ("minor" if minor else "major")


def transcribe(path: str, opts: TranscribeOptions | None = None) -> Transcription:
    opts = opts or TranscribeOptions()
    t0 = time.time()
    _log(opts, f"Decoding {path} ...")
    mix = audio.load_audio(path, opts.sr, opts.start, opts.duration)
    dur = mix.shape[1] / opts.sr
    _log(opts, f"  {dur:.1f} s of audio")

    vocals = accomp = None
    if opts.separate == "demucs":
        vocals, accomp = audio.separate_vocals_demucs(path, opts.sr, opts.start, opts.duration,
                                                      log=lambda *a: _log(opts, *a))
    _log(opts, "Spectral analysis ...")
    F = spectral.analyze(mix, opts.sr, opts.n_fft, opts.hop)
    fps, df = F.fps, F.df
    hmask = spectral.hpss_mask(F.mono)
    harm = F.mono * hmask

    tuning = opts.tuning if opts.tuning is not None else spectral.estimate_tuning(harm, df)
    ref_hz = 440.0 * 2 ** (tuning / 1200.0)
    _log(opts, f"  tuning: {tuning:+.1f} cents (A4 = {ref_hz:.1f} Hz)")

    # ---------------- lead melody / voice ----------------
    track = None
    mel_notes: list[Note] = []
    if opts.melody or (opts.accompaniment and opts.remove_voice_from_accompaniment):
        _log(opts, "Tracking the lead melody / voice ...")
        if vocals is not None:
            FV = spectral.analyze(vocals, opts.sr, opts.n_fft, opts.hop)
            n = min(FV.n_frames, F.n_frames)
            vspec = FV.mono[:n]
            vflat = FV.flatness[:n]
        else:
            # centre-panned harmonic content, emphasising partials that fluctuate
            # (vibrato / expressive pitch) over rock-steady sustained ones
            # (piano, pads, organ): two-stage HPSS (Tachibana et al. 2010)
            vspec = F.center * hmask
            vflat = F.flatness
        if opts.fluctuation_weight > 0:
            fspec = spectral.fluctuating_part(vspec, time_size=41)
        else:
            fspec = None
        track = melody.track_melody(vspec, df, fps, vflat, ref_hz=ref_hz, fluct_spec=fspec,
                                    fluct_weight=opts.fluctuation_weight,
                                    midi_lo=opts.melody_range[0], midi_hi=opts.melody_range[1],
                                    voicing_threshold=opts.voicing_threshold)
        if track.f0.size < F.n_frames:
            pad = F.n_frames - track.f0.size
            track = melody.MelodyTrack(np.pad(track.f0, (0, pad)),
                                       np.pad(track.midi, (0, pad), constant_values=np.nan),
                                       np.pad(track.salience, (0, pad)),
                                       np.pad(track.voiced, (0, pad)))
        mel_notes = melody.segment_melody(track, fps, min_note=opts.min_melody_note)
        if opts.octave_correction:
            mel_notes = melody.fix_octave_errors(mel_notes)
        mel_notes = melody.snap_onsets(mel_notes, rhythm.onset_envelope(F.flux, fps), fps)
        _log(opts, f"  voiced {100 * track.voiced.mean():.0f}% of frames, {len(mel_notes)} notes")

    # ---------------- accompaniment ----------------
    acc_notes: list[Note] = []
    onset = rhythm.onset_envelope(F.flux, fps)
    if opts.accompaniment:
        _log(opts, "Transcribing the accompaniment (multi-pitch) ...")
        if accomp is not None:
            FA = spectral.analyze(accomp, opts.sr, opts.n_fft, opts.hop)
            n = min(FA.n_frames, F.n_frames)
            aspec = np.zeros_like(harm)
            aspec[:n] = FA.mono[:n] * spectral.hpss_mask(FA.mono[:n])
        else:
            aspec = harm
            if track is not None and opts.remove_voice_from_accompaniment:
                vm = melody.vocal_harmonic_mask(F.n_frames, harm.shape[1], df, track.f0)
                aspec = harm * vm
        pitches, sals, energy = poly.multipitch_activations(
            aspec, df, ref_hz=ref_hz, midi_lo=opts.accomp_range[0],
            midi_hi=opts.accomp_range[1], max_poly=opts.max_polyphony)
        weights = None
        if track is not None and opts.remove_voice_from_accompaniment:
            weights = poly.voice_exclusion(track.midi, F.n_frames)
        acc_notes = poly.activations_to_notes(
            pitches, sals, fps, energy=energy, abs_threshold=opts.poly_threshold,
            rel_threshold=opts.poly_rel_threshold, min_note=opts.min_accomp_note,
            weights=weights)
        acc_notes = poly.prune_overtone_ghosts(acc_notes)
        _log(opts, f"  {len(acc_notes)} notes")

    # ---------------- rhythm ----------------
    _log(opts, "Tempo and beat tracking ...")
    num, den = opts.time_signature
    if opts.bpm:
        first = _first_onset(mel_notes + acc_notes)
        bm = rhythm.BeatMap.constant(opts.bpm, first, dur)
    else:
        bpm0 = rhythm.estimate_tempo(onset, fps)
        beats = rhythm.track_beats(onset, fps, bpm0)
        bm = rhythm.BeatMap.from_beats(beats, bpm0)
        bm = rhythm.fix_tempo_octave(bm, mel_notes if len(mel_notes) >= 8 else acc_notes,
                                     onset, fps)
    phase = rhythm.downbeat_phase(bm, onset, fps, num, acc_notes or mel_notes)
    # shift so that beat index `phase` becomes the start of a measure; make
    # sure nothing falls before the first bar
    first_note_beat = min([float(bm.to_beats(n.start)) for n in (mel_notes + acc_notes)] or [0.0])
    offset = -phase
    while first_note_beat + offset < -0.25:
        offset += num
    while first_note_beat + offset >= num:
        offset -= num
    _log(opts, f"  tempo ~ {bm.bpm:.1f} BPM, {bm.beat_times.size} beats, downbeat phase {phase}")

    tpb = 4 * 480 // den
    mel_q = rhythm.quantize(mel_notes, bm, offset, tpb, opts.melody_grid, monophonic=True,
                            fill_gaps=1)
    acc_q = rhythm.quantize(acc_notes, bm, offset, tpb, opts.accomp_grid, monophonic=False,
                            fill_gaps=0)

    # ---------------- key ----------------
    if opts.key:
        fifths, mode = _parse_key(opts.key)
        key_name = opts.key
    else:
        fifths, mode, key_name = estimate_key([mel_q, acc_q], [2.0, 1.0])
    _log(opts, f"  key: {key_name}")

    # ---------------- score ----------------
    parts = []
    if opts.melody:
        med = np.median([q.pitch for q in mel_q]) if mel_q else 67
        clef = "G8vb" if med < 61 else "G"  # tenor/baritone: treble-8vb clef
        lo, hi = int(opts.melody_range[0]), int(opts.melody_range[1])
        parts.append(PartSpec(opts.melody_name, opts.melody_name[:4] + ".", opts.melody_program,
                              [StaffSpec(mel_q, clef)], instrument_id="voice.vocals"
                              if opts.melody_name.lower().startswith(("voice", "vocal")) else "keyboard.piano",
                              min_pitch=lo, max_pitch=hi))
    if opts.accompaniment:
        up = [q for q in acc_q if q.pitch >= opts.split_point]
        dn = [q for q in acc_q if q.pitch < opts.split_point]
        parts.append(PartSpec("Piano", "Pno.", 0, [StaffSpec(up, "G"), StaffSpec(dn, "F")]))
    title = opts.title if opts.title is not None else os.path.splitext(os.path.basename(path))[0]
    score = ScoreSpec(parts=parts, title=title, subtitle="Transcribed from audio",
                      key_fifths=fifths, key_minor=(mode == "minor"),
                      time_num=num, time_den=den, bpm=round(bm.bpm))
    info = dict(duration=dur, tuning_cents=tuning, bpm=bm.bpm, key=key_name,
                melody_notes=len(mel_q), accomp_notes=len(acc_q),
                seconds=time.time() - t0)
    _log(opts, f"Done in {info['seconds']:.1f} s: {len(mel_q)} melody notes, "
               f"{len(acc_q)} accompaniment notes")
    return Transcription(score, mel_q, acc_q, info, track, mel_notes, acc_notes)


def _first_onset(notes: list[Note]) -> float:
    return min((n.start for n in notes), default=0.0)


__all__ = ["TranscribeOptions", "Transcription", "transcribe", "tpc_name", "spell"]
