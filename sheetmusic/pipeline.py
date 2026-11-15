"""End-to-end transcription: audio file -> ScoreSpec (-> .mscz / .mid)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import audio, melody, meter, poly, rhythm, spectral
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
    vocals: bool | None = None       # None = detect; True = vocal song; False = instrumental
    melody: bool = True              # lead line / vocal staff
    accompaniment: bool = True       # piano grand staff
    melody_name: str = "Voice"
    melody_program: int = 52         # GM "Choir Aahs"; 73 = flute, 0 = piano
    separate: str = "none"           # "none" | "demucs"
    # vocal handling
    remove_voice_from_accompaniment: bool = True
    voicing_threshold: float = 0.3
    voice_strictness: float = 0.72  # 0 = keep every tracked note
    fluctuation_weight: float = 0.5  # 0 = off; weight of the vibrato/fluctuation cue
    octave_correction: bool = True
    melody_range: tuple[float, float] = (45, 86)
    min_melody_note: float = 0.09
    # accompaniment
    max_polyphony: int = 5
    poly_method: str = "auto"        # "greedy" | "nmf" | "auto" (nmf for instrumentals)
    poly_threshold: float = 0.15
    poly_rel_threshold: float = 0.3
    accomp_range: tuple[int, int] = (28, 96)
    min_accomp_note: float = 0.09
    # rhythm / notation
    bpm: float | None = None
    time_signature: tuple[int, int] | None = None   # None = detect automatically
    melody_grid: int = 8             # finest subdivision per beat (8 = 32nds in x/4)
    accomp_grid: int = 8             # adaptive: coarser values are used when they fit
    tuning: float | None = None      # cents; None = auto
    key: str | None = None           # e.g. "G", "Eb", "F#m"; None = auto
    split_point: int = 60            # piano: >= split -> treble staff
    key_filter: bool = True          # drop weak, short out-of-key accompaniment notes
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
    if opts.vocals is not False and (opts.melody or opts.accompaniment):
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
            # noisiness of the *drum-free* vocal spectrum: consonants and
            # breaths are unvoiced, but a snare hit under a held note is not
            vflat = spectral.flatness(vspec, df)
        if opts.fluctuation_weight > 0:
            fspec = spectral.fluctuating_part(vspec, time_size=41)
        else:
            fspec = None
        track = melody.track_melody(vspec, df, fps, vflat, ref_hz=ref_hz, fluct_spec=fspec,
                                    mono_spec=(F.mono * hmask) if vocals is None else None,
                                    fluct_weight=opts.fluctuation_weight,
                                    midi_lo=opts.melody_range[0], midi_hi=opts.melody_range[1],
                                    voicing_threshold=opts.voicing_threshold)
        if track.f0.size < F.n_frames:
            pad = F.n_frames - track.f0.size
            track = melody.MelodyTrack(np.pad(track.f0, (0, pad)),
                                       np.pad(track.midi, (0, pad), constant_values=np.nan),
                                       np.pad(track.salience, (0, pad)),
                                       np.pad(track.voiced, (0, pad)))
        onset_mix = rhythm.onset_envelope(F.flux, fps)
        mel_notes = melody.segment_melody(track, fps, min_note=opts.min_melody_note,
                                          onset=onset_mix,
                                          voice_strictness=opts.voice_strictness)
        if opts.octave_correction:
            mel_notes = melody.fix_octave_errors(mel_notes)
        mel_notes = melody.snap_onsets(mel_notes, rhythm.onset_envelope(F.flux, fps), fps)
        _log(opts, f"  voiced {100 * track.voiced.mean():.0f}% of frames, {len(mel_notes)} notes")

    # ---------------- vocals or instrumental? ----------------
    has_vocals = opts.vocals
    vinfo = {}
    if track is not None and vocals is None:
        vinfo = melody.vocal_evidence(track, mel_notes, F.mono, df, fps, ref_hz)
        if has_vocals is None:
            has_vocals = vinfo["has_vocals"]
    elif vocals is not None:
        has_vocals = True if has_vocals is None else has_vocals
    if has_vocals is None:
        has_vocals = False
    _log(opts, f"  {'vocals detected' if has_vocals else 'no vocals: instrumental transcription'}"
               + (f" (timbre var {vinfo['timbre_var']}, wobble {vinfo['wobble']}, "
                  f"movement {vinfo['fluct']})" if vinfo else ""))
    instrumental = not has_vocals
    if instrumental:
        # everything goes into the piano score; nothing is removed as "voice"
        mel_notes = []
        track = None
        show_melody = False
        max_poly = max(opts.max_polyphony, 8)
        method = "nmf" if opts.poly_method == "auto" else opts.poly_method
        poly_thr = opts.poly_threshold if method == "greedy" else max(opts.poly_threshold, 0.2)
    else:
        show_melody = opts.melody
        max_poly = opts.max_polyphony
        method = "greedy" if opts.poly_method == "auto" else opts.poly_method
        poly_thr = opts.poly_threshold

    # ---------------- accompaniment ----------------
    acc_notes: list[Note] = []
    onset = rhythm.onset_envelope(F.flux, fps)
    if opts.accompaniment or instrumental:
        _log(opts, "Transcribing the " + ("music" if instrumental else "accompaniment")
             + " (multi-pitch) ...")
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
            midi_hi=opts.accomp_range[1], max_poly=max_poly)
        act_nmf = None
        if method == "nmf":
            _log(opts, "  harmonic NMF ...")
            act_nmf = poly.nmf_activations(aspec, df, ref_hz=ref_hz,
                                           midi_lo=opts.accomp_range[0],
                                           midi_hi=opts.accomp_range[1])
        weights = None
        if track is not None and opts.remove_voice_from_accompaniment:
            weights = poly.voice_exclusion(track.midi, F.n_frames)
        acc_notes = poly.activations_to_notes(
            pitches, sals, fps, energy=energy, abs_threshold=poly_thr,
            rel_threshold=opts.poly_rel_threshold, min_note=opts.min_accomp_note,
            weights=weights, spec=aspec, df=df, ref_hz=ref_hz,
            short_spec=F.short, short_df=opts.sr / F.short_nfft, onset_env=onset,
            act_in=act_nmf)
        acc_notes = poly.prune_overtone_ghosts(acc_notes)
        _log(opts, f"  {len(acc_notes)} notes")

    # ---------------- rhythm ----------------
    _log(opts, "Tempo and beat tracking ...")
    all_notes = mel_notes + acc_notes
    if opts.bpm:
        first = _first_onset(all_notes)
        bm = rhythm.BeatMap.constant(opts.bpm, first, dur)
    else:
        bpm0 = rhythm.estimate_tempo(onset, fps)
        beats = rhythm.track_beats(onset, fps, bpm0)
        bm = rhythm.BeatMap.from_beats(beats, bpm0)
        bm = rhythm.fix_tempo_octave(bm, mel_notes if len(mel_notes) >= 8 else acc_notes,
                                     onset, fps)
    only = None
    if opts.time_signature:
        only = {(4, 4): ["4/4"], (3, 4): ["3/4"], (2, 4): ["2/4"], (6, 8): ["6/8", "6/8t"],
                (12, 8): ["12/8t"], (9, 8): ["9/8t"]}.get(tuple(opts.time_signature))
        if only:
            only = only + [o + "x2" for o in only if o + "x2" in meter._HYP]
        if only is None:
            raise RuntimeError(f"unsupported time signature {opts.time_signature}; "
                               "use 2/4, 3/4, 4/4, 6/8, 9/8 or 12/8")
    met = meter.detect_meter(bm, onset, fps, all_notes, spec=harm, df=df, ref_hz=ref_hz,
                             only=only, debug=bool(os.environ.get("SHEETMUSIC_DEBUG")),
                             lead=mel_notes)
    num, den = met.num, met.den
    bm = met.beat_map
    phase = met.downbeat_phase
    bpb = met.beats_per_bar
    # shift so that beat index `phase` becomes the start of a measure; make
    # sure nothing falls before the first bar
    first_note_beat = min([float(bm.to_beats(n.start)) for n in all_notes] or [0.0])
    offset = -phase
    while first_note_beat + offset < -0.25:
        offset += bpb
    while first_note_beat + offset >= bpb:
        offset -= bpb
    _log(opts, f"  time signature {met.name}, tempo ~ {bm.bpm:.1f} "
               f"{'dotted-crotchet' if met.compound else 'beat'}s per minute, "
               f"{bm.beat_times.size} beats")

    tpb = met.beat_ticks
    # finest grid: 32nds (compound: 12 per dotted crotchet), never below a 64th
    def _grid(g):
        if met.compound:
            g = max(3, g // 2 * 3)     # 8 -> 12, 4 -> 6, 2 -> 3
        return max(1, min(g, tpb // 30))
    mgrid, agrid = _grid(opts.melody_grid), _grid(opts.accomp_grid)
    mel_q = rhythm.quantize(mel_notes, bm, offset, tpb, mgrid, monophonic=True,
                            fill_gaps=0.25)
    # accompaniment onsets are smeared (strums, arpeggios, long STFT windows):
    # snap them harder so chords line up instead of scattering into 32nds
    acc_q = rhythm.quantize(acc_notes, bm, offset, tpb, agrid, monophonic=False,
                            fill_gaps=0.25, snap_tol=0.18,
                            min_units=max(1, 120 // (tpb // agrid)))  # >= a 16th

    # ---------------- key ----------------
    if opts.key:
        fifths, mode = _parse_key(opts.key)
        key_name = opts.key
    else:
        cg = meter.chroma(harm, df, ref_hz).sum(axis=0)
        fifths, mode, key_name = estimate_key([mel_q, acc_q], [2.0, 1.0], chroma=cg)
    _log(opts, f"  key: {key_name}")
    if opts.key_filter:
        acc_q = _drop_weak_out_of_key(acc_q, fifths, mode, tpb)

    # ---------------- score ----------------
    parts = []
    if show_melody:
        med = np.median([q.pitch for q in mel_q]) if mel_q else 67
        clef = "G8vb" if med < 61 else "G"  # tenor/baritone: treble-8vb clef
        lo, hi = int(opts.melody_range[0]), int(opts.melody_range[1])
        parts.append(PartSpec(opts.melody_name, opts.melody_name[:4] + ".", opts.melody_program,
                              [StaffSpec(mel_q, clef)], instrument_id="voice.vocals"
                              if opts.melody_name.lower().startswith(("voice", "vocal")) else "keyboard.piano",
                              min_pitch=lo, max_pitch=hi))
    if opts.accompaniment or instrumental:
        up = [q for q in acc_q if q.pitch >= opts.split_point]
        dn = [q for q in acc_q if q.pitch < opts.split_point]
        parts.append(PartSpec("Piano", "Pno.", 0, [StaffSpec(up, "G"), StaffSpec(dn, "F")]))
    title = opts.title if opts.title is not None else os.path.splitext(os.path.basename(path))[0]
    score = ScoreSpec(parts=parts, title=title, subtitle="Transcribed from audio",
                      key_fifths=fifths, key_minor=(mode == "minor"),
                      time_num=num, time_den=den, bpm=round(bm.bpm),
                      beat_ticks_override=tpb)
    info = dict(duration=dur, tuning_cents=tuning, bpm=bm.bpm, key=key_name,
                time_signature=met.name, meter_scores=met.scores,
                has_vocals=has_vocals, vocal_evidence=vinfo,
                melody_notes=len(mel_q), accomp_notes=len(acc_q),
                seconds=time.time() - t0)
    _log(opts, f"Done in {info['seconds']:.1f} s: {len(mel_q)} melody notes, "
               f"{len(acc_q)} accompaniment notes")
    return Transcription(score, mel_q, acc_q, info, track, mel_notes, acc_notes)


def _drop_weak_out_of_key(notes: list[QNote], fifths: int, mode: str, tpb: int) -> list[QNote]:
    """Remove accompaniment notes that are outside the key's scale *and* both
    quieter than typical and shorter than a beat - in practice these are
    analysis artefacts (reverb, beating, drum tones), not real chromaticism."""
    if not notes:
        return notes
    tonic_major = (fifths * 7) % 12
    scale = {(tonic_major + i) % 12 for i in (0, 2, 4, 5, 7, 9, 11)}
    if mode == "minor":
        scale |= {(tonic_major + 6) % 12, (tonic_major + 8) % 12}  # raised 6th/7th of minor
    med_v = float(np.median([q.velocity for q in notes]))
    return [q for q in notes
            if q.pitch % 12 in scale or q.velocity >= med_v or q.end - q.start >= tpb]


def _first_onset(notes: list[Note]) -> float:
    return min((n.start for n in notes), default=0.0)


__all__ = ["TranscribeOptions", "Transcription", "transcribe", "tpc_name", "spell"]
