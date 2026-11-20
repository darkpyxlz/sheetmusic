"""End-to-end transcription: audio file -> ScoreSpec (-> .mscz / .mid)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import arrange, audio, depth as depth_mod, melody, meter, poly, rhythm, spectral
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
    arrangement: str = "playable"    # "playable", "easy", or "full" (everything heard); see arrange.py
    melody_in_piano: bool = False    # songs: also play the sung tune in the piano's right hand
    depth: int | str | None = "auto"  # notes sounding at once: "auto", "full" (no limit),
    #                                   "melody"/"sparse"/"normal"/"rich" (1/2/4/6) or a number
    max_polyphony: int = 5
    poly_method: str = "auto"        # "greedy" | "nmf" | "auto" (nmf for instrumentals)
    poly_threshold: float = 0.15
    poly_rel_threshold: float = 0.3
    accomp_range: tuple[int, int] = (28, 96)
    min_accomp_note: float = 0.09
    # rhythm / notation
    bpm: float | None = None
    time_signature: tuple[int, int] | None = None   # None = detect automatically; any N/D
    odd_meters: bool = False         # also consider bars of 5 and 7 beats (5/4, 7/8 ...)
    triplets: bool = True            # write triplets where the beat divides in three
    swing: str = "auto"              # "auto" (detect), "on", "off": long-short quavers written straight
    melody_grid: int = 8             # finest note value, per crotchet (8 = 32nds)
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
    beat_map: rhythm.BeatMap | None = None   # notated beats <-> seconds
    beat_offset: float = 0.0                 # beats added so bar 1 starts at tick 0

    def ticks_to_seconds(self, ticks):
        """Audio time of a score position (inverse of the quantiser's mapping)."""
        bm = self.beat_map
        b = np.asarray(ticks, float) / self.score.beat_ticks - self.beat_offset
        bt = bm.beat_times
        i = np.arange(bt.size, dtype=float)
        s = np.interp(b, i, bt)
        s = np.where(b < 0, bt[0] + b * bm.period, s)
        return np.where(b > i[-1], bt[-1] + (b - i[-1]) * bm.period, s)


def _log(opts, *a):
    if opts.verbose:
        print(*a, flush=True)


_BEAT_NAMES = {240: "quaver", 360: "dotted-quaver", 480: "crotchet", 720: "dotted-crotchet",
               960: "minim", 1440: "dotted-minim", 1920: "semibreve"}


def _parse_key(s: str):
    """(fifths of the key signature, mode) from 'G', 'Eb', 'F#m', 'Bbm',
    'A minor', 'eb major' ...  Raises KeyError / IndexError when unreadable."""
    t = s.strip()
    low = t.lower()
    minor = False
    for suffix, is_minor in ((" minor", True), (" major", False), ("min", True), ("maj", False),
                             ("m", True)):
        if low.endswith(suffix) and len(t) > len(suffix):
            t, minor = t[:len(t) - len(suffix)].strip(), is_minor
            break
    letter, acc = t[0].upper(), t[1:]
    if letter not in "FCGDAEB" or any(c not in "#b" for c in acc):
        raise KeyError(s)
    f = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}[letter]
    f += 7 * (acc.count("#") - acc.count("b"))
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
                  f"movement {vinfo['fluct']}, centre {vinfo['centre']}, range {vinfo['range']})"
                  if vinfo else ""))
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
    depth_used = "full"
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
        acc_notes = poly.merge_split_notes(acc_notes)
        n_found = len(acc_notes)
        d = depth_mod.parse_depth(opts.depth)
        if d == 0:
            d = depth_mod.auto_depth(acc_notes)
            depth_used = f"auto ({d})"
        else:
            depth_used = "full" if d is None else str(d)
            if d is not None and mel_notes and show_melody:
                d = max(1, d - 1)          # the voice is one of the sounding notes
        strict = None                      # per-depth default (measured on the NMF path)
        if method != "nmf":
            # greedy saliences: weak low notes are as often real as false
            strict = 0.0 if (depth_used.startswith("auto") or d is None or d >= 3) else 0.4
        acc_notes = depth_mod.limit_depth(acc_notes, d, strictness=strict)
        _log(opts, f"  {n_found} notes found, {len(acc_notes)} kept at depth {depth_used}")

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
    met = meter.detect_meter(bm, onset, fps, all_notes, spec=harm, df=df, ref_hz=ref_hz,
                             time_signature=(tuple(opts.time_signature)
                                             if opts.time_signature else None),
                             debug=bool(os.environ.get("SHEETMUSIC_DEBUG")),
                             lead=mel_notes, odd_meters=opts.odd_meters)
    if met.compound and not opts.time_signature and (opts.triplets or opts.swing != "off"):
        why = meter.simple_with_triplets(met.beat_map, all_notes)
        if why:
            alt = meter.detect_meter(bm, onset, fps, all_notes, spec=harm, df=df, ref_hz=ref_hz,
                                     lead=mel_notes, simple_only=True)
            if abs(alt.beat_map.period / met.beat_map.period - 1) < 0.1:
                _log(opts, f"  {met.name} reads as {alt.name} with {why}")
                met = alt
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
               f"{_BEAT_NAMES.get(met.beat_ticks, 'beat')}s per minute, "
               f"{bm.beat_times.size} beats (metre confidence {met.confidence:+.2f})")

    tpb = met.beat_ticks
    # Grids and snapping tolerances are defined on note values (``opts.*_grid``
    # subdivisions of a crotchet), so 2/2 and 3/8 snap like 4/4 and 3/4 at the
    # same note values; compound metres keep their dotted-crotchet tolerances.
    scale = (720 if met.compound else 480) / tpb

    def _grid(g):
        unit = max(30, 480 // max(1, g))        # finest value in ticks
        while unit > 30 and (tpb % unit or (met.compound and (tpb // unit) % 3)):
            unit //= 2
        return max(1, tpb // unit)
    mgrid, agrid = _grid(opts.melody_grid), _grid(opts.accomp_grid)
    swing = False
    if opts.swing != "off" and not met.compound:
        allpos = [float(bm.to_beats(n.start)) + offset for n in mel_notes + acc_notes]
        swing = opts.swing == "on" or rhythm.detect_swing(allpos, met.compound)
    trip = opts.triplets and not met.compound
    mel_q = rhythm.quantize(mel_notes, bm, offset, tpb, mgrid, monophonic=True,
                            fill_gaps=0.25 * scale, snap_tol=0.12 * scale, swing=swing,
                            triplets=trip)
    # accompaniment onsets are smeared (strums, arpeggios, long STFT windows):
    # snap them harder so chords line up instead of scattering into 32nds
    acc_q = rhythm.quantize_poly(acc_notes, bm, offset, tpb, agrid, snap_tol=0.18 * scale,
                                 min_units=max(1, 120 // (tpb // agrid)),  # >= a 16th
                                 split=opts.split_point, max_extend=4.0 * scale, swing=swing,
                                 triplets=trip, compound=met.compound)
    if swing:
        _log(opts, "  swing: long-short quavers written straight, marked 'Swing'")
    n_trip = sum(1 for q in mel_q + acc_q if (q.start % tpb) % 30)
    if n_trip:
        _log(opts, f"  {n_trip} notes in triplets")

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
    if opts.melody_in_piano and mel_q and (opts.accompaniment or instrumental):
        acc_q = _tune_into_right_hand(mel_q, acc_q)
        _log(opts, "  the sung tune is in the piano's right hand")
    qps = bm.bpm * tpb / 480.0 / 60.0                  # crotchets per second
    measure_ticks = num * 1920 // den
    if opts.arrangement != "full" and acc_q:
        n_before = len(acc_q)
        acc_q = arrange.arrange_piano(acc_q, opts.arrangement, beat_ticks=tpb,
                                      measure_ticks=measure_ticks, crotchets_per_second=qps,
                                      compound=met.compound, split=opts.split_point)
        _log(opts, f"  {opts.arrangement} arrangement: {len(acc_q)} of {n_before} piano notes kept")
    play = arrange.playability(acc_q, qps)

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
        # staves follow the hands, not a fixed pitch (see hands.assign_hands)
        up = [q for q in acc_q if (q.hand if q.hand is not None else q.pitch >= opts.split_point)]
        dn = [q for q in acc_q if not (q.hand if q.hand is not None else q.pitch >= opts.split_point)]
        parts.append(PartSpec("Piano", "Pno.", 0, [StaffSpec(up, "G"), StaffSpec(dn, "F")]))
    title = opts.title if opts.title is not None else os.path.splitext(os.path.basename(path))[0]
    score = ScoreSpec(parts=parts, title=title, subtitle="Transcribed from audio",
                      key_fifths=fifths, key_minor=(mode == "minor"),
                      time_num=num, time_den=den, bpm=round(bm.bpm),
                      beat_ticks_override=tpb, swing=swing)
    info = dict(duration=dur, tuning_cents=tuning, bpm=bm.bpm, key=key_name,
                time_signature=met.name, meter_scores=met.scores, meter_confidence=met.confidence,
                swing=swing,
                has_vocals=has_vocals, vocal_evidence=vinfo, depth=depth_used, playability=play,
                arrangement=opts.arrangement,
                melody_notes=len(mel_q), accomp_notes=len(acc_q),
                seconds=time.time() - t0)
    _log(opts, f"Done in {info['seconds']:.1f} s: {len(mel_q)} melody notes, "
               f"{len(acc_q)} accompaniment notes")
    return Transcription(score, mel_q, acc_q, info, track, mel_notes, acc_notes, bm, offset)


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


def _tune_into_right_hand(tune: list[QNote], piano: list[QNote]) -> list[QNote]:
    """Piano-solo version of a song: the sung line becomes the top of the
    right hand.  A low (male) tune is played an octave up, where a right hand
    plays.  Piano notes at or above the tune while it sounds would hide it,
    and notes a semitone below it are almost always the voice leaking into the
    accompaniment analysis (vibrato, scoops): both are left out."""
    shift = 12 if np.median([q.pitch for q in tune]) < 60 else 0
    t = [QNote(q.start, q.end, q.pitch + shift, max(q.velocity, 100), False, 1) for q in tune]
    starts = np.array([q.start for q in t])
    order = np.argsort(starts)
    t = [t[i] for i in order]
    starts = starts[order]
    kept = []
    for q in piano:
        i = int(np.searchsorted(starts, q.end))          # tune notes starting before q ends
        hidden = any(x.end > q.start and q.pitch >= x.pitch - 1 for x in t[max(0, i - 8):i])
        if not hidden:
            kept.append(q)
    return sorted(kept + t, key=lambda q: (q.start, q.pitch))


def _first_onset(notes: list[Note]) -> float:
    return min((n.start for n in notes), default=0.0)


__all__ = ["TranscribeOptions", "Transcription", "transcribe", "tpc_name", "spell"]
