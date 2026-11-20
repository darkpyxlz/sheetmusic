# Technical audit of `sheetmusic`

Audit of commit `0af0eb2`. No source file was changed. Every number below was measured on the
unmodified code unless it is marked "by inspection".

## 0. Headline findings

1. **On real piano recordings the default mode scores F = 0.32**, against 0.74–0.98 on the
   repo's own synthetic pieces. The automatic vocal detector called all 8 real solo-piano
   recordings "vocal songs", which routes them through the weaker path.
2. **With vocals forced off, real piano scores F = 0.59 (NMF) and 0.49 (greedy)** at the
   standard ±50 ms onset tolerance. With offsets it is 0.12–0.20.
3. **The existing benchmark cannot show any of this.** It uses only the repo's own additive
   synthesiser, a ±0.26-beat tolerance (±156 ms at 100 BPM), and the same pieces the thresholds
   were tuned on.
4. Tempo is reliable (23/24). Meter is right on 4/4 and wrong on every real 3/4 or 6/8 piano
   piece tried (0/4). Downbeats are right in 9/23 cases, key signatures in 16/24.
5. The README's synthetic numbers and runtime claims reproduce. Its claims about automatic
   vocal and time-signature detection do not survive real audio.

### Evidence base

Three tiers, kept separate throughout.

| tier | material | size | ground truth |
|---|---|---|---|
| **Real** | MAESTRO v3, Disklavier concert-grand recordings | 8 pieces, 13.9 min, 9,044 notes | aligned MIDI (notes, velocities, pedal) |
| **Sample-based** | Bach chorales from the music21 corpus, rendered through a General MIDI soundfont as piano, string quartet and wind quartet | 24 excerpts of 40 beats | score: notes, tempo, meter, key signature, pickup |
| **Synthetic** | the repo's `tests/synth*.py` | 12 pieces + 16 transpositions | exact |

Limits of this evidence:

- The real tier is small and is all virtuosic solo piano (5–20 notes per second). It says nothing
  about pop mixes, guitar or singing.
- **No real singing was tested.** MAPS, GuitarSet and MedleyDB could not be reached from this
  environment: `zenodo.org` and the MAPS host are blocked by the network policy, and MAPS and
  MedleyDB also need a registration. Only MAESTRO (Google Cloud Storage) was reachable.
- MAESTRO has no meter annotations. The meters quoted for those pieces come from the published
  scores as I know them, for the four pieces where I am certain.
- All 32 real and sample-based pieces have now been looked at, so they are a development set
  from here on. Section 6 proposes a held-out set.

Metrics: note precision/recall/F with onset within ±50 ms and exact pitch (the `mir_eval`
convention), "F100" with ±100 ms, "F-offset" also requiring the offset within max(50 ms, 20 % of
the note), all on the un-quantised notes. "Score F100" is the quantised score mapped back to
seconds through the beat map.

## 1. Architecture overview

```
audio ─ffmpeg─► stereo 22.05 kHz
  └─ spectral.analyze: STFT 4096/256 (5.4 Hz bins, 186 ms window, 86 fps)
        ├─ mono |M|, centre-weighted |M|·sim², 1024-window spectrum, flux, flatness
        └─ hpss_mask ─► harmonic spectrum ─► estimate_tuning
  ├─ melody.track_melody (always runs): whiten ─► harmonic summation ─► Viterbi ─► voicing
  │     └─ segment_melody ─► fix_octave_errors ─► snap_onsets ─► vocal_evidence ─► has_vocals?
  ├─ has_vocals:  voice comb notched out ─► poly.multipitch_activations (greedy, 5 voices)
  │  otherwise:   melody discarded       ─► poly.nmf_activations (fixed harmonic dictionary)
  │     └─ activations_to_notes (hysteresis, re-attacks, releases) ─► prune_overtone_ghosts
  ├─ rhythm: onset envelope ─► autocorrelation tempo ─► Ellis DP beats ─► fix_tempo_octave
  ├─ meter.detect_meter: 11 hypotheses × downbeat phases on a 6-slot-per-beat accent grid
  ├─ rhythm.quantize (adaptive, coarsest grid first) ─► keys.estimate_key ─► drop weak out-of-key
  └─ mscore (MuseScore 3.01 XML, up to 4 voices per staff) / midi
```

About 2,700 lines, numpy and scipy only, deterministic. `pipeline.transcribe` is one 220-line
function. Everything is held in memory.

## 2. Correctness review

Implemented correctly, by inspection:

- The STFT, the centre extraction `2·Re(L·R*) / (|L|²+|R|²)`, the median-filter HPSS soft mask
  and the Klapuri band whitening.
- The pitch Viterbi. The semitone HMM's shortcut (`melody.py:413-419`, switching only from the
  current best state) is exact, because the switching cost is uniform.
- The NMF update (`poly.py:125`) is the standard KL multiplicative update with an L1 penalty.
- The Ellis dynamic-programming beat tracker, the MIDI writer, tie and chord-index handling in
  the MuseScore writer.

Problems:

| # | where | issue | status |
|---|---|---|---|
| C1 | `melody.py:574-617` | The three vocal cues are confounded by polyphony and by decaying notes. See 3.2. | measured |
| C2 | `poly.py:119` | NMF runs on `|X|^0.6`. Compressed magnitudes are not additive where partials of two notes overlap, which is exactly the case NMF is meant to resolve. | by inspection |
| C3 | `poly.py:78-100` | The dictionary is three power-law tilts with one fixed inharmonicity (2e-4) and at most 20 partials. A piano A1 has about 90 partials below 5 kHz; partials 21+ have no atom and must be explained by other pitches. | by inspection, consistent with 3.3 |
| C4 | `poly.py:14-75` | Greedy cancellation removes only 10 harmonics, has a 20-cent tolerance with no inharmonicity model, and always runs a fixed number of iterations with no stopping rule. | by inspection |
| C5 | `spectral.py:219-240` | Harmonic weights `α^(h-1)` are not normalised by how many harmonics fit below 5 kHz, so high pitches get systematically less salience. | by inspection |
| C6 | `poly.py:192-197`, `melody.py:126-134` | Every threshold is relative to the recording's own 90th/95th percentile. There is no absolute noise floor: 8 s of white noise produces 253 notes. Soft passages disappear next to loud ones. | measured |
| C7 | `poly.py:132-143, 418-435` | Octave/twelfth "ghost" suppression deletes real notes. See 3.3. | measured |
| C8 | `rhythm.py:154-173` | `snap` tries coarse grids first with a 0.12-beat tolerance. Triplet eighths (1/3, 2/3) become 0.25 and 0.75; a swung eighth becomes a dotted rhythm; worst-case displacement is 0.12 beat against 0.06 for nearest-32nd. | measured |
| C9 | `pipeline.py:311-323` | Out-of-key notes that are softer than the median and shorter than a beat are deleted. This removes real chromatic notes, and depends on a key estimate that is wrong a third of the time. | by inspection |
| C10 | `pipeline.py:195-203` | In instrumental mode the greedy estimator still runs all 8 iterations although only its `energy` output is used. | by inspection |
| C11 | `poly.py:258`, `poly.py:418`, `meter.py:211` | Three O(n²) Python loops over notes or segments. Harmless at 2 minutes, quadratic for long files. | by inspection |
| C12 | `spectral.py:52`, `pipeline.py:44` | Analysis stops at 5.5 kHz and MIDI 96. Notes above C7 cannot be found (0 of 46). | measured |

## 3. Weaknesses by module, with measurements

### 3.1 Results on real piano (MAESTRO, 8 pieces)

| mode | P | R | F (±50 ms) | F100 | F-offset | score F100 | time |
|---|---|---|---|---|---|---|---|
| default (auto) | 0.30 | 0.35 | **0.32** | 0.38 | 0.06 | 0.38 | 0.16× |
| vocals off, NMF | 0.61 | 0.58 | **0.59** | 0.64 | 0.12 | 0.61 | 0.25× |
| vocals off, greedy | 0.50 | 0.48 | **0.49** | 0.54 | 0.09 | 0.52 | — |

Per-piece F: auto 0.24–0.41, NMF 0.49–0.68, greedy 0.37–0.66. NMF beats greedy on all 8.
Frame-level pitch-class agreement (a chord-accuracy proxy) is 0.38 for NMF. Mean absolute onset
error of matched notes is 25 ms, with NMF onsets about 13 ms early on average. Time is wall-clock relative
to audio length, single run, 4 cores.

Where the NMF path loses notes (recall by category):

| reference notes that are… | recall | vs. | recall |
|---|---|---|---|
| repeated within 0.5 s on the same pitch | **0.39** | not repeated | 0.65 |
| in the softest third | **0.43** | loudest third | 0.66 |
| struck with a note an octave or twelfth below | **0.41** | without | 0.57 |
| below C3 | 0.42 | C3–C5 | 0.62 |
| above C7 | 0.00 | C5–C7 | 0.51 |

Where its 3,333 false notes come from:

| kind | share |
|---|---|
| right pitch, onset 50–200 ms off, or a duplicate | 25 % |
| overtone of a sounding note (octave, twelfth, …) | 21 % |
| sub-harmonic of a sounding note | 20 % |
| pitch already sounding (spurious re-attack) | 12 % |
| other | 22 % |

Durations: matched notes are 2–4.5× longer than the key was held. Against a pedal-extended
reference F-offset rises only to 0.20, so the sustain pedal explains part of it, not most.

### 3.2 `melody.py`: vocal tracking and vocal detection

- **Detector false positives.** Thresholds are timbre variation ≥ 0.45, pitch wobble ≥ 6 cents,
  movement ≥ 0.6. Real piano measures 0.75–1.01, 16–68 cents and 0.68–0.91: all three cues say
  "voice" on all 8 pieces. On the soundfont set: wind quartet 6/8 false positives, string
  quartet 1/8, piano 0/8.
- Why each cue fails:
  - *Timbre variation* is the between-note spread of six harmonic amplitudes. On real instruments
    that spread comes from pitch, dynamics and from other notes landing in the same bins, not
    only from vowels.
  - *Wobble* is the jitter of the Viterbi path. In polyphony the path is pulled between
    simultaneous notes, which looks like jitter.
  - *Movement* is what survives subtracting a 0.5 s median. Every attack and decay survives it,
    so any percussive instrument "moves".
- The synthetic instruments have fixed spectra and exact frequencies, which is why the detector
  passes every existing test.
- The same movement cue drives half of the tracking emission and the voice-likeness filter. How
  a real voice over a real band fares is unmeasured.
- The Viterbi has no unvoiced state. Voicing is decided afterwards by a global percentile
  threshold, so the tracker always follows something.
- Not tested here: real singing, electric guitar, a solo violin or flute over accompaniment.

### 3.3 `poly.py`: greedy cancellation and NMF

- Both estimators were run on identical recordings (3.1). NMF is better by 0.10 F and is
  slower; a clean like-for-like timing of the two was not taken.
- The octave suppression heuristics cost recall on true octaves (0.41 vs 0.57) and still leave
  41 % of the false notes as overtones or sub-harmonics. They hide the dictionary mismatch
  instead of fixing it.
- Repeated notes are the largest single recall loss, and timing, duplicates and spurious
  re-attacks are 37 % of false notes. Onsets are inferred from activation thresholds in a
  186 ms window, not from an attack model.
- NMF frames are independent: no temporal continuity, no attack/sustain distinction, no atom for
  noise or transients, no adaptation of the dictionary to the recording.
- The 4096-sample window cannot separate partials closer than about 11 Hz, so adjacent
  semitones below F♯3 are not resolved at the fundamental.
- The estimated velocity is a usable confidence already: precision by velocity third is
  0.37 / 0.64 / 0.81. It tracks true key velocity only loosely (Spearman 0.52).

### 3.4 `rhythm.py` and `meter.py`

Soundfont set (exact tempo, meter and downbeat known):

| | piano | strings | winds | all 24 |
|---|---|---|---|---|
| note F (±50 ms) | 0.67 | 0.78 | 0.65 | 0.70 |
| note F100 | 0.86 | 0.83 | 0.72 | 0.80 |
| F-offset | 0.44 | 0.71 | 0.48 | 0.55 |
| onset bias | −28 ms | +6 ms | +10 ms | |
| tempo within 4 % | 8/8 | 7/8 | 8/8 | 23/24 |
| time signature | 7/8 | 8/8 | 8/8 | 23/24 |
| downbeat (of those with right meter) | 4/7 | 3/8 | 2/8 | 9/23 |
| key signature | 4/8 | 6/8 | 6/8 | 16/24 |
| false "vocals" | 0/8 | 1/8 | 6/8 | 7/24 |

- Tempo is solid on steady material. The one miss is a doubling (66 written as 132).
- **Meter.** 21/21 on 4/4, 2/3 on 3/4. On the real pieces whose meter I am sure of
  (Rachmaninoff Op. 32/8 and Schubert Op. 90/4 and Grieg Op. 38/7 in 3/4, Chopin Op. 10/9 in
  6/8) the result was 4/4 every time. The scoring weights and priors (+0.10 for 4/4, a tempo
  prior centred on 100 BPM, "tunes move in crotchets") were set on five synthetic pieces.
- **Downbeat** is right in 9 of 23 cases; 12 of the 14 errors are half a bar off.
- **Piano onsets are 28 ms early** on the soundfont set. F rises from 0.67 to 0.86 when the
  window widens to 100 ms: 19 points of piano F are notes found within 50–100 ms.
- One global tempo estimate feeds a beat tracker with a stiff tempo prior. Rubato and tempo
  changes are not modelled. There is no beat-annotated real data in this audit, so beat accuracy
  on expressive playing is unmeasured.
- No tuplets, swing, pickup bars, meter changes, 5/4 or 7/8. Polyphonic notes are snapped one by
  one, so a rolled chord can land on several grid points.
- Quantisation costs little note accuracy (0.64 → 0.61 on real piano). That measures timing, not
  whether the written rhythm is readable.

### 3.5 `keys.py`

- 16/24 key signatures on chorales, 5/6 on the MAESTRO pieces that have one well-known key. Of the 9 errors,
  6 are one fifth sharp (the dominant). The blended chroma is taken from the raw spectrum, where
  every note's third partial adds energy a fifth above it.
- Caveat: the chorale excerpts are short and chorales modulate, so a few of those "errors" are
  arguable.
- The tonic bonus relies on the first and last bass notes, which are wrong for excerpts
  (`--start`, `--duration`) and for any piece with a transcription error at its end.
- Modal music is forced into major or minor (D dorian gets one flat). No modulation handling.
- Spelling is a fixed window on the line of fifths with no melodic or harmonic context.

### 3.6 `mscore.py`

- Sound: measure filling, ties across barlines and inside chords, dotted values, hidden rests in
  secondary voices.
- Missing: tuplets, pickup measure, explicit beaming, dynamics.
- Voices are assigned "first free voice", with no pitch continuity, so a line can hop between
  voices. The staff split is a fixed pitch (MIDI 60), not a hand assignment.
- The only test that opens the file in MuseScore is skipped when MuseScore is absent. It was
  skipped here, so MuseScore 3/4 compatibility is unverified in this audit.

### 3.7 Engineering

- Runtime: 23 s (vocal path) and 36 s (instrumental) for 141 s of audio.
- Memory: 0.6–0.7 GB peak for 141 s, about 4.5 MB per second of audio. An hour-long file would
  need roughly 16 GB.
- Silence, a 0.3 s clip, a constant tone and a clipped DC signal all complete without error. A
  missing file raises a `RuntimeError` carrying ffmpeg's message.
- `Transcription` does not expose the beat map, so a score cannot be mapped back to audio time
  without patching. This blocks evaluation and per-note confidence.
- No CI configuration, no pinned test environment.

## 4. Assessment of the existing tests

27 pass, 1 skipped (MuseScore not installed).

- **Synthetic only.** Every acoustic test uses the repo's additive synth: geometric partial
  amplitudes, the same `sqrt(1 + B·h²)` inharmonicity formula the NMF templates use, an
  exponential-noise reverb, fixed or ±3 % tempo. The "guitar" in the hard song is the piano
  model.
- **No held-out data.** The thresholds were tuned on these pieces, so the reported scores are
  training scores.
- **Lenient metric.** Onset tolerance is ±0.26 beat after searching for the best global shift,
  on quantised output only. No offsets (except one melody check), no chords, no downbeat, no
  tempo-octave metric.
- Each instrumental piece is played twice, so its notes are counted twice.
- Nothing covers real timbres, dynamics, pedal, tuplets, pickups, key changes, noise or long
  files.
- Strengths worth keeping: the notation unit tests are precise, and the "hard" song encodes
  useful vocal failure cases (intro riff, slides, backing vocals).

## 5. Ten highest-priority problems

| # | problem | evidence | impact | difficulty |
|---|---|---|---|---|
| 1 | Vocal detector false positives send instrumental music down the wrong path | 8/8 real piano, 6/8 winds; F 0.59 → 0.32 | very high | medium |
| 2 | No real-data benchmark, lenient metric, no held-out set | section 4 | very high (blocks everything else) | low–medium |
| 3 | Repeated notes and onset detection | recall 0.39 vs 0.65; 37 % of false notes | high | medium–hard |
| 4 | Soft notes lost to global and frame-relative thresholds | recall 0.43 vs 0.66; noise gives 253 notes | high | medium |
| 5 | Overtone and sub-harmonic ghosts, and the heuristics that delete true octaves | 41 % of false notes; recall 0.41 vs 0.57 | high | hard |
| 6 | Onset timing bias on piano | −13 ms real, −28 ms soundfont; 19 points of F between 50 and 100 ms | medium–high | low |
| 7 | Meter and downbeat | 0/4 real triple or compound; downbeat 9/23 | medium–high for readability | hard |
| 8 | Note durations | F-offset 0.12–0.20 real, 0.44–0.71 soundfont | medium–high for readability | medium |
| 9 | Key signature biased to the dominant; chromatic notes deleted on that basis | 16/24; 6 of 9 errors one fifth sharp | medium | low–medium |
| 10 | Range and resolution: bass, top octave, no tuplets | recall 0.42 below C3, 0 above C7 | medium | medium |

Lower priority: memory scaling, the O(n²) loops, the redundant greedy pass, voice assignment
and pickup bars.

## 6. Roadmap with acceptance criteria

Rules for every step: one change at a time; report dev and held-out numbers separately for real,
sample-based and synthetic tiers; the held-out set is run only at milestone gates; all existing
tests keep passing; no new runtime dependency.

**M0. Evaluation harness (no algorithm change).**
- Metrics: P/R/F at ±50 ms, F with offsets (pedal-aware for piano), octave-agnostic F, onset and
  offset deviation, duration ratio, frame-level chord agreement, tempo (±4 %, with double/half
  classes), meter, downbeat phase, key (MIREX weighting), spelling accuracy against the corpus
  scores.
- Data: MAESTRO by manifest; the soundfont corpus set widened beyond chorales (triple and
  compound meters, triplets, pickups); the synthetic set; GuitarSet and MedleyDB once
  `zenodo.org` is reachable. The ASAP annotations (beats, downbeats, meter and key for MAESTRO
  performances) look like the right source of real rhythm ground truth; their reachability is
  not yet checked.
- Split: dev = the 32 pieces used here. Held-out = 20 further MAESTRO test-split pieces and 30
  other corpus works, fixed by manifest before any tuning.
- Add difficult synthetic cases: detuning drift, distortion, dense chords, fast repeated notes.
- Additive API: expose the beat map on `Transcription`.
- *Benefit:* every later claim becomes checkable. *Drawback:* test dependencies (`pretty_midi`,
  `music21`, a soundfont synth) and downloads, kept out of the runtime package.
- *Accept when:* one command reproduces the tables in this audit to ±0.01.

**M1. Vocal / instrumental decision.**
- Measure the cues only where one harmonic series dominates the frame, and add a cue polyphony
  cannot fake (formant-like envelope moving *within* a held note). When unsure, choose
  instrumental: that error costs far less.
- *Benefit:* default-mode F on real piano from 0.32 to about 0.59. *Drawback:* some real vocals
  may be missed until real singing data is available; `--mode vocal` remains.
- *Accept when:* 0 false positives on 20 held-out piano pieces, ≤ 1/16 on soundfont strings and
  winds, all synthetic vocal tests pass, default F on the real dev set ≥ 0.58.

**M2. Onset timing.**
- Place onsets from the short-window attack of the note's own partials instead of the long-window
  activation crossing.
- *Benefit:* notes already found start counting at ±50 ms. *Drawback:* soft attacks in dense
  textures may jitter more.
- *Accept when:* |bias| ≤ 5 ms on both tiers, soundfont piano F ≥ 0.80, real F up by ≥ 0.03.

**M3. Level-adaptive thresholds and a noise floor.**
- Replace global percentiles with a local reference (a running loudness estimate) and an
  absolute floor relative to the noise level.
- *Benefit:* soft-note recall. *Drawback:* more false notes in quiet passages.
- *Accept when:* softest-third recall ≥ 0.55 with precision ≥ 0.60 on real dev; white noise
  yields ≤ 5 notes.

**M4. NMF model.**
- In order, each measured separately: (a) dictionary adapted to the recording, with per-pitch
  harmonic amplitudes re-estimated under a smoothness constraint and per-register
  inharmonicity; (b) separate attack and sustain atoms plus a noise atom; (c) temporal
  continuity on activations; (d) retire the octave suppression heuristics if (a) makes them
  unnecessary. Review the 0.6 power compression against a β-divergence on plain magnitudes.
- Give the greedy estimator the same front end and keep comparing both on identical recordings.
- *Benefit:* fewer ghosts, true octaves kept, repeated notes separated. *Drawback:* more
  iterations and more runtime; an adaptive dictionary can drift, hence the constraint.
- *Accept when:* real dev F ≥ 0.68 and held-out ≥ 0.65; repeated-note recall ≥ 0.55;
  octave-partner recall ≥ 0.55; overtone plus sub-harmonic share of false notes halved; runtime
  ≤ 0.5× real time.

**M5. Note ends.**
- *Accept when:* pedal-aware F-offset ≥ 0.30 on real dev and ≥ 0.70 averaged on the soundfont
  set. *Drawback:* with the pedal down the written length is partly a notation choice.

**M6. Rhythm and meter.**
- Tempo-varying beat tracking; meter scoring re-fitted on dev data with real triple and compound
  pieces; tuplet-aware quantisation chosen per beat by model comparison, so a triplet grid is
  used only where it fits clearly better; chord-level snapping; pickup detection.
- *Benefit:* readable rhythm. *Drawback:* a richer grid can over-fit expressive timing, which is
  why the per-beat choice needs a complexity penalty.
- *Accept when:* meter ≥ 80 % on ≥ 40 pieces including ≥ 12 not in 4/4; downbeat ≥ 70 % on the
  soundfont set; triplet test pieces written with exact onsets for ≥ 90 % of notes; tempo
  accuracy not below 23/24.

**M7. Key and spelling.**
- Build the key profile from transcribed notes rather than raw-spectrum chroma, or remove
  third-partial energy first; segment-wise key for modulation; spell from local context.
- *Accept when:* key signature ≥ 21/24 on the chorale dev set and no worse on held-out; spelling
  ≥ 97 % against corpus scores. *Drawback:* note-based profiles inherit transcription errors.

**M8. Notation.** Pickup bars, tuplets, voice continuity, hand assignment.
- *Accept when:* MuseScore 3 and 4 open every benchmark score without repair (CI job with
  MuseScore installed), and score-level note F with exact onset and duration is reported.

**M9. Engineering.** Chunked analysis, linear-time note loops, per-note confidence in the
output, CI.
- *Accept when:* a 60-minute file stays under 1.5 GB; precision rises monotonically across
  confidence thirds on held-out data.

## 7. README claims that are overstated or unsupported

| claim | finding |
|---|---|
| Accuracy table under "Accuracy, honestly" | Reproduces (0.98, 0.94, 0.77, 0.74; melody 0.90–1.00), but only on the repo's own synth at ±0.26 beat. No real-recording number is given. Real piano: 0.59 forced, 0.32 default. |
| Vocals "detected automatically" | Fails on real piano (8/8) and wind ensembles (6/8). |
| Time signature "detected automatically", including 3/4 and 6/8 | 4/4 is reliable. 0/4 on real 3/4 and 6/8 piano pieces, 2/3 on soundfont 3/4. |
| "`--separate demucs` helps the most" | No measurement in the repo. The only test uses a fake Demucs module. Not testable here (model host blocked). |
| "Open the `.mscz` in MuseScore 3 or 4" | The validating test is skipped without MuseScore. Unverified in this audit. |
| "transposed to 8 other keys: 8/8" | Only 3 transpositions are in the test suite. The claim itself holds: 16/16 in my re-run. |
| "Note ends come from a steep fall… otherwise every quaver ends up as a crotchet" | On real piano, notes are still 2–4× too long. |
| "Only one voice per staff" (known limitations) | Stale: the writer uses up to 4 voices, as the README says elsewhere. |
| Slides, consonants, drums, tuning and other vocal claims | Each is supported only by the synthetic singer. |
| Runtime: 40 s vocal, 80 s instrumental for 4 minutes | Supported: 0.16× and 0.25× real time here. |
| "Instrumental mode: slower, more accurate" | Supported: NMF 0.59 vs greedy 0.49 on the same recordings. |

## 8. Addendum: first round of changes

Four changes were made after this audit, each benchmarked against the previous
version. The held-out sets (8 MAESTRO pieces by other composers, 24 other chorales)
were fixed before any change and run twice: once at the end, and once more after
correcting a regression that the synthetic vocal songs exposed.

| step | change | real piano dev F | chorales dev F |
|---|---|---|---|
| 0 | audited code, default mode | 0.318 | 0.701 |
| 1 | vocal detection: also require centre placement, vocal range, vibrato-sized wobble | 0.588 | 0.694 |
| 2 | depth limit with a stricter bass (`--depth`, default `auto`) | 0.595 | 0.720 |
| 3 | struck-note onsets placed from the short window | 0.616 | 0.764 |
| 4 | key profile without raw-spectrum chroma | 0.616 | 0.765 |

Held-out confirmation, audited code against current code:

| | note F | precision | recall | false "vocals" |
|---|---|---|---|---|
| real piano | 0.408 → **0.654** | 0.38 → 0.70 | 0.45 → 0.62 | 7/8 → 0/8 |
| chorales | 0.721 → **0.761** | 0.71 → 0.78 | 0.74 → 0.75 | 10/24 → 0/24 |

What did not improve, or got worse:

- Frame-level pitch-class agreement on chorales fell from 0.83 to 0.78 (held-out). Notes
  are now cut shorter by the depth limit and wind pieces no longer take the vocal path.
- Wind-quartet F is unchanged (0.64 → 0.65).
- Key signature (17 → 18 of 24), time signature (22 → 22) and downbeat (10 → 9) on the
  held-out chorales did not move. The key change in step 4 helped on 30-second dev
  excerpts (14 → 17) but not on the full-length runs.
- Two depth rules were withdrawn on evidence. The low-interval limit gained under 0.005
  on real piano and cost 0.05 on strummed-guitar voicings. The stricter bass removed
  real chord notes under vocals, so it applies to instrumental analysis only.
- The vocal check is now conservative and has still never seen real singing.

Roadmap status: M0 (harness) and M1 (vocal decision) are done to the extent data
allows; M2 (onset timing) is done for struck notes. M3–M9 are open. The largest
remaining losses are unchanged from section 5: repeated notes, soft notes, overtone
and sub-harmonic ghosts, note ends, meter and downbeat.

## 9. Addendum: second round (metre, note values, playable arrangements, CLI)

Requests: more time signatures, long and short notes recognised ("not every note a
semiquaver"), output a normal pianist can play, and every function on the command
line. New benchmark material, fixed before tuning and split dev / held-out:

- **ASAP** annotations for 67 MAESTRO performances (first 60 s): beats, downbeats,
  time signature and key of real recordings in eight metres. Real rhythm ground truth
  for the first time.
- **Mixed corpus**: 52 score excerpts rendered through the soundfont (piano pieces,
  string quartet movements, folk tunes) in 2/4, 3/4, 4/4, 2/2, 3/8, 6/8, 9/8, 12/8.
- New metrics: written note values against the score (exact / too short / too long),
  share of semiquavers on the page against the score's share, tune and bass-line
  recall of the written notes, and per-hand playability.

### What changed, with development-set evidence

| change | evidence (development pieces) |
|---|---|
| Metre as a hierarchy (beat = ½–3 pulses, divided in 2 or 3, bars of 2/3/4) scored by periodicity, phase-aware contrast, tempo, bar-length and readability preferences; weights fitted with every synthetic test piece required to keep its metre | time signature 9 → 12 of 34 real recordings, 9 → 12 of 26 mixed scores, 23/24 chorales kept; downbeat F 0.285 → 0.302 (real), 0.418 → 0.645 (chorales) |
| Grids and snapping per crotchet, tempo marks for every beat unit | 2/2, 3/8, 6/16 get the note values of 4/4, 3/4, 6/8 |
| Note values of the piano part from the onsets around each note (96 % of score notes end where another starts), struck vs sustained notes classified by decay rate (piano 11–22 dB/s, strings/winds 0–3), struck notes run to the next onset of their own hand | exact written values 0.81 → 0.85 (chorales), 0.67 → 0.70 (mixed scores) |
| Hands assigned by reach, movement and register instead of a fixed split | 91 % / 89 % (dev / held-out) of notes in the right hand against real piano scores, against 75 % / 81 % for a split at middle C |
| `--arrangement playable` (default) / `easy` / `full` | see the table below |

Arrangement levels on the development sets (written notes; the transcription
underneath is identical):

| set | level | written notes right | tune kept | bass kept | right hand: notes held (95 %) / span (95 %) / busiest second |
|---|---|---|---|---|---|
| real piano | full | 0.718 | 0.619 | 0.578 | 4.1 / 18.5 / 8.8 |
| real piano | playable | 0.747 | 0.565 | 0.478 | 3.0 / 11.5 / 6.9 |
| real piano | easy | 0.712 | 0.410 | 0.278 | 2.0 / 8.1 / 3.9 |
| chorales | full | 0.854 | 0.838 | 0.830 | 3.4 / 13.8 / 3.9 |
| chorales | playable | 0.866 | 0.808 | 0.821 | 2.9 / 10.4 / 3.9 |
| chorales | easy | 0.876 | 0.770 | 0.652 | 2.0 / 8.6 / 3.2 |
| mixed scores | full | 0.911 | 0.776 | 0.743 | 2.7 / 11.4 / 6.4 |
| mixed scores | playable | 0.917 | 0.759 | 0.705 | 2.5 / 9.1 / 6.0 |
| mixed scores | easy | 0.871 | 0.545 | 0.474 | 1.9 / 6.7 / 3.7 |

Two arrangement bugs were found by measuring and fixed before the held-out run:
overtone ghosts of inner voices (winds) were taken for the tune (wind tune kept at
"playable" 0.75 → 0.82); and a hand's limits were enforced by always cutting the held
notes, even for a ghost arriving with a chord (now the least important note gives way).

MuseScore 3.2.3 opened and exported (PDF, MusicXML) all 15 checked scores without
repairs (3/8, 2/2, 6/8, 9/8 and 3/4, real and soundfont, all three levels), and the
previously skipped MuseScore test now passes.

### Held-out confirmation (version at the start of the round → now)

| held-out set | measure | before | now |
|---|---|---|---|
| ASAP, 33 real performances | time signature / simple-compound / beat level / downbeat F / key signature | 9 / 15 / 12 / 0.253 / 14 of 30 | 10 / 18 / 12 / 0.273 / 16 of 30 |
| 26 scores in eight metres | time signature / simple-compound / tempo / downbeat | 11 / 15 / 17 / 10 | 17 / 20 / 18 / 11 |
| 26 scores in eight metres | written values exact / too short / semiquaver share (score 0.21) | 0.80 / 0.05 / 0.08 | 0.81 / 0.02 / 0.08 |
| 24 chorales | time signature / tempo / downbeat / key signature | 22 / 23 / 9 / 18 | 21 / 22 / 14 / 18 |
| 24 chorales | written values exact / too short / semiquaver share (score 0.01) | 0.84 / 0.10 / 0.08 | 0.86 / 0.04 / 0.05 |
| MAESTRO, 8 real pieces | note F | 0.654 | 0.654 (transcription stage unchanged) |

Arrangements on the held-out sets, tune / bass kept and right hand held (95 %) / span
(95 %) / busiest second: real piano full 0.64 / 0.60, 3.9 / 16 / 9.0 → playable
0.56 / 0.48, 3.0 / 12 / 6.8 → easy 0.37 / 0.28, 2.0 / 8 / 3.9; chorales full 0.82 /
0.81 → playable 0.80 / 0.80 → easy 0.75 / 0.62; scores in eight metres full 0.82 /
0.77 → playable 0.80 / 0.74 → easy 0.55 / 0.48. Written-note precision rises at
"playable" on every set (0.73 → 0.76, 0.84 → 0.86, 0.92 → 0.92).

What did not improve, or got worse:

- Real 4/4 recordings: time signature 8 → 6 of 8. The real-recording gain is all in
  compound time (6/8 0 → 2 of 5, 12/8 0 → 1 of 3); real 3/4 is still 1 of 8 and the
  beat level is right for only 12 of 33 real performances, unchanged. A single-tempo
  beat tracker remains the weak point on expressive playing.
- Chorales: one time signature and one tempo lost (22 → 21, 23 → 22).
- Synthetic: one of 16 transposed pieces now comes out in 12/8 instead of 6/8, and
  the "hard" song's key is reported as B minor instead of D major (same signature).
  Accompaniment F of the synthetic vocal songs moved from 0.59–0.72 to 0.58–0.73.
- "Easy" keeps only about half the tune of music with fast runs; "playable" gives up
  0.08 of the tune and 0.12 of the bass on virtuosic concert piano.
- Odd metres (5, 7) are implemented but unmeasured: no such pieces in the data.
- Still unmeasured: real singing, guitar, band mixes (no annotated data reachable).
