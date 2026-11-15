# sheetmusic: MP3 to MuseScore (.mscz)

`sheetmusic` turns an MP3 (or any audio file ffmpeg can read) into a MuseScore score:

* **songs with vocals:** a **Voice staff** with the sung line plus a **Piano grand
  staff** with the accompaniment (chords and bass);
* **instrumental music:** everything transcribed into a **piano score**.

Whether the song has vocals, the **time signature** (2/4, 3/4, 4/4, 6/8, 9/8, 12/8),
the **tempo**, the **key** and **key signature** are all detected automatically. Measures,
ties and rests are filled in. It also writes a MIDI file you can listen to, and a PDF
if MuseScore is installed.

It uses plain signal processing (numpy and scipy, plus ffmpeg for decoding), so there
are no models to download and no GPU is needed. A 4-minute song with vocals takes
about 40 s on a laptop CPU, an instrumental one about 80 s (it uses a slower, more
accurate analysis). Neural source separation (Demucs) is an optional extra.

How accurate it is on real recordings, measured, is in "Accuracy, honestly" below.

![example](examples/synthetic_song.png)

*Output for the synthetic test song in `tests/synth.py`. The "singer" there is made
hard to transcribe on purpose: a weak fundamental, vowel formants that change on
every note, vibrato, pitch scoops into each note, and consonant noise. Behind it are
piano chords that double the voice an octave lower, plus bass and drums. The voice
line comes out note-for-note correct.*

![instrumental example](examples/instrumental_waltz.png)

*An instrumental piano waltz (`tests/synth_instr.py`). The program detected on its
own that there are no vocals, that it is in 3/4 at 133 BPM, and that it is in
F major, then transcribed it as a piano score.*

## Install

```bash
pip install -r requirements.txt      # numpy, scipy
# ffmpeg must be on PATH for MP3 (WAV works without it):
#   Debian/Ubuntu: sudo apt install ffmpeg    macOS: brew install ffmpeg
# optional, for better vocal isolation in dense mixes:
pip install demucs
```

You can also `pip install .`, which adds an `mp3-to-mscz` command.

## Use

```bash
python -m sheetmusic song.mp3                       # -> song.mscz, everything automatic
python -m sheetmusic song.mp3 -o out.mscz --midi out.mid --pdf out.pdf
python -m sheetmusic song.mp3 --mode melody         # only the vocal line
python -m sheetmusic song.mp3 --mode instrumental   # force "no vocals": full piano score
python -m sheetmusic song.mp3 --mode vocal          # force "has vocals"
python -m sheetmusic song.mp3 --separate demucs     # neural vocal separation first
python -m sheetmusic song.mp3 --depth sparse        # fewer simultaneous notes (see below)
python -m sheetmusic song.mp3 --start 30 --duration 20 --bpm 92 --key Eb --time-signature 6/8
```

The program prints what it detected, e.g.

```
  no vocals: instrumental transcription (timbre var 0.27, wobble 24.4, movement 0.76)
  time signature 6/8, tempo ~ 66.4 dotted-crotchets per minute, 50 beats
  key: A minor
```

Open the `.mscz` in **MuseScore 3 or 4**. The file uses the MuseScore 3.01 format,
which both versions read; MuseScore 4 may offer to update the layout style when it opens.

Useful options (`python -m sheetmusic -h` lists them all):

| option | meaning |
|---|---|
| `--mode auto\|vocal\|instrumental\|melody\|accompaniment` | `auto` (default) detects vocals; the others force a choice or write only one part |
| `--separate demucs` | use Demucs to split vocals/accompaniment before transcribing |
| `--voicing 0.3` | lower it to get more (quieter) melody notes, raise it for fewer |
| `--voice-strictness 0.72` | how strictly instrument notes (intro riffs, fills between vocal lines) are kept out of the voice staff; raise if you still see extra notes, lower if real sung notes go missing |
| `--voice-range 45-86` | MIDI range searched for the melody (A2–D6; use `38-86` for a bass singer) |
| `--depth auto\|full\|melody\|sparse\|normal\|rich\|N` | how many notes may sound at once (see "Depth"). `auto` (default) picks it from the music; `melody`/`sparse`/`normal`/`rich` = 1/2/4/6; `full` keeps everything the analysis finds |
| `--poly-threshold 0.15` | accompaniment sensitivity (lower = more notes) |
| `--polyphony 5` | analysis polyphony: harmonic series searched per frame (use `--depth` to thin the result) |
| `--bpm`, `--time-signature 6/8`, `--key F#m`, `--tuning +20` | override what the program detects (time signature, tempo, key and tuning are all automatic by default) |
| `--grid 8`, `--accomp-grid 8` | finest note value as subdivisions per beat (8 = demisemiquavers / 32nds, 4 = 16ths, 2 = quavers). Quantisation is adaptive: each note snaps to the longest value that fits its timing, so 32nds only appear where the music really has them |
| `--keep-voice-in-accompaniment` | don't remove the voice's harmonics from the piano part |

## How it works

```
mp3 ─ffmpeg─► stereo PCM ─► STFT ─┬─► centre-panned harmonic spectrum ─► vocal f0 tracker ─► vocal notes ─┐
                                  │                                          │ f0 track                     │
                                  ├─► harmonic spectrum ─(voice comb notched out)─► multi-pitch ─► piano notes ─┤
                                  └─► onset envelope ─► tempo + beat tracking + downbeat ──────────────────────┤
                                                                                         quantise, key, spell ─► .mscz
```

### Why a voice garbles naive transcription, and what this program does instead

If you write down "the strongest frequency" in each moment, singing goes wrong for
four reasons:

1. **The loudest partial usually isn't the sung note.** A voice is a harmonic
   series (f0, 2·f0, 3·f0 …) shaped by the resonances of the vocal tract
   (*formants*). Whichever harmonic sits under a formant is the loudest, and the
   formants move with every vowel. The "melody" then jumps by octaves and fifths
   from syllable to syllable.
2. **Vibrato, scoops and slides** move the pitch continuously, so a
   frame-by-frame quantiser writes runs of short chromatic notes.
3. **Consonants and breaths** are noise and turn into random notes.
4. **The voice's many harmonics** show up as phantom notes in the accompaniment.

The program handles each of these:

* **Add the voice's resonant harmonics back into its fundamental (harmonic
  summation).** Every candidate pitch collects the energy at f0, 2f0, 3f0 … up to
  16 harmonics, so the sung note wins even when its fundamental is weak. Before
  that, the spectrum is **whitened** (Klapuri-style band normalisation). This
  flattens the formant envelope, so changing vowels no longer push one harmonic up
  and make it look like the melody (`spectral.whiten`, `spectral.HarmonicSalience`).
* **Look where the singer is.** Lead vocals are usually panned to the centre. The
  tracker weights each time-frequency bin by how similar the left and right
  channels are, and uses only the harmonic part of the sound (median-filter HPSS
  removes drums).
* **Use the voice's movement.** Sustained instruments keep their partials still,
  while a voice keeps moving (vibrato, intonation). The long-term steady part of
  the spectrum is subtracted, and the f0 tracker blends the plain and "moving"
  saliences. Without this, a piano doubling the singer an octave lower pulls the
  melody down an octave. Both the f0 track and the voicing decision use this cue
  (`spectral.fluctuating_part`).
* **Track smoothly.** A Viterbi search on a 20-cent grid with a cost for big jumps
  follows vibrato and glides without octave flips. Voicing uses the salience,
  spectral flatness (to drop consonants) and hysteresis.
* **Segment notes the way a musician hears them.** The pitch contour is averaged
  over about one vibrato cycle. An HMM over semitones then cuts it into stable
  notes, so scoops and glides don't become extra notes. Each note's pitch is the
  centre of its stable part, corrected for the recording's tuning (estimated
  automatically, so songs not tuned to A440 still work). Repeated notes are split
  where the voice re-attacks. Isolated octave slips are folded back. Note starts
  are moved back to the consonant / attack (`melody.segment_melody`).
* **Keep instruments out of the voice staff.** Every note gets a
  "voice-likeness" score: how much its partials move (vibrato, intonation)
  times how centre-panned its harmonic series is. Whole phrases that score
  clearly below the song's own typical vocal score (instrumental intros,
  guitar riffs, fills between vocal lines) are dropped, then single outlier
  notes. The threshold is relative to the song, so a voice with wide stereo
  reverb is not penalised (`--voice-strictness`).
* **Don't mistake drums for consonants.** Noisiness (used to detect
  consonants/breaths) is measured on the drum-free harmonic vocal spectrum, so a
  snare hit under a held note no longer chops it in two.
* **Slides are not notes.** Short notes between their neighbours' pitches
  (portamento) are absorbed into the note they lead to, and the boundary
  between slurred notes is placed where the raw pitch actually changes, so
  legato notes don't start late.
* **Keep the voice out of the piano part.** The voice's harmonic comb (f0 and all
  its overtones, widened for vibrato) is notched out of the spectrum before the
  accompaniment is transcribed. Accompaniment notes on the sung pitch (±1 semitone
  for scoops) or its first overtones are suppressed
  (`melody.vocal_harmonic_mask`, `poly.voice_exclusion`).

### Depth: how many notes at once

A multi-pitch analysis reports everything that looks like a harmonic series: real
notes, but also overtones, sub-harmonics and the ringing of notes that have ended.
Most music has two to four real notes sounding at a time, and fewer in the bass.
`--depth` (`sheetmusic/depth.py`) thins the result with two rules:

* **Depth.** At most N notes sound at once. When one more starts, the least important
  note gives way: one that is already sounding is cut short there (it was most likely
  ringing on), one that is just starting is dropped. Importance is the note's strength,
  with a bonus for the top line and the bass line. In a vocal song the voice counts as
  one of the N.
* **Stricter bass.** Below A3 a note must be clearly present next to the piece's other
  low notes. On real piano recordings two thirds of the notes found below E3 were
  false, and they are weaker than the real ones. The reference is the low register
  itself, so a soft accompaniment under a loud tune is kept. This rule was measured on
  instrumental music; under vocals it is only applied when you ask for a depth below 3.

`auto` counts how many clearly present notes sound together most of the time (80th
percentile) and allows one more, between 2 and 6. A lower depth trades recall for
precision: on the real-piano development set `--depth 2` gives precision 0.74 at
recall 0.39, `full` gives 0.61 at 0.58.

### Vocals or instrumental?

The lead line found by the melody tracker must pass six checks before a song is
treated as a vocal song (`melody.vocal_evidence`):

* its **timbre changes from note to note** (different vowels = different formants);
* its pitch **wobbles** inside notes by a vibrato-sized amount: not rock steady like a
  keyboard, and not jumping like a path that hops between the notes of a piano texture;
* its partials keep **moving** (vibrato and intonation);
* it is mixed to the **centre**, as lead vocals are (a stereo-miked instrument is not;
  mono files pass this check by construction);
* it stays within a **vocal range** of about an octave and a half.

The first three alone are what a synthetic piano fails, not what a voice is: every
real solo-piano recording in the benchmark passed them. A wrong "instrumental" only
costs the voice staff, while a wrong "vocal" removes a phantom voice from the music
and halves the note accuracy, so doubtful cases are treated as instrumental. Use
`--mode vocal` / `--mode instrumental` to override. **This check has not been
validated on real singing** (no annotated vocal recordings were available); if a
song with vocals comes out without a voice staff, use `--mode vocal`.

### Instrumental music

Nothing is removed as "voice", and the notes come from **non-negative matrix
factorisation with a fixed harmonic dictionary** (`poly.nmf_activations`). Every
semitone has several harmonic templates (bright to dull timbres), and the spectrum
is explained by all notes **jointly**. A partial shared by two notes, such as the
bass note's 3rd harmonic coinciding with the chord's fifth (very common), is split
between them instead of being swallowed by whichever note was found first. Overtone
residue a twelfth or octave above a much stronger note is suppressed afterwards.

For every note, attacks and releases are judged on that note's **own partials**:

* **Repeated notes are split** when most of its partials jump up together in a short
  (46 ms) analysis window, together with an onset in the mix. Vibrato wobble and a
  neighbour's attack don't count.
* **Note starts** of struck and plucked notes are placed at the attack of the note's
  own partials in the short window. The long window that finds the note smears its
  energy ahead of the attack, which made piano notes start 14–28 ms early.
* **Note ends** come from a steep fall of its partials (key or string released), not
  from waiting for the sound and room reverb to fade. Otherwise every quaver ends up
  written as a crotchet.

Partials that coincide with other notes sounding at the same time are left out of
both decisions.

### Accompaniment of vocal songs

The program uses iterative multi-pitch estimation with spectral-smoothness
cancellation (after Klapuri 2006). It finds the strongest harmonic series, records
it, subtracts its partials, and repeats up to `--polyphony` times per frame. Notes
are then tracked per pitch with hysteresis, so decaying piano notes don't flicker
on and off. A note gets a new attack only when the energy in its own partials rises
again. Leftover overtones of louder co-sounding notes are pruned, and short sub-bass
blips (kick-drum tails) are dropped. Notes at or above `--split` (default C4) go on
the treble staff, the rest on the bass staff.

### Rhythm and notation

* The onset envelope comes from a short-window spectral flux. Tempo comes from
  autocorrelation, and beats from dynamic programming (Ellis 2007). The program
  maps time to beats through the tracked beats, so the score stays in time even
  when the tempo drifts.
* **Time signature and downbeat** (`meter.py`): the tracked beats are divided into
  six slots each (so both halves and thirds of a beat are on the grid). An *accent*
  is measured in every slot from onsets, note starts (long notes and bass notes count
  more) and harmonic change (chroma novelty: chords tend to change on the "1").
  Every candidate meter (2/4, 3/4, 4/4, 6/8, 9/8, 12/8, each also with the beat
  twice as long as the tracker's pulse) is then scored over all bars and every
  possible downbeat position:
  * do its downbeats stand out from the other positions in the bar?
  * do its beats stand out from the off-beats?
  * does the beat divide in two (simple time) or three (compound time)?
  * does the tune move mostly in crotchets at that tempo (this settles "140 BPM
    in crotchets" against "70 BPM in quavers")?
  * is the tempo plausible (around 100 crotchets or 70 dotted crotchets per minute)?
* Double or half tempo is thereby decided together with the meter.
* Quantisation is adaptive down to demisemiquavers (32nds): each onset snaps to
  the coarsest grid (beat, half, quarter, eighth of a beat) it is close to, so a
  note sung slightly late still lands on the beat while genuinely fast notes keep
  their 16ths/32nds. Accompaniment onsets snap a little harder (strums and
  arpeggios smear them) and are at least a 16th long. Small gaps in the vocal
  line are closed (legato).
* **Key**: the transcribed notes (bass counted double) are correlated with
  Krumhansl–Kessler major/minor profiles. (The raw spectrum's chroma is no longer
  blended in: every note's third partial sounds a fifth above it, which pulled keys
  towards the dominant.) Relative keys (C major and
  A minor) share their notes, so the tonic decides between them: the final bass note,
  the last note, the first bass note and the most used bass notes. The key signature
  follows, and accidentals are spelled for the key: F♯ in G major, B♭ in F major, and
  the leading note G♯ in A minor.
* `mscore.py` writes MuseScore XML directly, using as few ties as possible:
  a note is written as one value whenever one exists (syncopated crotchets,
  dotted notes, minims ...); ties remain only across barlines or for lengths no
  single value can express. Notes of different lengths on the same staff go into
  separate MuseScore voices (up to 4), so a held bass note under moving chords
  stays one note instead of a chain of tied quavers. Also: full-measure rests, a
  tempo marking, and a G8vb clef when the melody is a low (male) voice.

## Accuracy, honestly

Transcribing polyphonic music automatically is still an unsolved research problem.
Expect a usable draft that you correct in MuseScore, not a finished engraving.

Accuracy is reported in three tiers, because they mean different things. "Note F" is
the F-measure of notes with the right pitch whose start is within ±50 ms of the truth
(the usual convention), measured on the notes before quantisation, with every setting
at its default.

**Real recordings** (MAESTRO v3: concert-grand piano with aligned MIDI, 8 development
and 8 held-out pieces by different composers, 13–15 minutes each set):

| | precision | recall | note F | F with note ends |
|---|---|---|---|---|
| development set | 0.68 | 0.57 | **0.62** | 0.20 |
| held-out set | 0.70 | 0.62 | **0.65** | 0.23 |

This is virtuosic solo piano only. There is **no measurement on real singing, guitar
or band recordings yet** (no annotated data was available), and none of how Demucs
affects accuracy.

**Sample-based** (Bach chorales rendered through a General MIDI soundfont, 24
development and 24 held-out excerpts; exact notes, tempo, meter and key are known):

| held-out set | note F | tempo | time signature | downbeat | key signature |
|---|---|---|---|---|---|
| piano | 0.81 | 8/8 | 7/8 | 2/8 | 3/8 |
| string quartet | 0.83 | 7/8 | 7/8 | 4/8 | 7/8 |
| wind quartet | 0.65 | 8/8 | 8/8 | 3/8 | 8/8 |

So tempo is reliable, the time signature is reliable for 4/4, and the downbeat and
the key signature are not: expect to correct them (`--key`, or move the barlines in
MuseScore). On four real piano pieces in 3/4 or 6/8 the detected meter was 4/4 every
time.

**Synthetic** (`tests/`, the repo's own additive synthesiser; these pieces were used
to tune the program, so read them as "it works on clean input", not as accuracy).
Scored on the quantised output with a ±0.26-beat tolerance:

| | note F-measure |
|---|---|
| vocal melody, simple song (100/140/72 BPM, +35-cent detuning, low male voice, mono) | 0.96 – 1.00 |
| vocal melody, "hard" song (intro riff, reverb, slides, backing vocals, strummed guitar, snare, tempo drift) | 0.90 |
| vocal melody, sung waltz (3/4) | 0.94 |
| piano waltz, 3/4, F major | 0.98 |
| solo piano, 4/4, D minor | 0.93 |
| strings + piano + bass + drums, 4/4, Eb major | 0.77 |
| flute + guitar arpeggios + bass, 6/8, A minor | 0.77 |
| accompaniment of vocal songs | 0.59 – 0.72 |

Tempo, meter, key and vocals/instrumental are detected correctly on all of these,
including 16 transpositions of the four instrumental pieces.

How to reproduce, and what changed these numbers, is in "Benchmarks" below and in
`docs/AUDIT.md`. If the tempo comes out doubled or halved, or the key is wrong,
override them with `--bpm` and `--key`.

Known limitations:

* Voices on a staff are assigned by "first free voice", so a line can move between
  voices; the piano staves are split at a fixed pitch, not by hand.
* No triplets or swing: the grid is binary (1/2/4/8/16 subdivisions per beat).
* No lyrics, dynamics or articulations.
* Only one time signature and tempo marking for the whole piece; the beat map
  follows tempo drift internally. Odd meters (5/4, 7/8) are not detected.
* Instrumental music is written as a piano reduction; individual instruments are
  not separated into their own staves.

## Development

```bash
pip install pytest
python -m pytest            # unit + end-to-end tests (needs ffmpeg; MuseScore optional)
python tests/synth.py test.mp3   # regenerate the synthetic test song
# tests/synth_hard.py: make_hard_song() builds the harder, pop-production-like one
# tests/synth_instr.py: make_piece(name) builds the instrumental pieces (and a sung waltz)
```

### Benchmarks

```bash
pip install pretty_midi music21 && pip install --no-deps tinysoundfont
python -m benchmarks.datasets            # fetch MAESTRO pieces (range requests), render chorales
python -m benchmarks.run maestro dev     # real piano, development set
python -m benchmarks.run corpus dev      # soundfont chorales, development set
python -m benchmarks.run maestro test    # held-out: only to confirm a finished change
python -m benchmarks.run maestro dev --opt depth='"full"' --tag full   # any TranscribeOptions field
```

Tune on `dev`, confirm on `test`, and report both. None of this is needed to use the
program.

Layout:

```
sheetmusic/
  audio.py      decoding (ffmpeg / WAV fallback), optional Demucs separation
  spectral.py   STFT, centre extraction, HPSS, whitening, harmonic salience, tuning
  melody.py     vocal f0 tracking, note segmentation, voice harmonic mask, vocal detection
  poly.py       multi-pitch transcription (iterative cancellation, harmonic NMF)
  rhythm.py     onsets, tempo, beat tracking, quantisation
  meter.py      time signature + downbeat detection
  keys.py       key estimation and pitch spelling
  depth.py      how many notes may sound at once (--depth)
  mscore.py     .mscx / .mscz writer
  midi.py       MIDI writer
  pipeline.py   glue (TranscribeOptions, transcribe())
  cli.py        command line
benchmarks/     real-recording and soundfont benchmarks (not installed)
```

Use it from Python:

```python
from sheetmusic import transcribe, TranscribeOptions
from sheetmusic.mscore import write_mscz

res = transcribe("song.mp3", TranscribeOptions(bpm=None, key=None))
print(res.info)                 # tempo, key, tuning, note counts
write_mscz(res.score, "song.mscz")
```
