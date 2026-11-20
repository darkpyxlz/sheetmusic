# sheetmusic: MP3 to MuseScore (.mscz)

`sheetmusic` turns an MP3 (or any audio file ffmpeg can read) into a MuseScore score:

* **songs with vocals:** a **Voice staff** with the sung line plus a **Piano grand
  staff** with the accompaniment (chords and bass);
* **instrumental music:** everything transcribed into a **piano score**.

Whether the song has vocals, the **time signature** (2/4, 3/4, 4/4, 3/8, 6/8, 9/8, 12/8;
any other can be forced), the **tempo**, the **key** and **key signature** are all
detected automatically. Measures, ties and rests are filled in, and note values are
chosen so that a crotchet is written as a crotchet, not as a semiquaver and rests.
By default the piano part is **arranged to be playable by two hands** (at most three
notes per hand within an octave; `--arrangement easy` for a simpler version, `full`
for every note that was heard). It also writes a MIDI file you can listen to, and a
PDF if MuseScore is installed.

It uses plain signal processing (numpy and scipy, plus ffmpeg for decoding), so there
are no models to download and no GPU is needed. A 4-minute song with vocals takes
about 40 s on a laptop CPU, an instrumental one about 80 s (it uses a slower, more
accurate analysis). Neural source separation (Demucs) is an optional extra.

How accurate it is on real recordings, measured, is in "Accuracy, honestly" below.

![example](examples/synthetic_song.png)

*Output for the synthetic test song in `tests/synth.py`, default settings. The
"singer" there is made hard to transcribe on purpose: a weak fundamental, vowel
formants that change on every note, vibrato, pitch scoops into each note, and
consonant noise. Behind it are piano chords that double the voice an octave lower,
plus bass and drums. The voice line comes out right except for one extra semiquaver
where the second phrase starts (bar 5). The piano part is the weaker half: it still
has stray notes where the voice leaks into it.*

![easy piano solo](examples/synthetic_song_easy_solo.png)

*The same song as an easy piano solo: `--mode accompaniment --melody-in-piano
--arrangement easy`. The sung tune is the right hand's top line; the left hand plays
the bass in minims.*

![instrumental example](examples/instrumental_waltz.png)

*An instrumental piano waltz (`tests/synth_instr.py`). The program detected on its
own that there are no vocals, that it is in 3/4 at 133 BPM, and that it is in
F major, then wrote it as a playable piano score. (The synthetic piece's
accompaniment simply stops before its tune does.)*

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
python -m sheetmusic song.mp3                       # -> song.mscz, everything automatic,
                                                    #    piano part playable by two hands
python -m sheetmusic song.mp3 -o out.mscz --midi out.mid --pdf out.pdf
python -m sheetmusic song.mp3 --mode melody         # only the vocal line
python -m sheetmusic song.mp3 --mode instrumental   # force "no vocals": full piano score
python -m sheetmusic song.mp3 --mode vocal          # force "has vocals"
python -m sheetmusic song.mp3 --separate demucs     # neural vocal separation first
python -m sheetmusic song.mp3 --arrangement easy       # the simplest version: tune, bass, few notes
python -m sheetmusic song.mp3 --arrangement full       # every note that was heard (not arranged)
python -m sheetmusic song.mp3 --mode accompaniment --melody-in-piano --arrangement easy
                                                       # a song as an easy piano solo
python -m sheetmusic song.mp3 --depth sparse        # fewer simultaneous notes (see below)
python -m sheetmusic song.mp3 --start 30 --duration 20 --bpm 92 --key Eb --time-signature 6/8
python -m sheetmusic song.mp3 --time-signature 7/8  # any signature can be forced (2/2, 5/4, 12/16 ...)
python -m sheetmusic song.mp3 --info-json song.json # what was detected, and how hard each hand is
```

The program prints what it detected, e.g.

```
  no vocals: instrumental transcription (timbre var 0.442, wobble 49.61, movement 0.907, centre 1.0, range 15.5)
  time signature 12/8, tempo ~ 107.7 dotted-crotchets per minute, 32 beats (metre confidence +0.23)
  key: D major
  playable arrangement: 74 of 74 piano notes kept
```

(a fiddle jig; it is written in 6/8, so the detector chose twice the bar length here:
the note values are right, the barlines come every second bar). The metre confidence
is the margin over the best different metre. On the development pieces, when it was
below +0.05 the time signature was right only 9 times in 16; above +0.4 the
simple/compound choice was right 17 times in 18. When it is low, check the metre;
`--time-signature` settles it.

Open the `.mscz` in **MuseScore 3 or 4**. The file uses the MuseScore 3.01 format,
which both versions read; MuseScore 4 may offer to update the layout style when it opens.
Files are checked with MuseScore 3.2.3, which opens and exports them without repairs
(the test suite does this when MuseScore is installed); MuseScore 4 has not been tested.

`python -m sheetmusic -h` lists every option with its default; every setting of
`TranscribeOptions` (below) has one. The ones you are most likely to need:

| option | meaning |
|---|---|
| `--arrangement playable\|easy\|full` | how much of the piano part to write (see "Playable arrangements"). `playable` (default): at most 3 notes per hand within an octave, nothing faster than semiquavers; `easy`: at most 2 notes per hand within a sixth, quavers at the shortest; `full`: every note that was heard |
| `--melody-in-piano` | songs: put the sung tune into the piano's right hand too (with `--mode accompaniment`: a piano solo) |
| `--mode auto\|vocal\|instrumental\|melody\|accompaniment` | `auto` (default) detects vocals; the others force a choice or write only one part |
| `--time-signature N/D`, `--odd-meters` | force any time signature, or let detection also choose bars of 5 and 7 beats |
| `--bpm`, `--key F#m` / `--key "A minor"`, `--tuning +20` | override what the program detects |
| `--shortest 16th` | shortest note value written (64th, 32nd, 16th, 8th, quarter); values still snap to the longest value that fits |
| `--depth auto\|full\|melody\|sparse\|normal\|rich\|N` | how many notes may sound at once before arranging (see "Depth") |
| `--separate demucs` | use Demucs to split vocals/accompaniment before transcribing |
| `--voicing 0.3`, `--voice-strictness 0.72`, `--voice-range A2-D6` | melody sensitivity, how strictly instrument notes are kept out of the voice staff, and the range searched (MIDI numbers or note names) |
| `--poly-method auto\|greedy\|nmf`, `--poly-threshold 0.15` | multi-pitch method and sensitivity of the piano part |
| `--midi PATH`, `--pdf PATH`, `--info-json PATH` | extra outputs: MIDI to listen to, PDF (needs MuseScore), and what was detected (tempo, metre and its confidence, key, tuning, vocals, how hard each hand is) as JSON |

Everything is also callable from Python:

```python
from sheetmusic import transcribe, TranscribeOptions, write_mscz, write_midi
from sheetmusic import arrange_piano, playability, detect_meter, assign_hands, estimate_key

res = transcribe("song.mp3", TranscribeOptions(arrangement="easy", time_signature=(6, 8)))
print(res.info["time_signature"], res.info["bpm"], res.info["key"], res.info["playability"])
write_mscz(res.score, "song.mscz")
easier = arrange_piano(res.accomp_notes, "easy", beat_ticks=res.score.beat_ticks,
                       measure_ticks=res.score.measure_ticks,      # re-arrange without
                       crotchets_per_second=res.info["bpm"] * res.score.beat_ticks / 480 / 60)
                                                                   # re-analysing
```

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

### Playable arrangements

A transcription writes down what sounds: every note of a chord, every passing note,
every repeated strum, which is more than two hands can play and more than anyone
wants to read. `--arrangement` (`arrange.py`) keeps the main notes within what one
player can manage:

| | `full` | `playable` | `easy` |
|---|---|---|---|
| notes held at once, per hand | all | 3 | 2 |
| widest interval in one hand | any | an octave | a sixth |
| shortest note value, onset grid | 32nd | semiquaver | quaver |
| closest onsets: right / left hand | – | ¼ / ½ beat | ½ / 1 beat |
| onsets per second: right / left hand | – | 7 / 4 | 4 / 2.5 |

What stays is decided by importance. The top note of the right hand (the tune) and
the bottom note of the left hand (the bass) go first, then notes by strength, length
and metrical position (downbeat > beat > off-beat). Onsets are kept greedily, most
important first, as long as they respect the hand's gap and rate limits, so a fast
tremolo or strum becomes one chord. Within each kept onset the outer note comes
first, then other notes while the hand's chord size and reach allow. Notes are only
removed, shortened, or lengthened to the next kept note of the same hand (a dropped
passing note leaves its predecessor sounding), never moved to another pitch, so every
note on the page is one the transcription heard. The left hand stays below the right.

For songs, `--melody-in-piano` puts the sung tune into the right hand as its top line
(a low male tune an octave up), so `--mode accompaniment --melody-in-piano
--arrangement easy` gives a piano-solo version of the song.

`--info-json` reports how hard the result is: per hand, onsets per second, the
busiest second, notes held at once and the widest interval held.

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
blips (kick-drum tails) are dropped. Notes go to the treble or bass staff by the
hand that plays them (see "Hands" below).

### Rhythm and notation

* The onset envelope comes from a short-window spectral flux. Tempo comes from
  autocorrelation, and beats from dynamic programming (Ellis 2007). The program
  maps time to beats through the tracked beats, so the score stays in time even
  when the tempo drifts.
* **Time signature, beat level and downbeat** (`meter.py`). The beat tracker gives a
  steady pulse, but on real recordings that pulse is often half the beat, a third of
  it (the quavers of 6/8), or three semiquavers of a syncopated tune. Music repeats at
  every level of its metre, so the metre is found as the hierarchy
  *subdivision → beat → bar* that the music supports best. The pulse is cut into
  twelve slots (halves, thirds and quarters of it are all on the grid) and three
  accent signals are sampled there: attacks (onsets and note starts, long notes
  counting more), harmony (chord changes and bass notes) and melodic shape. Every
  hierarchy (beat = ½ to 3 pulses, divided in 2 or 3, bars of 2, 3 or 4 beats; 5 and
  7 with `--odd-meters`) is scored by:
  * periodicity: the autocorrelation of the accents at the beat, at the bar and at the
    rival groupings (triple time repeats after 3 and 6 beats, not after 2 and 4);
  * phase-aware contrast: do the beats stand out from the subdivisions between them,
    does the downbeat stand out from the other beats, and do the subdivisions stand
    out from the positions of the rival division (thirds against halves)? An even
    stream of quavers repeats at every multiple of a quaver, so periodicity alone
    cannot tell 4/4 from 12/8; contrast can;
  * a tempo preference (a plain beat near 110 per minute, a dotted beat near 70), a
    bar-length preference, and readability: the usual gap between notes should come
    out as a quaver or a crotchet, not a semiquaver, so fast music is not written at
    half speed in a blizzard of short notes.

  The weights were fitted on development pieces (real recordings with annotated
  beats, and scores in eight metres) with every synthetic test piece required to keep
  its metre, and checked on held-out pieces (see "Accuracy, honestly"). The downbeat
  is then the bar phase whose first beat carries the harmony changes and bass notes,
  at which long notes start, and across whose barlines the fewest notes are tied.
  `--time-signature` forces any signature (2/2, 3/8, 5/4, 7/8, 12/16 ...): beat level
  and downbeat are still found from the audio.
* Quantisation is adaptive down to demisemiquavers (32nds): each onset snaps to
  the coarsest grid (beat, half, quarter, eighth of a beat) it is close to, so a
  note sung slightly late still lands on the beat while genuinely fast notes keep
  their 16ths/32nds. Grids and tolerances are set per crotchet, so 2/2 and 3/8 get
  the same note values as 4/4 and 3/4. Notes of the piano part struck together
  snap together. Small gaps in the vocal line are closed (legato).
* **Note values of the piano part** (`rhythm.quantize_poly`). In real scores a note
  almost always ends exactly where another note begins (96 % of the notes in the
  benchmark scores). So a note's end is chosen among the starts of later notes, plus
  the grid position where its sound stopped (a rest follows), by a cost fitted to
  how written lengths relate to sounding lengths in those scores. A freely decaying
  note (piano, guitar: recognised by how fast its partials fade, 11–22 dB/s against
  0–3 dB/s for bowed and blown notes) may be written much longer than it sounds,
  because it fades or is damped early, and it runs to the next note *of its own
  hand*; a sustained note is written as long as it sounds. This is what keeps a
  crotchet from being written as a semiquaver followed by rests.
* **Hands** (`hands.py`): the piano part is split between the hands by reach (a hand
  spans about an octave), by movement (a hand stays where it is unless it has to
  move) and by register, instead of at a fixed pitch. Against the hands in real
  piano scores this is right for 91 % of notes, against 75 % for a split at middle C.
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

Accuracy is reported in tiers, because they mean different things. "Note F" is the
F-measure of notes with the right pitch whose start is within ±50 ms of the truth
(the usual convention), measured on the notes before quantisation. All numbers below
are for **held-out** pieces (fixed before any tuning, by other composers / other
works than the development pieces), default settings unless a row says otherwise;
"before" is the version at the start of the latest round of changes. Development-set
numbers and the full method are in `docs/AUDIT.md`.

**Real recordings: notes** (MAESTRO v3, concert-grand piano with aligned MIDI, 8
pieces, 15 minutes):

| | precision | recall | note F | F with note ends |
|---|---|---|---|---|
| held-out set | 0.70 | 0.62 | **0.65** | 0.23 |

This is virtuosic solo piano only. There is **no measurement on real singing, guitar
or band recordings yet** (no annotated data was available), and none of how Demucs
affects accuracy.

**Real recordings: rhythm** (ASAP annotations of 33 MAESTRO performances, first
minute each, in eight metres):

| | before | now |
|---|---|---|
| time signature | 9/33 | 10/33 |
| simple or compound time | 15/33 | 18/33 |
| beat at the right level (±8 %) | 12/33 | 12/33 |
| downbeat F (±70 ms) | 0.25 | 0.27 |
| key signature | 14/30 | 16/30 |

The gain is in compound time (6/8: 0 → 2 of 5, 12/8: 0 → 1 of 3); 4/4 went from 8 to 6
of 8, and real 3/4 is still recognised only 1 time in 8. Expressive real playing is
hard for a single-tempo beat tracker: check the metre and use `--time-signature`.

**Scores in eight metres** (26 excerpts: piano pieces, string-quartet movements, folk
tunes in 2/4, 3/4, 4/4, 2/2, 3/8, 6/8, 9/8, 12/8, rendered through a soundfont in
strict tempo):

| | before | now |
|---|---|---|
| note F | 0.79 | 0.79 |
| time signature | 11/26 | **17/26** |
| simple or compound time | 15/26 | 20/26 |
| tempo (beat level right) | 17/26 | 18/26 |
| downbeat | 10/26 | 11/26 |
| written note values exact (pieces at the right tempo) | 0.80 | 0.81 |
| semiquavers or shorter on the page (the scores: 0.21) | 0.08 | 0.08 |

**Bach chorales** (24 excerpts through a soundfont: piano, string quartet, wind
quartet):

| | before | now |
|---|---|---|
| note F (piano / strings / winds) | 0.81 / 0.83 / 0.65 | 0.81 / 0.83 / 0.65 |
| time signature | 22/24 | 21/24 |
| tempo | 23/24 | 22/24 |
| downbeat | 9/24 | **14/24** |
| key signature | 18/24 | 18/24 |
| written note values exact | 0.84 | 0.86 |
| written too short | 0.10 | **0.04** |
| semiquavers or shorter on the page (the scores: 0.01) | 0.08 | 0.05 |

**Playable arrangements** (same held-out pieces; the transcription underneath is the
same at every level, only what is written differs). "Tune kept" is the share of the
true top line (melody) on the page, "bass kept" the same for the bottom line; the
hand columns give, for the right hand, the notes held at once and the span held
(95th percentiles, semitones), and the busiest second:

| | level | written notes right | tune kept | bass kept | right hand: held / span / busiest second | left hand: held / span / busiest second |
|---|---|---|---|---|---|---|
| real piano | full | 0.73 | 0.64 | 0.60 | 3.9 / 16 / 9.0 | 3.6 / 20 / 7.2 |
| real piano | **playable** | 0.76 | 0.56 | 0.48 | 3.0 / 12 / 6.8 | 2.9 / 12 / 3.9 |
| real piano | easy | 0.75 | 0.37 | 0.28 | 2.0 / 8 / 3.9 | 2.0 / 7 / 2.0 |
| chorales | full | 0.84 | 0.82 | 0.81 | 3.3 / 14 / 4.2 | 3.1 / 13 / 3.9 |
| chorales | **playable** | 0.86 | 0.80 | 0.80 | 2.9 / 11 / 4.2 | 2.7 / 11 / 3.4 |
| chorales | easy | 0.86 | 0.75 | 0.62 | 2.0 / 9 / 3.5 | 2.0 / 8 / 2.0 |
| scores in eight metres | full | 0.92 | 0.82 | 0.77 | 2.6 / 11 / 6.1 | 2.5 / 11 / 4.5 |
| scores in eight metres | **playable** | 0.92 | 0.80 | 0.74 | 2.3 / 9 / 5.8 | 2.2 / 10 / 3.4 |
| scores in eight metres | easy | 0.91 | 0.55 | 0.48 | 1.8 / 5 / 3.6 | 1.9 / 6 / 2.0 |

So "playable" costs little of the tune on ordinary music (−0.01 to −0.02) and keeps
every hand within reach; on virtuosic concert piano it gives up some tune (−0.08) and
bass (−0.12) to get there. "Easy" halves the busiest passages again; on music with
fast runs it keeps only about half of the tune.

**Synthetic** (`tests/`, the repo's own additive synthesiser; these pieces were used
to tune the program, so read them as "it works on clean input", not as accuracy).
Scored on the quantised output (`--arrangement full`) with a ±0.26-beat tolerance:

| | note F-measure |
|---|---|
| vocal melody, simple song (100/140/72 BPM, +35-cent detuning, low male voice, mono) | 0.96 – 1.00 |
| vocal melody, "hard" song (intro riff, reverb, slides, backing vocals, strummed guitar, snare, tempo drift) | 0.90 |
| vocal melody, sung waltz (3/4) | 0.94 |
| piano waltz, 3/4, F major | 0.98 |
| solo piano, 4/4, D minor | 0.93 |
| strings + piano + bass + drums, 4/4, Eb major | 0.77 |
| flute + guitar arpeggios + bass, 6/8, A minor | 0.77 |
| accompaniment of vocal songs | 0.58 – 0.73 |

Tempo, metre and vocals/instrumental are detected correctly on all of these; of 16
transpositions of the four instrumental pieces, one 6/8 comes out as 12/8 (same note
values, barlines every second bar). The keys are right except the "hard" song, which
is reported as B minor instead of D major (the same key signature).

How to reproduce is in "Benchmarks" below. If the tempo comes out doubled or halved,
or the key is wrong, override them with `--bpm` and `--key`.

Known limitations:

* Voices on a staff are assigned by "first free voice", so a line can move between
  voices.
* No triplets or swing in simple time: the grid there is binary (halves, quarters ...
  of a beat); compound metres (6/8, 9/8, 12/8) divide in three.
* No lyrics, dynamics or articulations.
* Only one time signature and tempo marking for the whole piece; the beat map
  follows tempo drift internally. Bars of 5 and 7 beats are only considered with
  `--odd-meters` (and can always be forced with `--time-signature`), and have not
  been benchmarked: there were no such pieces in the data.
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
python -m benchmarks.datasets            # fetch MAESTRO pieces and ASAP annotations, render scores
python -m benchmarks.run maestro dev     # real piano: notes
python -m benchmarks.run asap dev        # real piano: beats, downbeats, time signature, key (ASAP)
python -m benchmarks.run corpus dev      # soundfont chorales: notes, note values, metre, key
python -m benchmarks.run mixed dev       # soundfont scores in eight metres (piano, quartets, folk tunes)
python -m benchmarks.run maestro test    # held-out: only to confirm a finished change
python -m benchmarks.run maestro dev --opt arrangement='"easy"' --tag easy   # any TranscribeOptions field
python -m benchmarks.report test before after   # markdown tables comparing saved runs
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
  meter.py      time signature, beat level and downbeat (metrical hierarchy)
  hands.py      left/right hand assignment
  arrange.py    playable piano arrangements (--arrangement), playability measures
  keys.py       key estimation and pitch spelling
  depth.py      how many notes may sound at once (--depth)
  mscore.py     .mscx / .mscz writer
  midi.py       MIDI writer
  pipeline.py   glue (TranscribeOptions, transcribe())
  cli.py        command line
benchmarks/     real-recording and soundfont benchmarks (not installed)
```
