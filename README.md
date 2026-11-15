# sheetmusic: MP3 to MuseScore (.mscz)

`sheetmusic` turns an MP3 (or any audio file ffmpeg can read) into a MuseScore score:

* a **Voice / melody staff** with the sung (or lead) line, and
* a **Piano grand staff** with the accompaniment (chords and bass),

with tempo, time signature, key signature, measures, ties and rests filled in. It also
writes a MIDI file you can listen to, and a PDF if MuseScore is installed.

It uses plain signal processing (numpy and scipy, plus ffmpeg for decoding), so there
are no models to download and no GPU is needed. A 4-minute song takes about 40 s on a
laptop CPU. Neural source separation (Demucs) is an optional extra.

![example](examples/synthetic_song.png)

*Output for the synthetic test song in `tests/synth.py`. The "singer" there is made
hard to transcribe on purpose: a weak fundamental, vowel formants that change on
every note, vibrato, pitch scoops into each note, and consonant noise. Behind it are
piano chords that double the voice an octave lower, plus bass and drums. The voice
line comes out note-for-note correct.*

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
python -m sheetmusic song.mp3                       # -> song.mscz
python -m sheetmusic song.mp3 -o out.mscz --midi out.mid --pdf out.pdf
python -m sheetmusic song.mp3 --mode melody         # only the vocal line
python -m sheetmusic song.mp3 --mode instrumental   # lead line called "Melody" + piano
python -m sheetmusic song.mp3 --separate demucs     # neural vocal separation first
python -m sheetmusic song.mp3 --start 30 --duration 20 --bpm 92 --key Eb
```

Open the `.mscz` in **MuseScore 3 or 4**. The file uses the MuseScore 3.01 format,
which both versions read; MuseScore 4 may offer to update the layout style when it opens.

Useful options (`python -m sheetmusic -h` lists them all):

| option | meaning |
|---|---|
| `--mode vocal\|instrumental\|melody\|accompaniment` | which staves to write (default `vocal`) |
| `--separate demucs` | use Demucs to split vocals/accompaniment before transcribing |
| `--voicing 0.3` | lower it to get more (quieter) melody notes, raise it for fewer |
| `--voice-strictness 0.72` | how strictly instrument notes (intro riffs, fills between vocal lines) are kept out of the voice staff; raise if you still see extra notes, lower if real sung notes go missing |
| `--voice-range 45-86` | MIDI range searched for the melody (A2–D6; use `38-86` for a bass singer) |
| `--poly-threshold 0.15` | accompaniment sensitivity (lower = more notes) |
| `--polyphony 5` | max simultaneous accompaniment notes |
| `--bpm`, `--time-signature 3/4`, `--key F#m`, `--tuning +20` | override what the program estimates |
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

### Accompaniment

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
* Double or half tempo (e.g. locking onto hi-hat 8ths) is corrected using the
  melody's rhythm. The downbeat is the beat phase with the strongest
  onsets and bass-note starts.
* Quantisation is adaptive down to demisemiquavers (32nds): each onset snaps to
  the coarsest grid (beat, half, quarter, eighth of a beat) it is close to, so a
  note sung slightly late still lands on the beat while genuinely fast notes keep
  their 16ths/32nds. Accompaniment onsets snap a little harder (strums and
  arpeggios smear them) and are at least a 16th long. Small gaps in the vocal
  line are closed (legato).
* The key is estimated with Krumhansl–Kessler profiles. Accidentals are spelled for
  that key (F♯ in G major, B♭ in F major …).
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

On the synthetic test songs (`tests/`):

| | note F-measure |
|---|---|
| vocal melody, simple song (100/140/72 BPM, +35-cent detuning, low male voice, mono) | 0.96 – 1.00 |
| vocal melody, "hard" song (`tests/synth_hard.py`: instrumental intro riff, reverb, slides, backing vocals, strummed guitar, snare, tempo drift) | 0.90 |
| piano accompaniment, simple song | 0.63 – 0.75 |
| piano accompaniment, hard song | ~0.5 (precision 0.73) |

Real recordings are harder. Reverb, backing vocals, distorted guitars and dense
mixes all lower accuracy. For those, `--separate demucs` helps the most. If the
tempo comes out doubled or halved, or the key is wrong, override them with `--bpm`
and `--key`.

Known limitations:

* Only one voice per staff (chords are fine). Simultaneous notes of different
  lengths are written as tied chords.
* No triplets or swing: the grid is binary (1/2/4/8/16 subdivisions per beat).
* No lyrics, dynamics or articulations.
* Only one time signature and tempo marking for the whole piece; the beat map
  follows tempo drift internally.

## Development

```bash
pip install pytest
python -m pytest            # unit + end-to-end tests (needs ffmpeg; MuseScore optional)
python tests/synth.py test.mp3   # regenerate the synthetic test song
# tests/synth_hard.py: make_hard_song() builds the harder, pop-production-like one
```

Layout:

```
sheetmusic/
  audio.py      decoding (ffmpeg / WAV fallback), optional Demucs separation
  spectral.py   STFT, centre extraction, HPSS, whitening, harmonic salience, tuning
  melody.py     vocal f0 tracking, note segmentation, voice harmonic mask
  poly.py       multi-pitch accompaniment transcription
  rhythm.py     onsets, tempo, beat tracking, downbeat, quantisation
  keys.py       key estimation and pitch spelling
  mscore.py     .mscx / .mscz writer
  midi.py       MIDI writer
  pipeline.py   glue (TranscribeOptions, transcribe())
  cli.py        command line
```

Use it from Python:

```python
from sheetmusic import transcribe, TranscribeOptions
from sheetmusic.mscore import write_mscz

res = transcribe("song.mp3", TranscribeOptions(bpm=None, key=None))
print(res.info)                 # tempo, key, tuning, note counts
write_mscz(res.score, "song.mscz")
```
