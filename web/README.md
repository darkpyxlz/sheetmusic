# MP3 to MIDI (web app)

A static site that converts an audio file (MP3, WAV, OGG, FLAC, M4A, anything the
browser can decode) to a MIDI file, entirely in the browser. It uses Spotify's
[Basic Pitch](https://github.com/spotify/basic-pitch-ts) model (Apache-2.0) on
TensorFlow.js; no audio is uploaded anywhere.

Features: polyphonic transcription, piano-roll preview, playback of the MIDI
(built-in synth) and of the original, settings that re-segment notes instantly
without re-running the model (onset/frame thresholds, minimum note length, pitch
range, ghost-note removal), optional pitch bends, velocity from loudness.

## Run locally

```bash
cd web
npm install
npm run build
python3 -m http.server -d dist 8080   # then open http://localhost:8080
```

## Deploy

`.github/workflows/pages.yml` builds `web/` and publishes `web/dist` to GitHub
Pages on every push to the default branch (`claude/mp3-to-sheet-music-gvz6gj`,
or `main` if it is renamed) that touches `web/`. One-time setup: in the
repository's **Settings → Pages**, set **Source** to **GitHub Actions**.

## Files

* `src/app.js`: UI flow and settings
* `src/audio.js`: decoding, mono mix-down and resampling to 22 050 Hz
* `src/model.js`: runs the model in 2-second windows (frees GPU memory as it goes)
* `src/midi.js`: writes the MIDI file (one channel per overlapping note when pitch bends are on)
* `src/roll.js`, `src/player.js`: piano roll and playback
