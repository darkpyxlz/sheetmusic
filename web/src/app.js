import * as tf from '@tensorflow/tfjs';
import { addPitchBendsToNoteEvents, noteFramesToTime, outputToNotesPoly } from '@spotify/basic-pitch';
import { decodeFile, resampleMono } from './audio.js';
import { runModel, SAMPLE_RATE, FPS } from './model.js';
import { notesToMidi } from './midi.js';
import { PianoRoll } from './roll.js';
import { Player } from './player.js';

const $ = (id) => document.getElementById(id);

const DEFAULTS = {
  onset: 0.5, frame: 0.3, minlen: 120, minf: 21, maxf: 108, bpm: 120,
  melodia: true, bends: false, velocity: true,
};

const NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'];
const noteName = (m) => NAMES[m % 12] + (Math.floor(m / 12) - 1);

const state = {
  fileName: '',
  original: null,     // AudioBuffer at its own sample rate, for playback
  frames: null,       // model outputs, one row per frame
  onsets: null,
  contours: null,
  notes: [],          // NoteEventTime[]
};

const roll = new PianoRoll($('roll'));
const player = new Player();

// ---------- settings ----------

function readSettings() {
  return {
    onset: +$('onset').value,
    frame: +$('frame').value,
    minlen: +$('minlen').value,
    minf: +$('minf').value,
    maxf: +$('maxf').value,
    bpm: Math.min(300, Math.max(20, +$('bpm').value || 120)),
    melodia: $('melodia').checked,
    bends: $('bends').checked,
    velocity: $('velocity').checked,
  };
}

function showSettingValues() {
  const s = readSettings();
  $('onset-v').textContent = s.onset.toFixed(2);
  $('frame-v').textContent = s.frame.toFixed(2);
  $('minlen-v').textContent = `${s.minlen} ms`;
  $('minf-v').textContent = noteName(s.minf);
  $('maxf-v').textContent = noteName(s.maxf);
  $('bpm-v').textContent = '';
}

function applyDefaults() {
  for (const [k, v] of Object.entries(DEFAULTS)) {
    const el = $(k);
    if (el.type === 'checkbox') el.checked = v; else el.value = v;
  }
  showSettingValues();
}

// ---------- notes from model output ----------

// Copy the model output, zeroing pitches outside [lo, hi] (the library's own
// frequency limit edits its input in place, and we keep the originals to re-run).
function constrained(rows, lo, hi) {
  const a = lo - 21, b = hi - 21 + 1;
  return rows.map((r) => {
    const c = r.slice();
    if (a > 0) c.fill(0, 0, a);
    if (b < c.length) c.fill(0, b);
    return c;
  });
}

function extractNotes() {
  if (!state.frames || state.frames.length === 0) return;
  const s = readSettings();
  const lo = Math.min(s.minf, s.maxf), hi = Math.max(s.minf, s.maxf);
  const minLenFrames = Math.max(1, Math.round((s.minlen / 1000) * FPS));
  let events = outputToNotesPoly(
    constrained(state.frames, lo, hi),
    constrained(state.onsets, lo, hi),
    s.onset, s.frame, minLenFrames,
    true, null, null, s.melodia,
  );
  if (s.bends) events = addPitchBendsToNoteEvents(state.contours, events);
  state.notes = noteFramesToTime(events).sort((x, y) => x.startTimeSeconds - y.startTimeSeconds);
  render();
}

function render() {
  const n = state.notes.length;
  const dur = state.original ? state.original.duration : 0;
  let summary = `${n} note${n === 1 ? '' : 's'} · ${fmtTime(dur)}`;
  if (n) {
    let lo = 127, hi = 0;
    for (const x of state.notes) { lo = Math.min(lo, x.pitchMidi); hi = Math.max(hi, x.pitchMidi); }
    summary += ` · ${noteName(lo)}–${noteName(hi)}`;
  }
  $('summary').textContent = summary;
  roll.setNotes(state.notes, dur);
}

let extractTimer = 0;
function scheduleExtract() {
  showSettingValues();
  clearTimeout(extractTimer);
  extractTimer = setTimeout(() => {
    player.stop();
    updatePlayButtons();
    extractNotes();
  }, 120);
}

// ---------- UI flow ----------

function setStatus(text, pct, isError = false) {
  $('status').hidden = false;
  $('status').classList.toggle('error', isError);
  $('status-text').textContent = text;
  $('status-pct').textContent = pct == null ? '' : `${Math.round(pct * 100)}%`;
  $('bar-fill').style.width = `${Math.round((pct ?? 0) * 100)}%`;
  $('bar-fill').parentElement.hidden = isError;
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

let busy = false;

async function handleFile(file) {
  if (!file || busy) return;
  busy = true;
  player.stop();
  updatePlayButtons();
  $('drop').hidden = true;
  $('result').hidden = true;
  state.fileName = file.name;
  try {
    setStatus('Decoding audio…', 0);
    const original = await decodeFile(file);
    state.original = original;
    const mono = await resampleMono(original, SAMPLE_RATE);

    setStatus('Loading model…', 0);
    const t0 = performance.now();
    const out = await runModel(mono, (p) => {
      const elapsed = (performance.now() - t0) / 1000;
      const eta = p > 0.05 ? elapsed / p - elapsed : null;
      setStatus(`Transcribing… ${eta != null ? `about ${Math.ceil(eta)} s left` : ''} (${tf.getBackend()})`, p);
    });
    state.frames = out.frames;
    state.onsets = out.onsets;
    state.contours = out.contours;

    setStatus('Finding notes…', 1);
    await new Promise((r) => setTimeout(r, 0));
    $('file-name').textContent = file.name;
    $('result').hidden = false;
    extractNotes();
    $('status').hidden = true;
  } catch (err) {
    console.error(err);
    const msg = err && err.name === 'EncodingError'
      ? 'This browser could not decode that file. Try an MP3 or WAV.'
      : `Something went wrong: ${err && err.message ? err.message : err}`;
    setStatus(msg, null, true);
    $('drop').hidden = false;
  } finally {
    busy = false;
  }
}

function download() {
  if (!state.notes.length) return;
  const s = readSettings();
  const bytes = notesToMidi(state.notes, {
    bpm: s.bpm, pitchBends: s.bends, velocityFromAmplitude: s.velocity,
    name: state.fileName.replace(/\.[^.]+$/, ''),
  });
  const blob = new Blob([bytes], { type: 'audio/midi' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (state.fileName.replace(/\.[^.]+$/, '') || 'transcription') + '.mid';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

function updatePlayButtons() {
  $('play').textContent = player.mode === 'midi' ? '■ Stop' : '▶ Play MIDI';
  $('play-orig').textContent = player.mode === 'audio' ? '■ Stop' : '▶ Play original';
}

player.onTick = (t) => roll.setPlayhead(t);
player.onEnd = () => { roll.setPlayhead(null); updatePlayButtons(); };

$('play').addEventListener('click', () => {
  if (player.mode === 'midi') player.stop();
  else player.playNotes(state.notes, roll.seekTime());
  updatePlayButtons();
});
$('play-orig').addEventListener('click', () => {
  if (player.mode === 'audio') player.stop();
  else player.playBuffer(state.original, roll.seekTime());
  updatePlayButtons();
});
roll.onSeek = () => {
  if (player.mode === 'midi') player.playNotes(state.notes, roll.seekTime());
  else if (player.mode === 'audio') player.playBuffer(state.original, roll.seekTime());
};
$('download').addEventListener('click', download);
$('another').addEventListener('click', () => {
  player.stop();
  updatePlayButtons();
  $('result').hidden = true;
  $('drop').hidden = false;
  $('file').value = '';
  $('file').click();
});
$('reset').addEventListener('click', () => { applyDefaults(); scheduleExtract(); });

for (const id of ['onset', 'frame', 'minlen', 'minf', 'maxf', 'melodia', 'bends']) {
  $(id).addEventListener('input', scheduleExtract);
}
$('bpm').addEventListener('input', showSettingValues);

const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); }
});
$('file').addEventListener('change', (e) => handleFile(e.target.files[0]));
for (const ev of ['dragenter', 'dragover']) {
  document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
}
for (const ev of ['dragleave', 'drop']) {
  document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); });
}
document.addEventListener('drop', (e) => {
  const f = e.dataTransfer && e.dataTransfer.files[0];
  if (f) handleFile(f);
});

applyDefaults();

// Exposed for automated tests.
window.__mp3midi = { state, handleFile };
