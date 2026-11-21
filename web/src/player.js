// Plays the transcription with a small built-in synth, or the original audio.
import { audioContext } from './audio.js';

const LOOKAHEAD = 0.4; // seconds of notes scheduled ahead of the clock

export class Player {
  constructor() {
    this.mode = null; // 'midi' | 'audio' | null
    this.onTick = null;
    this.onEnd = null;
    this._timer = 0;
    this._raf = 0;
    this._nodes = [];
  }

  _begin(mode, offset, duration) {
    this.stop(true);
    const ctx = audioContext();
    if (ctx.state === 'suspended') ctx.resume();
    this.mode = mode;
    this._t0 = ctx.currentTime + 0.05 - offset;
    this._duration = duration;
    this._master = ctx.createGain();
    this._master.gain.value = mode === 'midi' ? 0.25 : 1;
    this._master.connect(ctx.destination);
    const tick = () => {
      const t = ctx.currentTime - this._t0;
      if (t >= this._duration + 0.3) { this.stop(); return; }
      if (this.onTick) this.onTick(Math.max(0, t));
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  playBuffer(buffer, offset = 0) {
    if (!buffer) return;
    if (offset >= buffer.duration) offset = 0;
    this._begin('audio', offset, buffer.duration);
    const ctx = audioContext();
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this._master);
    src.start(this._t0 + offset, offset);
    this._nodes.push(src);
  }

  playNotes(notes, offset = 0) {
    if (!notes.length) return;
    const end = Math.max(...notes.map((n) => n.startTimeSeconds + n.durationSeconds));
    if (offset >= end) offset = 0;
    this._begin('midi', offset, end);
    const ctx = audioContext();
    let i = 0;
    while (i < notes.length && notes[i].startTimeSeconds + notes[i].durationSeconds < offset) i++;
    const pump = () => {
      const horizon = ctx.currentTime - this._t0 + LOOKAHEAD;
      while (i < notes.length && notes[i].startTimeSeconds < horizon) {
        this._voice(notes[i], offset);
        i++;
      }
      this._nodes = this._nodes.filter((n) => n._end > ctx.currentTime);
    };
    pump();
    this._timer = setInterval(pump, 100);
  }

  _voice(n, offset) {
    const ctx = audioContext();
    const start = Math.max(this._t0 + n.startTimeSeconds, this._t0 + offset, ctx.currentTime);
    const stop = this._t0 + n.startTimeSeconds + n.durationSeconds;
    if (stop <= start) return;
    const f = 440 * 2 ** ((n.pitchMidi - 69) / 12);
    const vel = 0.25 + 0.75 * Math.min(1, n.amplitude * 1.3);
    const g = ctx.createGain();
    // Plucked envelope: fast attack, decay to a sustain, short release.
    g.gain.setValueAtTime(0, start);
    g.gain.linearRampToValueAtTime(vel, start + 0.006);
    g.gain.exponentialRampToValueAtTime(vel * 0.35, start + 0.35);
    g.gain.setValueAtTime(vel * 0.35, Math.max(start + 0.35, stop));
    g.gain.exponentialRampToValueAtTime(0.0001, Math.max(start + 0.36, stop) + 0.12);
    const end = Math.max(start + 0.36, stop) + 0.15;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = Math.min(12000, f * 6);
    const o1 = ctx.createOscillator();
    o1.type = 'triangle';
    o1.frequency.value = f;
    const o2 = ctx.createOscillator();
    o2.type = 'sine';
    o2.frequency.value = f * 2;
    const g2 = ctx.createGain();
    g2.gain.value = 0.3;
    o1.connect(lp); o2.connect(g2); g2.connect(lp); lp.connect(g); g.connect(this._master);
    o1.start(start); o2.start(start); o1.stop(end); o2.stop(end);
    o1._end = end; o2._end = end;
    this._nodes.push(o1, o2);
  }

  stop(silent = false) {
    clearInterval(this._timer);
    cancelAnimationFrame(this._raf);
    for (const n of this._nodes) { try { n.stop(); } catch { /* already stopped */ } }
    this._nodes = [];
    if (this._master) { this._master.disconnect(); this._master = null; }
    const was = this.mode;
    this.mode = null;
    if (!silent && was && this.onEnd) this.onEnd();
  }
}
