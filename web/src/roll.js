// Piano-roll view of the transcribed notes. Click to set where playback starts.
const MAX_CANVAS_PX = 32000;
const HEIGHT = 320;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export class PianoRoll {
  constructor(canvas) {
    this.canvas = canvas;
    this.wrap = canvas.parentElement;
    this.inner = document.createElement('div');
    this.inner.className = 'roll-inner';
    this.wrap.replaceChild(this.inner, canvas);
    this.inner.appendChild(canvas);
    this.head = document.createElement('div');
    this.head.className = 'playhead';
    this.head.hidden = true;
    this.inner.appendChild(this.head);
    this.cursor = document.createElement('div');
    this.cursor.className = 'seek-cursor';
    this.cursor.hidden = true;
    this.inner.appendChild(this.cursor);

    this.notes = [];
    this.duration = 0;
    this.pps = 60;
    this.seek = 0;
    this.onSeek = null;

    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      this.seek = Math.max(0, Math.min(this.duration, (e.clientX - r.left) / this.pps));
      this.cursor.style.left = `${this.seek * this.pps}px`;
      this.cursor.hidden = this.seek === 0;
      if (this.onSeek) this.onSeek(this.seek);
    });
    window.addEventListener('resize', () => this.draw());
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => this.draw());
    }
  }

  seekTime() { return this.seek; }

  setNotes(notes, duration) {
    this.notes = notes;
    this.duration = Math.max(duration, 0.5);
    if (this.seek > this.duration) this.seek = 0;
    this.draw();
  }

  draw() {
    const visible = this.wrap.clientWidth || 600;
    const dpr = window.devicePixelRatio || 1;
    this.pps = Math.max(visible / this.duration, Math.min(80, MAX_CANVAS_PX / this.duration / dpr));
    const width = Math.max(visible, Math.ceil(this.duration * this.pps));
    this.inner.style.width = `${width}px`;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${HEIGHT}px`;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(HEIGHT * dpr);
    const g = this.canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    let lo = 127, hi = 0;
    for (const n of this.notes) { lo = Math.min(lo, n.pitchMidi); hi = Math.max(hi, n.pitchMidi); }
    if (lo > hi) { lo = 48; hi = 84; }
    lo = Math.max(0, lo - 2); hi = Math.min(127, hi + 2);
    if (hi - lo < 24) { const mid = (hi + lo) / 2; lo = Math.max(0, Math.round(mid - 12)); hi = lo + 24; }
    const rows = hi - lo + 1;
    const rh = HEIGHT / rows;
    const y = (p) => HEIGHT - (p - lo + 1) * rh;

    g.fillStyle = cssVar('--roll-bg');
    g.fillRect(0, 0, width, HEIGHT);
    const black = new Set([1, 3, 6, 8, 10]);
    g.fillStyle = cssVar('--roll-black');
    for (let p = lo; p <= hi; p++) if (black.has(p % 12)) g.fillRect(0, y(p), width, rh);
    g.fillStyle = cssVar('--roll-grid');
    for (let p = lo; p <= hi; p++) if (p % 12 === 0) g.fillRect(0, y(p) + rh - 1, width, 1);
    const step = this.pps > 30 ? 1 : this.pps > 8 ? 5 : 15;
    for (let t = step; t < this.duration; t += step) g.fillRect(Math.round(t * this.pps), 0, 1, HEIGHT);

    // Notes, coloured by pitch class, brighter when louder.
    for (const n of this.notes) {
      const hue = (n.pitchMidi % 12) * 30;
      const a = 0.45 + 0.55 * Math.min(1, n.amplitude * 1.4);
      g.fillStyle = `hsla(${hue}, 75%, 52%, ${a})`;
      const x = n.startTimeSeconds * this.pps;
      const w = Math.max(2, n.durationSeconds * this.pps - 1);
      g.fillRect(x, y(n.pitchMidi) + 0.5, w, Math.max(1.5, rh - 1));
    }

    // Octave labels down the left edge of each screenful.
    g.fillStyle = cssVar('--muted');
    g.font = '11px system-ui, sans-serif';
    for (let x0 = 0; x0 < width; x0 += visible) {
      for (let p = lo; p <= hi; p++) if (p % 12 === 0) g.fillText(`C${p / 12 - 1}`, x0 + 4, y(p) + rh - 3);
    }
    this.cursor.style.left = `${this.seek * this.pps}px`;
    this.cursor.hidden = this.seek === 0;
  }

  setPlayhead(t) {
    if (t == null) { this.head.hidden = true; return; }
    this.head.hidden = false;
    const x = t * this.pps;
    this.head.style.left = `${x}px`;
    const w = this.wrap;
    if (x < w.scrollLeft || x > w.scrollLeft + w.clientWidth * 0.85) {
      w.scrollLeft = Math.max(0, x - w.clientWidth * 0.15);
    }
  }
}
