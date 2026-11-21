import { Midi } from '@tonejs/midi';

const BINS_PER_SEMITONE = 3;   // Basic Pitch contour resolution
const BEND_RANGE = 2;          // semitones, the General MIDI default

/**
 * @param {Array<{startTimeSeconds:number,durationSeconds:number,pitchMidi:number,amplitude:number,pitchBends?:number[]}>} notes
 * @param {{bpm:number, pitchBends:boolean, velocityFromAmplitude:boolean, name?:string}} opts
 * @returns {Uint8Array}
 */
export function notesToMidi(notes, opts) {
  const midi = new Midi();
  midi.header.setTempo(opts.bpm);
  midi.header.timeSignatures.push({ ticks: 0, timeSignature: [4, 4] });
  if (opts.name) midi.header.name = opts.name;

  const velocity = (n) => opts.velocityFromAmplitude
    ? Math.min(1, Math.max(0.08, n.amplitude))
    : 0.8;

  if (!opts.pitchBends) {
    const track = midi.addTrack();
    track.name = 'Piano';
    track.channel = 0;
    track.instrument.number = 0;
    for (const n of notes) {
      track.addNote({ midi: n.pitchMidi, time: n.startTimeSeconds, duration: n.durationSeconds, velocity: velocity(n) });
    }
    return midi.toArray();
  }

  // Pitch bend affects a whole channel, so overlapping notes need separate
  // channels. Assign each note to the first channel that is free at its start.
  const channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]; // skip drums (9)
  const tracks = [];
  const freeAt = [];
  for (const n of notes) {
    let k = freeAt.findIndex((t) => t <= n.startTimeSeconds + 1e-6);
    if (k < 0) {
      if (tracks.length < channels.length) {
        k = tracks.length;
        const tr = midi.addTrack();
        tr.name = `Voice ${k + 1}`;
        tr.channel = channels[k];
        tr.instrument.number = 0;
        tracks.push(tr);
        freeAt.push(0);
      } else {
        k = freeAt.indexOf(Math.min(...freeAt)); // more than 15 voices: share the earliest
      }
    }
    const tr = tracks[k];
    const end = n.startTimeSeconds + n.durationSeconds;
    freeAt[k] = end;
    tr.addPitchBend({ time: n.startTimeSeconds, value: 0 });
    const bends = n.pitchBends || [];
    for (let i = 0; i < bends.length; i++) {
      const t = n.startTimeSeconds + (bends.length > 1 ? (i / (bends.length - 1)) * n.durationSeconds : 0);
      const v = bends[i] / BINS_PER_SEMITONE / BEND_RANGE;
      tr.addPitchBend({ time: t, value: Math.max(-1, Math.min(1, v)) });
    }
    tr.addNote({ midi: n.pitchMidi, time: n.startTimeSeconds, duration: n.durationSeconds, velocity: velocity(n) });
  }
  return midi.toArray();
}
