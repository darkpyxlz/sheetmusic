// Runs the Basic Pitch model over the audio in 2-second windows. This mirrors
// BasicPitch.evaluateModel from @spotify/basic-pitch, but frees every tensor
// as it goes so long songs do not run out of GPU memory.
import * as tf from '@tensorflow/tfjs';

export const SAMPLE_RATE = 22050;
const FFT_HOP = 256;
export const FPS = Math.floor(SAMPLE_RATE / FFT_HOP);
const WINDOW_SAMPLES = SAMPLE_RATE * 2 - FFT_HOP;
const OVERLAP_FRAMES = 30;
const HALF_OVERLAP = OVERLAP_FRAMES / 2;
const OVERLAP_SAMPLES = OVERLAP_FRAMES * FFT_HOP;
const HOP = WINDOW_SAMPLES - OVERLAP_SAMPLES;
const OUTPUTS = ['Identity_1', 'Identity_2', 'Identity']; // frames, onsets, contours

let modelPromise = null;

async function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      try { await tf.setBackend('webgl'); } catch { /* fall through to CPU */ }
      await tf.ready();
      return tf.loadGraphModel(new URL('model/model.json', document.baseURI).href);
    })();
    modelPromise.catch(() => { modelPromise = null; });
  }
  return modelPromise;
}

/**
 * @param {Float32Array} audio mono samples at SAMPLE_RATE
 * @param {(p:number)=>void} onProgress
 * @returns {Promise<{frames:number[][], onsets:number[][], contours:number[][]}>}
 */
export async function runModel(audio, onProgress) {
  const model = await loadModel();
  const nFrames = Math.floor(audio.length * (FPS / SAMPLE_RATE));

  // Pad the front by half an overlap, then cut into overlapping windows.
  const padded = new Float32Array(OVERLAP_SAMPLES / 2 + audio.length);
  padded.set(audio, OVERLAP_SAMPLES / 2);
  const nWindows = Math.max(1, Math.ceil(padded.length / HOP));

  const frames = [], onsets = [], contours = [];
  for (let i = 0; i < nWindows && frames.length < nFrames; i++) {
    onProgress(i / nWindows);
    const win = new Float32Array(WINDOW_SAMPLES);
    win.set(padded.subarray(i * HOP, Math.min(padded.length, i * HOP + WINDOW_SAMPLES)));

    const results = tf.tidy(() => {
      const input = tf.tensor3d(win, [1, WINDOW_SAMPLES, 1]);
      const outs = model.execute(input, OUTPUTS);
      return outs.map((t) => {
        const kept = t.slice([0, HALF_OVERLAP, 0], [-1, t.shape[1] - 2 * HALF_OVERLAP, -1]);
        return kept.reshape([kept.shape[1], kept.shape[2]]);
      });
    });
    const arrays = await Promise.all(results.map((t) => t.array()));
    results.forEach((t) => t.dispose());

    const take = Math.min(arrays[0].length, nFrames - frames.length);
    for (let j = 0; j < take; j++) {
      frames.push(arrays[0][j]);
      onsets.push(arrays[1][j]);
      contours.push(arrays[2][j]);
    }
    await tf.nextFrame();
  }
  onProgress(1);
  return { frames, onsets, contours };
}
