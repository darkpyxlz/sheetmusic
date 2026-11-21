let ctx = null;

/** A shared AudioContext (created lazily; decoding does not need a user gesture). */
export function audioContext() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}

/** Decode any file the browser understands (MP3, WAV, OGG, FLAC, M4A...). */
export async function decodeFile(file) {
  const data = await file.arrayBuffer();
  return await audioContext().decodeAudioData(data);
}

/** Mix down to mono and resample with an OfflineAudioContext. */
export async function resampleMono(buffer, rate) {
  const length = Math.max(1, Math.ceil(buffer.duration * rate));
  const off = new OfflineAudioContext(1, length, rate);
  const src = off.createBufferSource();
  src.buffer = buffer;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0);
}
