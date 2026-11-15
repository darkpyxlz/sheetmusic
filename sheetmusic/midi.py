"""Minimal Standard MIDI File writer (for listening back to a transcription)."""

from __future__ import annotations

import struct

from .notes import QNote


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def write_midi(path: str, tracks: list[tuple[str, int, list[QNote]]], bpm: float,
               division: int = 480, time_sig: tuple[int, int] = (4, 4)) -> None:
    """tracks: list of (name, gm_program, notes)."""
    chunks = []
    # conductor track
    tempo = int(round(60_000_000 / bpm))
    num, den = time_sig
    ev = b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big")
    ev += b"\x00\xff\x58\x04" + bytes([num, den.bit_length() - 1, 24, 8])
    ev += b"\x00\xff\x2f\x00"
    chunks.append(ev)
    for ch, (name, program, notes) in enumerate(tracks):
        ch = ch if ch < 9 else ch + 1
        events = []
        for q in notes:
            events.append((q.start, 1, bytes([0x90 | ch, q.pitch & 0x7F, max(1, min(127, q.velocity))])))
            events.append((q.end, 0, bytes([0x80 | ch, q.pitch & 0x7F, 0])))
        events.sort(key=lambda e: (e[0], e[1]))
        data = b"\x00\xff\x03" + _vlq(len(name.encode())) + name.encode()
        data += b"\x00" + bytes([0xC0 | ch, program & 0x7F])
        t = 0
        for tick, _, msg in events:
            data += _vlq(tick - t) + msg
            t = tick
        data += b"\x00\xff\x2f\x00"
        chunks.append(data)
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), division))
        for c in chunks:
            f.write(b"MTrk" + struct.pack(">I", len(c)) + c)
