"""Run the transcriber over a benchmark set and print note / rhythm / key metrics.

    python -m benchmarks.run maestro dev                 # real piano
    python -m benchmarks.run corpus dev                  # soundfont-rendered chorales
    python -m benchmarks.run maestro dev --opt vocals=false --opt poly_method=greedy --tag greedy

Results go to benchmarks/results/<tier>_<split>_<tag>.json (notes included, so
error breakdowns need no re-run: python -m benchmarks.breakdown <file>).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

from .datasets import DATA, maestro_reference
from .metrics import frame_chroma_acc, metrics

OUT = os.path.join(os.path.dirname(__file__), "results")


def _run(job):
    tier, x, opts = job
    warnings.filterwarnings("ignore")
    from sheetmusic import TranscribeOptions, transcribe
    pid = x["id"]
    if tier == "maestro":
        full = maestro_reference(pid)
        ref = [(a, e, p) for a, _b, p, _v, e in full]      # pedal-extended offsets
    else:
        spb = 60.0 / x["bpm"]
        full = None
        ref = [(x["lead"] + b * spb, x["lead"] + (b + d) * spb, p) for b, d, p in x["notes"]]
    t0 = time.time()
    r = transcribe(os.path.join(DATA, tier, pid + ".wav"), TranscribeOptions(verbose=False, **opts))
    secs = time.time() - t0
    raw = [(n.start, n.end, n.pitch, n.velocity) for n in r.melody_raw + r.accomp_raw]
    qn = r.melody_notes + r.accomp_notes
    quant = [(float(r.ticks_to_seconds(q.start)), float(r.ticks_to_seconds(q.end)), q.pitch) for q in qn]
    dur = r.info["duration"]
    sc = r.score
    out = dict(id=pid, secs=round(secs, 1), dur=round(dur, 1), has_vocals=bool(r.info["has_vocals"]),
               vev=r.info["vocal_evidence"], ts=r.info["time_signature"], key=r.info["key"],
               fifths=sc.key_fifths, bpm=round(r.info["bpm"] * sc.beat_ticks / 480, 1),
               raw=metrics(ref, [e[:3] for e in raw]), quant=metrics(ref, quant),
               chord=frame_chroma_acc(ref, [e[:3] for e in raw], dur)[0],
               est_polyphony=sum(e[1] - e[0] for e in raw) / dur,
               ref_polyphony=sum(e[1] - e[0] for e in ref) / dur,
               notes=raw, ref=full or ref, info=r.info.get("depth"))
    if tier == "corpus":
        t_down = x["lead"] + x["pickup"] * spb
        pos = (float(r.beat_map.to_beats(t_down)) + r.beat_offset) * sc.beat_ticks
        ph = (pos % sc.measure_ticks) / sc.measure_ticks
        out.update(style=x["style"], tempo_ok=abs(out["bpm"] / x["bpm"] - 1) < 0.04,
                   meter_ok=out["ts"] == x["ts"], key_ok=sc.key_fifths == x["fifths"],
                   downbeat_ok=bool(out["ts"] == x["ts"] and min(ph, 1 - ph) < 0.06))
    return out


def summarise(res, label=""):
    def m(rs, part, k):
        return float(np.mean([r[part][k] for r in rs]))
    groups = {"all": res}
    for r in res:
        if "style" in r:
            groups.setdefault(r["style"], []).append(r)
    for g, rs in groups.items():
        line = (f"{label}{g:8s} n={len(rs):2d}  P {m(rs, 'raw', 'P'):.3f} R {m(rs, 'raw', 'R'):.3f} "
                f"F {m(rs, 'raw', 'F'):.3f}  F100 {m(rs, 'raw', 'F100'):.3f}  Foff {m(rs, 'raw', 'F_off'):.3f}  "
                f"scoreF100 {m(rs, 'quant', 'F100'):.3f}  chord {np.mean([r['chord'] for r in rs]):.3f}  "
                f"on_bias {np.mean([r['raw'].get('on_bias', 0) for r in rs]):+.0f}ms  "
                f"poly est/ref {np.mean([r['est_polyphony'] for r in rs]):.1f}/"
                f"{np.mean([r['ref_polyphony'] for r in rs]):.1f}  "
                f"vocalsFP {sum(r['has_vocals'] for r in rs)}")
        if "tempo_ok" in rs[0]:
            line += "  tempo %d meter %d downbeat %d key %d" % tuple(
                sum(r[k] for r in rs) for k in ("tempo_ok", "meter_ok", "downbeat_ok", "key_ok"))
        line += f"  x{sum(r['secs'] for r in rs) / sum(r['dur'] for r in rs):.2f}rt"
        print(line)


def _val(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", choices=["maestro", "corpus"])
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="TranscribeOptions override (JSON value), repeatable")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    opts = {k: _val(v) for k, v in (o.split("=", 1) for o in a.opt)}
    index = json.load(open(os.path.join(DATA, a.tier, a.split + ".json")))
    jobs = [(a.tier, x, opts) for x in index]
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as p:
            res = p.map(_run, jobs, chunksize=1)
    else:
        res = [_run(j) for j in jobs]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{a.tier}_{a.split}_{a.tag}.json"), "w"))
    if a.verbose:
        for r in res:
            print(r["id"], r.get("style", ""), "voc" if r["has_vocals"] else "ins", r["ts"], r["bpm"], r["key"],
                  "P %.2f R %.2f F %.2f" % (r["raw"]["P"], r["raw"]["R"], r["raw"]["F"]), r["vev"])
    summarise(res, f"{a.tier}/{a.split}/{a.tag}: ")


if __name__ == "__main__":
    sys.exit(main())
