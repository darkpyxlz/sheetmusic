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
from .metrics import (beat_f_any_level, event_f, frame_chroma_acc, line_recall, metrics, note_values,
                      ts_class)

OUT = os.path.join(os.path.dirname(__file__), "results")


def _run(job):
    tier, x, opts = job
    warnings.filterwarnings("ignore")
    from sheetmusic import TranscribeOptions, transcribe
    pid = x["id"]
    full = None
    if tier == "maestro":
        full = maestro_reference(pid)
        ref = [(a, e, p) for a, _b, p, _v, e in full]      # pedal-extended offsets
    elif tier == "asap":
        ref = []
    else:
        spb = 60.0 / x["bpm"]
        ref = [(x["lead"] + b * spb, x["lead"] + (b + d) * spb, p) for b, d, p in x["notes"]]
    t0 = time.time()
    r = transcribe(os.path.join(DATA, tier, pid + ".wav"), TranscribeOptions(verbose=False, **opts))
    secs = time.time() - t0
    raw = [(n.start, n.end, n.pitch, n.velocity, bool(n.struck),
            None if n.decay != n.decay else round(n.decay, 1)) for n in r.melody_raw + r.accomp_raw]
    qn = [q for st in (s_ for p_ in r.score.parts for s_ in p_.staves) for q in st.notes]   # as written
    quant = [(float(r.ticks_to_seconds(q.start)), float(r.ticks_to_seconds(q.end)), q.pitch) for q in qn]
    dur = r.info["duration"]
    sc = r.score
    out = dict(id=pid, secs=round(secs, 1), dur=round(dur, 1), has_vocals=bool(r.info["has_vocals"]),
               vev=r.info["vocal_evidence"], ts=r.info["time_signature"], key=r.info["key"],
               fifths=sc.key_fifths, bpm=round(r.info["bpm"] * sc.beat_ticks / 480, 1),
               notes=raw, ref=full or ref, info=r.info.get("depth"),
               qnotes=[(q.start, q.end, q.pitch) for q in qn],
               beatmap=dict(times=[round(float(t), 4) for t in r.beat_map.beat_times],
                            period=r.beat_map.period, offset=r.beat_offset, beat_ticks=sc.beat_ticks,
                            measure_ticks=sc.measure_ticks),
               meter_scores=r.info.get("meter_scores"))
    # how much of the page is semiquavers or shorter (the score's own share where known)
    out["short_share"] = float(np.mean([q.end - q.start <= 120 for q in qn])) if qn else 0.0
    if tier in ("corpus", "mixed"):
        out["short_share_ref"] = float(np.mean([d <= 0.25 + 1e-9 for _b, d, _p in x["notes"]]))
    out["playability"] = r.info.get("playability")
    if ref:
        out.update(melody_recall=line_recall(ref, quant, "top"), bass_recall=line_recall(ref, quant, "bottom"))
        out.update(raw=metrics(ref, [e[:3] for e in raw]), quant=metrics(ref, quant),
                   chord=frame_chroma_acc(ref, [e[:3] for e in raw], dur)[0],
                   est_polyphony=sum(e[1] - e[0] for e in raw) / dur,
                   ref_polyphony=sum(e[1] - e[0] for e in ref) / dur)
    if tier in ("corpus", "mixed"):
        t_down = x["lead"] + x["pickup"] * spb
        pos = (float(r.beat_map.to_beats(t_down)) + r.beat_offset) * sc.beat_ticks
        ph = (pos % sc.measure_ticks) / sc.measure_ticks
        out.update(style=x["style"], ts_true=x["ts"], tempo_ok=abs(out["bpm"] / x["bpm"] - 1) < 0.04,
                   tempo_ratio=round(out["bpm"] / x["bpm"], 3),
                   meter_ok=out["ts"] == x["ts"], class_ok=ts_class(out["ts"]) == ts_class(x["ts"]),
                   key_ok=sc.key_fifths == x["fifths"],
                   downbeat_ok=bool(out["ts"] == x["ts"] and min(ph, 1 - ph) < 0.06))
        if out["tempo_ok"]:
            out["values"] = note_values(x["notes"], out["qnotes"], r.ticks_to_seconds, x["lead"], spb)
    if tier == "asap":
        beats = list(r.beat_map.beat_times)
        n_bars = int(dur * 480 / 60 * out["bpm"] / sc.measure_ticks) + 3
        downs = [float(r.ticks_to_seconds(k * sc.measure_ticks)) for k in range(n_bars)]
        downs = [t for t in downs if 0 <= t <= dur]
        true_period = float(np.median(np.diff(x["beats"])))
        ratio = true_period / r.beat_map.period          # estimated tempo / annotated tempo
        out.update(style=x["ts"], ts_true=x["ts"], meter_ok=out["ts"] == x["ts"],
                   class_ok=ts_class(out["ts"]) == ts_class(x["ts"]),
                   tempo_ratio=round(ratio, 3), tempo_ok=abs(ratio - 1) < 0.08,
                   beat_f=event_f(x["beats"], beats), beat_f_any=beat_f_any_level(x["beats"], beats),
                   downbeat_f=event_f(x["downbeats"], downs),
                   key_ok=(None if x["fifths"] is None or x["key_changes"] else sc.key_fifths == x["fifths"]))
    return out


def summarise(res, label=""):
    def m(rs, part, k):
        return float(np.mean([r[part][k] for r in rs]))
    groups = {"all": res}
    for r in res:
        if "style" in r:
            groups.setdefault(r["style"], []).append(r)
    for g, rs in groups.items():
        line = f"{label}{g:9s} n={len(rs):2d}"
        if "raw" in rs[0]:
            line += (f"  P {m(rs, 'raw', 'P'):.3f} R {m(rs, 'raw', 'R'):.3f} "
                     f"F {m(rs, 'raw', 'F'):.3f}  F100 {m(rs, 'raw', 'F100'):.3f}  Foff {m(rs, 'raw', 'F_off'):.3f}  "
                     f"scoreF100 {m(rs, 'quant', 'F100'):.3f}  chord {np.mean([r['chord'] for r in rs]):.3f}  "
                     f"on_bias {np.mean([r['raw'].get('on_bias', 0) for r in rs]):+.0f}ms  "
                     f"poly est/ref {np.mean([r['est_polyphony'] for r in rs]):.1f}/"
                     f"{np.mean([r['ref_polyphony'] for r in rs]):.1f}")
        line += f"  vocalsFP {sum(r['has_vocals'] for r in rs)}"
        if "beat_f" in rs[0]:
            ks = [r["key_ok"] for r in rs if r["key_ok"] is not None]
            line += ("  meter %d class %d tempo %d  beatF %.3f (any level %.3f) downbeatF %.3f  key %d/%d" % (
                sum(r["meter_ok"] for r in rs), sum(r["class_ok"] for r in rs), sum(r["tempo_ok"] for r in rs),
                np.mean([r["beat_f"] for r in rs]), np.mean([r["beat_f_any"] for r in rs]),
                np.mean([r["downbeat_f"] for r in rs]), sum(ks), len(ks)))
        elif "tempo_ok" in rs[0]:
            line += "  tempo %d meter %d class %d downbeat %d key %d" % tuple(
                sum(r[k] for r in rs) for k in ("tempo_ok", "meter_ok", "class_ok", "downbeat_ok", "key_ok"))
        if "melody_recall" in rs[0]:
            line += "  | written P100 %.3f R100 %.3f tune %.3f bass %.3f" % (
                m(rs, "quant", "P100"), m(rs, "quant", "R100"),
                np.mean([r["melody_recall"] for r in rs]), np.mean([r["bass_recall"] for r in rs]))
        pl = [r["playability"] for r in rs if r.get("playability")]
        if pl:
            def hp(h, k):
                v = [p_[h][k] for p_ in pl if p_[h].get("notes")]
                return float(np.mean(v)) if v else 0.0
            line += "  | RH %.1f/s busiest %.1f held95 %.1f span %.1f  LH %.1f/s busiest %.1f held95 %.1f span %.1f" % (
                hp("right", "onsets_per_s"), hp("right", "busiest_second"), hp("right", "held_p95"), hp("right", "span_p95"),
                hp("left", "onsets_per_s"), hp("left", "busiest_second"), hp("left", "held_p95"), hp("left", "span_p95"))
        line += "  16ths %.2f" % np.mean([r["short_share"] for r in rs])
        if "short_share_ref" in rs[0]:
            line += " (score %.2f)" % np.mean([r["short_share_ref"] for r in rs])
        vs = [r["values"] for r in rs if "values" in r]
        if vs:
            line += ("  | values (right tempo only) exact %.2f short %.2f long %.2f valueF %.3f" % tuple(
                float(np.mean([v[k] for v in vs])) for k in
                ("value_exact", "too_short", "too_long", "value_F")))
        line += f"  x{sum(r['secs'] for r in rs) / sum(r['dur'] for r in rs):.2f}rt"
        print(line)


def _val(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", choices=["maestro", "corpus", "mixed", "asap"])
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
            print(r["id"], r.get("style", ""), "voc" if r["has_vocals"] else "ins", r["ts"],
                  "(true %s)" % r["ts_true"] if "ts_true" in r else "", r["bpm"],
                  "x%.2f" % r["tempo_ratio"] if "tempo_ratio" in r else "", r["key"],
                  "P %.2f R %.2f F %.2f" % (r["raw"]["P"], r["raw"]["R"], r["raw"]["F"]) if "raw" in r else "",
                  "beatF %.2f/%.2f db %.2f" % (r["beat_f"], r["beat_f_any"], r["downbeat_f"]) if "beat_f" in r else "",
                  ("values %.2f" % r["values"]["value_exact"]) if "values" in r else "")
    summarise(res, f"{a.tier}/{a.split}/{a.tag}: ")


if __name__ == "__main__":
    sys.exit(main())
