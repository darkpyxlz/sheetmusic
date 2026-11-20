"""Markdown tables from saved benchmark results (benchmarks/results/*.json).

    python -m benchmarks.report test before fin_full fin_playable fin_easy

prints, for the given split and tags, note accuracy, metre and rhythm, note
values and playability, one row per tag.  Missing result files are skipped.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from .run import OUT


def _load(tier, split, tag):
    p = os.path.join(OUT, f"{tier}_{split}_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _m(rs, part, key):
    v = [r[part][key] for r in rs if part in r and key in r[part]]
    return float(np.mean(v)) if v else float("nan")


def _hand(rs, hand, key, agg=np.mean):
    v = [r["playability"][hand][key] for r in rs
         if r.get("playability") and r["playability"][hand].get("notes")]
    return float(agg(v)) if v else float("nan")


def notes_table(split, tags):
    print("\n| set | version | precision | recall | note F | written notes right (P) | tune kept | bass kept |")
    print("|---|---|---|---|---|---|---|---|")
    for tier in ("maestro", "corpus", "mixed"):
        for tag in tags:
            rs = _load(tier, split, tag)
            if not rs:
                continue
            print("| %s | %s | %.3f | %.3f | %.3f | %.3f | %s | %s |" % (
                tier, tag, _m(rs, "raw", "P"), _m(rs, "raw", "R"), _m(rs, "raw", "F"),
                _m(rs, "quant", "P100"),
                "%.3f" % np.mean([r["melody_recall"] for r in rs]) if "melody_recall" in rs[0] else "–",
                "%.3f" % np.mean([r["bass_recall"] for r in rs]) if "bass_recall" in rs[0] else "–"))


def rhythm_table(split, tags):
    print("\n| set | version | time signature | simple/compound | tempo | downbeat | key signature |")
    print("|---|---|---|---|---|---|---|")
    for tier in ("asap", "mixed", "corpus"):
        for tag in tags:
            rs = _load(tier, split, tag)
            if not rs:
                continue
            n = len(rs)
            if tier == "asap":
                ks = [r["key_ok"] for r in rs if r["key_ok"] is not None]
                print("| %s | %s | %d/%d | %d/%d | %d/%d | F %.3f | %d/%d |" % (
                    tier, tag, sum(r["meter_ok"] for r in rs), n, sum(r["class_ok"] for r in rs), n,
                    sum(r["tempo_ok"] for r in rs), n, np.mean([r["downbeat_f"] for r in rs]),
                    sum(ks), len(ks)))
            else:
                print("| %s | %s | %d/%d | %d/%d | %d/%d | %d/%d | %d/%d |" % (
                    tier, tag, sum(r["meter_ok"] for r in rs), n, sum(r["class_ok"] for r in rs), n,
                    sum(r["tempo_ok"] for r in rs), n, sum(r["downbeat_ok"] for r in rs), n,
                    sum(r["key_ok"] for r in rs), n))


def values_table(split, tags):
    print("\n| set | version | value exact | too short | too long | semiquavers or shorter (score) |")
    print("|---|---|---|---|---|---|")
    for tier in ("corpus", "mixed"):
        for tag in tags:
            rs = _load(tier, split, tag)
            if not rs:
                continue
            vs = [r["values"] for r in rs if "values" in r]
            ss = [r["short_share"] for r in rs if "short_share" in r]
            sr = [r["short_share_ref"] for r in rs if "short_share_ref" in r]
            print("| %s | %s | %.2f | %.2f | %.2f | %s |" % (
                tier, tag, *(float(np.mean([v[k] for v in vs])) for k in ("value_exact", "too_short", "too_long")),
                ("%.2f (%.2f)" % (np.mean(ss), np.mean(sr))) if ss and sr else "–"))


def playability_table(split, tags):
    print("\n| set | version | right hand: notes/s, busiest s, held (95 %), span (95 %) "
          "| left hand: notes/s, busiest s, held (95 %), span (95 %) |")
    print("|---|---|---|---|")
    for tier in ("maestro", "corpus", "mixed"):
        for tag in tags:
            rs = _load(tier, split, tag)
            if not rs or not rs[0].get("playability"):
                continue
            cells = []
            for h in ("right", "left"):
                cells.append("%.1f, %.1f, %.1f, %.1f" % (_hand(rs, h, "onsets_per_s"), _hand(rs, h, "busiest_second"),
                                                         _hand(rs, h, "held_p95"), _hand(rs, h, "span_p95")))
            print("| %s | %s | %s | %s |" % (tier, tag, *cells))


if __name__ == "__main__":
    split, tags = sys.argv[1], sys.argv[2:]
    notes_table(split, tags)
    rhythm_table(split, tags)
    values_table(split, tags)
    playability_table(split, tags)
