"""Note-level metrics in seconds (mir_eval conventions: onset +-50 ms, offset max(50 ms, 20 %))."""
import numpy as np

def match(ref, est, onset_tol=0.05, pitch_class=False):
    """ref/est: arrays (n,3) = onset, offset, pitch. Returns list of (i_ref, j_est)."""
    pairs = []
    key = (lambda p: int(p) % 12) if pitch_class else (lambda p: int(p))
    byp = {}
    for j, e in enumerate(est):
        byp.setdefault(key(e[2]), []).append(j)
    used = set()
    for i in np.argsort([r[0] for r in ref]):
        r = ref[i]
        best = None
        for j in byp.get(key(r[2]), ()):
            if j in used:
                continue
            d = abs(est[j][0] - r[0])
            if d <= onset_tol and (best is None or d < best[0]):
                best = (d, j)
        if best:
            used.add(best[1]); pairs.append((i, best[1]))
    return pairs

def prf(nm, nr, ne):
    p = nm / ne if ne else 0.0
    r = nm / nr if nr else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def metrics(ref, est):
    ref = np.asarray(ref, float).reshape(-1, 3); est = np.asarray(est, float).reshape(-1, 3)
    out = {}
    pairs = match(ref, est, 0.05)
    out["P"], out["R"], out["F"] = prf(len(pairs), len(ref), len(est))
    # with offsets
    ok = [(i, j) for i, j in pairs
          if abs(est[j][1] - ref[i][1]) <= max(0.05, 0.2 * (ref[i][1] - ref[i][0]))]
    out["F_off"] = prf(len(ok), len(ref), len(est))[2]
    # looser onset window (what survives quantisation / window smear)
    wide = match(ref, est, 0.10)
    out["P100"], out["R100"], out["F100"] = prf(len(wide), len(ref), len(est))
    out["F_chroma"] = prf(len(match(ref, est, 0.05, True)), len(ref), len(est))[2]
    if wide:
        do = np.array([est[j][0] - ref[i][0] for i, j in wide]) * 1000
        dn = np.array([est[j][1] - ref[i][1] for i, j in wide]) * 1000
        dd = np.array([(est[j][1] - est[j][0]) / max(1e-3, ref[i][1] - ref[i][0]) for i, j in wide])
        out.update(on_bias=float(np.mean(do)), on_mad=float(np.mean(np.abs(do))),
                   off_mad=float(np.median(np.abs(dn))), dur_ratio=float(np.median(dd)))
    out["n_ref"], out["n_est"] = len(ref), len(est)
    return out

def frame_chroma_acc(ref, est, dur, fps=20):
    """'Chord' accuracy: per-frame pitch-class-set Jaccard between ref and est piano rolls."""
    T = int(dur * fps) + 1
    def roll(ns):
        R = np.zeros((T, 12), bool)
        for a, b, p in ns:
            R[int(a * fps):max(int(a * fps) + 1, int(b * fps)), int(p) % 12] = True
        return R
    A, B = roll(ref), roll(est)
    act = A.any(1) | B.any(1)
    inter = (A & B).sum(1)[act]; union = (A | B).sum(1)[act]
    return float(np.mean(inter / np.maximum(union, 1))), float(np.mean((A == B).all(1)[act]))


def note_values(ref_beats, qnotes, to_seconds, lead, spb, tol=0.10):
    """Written note values against the score (soundfont tier, tempo known).

    ref_beats: [(onset_beat, dur_beats, pitch)] in crotchets; qnotes: [(start_tick, end_tick, pitch)]
    with 480 ticks per crotchet.  A written note is matched to a score note of the same pitch whose
    onset is within ``tol`` seconds.  Returns the share of matched notes with exactly the right
    value, too short (< 3/4) and too long (> 3/2), and the F-measure of notes that have the right
    pitch, onset *and* value."""
    ref = [(lead + b * spb, d, p) for b, d, p in ref_beats]
    est = [(float(to_seconds(a)), (b - a) / 480.0, p) for a, b, p in qnotes]
    pairs = match([(o, o, p) for o, _d, p in ref], [(o, o, p) for o, _d, p in est], tol)
    if not pairs:
        return dict(value_exact=0.0, too_short=0.0, too_long=0.0, value_F=0.0, short_share=0.0,
                    short_share_ref=0.0)
    ratio = np.array([est[j][1] / ref[i][1] for i, j in pairs])
    exact = np.abs(ratio - 1.0) < 1e-6
    return dict(value_exact=float(exact.mean()), too_short=float((ratio < 0.75).mean()),
                too_long=float((ratio > 1.5).mean()),
                value_F=prf(int(exact.sum()), len(ref), len(est))[2],
                # how much of the page is semiquavers or shorter, against the score
                short_share=float(np.mean([d <= 0.25 + 1e-9 for _o, d, _p in est])),
                short_share_ref=float(np.mean([d <= 0.25 + 1e-9 for _o, d, _p in ref])))


def ts_class(ts: str):
    """(grouping, subdivision) of a time signature: 2/4, 4/4 and 2/2 are all
    ("duple", "simple"); 6/8 and 12/8 ("duple", "compound"); 3/4 and 3/8
    ("triple", "simple"); 9/8 ("triple", "compound")."""
    n, d = (int(x) for x in ts.split("/"))
    if (d >= 8 and n % 3 == 0 and n > 3) or (n, d) == (6, 4):
        return ("triple" if n == 9 else "duple", "compound")
    return ("triple" if n % 3 == 0 else "duple", "simple")


def event_f(ref, est, tol=0.07):
    """F-measure of event times (beats, downbeats) matched one-to-one within ``tol`` s."""
    ref, est = sorted(ref), sorted(est)
    if not ref or not est:
        return 0.0
    used, hit, j0 = set(), 0, 0
    for t in ref:
        best = None
        while j0 < len(est) and est[j0] < t - tol:
            j0 += 1
        j = j0
        while j < len(est) and est[j] <= t + tol:
            if j not in used and (best is None or abs(est[j] - t) < abs(est[best] - t)):
                best = j
            j += 1
        if best is not None:
            used.add(best)
            hit += 1
    return prf(hit, len(ref), len(est))[2]


def beat_f_any_level(ref, est, tol=0.07):
    """Best beat F over metrical levels of the estimate (x2, x3, /2, /3):
    separates 'right pulse, wrong level' from 'lost the pulse'."""
    est = np.asarray(sorted(est), float)
    cands = [est]
    if est.size >= 2:
        mid = 0.5 * (est[:-1] + est[1:])
        cands.append(np.sort(np.concatenate([est, mid])))
        third = np.concatenate([est, est[:-1] + np.diff(est) / 3, est[:-1] + 2 * np.diff(est) / 3])
        cands.append(np.sort(third))
        cands += [est[k::2] for k in range(2)] + [est[k::3] for k in range(3)]
    return max(event_f(ref, list(c), tol) for c in cands)


def outer_line(notes, which="top", together=0.03):
    """The top (tune) or bottom (bass) line of a polyphonic part: at each onset
    the highest / lowest note, kept only if it is above / below everything
    still sounding then.  notes: [(onset, offset, pitch)]."""
    ns = sorted(notes)
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1][0] - ns[i][0] <= together:
            j += 1
        grp = ns[i:j + 1]
        cand = max(grp, key=lambda n: n[2]) if which == "top" else min(grp, key=lambda n: n[2])
        t = ns[i][0]
        sounding = [n[2] for n in ns[:i] if n[1] > t + 0.05]
        if not sounding or (cand[2] >= max(sounding) if which == "top" else cand[2] <= min(sounding)):
            out.append(cand)
        i = j + 1
    return out


def line_recall(ref, est, which="top", tol=0.10):
    """Share of the reference's top or bottom line that the estimate contains
    (same pitch, onset within ``tol`` s): does an arrangement keep the tune
    and the bass?"""
    line = outer_line(ref, which)
    if not line:
        return 0.0
    return len(match(line, est, tol)) / len(line)
