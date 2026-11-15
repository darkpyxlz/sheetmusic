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
    out["F100"] = prf(len(wide), len(ref), len(est))[2]
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
