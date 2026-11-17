"""Learned stoppage detector: anchors + clean-label model + HMM, decoded as one continuous path.

It replaces the calm-run flow (Flow B) for every dead frame that is not inside a set-piece
segment. Set pieces, kickoffs and the second-half wait keep the rule flow in dead_ball.py.

  1. Anchors    frames the ball track and the touches PROVE dead (a tag-confirmed exit, the wait
                from it to a confirmed strike, the 3 s before a kickoff strike, a ball resting
                3 s outside the penalty areas) or live (4 s of a ball being passed around).
                Fixed votes of +-ANCHOR log-odds.
  2. Evidence   everywhere else, the average log-odds of
                  - the clean-label model: a LightGBM on player features, trained only on the
                    anchors (labels made from the tracking by dead_ball_ml.anchors), and
                  - an unsupervised 2-state HMM on 23 player-movement features whose "stopped"
                    state is the slowest one.
  3. Decode     one best dead/live path per half that pays SWITCH log-odds every time it changes,
                then dead spells under 5 s are dropped and live gaps under 2 s are closed. Dead is
                continuous; a brief contradiction cannot split a stoppage.
  4. Guard      label-free checks against this match's own anchors. If the clean-label model, the
                HMM or the dead share looks wrong, stoppage_dead() reports ok=False and
                dead_ball.py falls back to the calm-run flow.

Training is train_dead_ball_ml.py, run once; the models live in solution/models/dead_ball_ml/.
On the match they were trained on, the evidence is cross-fitted (each block is scored by a model
that never saw it) and cached; on any other match the stored models are applied as they are.
See solution_documentation/DEAD_BALL_ML.md.
"""

import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd

import dead_ball_features
import seq_net
import stoppage

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "dead_ball_ml")
CACHE_DIR = os.path.join(HERE, "cache")
FPS = 25
EXIT_TYPES = ("throw-in", "corner-kick", "goal-kick")
KICKOFF_FRAMES = 75
REST_FRAMES, REST_RADIUS = 75, 0.7        # 3 s within 0.7 m
PLAY_FRAMES, PLAY_TRACKED, PLAY_INSIDE, PLAY_PATH, PLAY_TOUCHES = 100, 0.9, 1.0, 10.0, 2
BOX_D, BOX_HW = 16.5, 20.16
HMM_STEP = 5                              # the HMM runs at 5 frames per second
CLIP = (0.02, 0.98)
ANCHOR = 8.0            # log-odds an anchored frame carries
SWITCH = 60.0           # log-odds paid per dead/live change (about 2.4 s of firm evidence each way)
MIN_SPELL, CLOSE_GAP = 125, 50   # 5 s, 2 s
CLEAN_PARAMS = dict(objective="binary", is_unbalance=True, num_leaves=15, learning_rate=0.05,
                    n_estimators=300, min_child_samples=30, colsample_bytree=0.4, subsample=0.8,
                    subsample_freq=1, reg_lambda=5, verbose=-1)
BLOCK, GAP, TRAIN_STEP = 3000, 250, 3   # 2-minute blocks, 10 s gaps, every 3rd frame
# the guard: label-free checks against the match's own anchors, set from this match's values
# with a margin (0.93 / 0.88 / 44% here)
CHECKS = dict(min_dead_anchors=1500, min_live_anchors=5000, min_clean_auc=0.85, min_hmm_auc=0.75, min_gru_auc=0.80,
              dead_share=(0.20, 0.60))
DURATION_COLUMNS = ("h_still", "h_since", "h_until")


# ----------------------------------------------------------------------------- anchors

def anchors(ball, touches, segments, exits, length, width):
    """(dead, alive, source) boolean arrays over every video frame, plus a per-source count."""
    n = len(ball)
    src = {k: np.zeros(n, bool) for k in ("exit", "wait", "kickoff", "resting")}
    by_first = {ex["first"]: ex for ex in exits}
    for s in segments:
        if s["rule"] in EXIT_TYPES and s.get("exit_first") is not None:
            ex = by_first.get(s["exit_first"])
            if ex is None or ex["projected"] or ex.get("near"):
                continue
            src["exit"][ex["first"]:ex["last"] + 1] = True
            if "strike" in s.get("triggers", []):
                src["wait"][ex["first"]:s["end"] + 1] = True
        elif s["rule"] == "kickoff":
            src["kickoff"][max(s["start"], s["end"] + 1 - KICKOFF_FRAMES):s["end"] + 1] = True

    x, y = ball[:, 0], ball[:, 1]
    seen = ~np.isnan(x)
    in_box = (((x <= BOX_D) | (x >= length - BOX_D)) & (np.abs(y - width / 2) <= BOX_HW))
    for t in range(0, n - REST_FRAMES, 5):
        w = slice(t, t + REST_FRAMES)
        if seen[w].mean() < 0.9 or in_box[w][seen[w]].any():
            continue
        cx, cy = np.nanmedian(x[w]), np.nanmedian(y[w])
        if np.nanmax(np.hypot(x[w] - cx, y[w] - cy)) <= REST_RADIUS:
            src["resting"][w] = True

    step = np.zeros(n)
    step[1:] = np.nan_to_num(np.hypot(np.diff(x), np.diff(y)))
    cs_step = np.concatenate([[0], np.cumsum(step)])
    cs_seen = np.concatenate([[0], np.cumsum(seen)])
    inside = seen & (x >= PLAY_INSIDE) & (x <= length - PLAY_INSIDE) & (y >= PLAY_INSIDE) & (y <= width - PLAY_INSIDE)
    cs_in = np.concatenate([[0], np.cumsum(inside)])
    tf = np.array([t["frame"] for t in touches if t["team"] is not None])
    cs_t = np.concatenate([[0], np.cumsum(np.bincount(tf, minlength=n)[:n])])
    alive = np.zeros(n, bool)
    for t in range(0, n - PLAY_FRAMES, 5):
        e = t + PLAY_FRAMES
        if (cs_seen[e] - cs_seen[t]) / PLAY_FRAMES < PLAY_TRACKED:
            continue
        if cs_in[e] - cs_in[t] != cs_seen[e] - cs_seen[t]:
            continue
        if cs_step[e] - cs_step[t] >= PLAY_PATH and cs_t[e] - cs_t[t] >= PLAY_TOUCHES:
            alive[t:e] = True
    dead = np.zeros(n, bool)
    for v in src.values():
        dead |= v
    both = dead & alive
    dead[both] = alive[both] = False
    return dead, alive, src, int(both.sum())




# ----------------------------------------------------------------------------- models

def available():
    return os.path.isfile(os.path.join(MODEL_DIR, "config.json"))


def load():
    import lightgbm as lgb
    with open(os.path.join(MODEL_DIR, "config.json")) as fh:
        config = json.load(fh)
    hmm = np.load(os.path.join(MODEL_DIR, "hmm.npz"))
    with open(os.path.join(MODEL_DIR, "transforms.pkl"), "rb") as fh:
        transforms = pickle.load(fh)
    with open(os.path.join(MODEL_DIR, "clean_features.txt")) as fh:
        clean_features = [line.strip() for line in fh if line.strip()]
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "clean_model.txt"))
    oof_path = os.path.join(MODEL_DIR, "oof.npz")
    oof = dict(np.load(oof_path)) if os.path.isfile(oof_path) else None
    return dict(config=config, hmm={k: hmm[k] for k in hmm.files}, transforms=transforms,
                clean_features=clean_features, booster=booster,
                gru=seq_net.load(os.path.join(MODEL_DIR, "gru.npz")), oof=oof)


def hmm_inputs(feat, transforms):
    """The player features, rank-normalised: what the HMM projects and the GRU reads."""
    cols = transforms["columns"]
    x = feat[cols].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(pd.Series(transforms["medians"], index=cols))
    for c in cols:
        if c.startswith(DURATION_COLUMNS):
            x[c] = np.log1p(x[c])
    return transforms["qt"].transform(x.values)


def hmm_embedding(feat, transforms):
    """Z: the player features, rank-normalised and projected the way the HMM was trained."""
    return transforms["pca"].transform(hmm_inputs(feat, transforms))


def hmm_posterior(z, half, params):
    """P(stopped) per frame: forward-backward on a diagonal-Gaussian HMM at HMM_STEP, held to 25 fps."""
    means, variances = params["means"], params["covars"]
    trans, start, stopped = params["transmat"], params["startprob"], int(params["stopped"])
    out = np.zeros(len(z))
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        sample = idx[::HMM_STEP]
        x = z[sample]
        loge = -0.5 * (np.log(2 * np.pi * variances)[None] + ((x[:, None, :] - means[None]) ** 2) / variances[None]).sum(2)
        e = np.exp(loge - loge.max(1, keepdims=True))
        n, k = e.shape
        alpha = np.zeros((n, k)); beta = np.ones((n, k))
        a = start * e[0]; alpha[0] = a / a.sum()
        for t in range(1, n):
            a = (alpha[t - 1] @ trans) * e[t]
            alpha[t] = a / a.sum()
        for t in range(n - 2, -1, -1):
            b = trans @ (e[t + 1] * beta[t + 1])
            beta[t] = b / b.sum()
        post = alpha * beta
        post /= post.sum(1, keepdims=True)
        pos = np.clip(np.searchsorted(sample, idx, side="right") - 1, 0, n - 1)
        out[idx] = post[pos, stopped]
    return out


def cross_fit(x, y, train_ok, frames):
    """Out-of-fold probabilities: every 2-minute block is scored by a model that never saw it."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    p = np.full(len(x), np.nan)
    for _, te in GroupKFold(5).split(x, groups=frames // BLOCK):
        test = np.zeros(len(x), bool); test[te] = True
        near = np.convolve(test.astype(int), np.ones(2 * GAP + 1, int), "same") > 0
        rows = train_ok & ~near
        model = lgb.LGBMClassifier(**CLEAN_PARAMS, n_jobs=-1).fit(x[rows], y[rows])
        p[te] = model.predict_proba(x[te])[:, 1]
    return p


def roc_auc(y, score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def _cache_path(game):
    return os.path.join(CACHE_DIR, "%s_dead_ball_ml_evidence.npz" % game)


def _cache_key(dead_anchor, live_anchor, n_frames, version):
    return hashlib.sha1(dead_anchor.tobytes() + live_anchor.tobytes() + np.int64(n_frames).tobytes() +
                        json.dumps(version).encode()).hexdigest()[:16]


def cached_evidence(game, key):
    """(p_clean, p_hmm, p_gru) if the cache holds this match's evidence for these anchors, else None."""
    path = _cache_path(game)
    if os.path.isfile(path):
        cached = np.load(path)
        if str(cached["key"]) == key and "p_gru" in cached.files:
            return cached["p_clean"], cached["p_hmm"], cached["p_gru"]
    return None


def evidence(feat, frames, dead_anchor, live_anchor, game, models, key):
    """(p_clean, p_stopped, p_gru, how) per in-play frame, or None if the stored evidence is stale. Writes the cache.

    On the match the models were trained on, the clean-label model and the GRU are read from their stored
    out-of-fold scores (every block scored by a model that never saw it); on another match the stored models
    are applied.
    """
    half = feat["half"].values
    p_hmm = hmm_posterior(hmm_embedding(feat, models["transforms"]), half, models["hmm"])
    if game == models["config"]["trained_on"]:
        oof = models["oof"]
        if oof is None or str(oof["key"]) != key:
            return None
        p_clean, p_gru, how = oof["p_clean"], oof["p_gru"], "stored out-of-fold scores"
    else:
        x = feat[models["clean_features"]].values.astype(np.float32)
        p_clean = models["booster"].predict(x)
        p_gru = 1.0 / (1.0 + np.exp(-seq_net.predict(models["gru"], hmm_inputs(feat, models["transforms"]), half)))
        how = "stored models"
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(_cache_path(game), key=key, p_clean=p_clean, p_hmm=p_hmm, p_gru=p_gru)
    return p_clean, p_hmm, p_gru, how


def half_of(frames):
    """1, 2, ... for each in-play frame: a new half starts wherever the frames jump."""
    return (np.concatenate([[0], np.cumsum(np.diff(frames) > 1)]) + 1).astype(np.int8)


def logit(p):
    p = np.clip(p, *CLIP)
    return np.log(p / (1 - p))


def decode(z, half, switch=SWITCH):
    """Best dead/live path per half: maximise sum(+z dead, -z live) minus switch per change."""
    out = np.zeros(len(z), bool)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        e = z[idx]
        n = len(idx)
        dead, live = e[0], -e[0]
        back_d = np.zeros(n, bool); back_l = np.zeros(n, bool)   # True = came from the other state
        for t in range(1, n):
            cd, cl = dead, live - switch
            back_d[t] = cl > cd
            nd = max(cd, cl) + e[t]
            cd2, cl2 = dead - switch, live
            back_l[t] = cd2 > cl2
            nl = max(cd2, cl2) - e[t]
            dead, live = nd, nl
        state = dead >= live
        path = np.zeros(n, bool)
        for t in range(n - 1, -1, -1):
            path[t] = state
            if t:
                state = (not state) if (back_d[t] if state else back_l[t]) else state
        out[idx] = path
    return out


def tidy(b):
    b = b.copy()
    for s, e in runs(~b):
        if e - s + 1 < CLOSE_GAP and s > 0 and e < len(b) - 1:
            b[s:e + 1] = True
    for s, e in runs(b):
        if e - s + 1 < MIN_SPELL:
            b[s:e + 1] = False
    return b




def runs(b):
    edges = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))


def stoppage_dead(ball_xy, players_xy, players_speed, team_of_track, touches, segments, exits,
                  game, length, width, in_play):
    """dict(ok, reasons, dead, spells, report): dead is a bool mask over every video frame,
    already clipped to the frames outside the set-piece segments in `segments`."""
    models = load()
    frames = np.flatnonzero(in_play)
    dead_a, live_a, _, _ = anchors(ball_xy, touches, segments, exits, length, width)
    da, la = dead_a[frames], live_a[frames]
    key = _cache_key(da, la, len(frames), models["config"].get("version"))
    hit = cached_evidence(game, key)
    if hit is not None:
        (p_clean, p_hmm, p_gru), how = hit, "cache"
    else:
        feat = dead_ball_features.build(players_xy, players_speed, team_of_track, frames, length, width)
        made = evidence(feat, frames, da, la, game, models, key)
        if made is None:
            return dict(ok=False, reasons=["the stored out-of-fold scores do not match this match's anchors; retrain"],
                        dead=np.zeros(len(ball_xy), bool), spells=[], report={})
        p_clean, p_hmm, p_gru, how = made
    half = half_of(frames)

    z = (logit(p_clean) + logit(p_hmm) + logit(p_gru)) / 3.0
    z[da] = ANCHOR
    z[la] = -ANCHOR
    path = tidy(decode(z, half))

    setpiece = np.zeros(len(ball_xy), bool)
    for s in segments:
        if s["rule"] not in ("stoppage", "outside"):
            setpiece[s["start"]:s["end"] + 1] = True
    dead = np.zeros(len(ball_xy), bool)
    dead[frames] = path
    dead &= ~setpiece

    certain = da | la
    reasons = []
    c = models["config"]["checks"]
    if da.sum() < c["min_dead_anchors"] or la.sum() < c["min_live_anchors"]:
        reasons.append("too few anchors (%d dead, %d live)" % (da.sum(), la.sum()))
        clean_auc = hmm_auc = gru_auc = float("nan")
    else:
        clean_auc, hmm_auc, gru_auc = (roc_auc(da[certain], p[certain]) for p in (p_clean, p_hmm, p_gru))
        if gru_auc < c["min_gru_auc"]:
            reasons.append("GRU ROC %.2f on the anchors < %.2f" % (gru_auc, c["min_gru_auc"]))
        if clean_auc < c["min_clean_auc"]:
            reasons.append("clean-label model ROC %.2f on the anchors < %.2f" % (clean_auc, c["min_clean_auc"]))
        if hmm_auc < c["min_hmm_auc"]:
            reasons.append("HMM ROC %.2f on the anchors < %.2f" % (hmm_auc, c["min_hmm_auc"]))
    share = float(path.mean())
    lo, hi = c["dead_share"]
    if not lo <= share <= hi:
        reasons.append("dead share %.0f%% outside %.0f-%.0f%%" % (100 * share, 100 * lo, 100 * hi))
    report = dict(evidence=how, dead_anchor_frames=int(da.sum()), live_anchor_frames=int(la.sum()),
                  clean_auc=clean_auc, hmm_auc=hmm_auc, gru_auc=gru_auc, dead_share=share)
    return dict(ok=not reasons, reasons=reasons, dead=dead, spells=runs(dead), report=report)
