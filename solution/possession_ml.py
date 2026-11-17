"""Learned possession (v3): the touch logic stays primary, a model fixes weak touches and fills the doubt.

Everything is made by an algorithm from the tracking. Steps (run by `compute`):

  1. Clean labels. A frame is labelled with a team only when the tracking is clear:
       touch   a credited touch within 1.5 m of the ball and no opponent within 3 m of the ball
       bridge  two consecutive credited touches by the same team, at most 6 s apart, in one half,
               no dead frame between: every frame between is that team's
       edge    0.5 s either side of a confirmed touch
     Everything else (contested touches, the flight between touches of opposite teams, dead
     frames) stays unlabelled.
  2. Model. A LightGBM on team-shape features (where each team stands and moves, and the same
     numbers over the past 2 s, next 2 s and a centred 6 s) gives P(Home has it) for every frame.
     No ball column is an input. On the match it was trained on every frame is scored by a model
     that never saw its 2-minute block; on another match the stored model is applied.
  3. Touch fix. A credited touch that is not clear (toucher farther than 1.5 m, or an opponent
     within 3 m) is flipped to the other team when the model, averaged over 2 s around the touch,
     is at least 82% sure of the other team. The rule pipeline's possession logic (control.py)
     then runs on the corrected touches.
  4. Decode. Those labels vote +-2 log-odds, the model adds its own log-odds, and one best Home/Away
     path per half pays a cost for each change. The cost is chosen so the path makes as many
     changes as the touches confirm (a change that the same team's next credited touch repeats,
     with no dead stretch between). Possessions under 1 s take the previous team.

If the models are missing or a guard fails, nothing here is used and the rule logic runs as before.
"""
import json
import os

import numpy as np
import pandas as pd

import dead_ball_ml as dml
import seq_net

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "possession_ml")
CACHE_DIR = os.path.join(HERE, "cache")

TOUCH_RADIUS, CLEAR_RADIUS = 1.5, 3.0   # m
BRIDGE, EDGE = 150, 12                  # frames
PAST, NEXT, WIDE = 50, 50, 150          # window half-lengths for the features
CONFIDENT = 0.82                        # the model's side must be this sure to override a weak touch
AVG = 25                                # frames either side of a touch averaged for that check
VOTE = 2.0                              # log-odds of the touch-fixed team in the decode
SMOOTH = 12                             # centred window half-length for the model's log-odds
MIN_SPELL = 25                          # frames
TRAIN_STEP = 2
CHECKS = dict(min_clean_frames=3000, min_auc=0.80)
CLASSIFIER = dict(objective="binary", num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=40,
                  colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=5, verbose=-1)
PREFIXES = ("f_h_", "f_a_")
KEY = ("diff_cx", "diff_cy", "diff_spd_mean", "diff_vx", "diff_vy", "diff_x_range", "diff_y_range", "diff_n_sprint",
       "diff_minx", "diff_maxx")
DIFF_OF = ("cx", "cy", "sx", "sy", "spd_mean", "spd_std", "spd_med", "spd_max", "spd_p90", "n_sprint", "n_still",
           "vx", "vy", "align", "x_range", "y_range", "minx", "maxx", "miny", "maxy")


def available():
    return os.path.isfile(os.path.join(MODEL_DIR, "config.json"))


def load():
    import lightgbm as lgb
    with open(os.path.join(MODEL_DIR, "config.json")) as fh:
        config = json.load(fh)
    with open(os.path.join(MODEL_DIR, "features.txt")) as fh:
        features = [line.strip() for line in fh if line.strip()]
    oof_path = os.path.join(MODEL_DIR, "oof.npz")
    scaler = np.load(os.path.join(MODEL_DIR, "scaler.npz"))
    return dict(config=config, features=features, booster=lgb.Booster(model_file=os.path.join(MODEL_DIR, "model.txt")),
                gru=seq_net.load(os.path.join(MODEL_DIR, "gru.npz")), scaler=dict(scaler),
                oof=dict(np.load(oof_path)) if os.path.isfile(oof_path) else None)


def scale(x, scaler):
    """Features as the GRU reads them: robust-scaled and clipped."""
    x = np.where(np.isnan(x), scaler["median"][None], x)
    return np.clip((x - scaler["median"]) / scaler["iqr"], -5, 5).astype(np.float32)


def fit_scaler(x):
    col = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), col[None], x)
    return dict(median=np.median(x, 0), iqr=(np.percentile(x, 75, 0) - np.percentile(x, 25, 0)) / 1.35 + 1e-6)


def blend(p_lgbm, p_gru):
    """Average in log-odds: the LightGBM and the GRU."""
    return 1.0 / (1.0 + np.exp(-(dml.logit(p_lgbm) + dml.logit(p_gru)) / 2.0))


# ----------------------------------------------------------------------------- labels

def is_clear(touch, ball, players_xy, team_of_track):
    f = touch["frame"]
    d = np.hypot(players_xy[f, :, 0] - ball[f, 0], players_xy[f, :, 1] - ball[f, 1])
    d = np.where(np.isnan(d), np.inf, d)
    return touch["distance"] <= TOUCH_RADIUS and not (d[team_of_track != touch["team"]] <= CLEAR_RADIUS).any()


def clean_labels(credited, clear, frames, half, dead):
    """-1 unlabelled, 0 Home, 1 Away, on the rows of `frames`."""
    n = len(frames)
    pos = {int(f): i for i, f in enumerate(frames)}
    lab = np.full(n, -1, np.int8)
    for i, t in enumerate(credited):
        if not clear[i] or t["frame"] not in pos:
            continue
        p = pos[t["frame"]]
        seg = slice(max(0, p - EDGE), min(n, p + EDGE + 1))
        lab[seg] = np.where((half[seg] == half[p]) & ~dead[seg], t["team"], lab[seg])
        if i + 1 < len(credited):
            u = credited[i + 1]
            if clear[i + 1] and u["team"] == t["team"] and u["frame"] in pos:
                q = pos[u["frame"]]
                if 0 < q - p <= BRIDGE and half[q] == half[p] and not dead[p:q + 1].any():
                    lab[p:q + 1] = t["team"]
    return lab


def confirmed_changes(credited, dead_mask):
    """Changes of team that the same team's next credited touch repeats, with no dead frame between."""
    n = 0
    for a, b, c in zip(credited[:-2], credited[1:-1], credited[2:]):
        if b["team"] != a["team"] and c["team"] == b["team"] and not dead_mask[a["frame"]:b["frame"] + 1].any():
            n += 1
    return n


# ----------------------------------------------------------------------------- features

def team_features(feat):
    """Instant and window features of both teams from the player feature table (no ball)."""
    half = feat["half"].values
    cols = [c for c in feat.columns if c.startswith(PREFIXES)]
    x = feat[cols].copy()
    for s in DIFF_OF:
        if "f_h_" + s in feat.columns and "f_a_" + s in feat.columns:
            x["diff_" + s] = feat["f_h_" + s] - feat["f_a_" + s]
    win = {}
    for c in KEY:
        v = x[c]
        win[c + "_past"] = v.groupby(half).transform(lambda s: s.rolling(PAST, min_periods=1).mean())
        win[c + "_next"] = v.groupby(half).transform(lambda s: s[::-1].rolling(NEXT, min_periods=1).mean()[::-1])
        win[c + "_wide"] = v.groupby(half).transform(lambda s: s.rolling(WIDE, center=True, min_periods=1).mean())
        win[c + "_chg"] = win[c + "_next"] - win[c + "_past"]
    return pd.concat([x.reset_index(drop=True), pd.DataFrame(win).reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)


def cross_fit(x, y, train_ok, frames, gap=250):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    p = np.full(len(x), np.nan)
    for _, te in GroupKFold(5).split(x, groups=frames // 3000):
        test = np.zeros(len(x), bool)
        test[te] = True
        near = np.convolve(test.astype(int), np.ones(2 * gap + 1, int), "same") > 0
        rows = train_ok & ~near
        model = lgb.LGBMClassifier(**CLASSIFIER, n_jobs=-1).fit(x[rows], y[rows])
        p[te] = model.predict_proba(x[te])[:, 1]
    return p


def _cache_path(game):
    return os.path.join(CACHE_DIR, "%s_possession_ml_evidence.npz" % game)


def _cache_key(lab, n, version):
    return "%d|%d|%s|%d" % (n, int((lab == 0).sum()), version, int((lab == 1).sum()))


def probabilities(make_features, lab, frames, half, game, models):
    """(P(Home) per row of `frames`, how it was made, AUC on the clean labels), or None if stale.

    On the match the models were trained on, the stored out-of-fold scores are used (every block scored by
    models that never saw it); on another match the stored LightGBM and GRU are applied.
    """
    key = _cache_key(lab, len(frames), models["config"].get("version"))
    path = _cache_path(game)
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["p"], "cache", float(z["auc"])
    if game == models["config"].get("trained_on"):
        oof = models["oof"]
        if oof is None or str(oof["key"]) != key:
            return None
        p, how = blend(oof["p_lgbm"], oof["p_gru"]), "stored out-of-fold scores"
    else:
        x = team_features(make_features())[models["features"]].values.astype(np.float32)
        p_gru = 1.0 / (1.0 + np.exp(-seq_net.predict(models["gru"], scale(x, models["scaler"]), half)))
        p, how = blend(models["booster"].predict(x), p_gru), "stored models"
    labelled = lab >= 0
    auc = dml.roc_auc((lab[labelled] == 0), p[labelled])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, key=np.array(key), p=p, auc=np.array(auc))
    return p, how, auc


# ----------------------------------------------------------------------------- touch fix and decode

def fix_touches(touches, clear_of, p_abs):
    """Flip weak touches the model confidently contradicts. (new touches, number flipped)."""
    out, flips = [], 0
    for t in touches:
        t = dict(t)
        if t["team"] is not None and not clear_of(t):
            w = p_abs[max(0, t["frame"] - AVG):t["frame"] + AVG + 1]
            p = np.nanmean(w) if np.isfinite(w).any() else np.nan
            if (t["team"] == 0 and p < 1 - CONFIDENT) or (t["team"] == 1 and p > CONFIDENT):
                t["team"] = 1 - t["team"]
                flips += 1
        out.append(t)
    return out, flips


def _smooth(v, half, k):
    return pd.Series(v).groupby(half).transform(lambda s: s.rolling(2 * k + 1, center=True, min_periods=1).mean()).values


def merge_short(path, alive, min_len):
    p = path[alive].copy()
    ed = np.flatnonzero(np.diff(np.r_[-1, p.astype(int), -1]))
    runs = list(zip(ed[:-1], ed[1:]))
    for i, (s, e) in enumerate(runs):
        if e - s < min_len and i > 0:
            p[s:e] = p[runs[i - 1][0]]
    out = path.copy()
    out[alive] = p
    return out


def decode_matched(z, half, alive, target):
    """The decode whose number of changes is closest to `target`, searching the switch cost."""
    lo, hi = 5.0, 400.0
    best = None
    for _ in range(9):
        mid = float(np.sqrt(lo * hi))
        path = merge_short(dml.decode(np.where(alive, z, 0.0), half, mid), alive, MIN_SPELL)
        changes = int((np.diff(path[alive].astype(int)) != 0).sum())
        best = (mid, path, changes)
        if changes > target:
            lo = mid
        else:
            hi = mid
    return best


def combine(p, fixed_labels, frames, half, alive, target):
    """Per-row Home (True) / Away path from the model plus the touch-fixed labels."""
    z = dml.logit(_smooth(p, half, SMOOTH))
    lab = fixed_labels[frames]
    z = z + np.where(lab == "H", VOTE, np.where(lab == "A", -VOTE, 0.0))
    return decode_matched(z, half, alive, target)


# ----------------------------------------------------------------------------- the pipeline step

def context(touches, ball, players_xy, team_of_track, dead_mask, restart_team_at, in_play, radius):
    """Everything the touch fix and the decode need, apart from the model's probabilities."""
    frames = np.flatnonzero(in_play)
    half = dml.half_of(frames)
    dead = dead_mask[frames]
    credited = [t for t in touches if t["team"] is not None]
    clear = [is_clear(t, ball, players_xy, team_of_track) for t in credited]
    return dict(touches=touches, ball=ball, players_xy=players_xy, team_of_track=team_of_track, dead_mask=dead_mask,
                restart_team_at=restart_team_at, radius=radius, frames=frames, half=half, dead=dead, alive=~dead,
                credited=credited, clear=clear, n=len(ball), lab=clean_labels(credited, clear, frames, half, dead))


def flow(ctx, p):
    """Touch fix, the rule pipeline's possession logic on the corrected touches, then the decode.

    Returns (labels, flips, touch-confirmed changes, switch cost, decoded changes, touch-fixed labels per row,
    path per row).
    """
    import control
    from assemble import build_labels
    n, frames, half, alive = ctx["n"], ctx["frames"], ctx["half"], ctx["alive"]
    p_abs = np.full(n, np.nan)
    p_abs[frames] = p
    ball, players_xy, team = ctx["ball"], ctx["players_xy"], ctx["team_of_track"]
    fixed, flips = fix_touches(ctx["touches"], lambda t: is_clear(t, ball, players_xy, team), p_abs)
    changes = control.confirm_changes(fixed, ball, players_xy, team, radius=ctx["radius"],
                                      dead_mask=ctx["dead_mask"], restart_team_at=ctx["restart_team_at"])
    fixed_labels = build_labels(n, changes, ctx["dead_mask"], ctx["restart_team_at"]).astype(object)
    target = confirmed_changes(ctx["credited"], ctx["dead_mask"])
    switch, path, n_changes = combine(p, fixed_labels, frames, half, alive, target)
    out = fixed_labels.copy()
    out[frames[alive]] = np.where(path, "H", "A")[alive]
    return out, flips, target, switch, n_changes, fixed_labels[frames], path


def compute(touches, ball, players_xy, players_speed, team_of_track, dead_mask, restart_team_at, in_play,
            game, length, width, radius):
    """(labels, report) with the learned possession, or (None, reason) to fall back to the rule logic."""
    import dead_ball_features
    if not available():
        return None, "no stored possession model"
    models = load()
    ctx = context(touches, ball, players_xy, team_of_track, dead_mask, restart_team_at, in_play, radius)
    lab = ctx["lab"]
    if (lab >= 0).sum() < CHECKS["min_clean_frames"] or len(np.unique(lab[lab >= 0])) < 2:
        return None, "only %d clean-labelled frames" % (lab >= 0).sum()
    made = probabilities(
        lambda: dead_ball_features.build(players_xy, players_speed, team_of_track, ctx["frames"], length, width),
        lab, ctx["frames"], ctx["half"], game, models)
    if made is None:
        return None, "the stored out-of-fold scores do not match this match's clean labels; retrain"
    p, how, auc = made
    if auc < CHECKS["min_auc"]:
        return None, "model ROC on the clean labels %.2f is under %.2f" % (auc, CHECKS["min_auc"])
    out, flips, target, switch, n_changes, _, _ = flow(ctx, p)
    return out.astype("<U1"), dict(evidence=how, clean_frames=int((lab >= 0).sum()), auc=round(auc, 3),
                                  weak_touches_flipped=flips, touch_changes=target, switch=round(switch, 1),
                                  decoded_changes=n_changes)
