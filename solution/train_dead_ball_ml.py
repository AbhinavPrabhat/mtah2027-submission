"""Train the learned stoppage detector once and store it in solution/models/dead_ball_ml/.

    .venv/bin/python solution/train_dead_ball_ml.py            # needs lightgbm, scikit-learn, hmmlearn

Every label it trains on is made by an algorithm from the tracking: the anchors
(dead_ball_ml.anchors). Steps:

  1. Run the rule pipeline's set-piece flow, so the exits and set-piece segments exist.
  2. Build the player features for every in-play frame (dead_ball_features.build).
  3. Anchors: frames the ball track and touches prove dead or live.
  4. HMM (unsupervised): rank-normalise 23 player-movement features, project to 8 whitened
     components, fit a 2-state diagonal-Gaussian HMM at 5 fps (best of 3 seeds). The "stopped"
     state is the one with the lowest mean collective player speed, fixed in advance.
  5. Clean-label model: a LightGBM on 150 player features, trained on the anchors only. Scored
     out of fold on 2-minute blocks with 10 s gaps (guard numbers), then refit on all anchors.
  6. Save the models and the numbers they must meet (config.json). Run again after any change
     to the exit, strike or anchor rules, and delete solution/cache/*_dead_ball_ml_evidence.npz.
"""
import contextlib
import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]

import dead_ball  # noqa: E402
import dead_ball_features  # noqa: E402
import dead_ball_ml as ml  # noqa: E402
import parse_atd  # noqa: E402
import restart_triggers  # noqa: E402
import seq_net  # noqa: E402
import touch_detector  # noqa: E402
from gap_fill import fill_short_gaps  # noqa: E402
from games import GAME, GAMES, require  # noqa: E402

VERSION = 2
SEED = 0
STOPPED_REFERENCE = "w100_mean_all_spd_mean"     # the stopped state is the slowest on this
HMM_COLUMNS = ["w25_mean_all_spd_med", "w25_mean_all_spd_p90", "w25_mean_all_n_sprint", "w25_mean_all_n_still",
               "w25_mean_n_standing_3s", "w25_mean_all_align", "w25_mean_all_rad", "w25_mean_mark_mean",
               "w25_mean_n_pairs_2m", "w25_mean_dense_n", "w100_mean_all_spd_mean", "w100_std_all_spd_mean",
               "w100_mean_n_braking", "w100_mean_n_accel", "w100_mean_team_dist",
               "h_still_past", "h_still_future", "h_since_sprint", "h_until_sprint", "h_disp_w50",
               "h_disp_w125", "h_env_past", "h_env_future"]


def main():
    from hmmlearn.hmm import GaussianHMM
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer
    import lightgbm as lgb

    length, width = GAMES[GAME]["pitch"]
    data = parse_atd.parse(GAME)
    filled, status = fill_short_gaps(data["ball_xy"], GAME)
    touches = touch_detector.detect(filled, status, data["players_xy"], data["team_of_track"])
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, segments, _ = dead_ball.compute(
            data["ball_xy"], data["players_xy"], data["team_of_track"], require(["atd_meta"], GAME)[0],
            game=GAME, touches=touches, players_speed=data["players_speed"], use_ml=False)
    exits = restart_triggers.exits(data["ball_xy"], length, width)
    in_play = dead_ball._in_play(len(data["ball_xy"]), GAME)
    frames = np.flatnonzero(in_play)
    print("1. set-piece flow: %d segments, %d exits" % (len(segments), len(exits)))

    feat = dead_ball_features.build(data["players_xy"], data["players_speed"], data["team_of_track"], frames, length, width)
    print("2. features: %d frames x %d columns" % feat.shape)

    dead_a, live_a, sources, _ = ml.anchors(data["ball_xy"], touches, segments, exits, length, width)
    da, la = dead_a[frames], live_a[frames]
    certain = da | la
    print("3. anchors: %d dead frames (%.1f%%), %d live frames (%.1f%%); dead by source: %s" % (
        da.sum(), 100 * da.mean(), la.sum(), 100 * la.mean(), {k: int(v[frames].sum()) for k, v in sources.items()}))

    # 4. HMM
    x = feat[HMM_COLUMNS].replace([np.inf, -np.inf], np.nan)
    medians = x.median()
    x = x.fillna(medians)
    for c in HMM_COLUMNS:
        if c.startswith(ml.DURATION_COLUMNS):
            x[c] = np.log1p(x[c])
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=100000, random_state=SEED)
    pca = PCA(8, whiten=True, random_state=SEED)
    z = pca.fit_transform(qt.fit_transform(x.values))
    half = feat["half"].values
    sel = [np.flatnonzero(half == h)[::ml.HMM_STEP] for h in np.unique(half)]
    zs, lengths = np.vstack([z[i] for i in sel]), [len(i) for i in sel]
    best = None
    for seed in range(3):
        m = GaussianHMM(2, covariance_type="diag", n_iter=100, tol=1e-3, random_state=SEED + seed,
                        init_params="mc", params="stmc", min_covar=1e-3)
        m.startprob_ = np.full(2, 0.5)
        m.transmat_ = np.array([[0.98, 0.02], [0.02, 0.98]])
        m.fit(zs, lengths)
        score = m.score(zs, lengths)
        if best is None or score > best[0]:
            best = (score, m)
    model = best[1]
    ref = feat[STOPPED_REFERENCE].values
    path = np.concatenate([model.predict(z[i]) for i in sel])
    ref_s = np.concatenate([ref[i] for i in sel])
    stopped = int(np.argmin([ref_s[path == s].mean() for s in range(2)]))
    covars = model.covars_
    if covars.ndim == 3:                      # hmmlearn expands a diagonal covariance to full matrices
        covars = np.einsum("kii->ki", covars)
    params = dict(means=model.means_, covars=covars, transmat=model.transmat_,
                  startprob=model.startprob_, stopped=np.array(stopped))
    ours = ml.hmm_posterior(z, half, params)
    ref_post = np.zeros(len(z))
    for h, i in zip(np.unique(half), sel):
        full = np.flatnonzero(half == h)
        pos = np.clip(np.searchsorted(i, full, side="right") - 1, 0, len(i) - 1)
        ref_post[full] = model.predict_proba(z[i])[pos, stopped]
    print("4. HMM: stopped state S%d, mean player speed %.2f vs %.2f m/s; numpy forward-backward differs from hmmlearn by %.1e" % (
        stopped, ref_s[path == stopped].mean(), ref_s[path != stopped].mean(), np.abs(ours - ref_post).max()))

    # 5. clean-label model, round 1 (anchors only)
    with open(os.path.join(ml.MODEL_DIR, "clean_features.txt")) as fh:
        clean_features = [line.strip() for line in fh if line.strip()]
    xc = feat[clean_features].values.astype(np.float32)
    train_ok = certain & (frames % ml.TRAIN_STEP == 0)
    p_clean1 = ml.cross_fit(xc, da, train_ok, frames)
    print("5. clean-label model, round 1: out-of-fold ROC on the anchors %.3f (HMM %.3f)" % (
        ml.roc_auc(da[certain], p_clean1[certain]), ml.roc_auc(da[certain], ours[certain])))

    # 6. self-training: round 2 also learns from frames round 1 is sure about
    z1 = 0.5 * (ml.logit(p_clean1) + ml.logit(ours))
    z1[da], z1[la] = ml.ANCHOR, -ml.ANCHOR
    path1 = ml.tidy(ml.decode(z1, half))
    pseudo_dead = path1 & ~certain & (p_clean1 >= 0.7) & (ours >= 0.6)
    pseudo_live = ~path1 & ~certain & (p_clean1 <= 0.3) & (ours <= 0.4)
    use = certain | pseudo_dead | pseudo_live
    y2 = da | pseudo_dead
    train2 = use & (frames % ml.TRAIN_STEP == 0)
    p_clean = ml.cross_fit(xc, y2, train2, frames)
    final = lgb.LGBMClassifier(**ml.CLEAN_PARAMS, n_jobs=-1).fit(xc[train2], y2[train2])
    clean_auc, hmm_auc = ml.roc_auc(da[certain], p_clean[certain]), ml.roc_auc(da[certain], ours[certain])
    print("6. self-training: +%d pseudo-dead, +%d pseudo-live frames; round 2 out-of-fold ROC on the anchors %.3f" % (
        pseudo_dead.sum(), pseudo_live.sum(), clean_auc))

    # 7. the GRU on the same rank-normalised movement features, trained on the anchors only
    xd = ml.hmm_inputs(feat, dict(columns=HMM_COLUMNS, medians=medians.values, qt=qt))
    y_gru = np.where(da, 1, np.where(la, 0, -1)).astype(np.int8)
    logit_gru, gru_models, gru_report = seq_net.train(xd, y_gru, frames, half, True)
    p_gru = 1.0 / (1.0 + np.exp(-logit_gru))
    gru_auc = ml.roc_auc(da[certain], p_gru[certain])
    print("7. GRU: %s" % gru_report)

    os.makedirs(ml.MODEL_DIR, exist_ok=True)
    final.booster_.save_model(os.path.join(ml.MODEL_DIR, "clean_model.txt"))
    seq_net.save(os.path.join(ml.MODEL_DIR, "gru.npz"), gru_models)
    key = ml._cache_key(da, la, len(frames), VERSION)
    np.savez_compressed(os.path.join(ml.MODEL_DIR, "oof.npz"), key=key, p_clean=p_clean.astype(np.float32),
                        p_gru=p_gru.astype(np.float32))
    np.savez(os.path.join(ml.MODEL_DIR, "hmm.npz"), **params)
    with open(os.path.join(ml.MODEL_DIR, "transforms.pkl"), "wb") as fh:
        pickle.dump(dict(columns=HMM_COLUMNS, medians=medians.values, qt=qt, pca=pca), fh)
    import sklearn
    config = dict(version=VERSION, trained_on=GAME, checks=ml.CHECKS, stopped_reference=STOPPED_REFERENCE,
                  anchors=dict(dead=int(da.sum()), live=int(la.sum())),
                  numbers_at_training=dict(clean_auc=round(clean_auc, 3), hmm_auc=round(hmm_auc, 3), gru_auc=round(gru_auc, 3),
                                           pseudo_dead=int(pseudo_dead.sum()), pseudo_live=int(pseudo_live.sum()), gru=gru_report),
                  libraries=dict(numpy=np.__version__, scikit_learn=sklearn.__version__, lightgbm=lgb.__version__))
    with open(os.path.join(ml.MODEL_DIR, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    stale = os.path.join(ml.CACHE_DIR, "%s_dead_ball_ml_evidence.npz" % GAME)
    if os.path.isfile(stale):
        os.remove(stale)
    print("8. saved to %s" % os.path.relpath(ml.MODEL_DIR, ROOT))


if __name__ == "__main__":
    main()
