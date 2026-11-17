"""Train the learned possession models once and store them in solution/models/possession_ml/.

    .venv/bin/python solution/train_possession_ml.py        # needs lightgbm, scikit-learn, torch (training only)

Every label it trains on is made by an algorithm from the tracking. Steps:

  1. Run the rule pipeline up to the dead-ball mask (ball gaps, touches, set pieces, learned stoppages).
  2. Clean labels (possession_ml.clean_labels): frames where a clear touch, or a pass between two clear
     touches of the same team, shows who has the ball.
  3. Team-shape features for every in-play frame.
  4. Round 1: a LightGBM and a GRU, each scored out of fold on 2-minute blocks with 10 s gaps; their
     log-odds are averaged and run through the whole flow (touch fix, rule logic, decode).
  5. Self-training: frames where that flow's path agrees with the touch-fixed labels and the round-1
     probability is at least 70% sure join the clean labels, and the LightGBM is trained again.
  6. Save: the round-2 LightGBM, the GRU weights (numpy, no torch needed to run them), the feature
     scaler, and the out-of-fold scores for this match (so running this match needs no retraining).

Run again after any change to the touch, dead-ball or clean-label rules, and delete
solution/cache/*_possession_ml_evidence.npz (the script does this).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]

import dead_ball  # noqa: E402
import dead_ball_features  # noqa: E402
import dead_ball_ml as dml  # noqa: E402
import parse_atd  # noqa: E402
import possession_ml as pm  # noqa: E402
import seq_net  # noqa: E402
import touch_detector  # noqa: E402
from gap_fill import fill_short_gaps  # noqa: E402
from games import GAME, GAMES, require  # noqa: E402

VERSION = 2
PSEUDO_LOGIT = float(np.log(0.7 / 0.3))      # round 1 must be at least 70% sure


def main():
    import lightgbm as lgb
    import sklearn
    length, width = GAMES[GAME]["pitch"]
    data = parse_atd.parse(GAME)
    ball, status = fill_short_gaps(data["ball_xy"], GAME)
    touches = touch_detector.detect(ball, status, data["players_xy"], data["team_of_track"])
    dead_mask, restart_team_at, _, _ = dead_ball.compute(
        data["ball_xy"], data["players_xy"], data["team_of_track"], require(["atd_meta"], GAME)[0], game=GAME,
        control_changes=None, touches=touches, players_speed=data["players_speed"])
    ctx = pm.context(touches, ball, data["players_xy"], data["team_of_track"], dead_mask, restart_team_at,
                     dead_ball._in_play(len(ball), GAME), touch_detector.CONTROL_RADIUS)
    frames, half, alive, lab = ctx["frames"], ctx["half"], ctx["alive"], ctx["lab"]
    labelled = lab >= 0
    print("1. dead mask: %d of %d in-play frames dead" % (ctx["dead"].sum(), len(frames)))
    print("2. clean labels: %d credited touches, %d clear; %d frames labelled (%.1f%% of live), %.0f%% Home" % (
        len(ctx["credited"]), sum(ctx["clear"]), labelled.sum(), 100 * labelled.sum() / alive.sum(), 100 * (lab[labelled] == 0).mean()))

    feat = dead_ball_features.build(data["players_xy"], data["players_speed"], data["team_of_track"], frames, length, width)
    x_df = pm.team_features(feat)
    features = list(x_df.columns)
    x = x_df.values.astype(np.float32)
    scaler = pm.fit_scaler(x)
    xs = pm.scale(x, scaler)
    print("3. features: %d columns" % len(features))

    # 4. round 1
    y = (lab == 0).astype(int)
    train1 = labelled & (frames % pm.TRAIN_STEP == 0)
    p_lgbm1 = pm.cross_fit(x, y, train1, frames)
    y_gru = np.where(lab == 0, 1, np.where(lab == 1, 0, -1)).astype(np.int8)
    logit_gru, gru_models, gru_report = seq_net.train(xs, y_gru, frames, half, False)
    p_gru = 1.0 / (1.0 + np.exp(-logit_gru))
    p1 = pm.blend(p_lgbm1, p_gru)
    print("4. round 1: out-of-fold ROC on the clean labels LightGBM %.3f, GRU %.3f, both %.3f; GRU %s" % (
        dml.roc_auc(y[labelled], p_lgbm1[labelled]), dml.roc_auc(y[labelled], p_gru[labelled]),
        dml.roc_auc(y[labelled], p1[labelled]), gru_report))

    # 5. self-training
    out, flips, target, switch, n_changes, flab, path = pm.flow(ctx, p1)
    team = np.where(path, 0, 1)
    z1 = dml.logit(p1)
    flab_team = np.where(flab == "H", 0, np.where(flab == "A", 1, -1))
    agree = alive & (flab_team == team) & (np.abs(z1) >= PSEUDO_LOGIT) & ((z1 > 0) == path)
    lab2 = np.where(lab >= 0, lab, np.where(agree, team, -1)).astype(np.int8)
    train2 = (lab2 >= 0) & (frames % pm.TRAIN_STEP == 0)
    y2 = (lab2 == 0).astype(int)
    p_lgbm = pm.cross_fit(x, y2, train2, frames)
    final = lgb.LGBMClassifier(**pm.CLASSIFIER, n_jobs=-1).fit(x[train2], y2[train2])
    p2 = pm.blend(p_lgbm, p_gru)
    auc2 = dml.roc_auc(y[labelled], p2[labelled])
    print("5. self-training: %d clean + %d pseudo labels; round 2 out-of-fold ROC on the clean labels %.3f" % (
        labelled.sum(), (lab2 >= 0).sum() - labelled.sum(), auc2))

    os.makedirs(pm.MODEL_DIR, exist_ok=True)
    final.booster_.save_model(os.path.join(pm.MODEL_DIR, "model.txt"))
    seq_net.save(os.path.join(pm.MODEL_DIR, "gru.npz"), gru_models)
    np.savez(os.path.join(pm.MODEL_DIR, "scaler.npz"), **scaler)
    key = pm._cache_key(lab, len(frames), VERSION)
    np.savez_compressed(os.path.join(pm.MODEL_DIR, "oof.npz"), key=np.array(key),
                        p_lgbm=p_lgbm.astype(np.float32), p_gru=p_gru.astype(np.float32))
    with open(os.path.join(pm.MODEL_DIR, "features.txt"), "w") as fh:
        fh.write("\n".join(features) + "\n")
    config = dict(version=VERSION, trained_on=GAME, checks=pm.CHECKS, clean_frames=int(labelled.sum()),
                  pseudo_frames=int((lab2 >= 0).sum() - labelled.sum()),
                  numbers_at_training=dict(auc=round(float(auc2), 3), gru=gru_report),
                  libraries=dict(numpy=np.__version__, scikit_learn=sklearn.__version__, lightgbm=lgb.__version__))
    with open(os.path.join(pm.MODEL_DIR, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    stale = pm._cache_path(GAME)
    if os.path.isfile(stale):
        os.remove(stale)
    print("6. saved to %s" % os.path.relpath(pm.MODEL_DIR, ROOT))


if __name__ == "__main__":
    main()
