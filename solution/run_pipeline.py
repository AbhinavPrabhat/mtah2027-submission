"""End-to-end entry point: raw ATD -> touches -> control -> dead-ball -> submission CSV.

    python solution/run_pipeline.py                         # writes evaluation/results/solution_v1.csv
    python solution/run_pipeline.py --out my_submission.csv

Then, same as the baseline workflow:
    python evaluation/validate_submission.py evaluation/results/solution_v1.csv
    python evaluation/run_evaluation.py
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require  # noqa: E402
from baseline import write_submission  # noqa: E402

import parse_atd  # noqa: E402
from gap_fill import fill_short_gaps  # noqa: E402
import touch_detector  # noqa: E402
import control  # noqa: E402
import dead_ball  # noqa: E402
import possession_ml  # noqa: E402
import post_process  # noqa: E402
from assemble import build_labels  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "solution_v1.csv")


def run(game=GAME, data_dir=None, radius=None, use_learned_possession=True, use_post_process=True):
    if radius is None:
        radius = touch_detector.CONTROL_RADIUS
    data = parse_atd.parse(game, data_dir)

    print("\nPhase 1 -- filling short ball gaps...")
    filled_xy, status = fill_short_gaps(data["ball_xy"], game)
    for code, name in ((0, "observed"), (1, "filled"), (2, "still missing")):
        count = int(np.sum(status == code))
        print("  {:<10} {:>7}  {:5.1f}%".format(name, count, 100.0 * count / len(status)))

    print("\nPhase 2 -- detecting touches...")
    touches = touch_detector.detect(
        filled_xy, status, data["players_xy"], data["team_of_track"], radius=radius)
    resolved = sum(1 for t in touches if t["team"] is not None)
    print("  {} touch candidates, {} with a credited toucher (radius {:.1f} m)".format(
        len(touches), resolved, radius))

    print("\nPhase 3 -- finding dead-ball windows...")
    meta_path = require(["atd_meta"], game, data_dir)[0]
    dead_mask, restart_team_at, _segments, _restarts = dead_ball.compute(
        data["ball_xy"], data["players_xy"], data["team_of_track"], meta_path, game=game,
        control_changes=None, touches=touches,
        players_speed=data["players_speed"])
    print("  {} / {} frames dead ({:.1f}%), {} restarts".format(
        int(dead_mask.sum()), len(dead_mask), 100.0 * dead_mask.sum() / len(dead_mask),
        len(restart_team_at)))

    print("\nPhase 4 -- confirming possession on live frames...")
    control_changes = control.confirm_changes(
        touches, filled_xy, data["players_xy"], data["team_of_track"], radius=radius,
        dead_mask=dead_mask, restart_team_at=restart_team_at)
    print("  {} confirmed changes, restarts included".format(len(control_changes)))

    print("\nPhase 5 -- assembling final labels...")
    labels = build_labels(len(data["ball_xy"]), control_changes, dead_mask, restart_team_at)
    if use_learned_possession:
        length, width = GAMES[game]["pitch"]
        learned, report = possession_ml.compute(
            touches, filled_xy, data["players_xy"], data["players_speed"], data["team_of_track"], dead_mask,
            restart_team_at, dead_ball._in_play(len(filled_xy), game), game, length, width, radius)
        if learned is None:
            print("  learned possession not used ({}); rule possession kept".format(report))
        else:
            labels = learned
            print("  learned possession: {}".format(", ".join("{} {}".format(k, v) for k, v in report.items())))
    if use_post_process:
        labels, report = post_process.run(
            labels, touches, filled_xy, dead_ball._in_play(len(filled_xy), game))
        print("  post-processor: {}".format(", ".join("{} {}".format(k, v) for k, v in report.items())))
    for code in ("H", "A", "D", "X"):
        count = int(np.sum(labels == code))
        print("  {}  {:>7}  {:5.1f}%".format(code, count, 100.0 * count / len(labels)))

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the submission CSV")
    parser.add_argument("--radius", type=float, default=None,
                        help="metres for who touched the ball and who still has it")
    args = parser.parse_args()

    labels = run(radius=args.radius)
    path = write_submission(args.out, labels)
    print("\n-> wrote {}".format(os.path.relpath(path, ROOT)))
    print("\nNow:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
    print("      python evaluation/run_evaluation.py")
