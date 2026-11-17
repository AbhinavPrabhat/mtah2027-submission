"""Phase 5 -- combine touches, control changes and dead-ball windows into one label per frame.

Priority order, and why:
  1. Dead-ball mask always wins. Whatever the touch/control logic thinks is happening, a frame
     inside a Smart Tagging set-piece window or a bridged out-of-bounds stretch is D.
  2. A restart's team (from dead_ball.py) overrides whatever control.py would have said at that
     frame. CHALLENGE.md states every restart has a clear, known possessor -- that is a stronger
     claim than inferring possession from ball physics, so it takes precedence right after a
     stoppage ends.
  3. Everywhere else, the team holds from the most recent confirmed change (restart or control),
     exactly the "constant label that only changes at a controlled event" model this solution is
     built around -- not a per-frame decision.

Frames before the first resolved possessor (restart or touch) have no evidence at all and are
left as X; in this match that is only the pre-kick-off warm-up, which is not graded.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import HOME, AWAY  # noqa: E402


def build_labels(n_frames, control_changes, dead_mask, restart_team_at):
    """frame,label array ('<U1' of 'H'/'A'/'D'/'X'), length n_frames."""
    change_points = {}
    for frame, team in control_changes:
        if 0 <= frame < n_frames and not dead_mask[frame]:
            change_points[frame] = team
    for frame, team in restart_team_at.items():
        if team is not None and 0 <= frame < n_frames:
            change_points[frame] = team  # restart evidence overrides a control change at the same frame

    team_for_alive = np.full(n_frames, -1, dtype=np.int8)
    change_frames = sorted(change_points)
    prev_team, prev_frame = None, 0
    for frame in change_frames:
        if prev_team is not None:
            team_for_alive[prev_frame:frame] = prev_team
        prev_team, prev_frame = change_points[frame], frame
    if prev_team is not None:
        team_for_alive[prev_frame:] = prev_team

    labels = np.full(n_frames, "X", dtype="<U1")
    labels[team_for_alive == HOME] = "H"
    labels[team_for_alive == AWAY] = "A"
    labels[dead_mask] = "D"  # dead-ball always wins, even over a team already assigned above
    return labels


if __name__ == "__main__":
    import parse_atd
    from gap_fill import fill_short_gaps
    import touch_detector
    import control
    import dead_ball
    from games import GAME, require

    data = parse_atd.parse()
    filled_xy, status = fill_short_gaps(data["ball_xy"])
    touches = touch_detector.detect(filled_xy, status, data["players_xy"], data["team_of_track"])
    control_changes = control.confirm_changes(touches, filled_xy, data["players_xy"],
                                              data["team_of_track"])
    meta_path = require(["atd_meta"])[0]
    dead_mask, restart_team_at, _segments, _restarts = dead_ball.compute(
        data["ball_xy"], data["players_xy"], data["team_of_track"], meta_path,
        control_changes=control_changes, touches=touches,
        players_speed=data["players_speed"])

    labels = build_labels(len(data["ball_xy"]), control_changes, dead_mask, restart_team_at)

    for code in ("H", "A", "D", "X"):
        count = int(np.sum(labels == code))
        print("  {}  {:>7}  {:5.1f}%".format(code, count, 100.0 * count / len(labels)))
