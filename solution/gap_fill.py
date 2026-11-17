"""Phase 1b -- fill short gaps in the ball's tracked position.

The ball is missing on ~43% of all frames (occlusion, a pass in the air, a crowded box). The
baseline treats every one of those as dead ball, which is its single biggest source of error.
Most of that 43% is really the ball still in play but unseen for a fraction of a second.

This module does the narrow, defensible part of that fix: short gaps (under MAX_GAP_FRAMES) are
filled by straight-line interpolation between the last observation before the gap and the first
after it. Long gaps are left as NaN -- a ball that vanishes for seconds was probably dealt with
(cleared, saved, out of play) in a way a straight line cannot represent, so Phase 4 (dead-ball
detection) handles those, not this interpolation.

Interpolation never crosses a half boundary: the two halves are separate blocks of play with
halftime between them, and nothing about the ball's motion before the break predicts its position
after it.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402

MAX_GAP_FRAMES = 25  # 1 s at 25 fps -- beyond this, a straight line stops being a safe guess


def _runs_of_nan(is_missing):
    """[(start, end)] half-open spans where is_missing is True, in index order."""
    missing = np.concatenate(([False], is_missing, [False]))
    edges = np.flatnonzero(missing[1:] != missing[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def fill_short_gaps(ball_xy, game=GAME, max_gap=MAX_GAP_FRAMES):
    """(filled_xy, status) -- status is 0 observed, 1 filled, 2 still missing, per frame.

    Only fills a gap that has a real observation immediately before AND after it, and that lies
    entirely inside one half -- a gap touching kickoff, full time or the halftime break is left
    missing rather than guessed.
    """
    n_frames = len(ball_xy)
    filled = ball_xy.copy()
    status = np.where(np.isnan(ball_xy[:, 0]), 2, 0).astype(np.int8)

    half_of_frame = np.full(n_frames, -1, dtype=np.int32)
    for half, first, last in half_video_blocks(game):
        half_of_frame[max(first, 0):min(last, n_frames - 1) + 1] = half

    is_missing = status == 2
    for start, end in _runs_of_nan(is_missing):
        length = end - start
        if length > max_gap:
            continue
        if start == 0 or end == n_frames:
            continue  # no observation on one side -- nothing to interpolate from
        if half_of_frame[start - 1] == -1 or half_of_frame[start - 1] != half_of_frame[end]:
            continue  # touches halftime or the edge of play -- leave missing

        before, after = filled[start - 1], filled[end]
        t = np.arange(1, length + 1, dtype=np.float64) / (length + 1)
        filled[start:end] = before + t[:, None] * (after - before)
        status[start:end] = 1

    return filled, status


if __name__ == "__main__":
    import parse_atd

    data = parse_atd.parse()
    filled, status = fill_short_gaps(data["ball_xy"])

    n = len(status)
    for code, name in ((0, "observed"), (1, "filled (short gap)"), (2, "still missing")):
        count = int(np.sum(status == code))
        print("  {:<20} {:>7}  {:5.1f}%".format(name, count, 100.0 * count / n))
