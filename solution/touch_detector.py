"""Phase 2 -- detect ball touches from breaks in the ball's own motion.

The baseline (and a naive improvement on it) asks "who is nearest the ball this frame?" on every
one of 199,319 frames. That is the wrong question: a defender sprinting past a moving ball is
nearest to it for a dozen frames without ever touching it, and each of those frames becomes a
fake, flickering possession change under that rule.

The right question is "when did the ball's motion change?" Between touches the ball is a free
body: it travels in roughly a straight line and slows down smoothly (friction, air resistance). A
touch is the thing that breaks that pattern -- a kick, a header, a save, a deflection. So instead
of 199k per-frame guesses, this produces a short list (a few hundred) of touch events, each with a
frame and (when resolvable) the team that made it. Everything downstream reasons about touches,
not frames.

Method: at each frame t, compare the incoming heading/speed (from t-K to t) with the outgoing
heading/speed (from t to t+K), only where the ball's position is trustworthy (observed or
short-gap-filled, from gap_fill.py) for the whole window. A big change in heading and/or speed is
a touch candidate; candidates within MIN_GAP_FRAMES of each other collapse to their strongest
point (the ball cannot be touched twice in 0.1 s). The toucher is whichever player is closest to
the ball at that instant, but only within CONTROL_RADIUS -- beyond that, nobody is reasonably
called the toucher, and the event is kept with team=None rather than guessed.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

K_FRAMES = 4            # 0.16 s half-window either side of the candidate frame
MIN_SPEED = 0.3         # m/s -- below this the ball is nearly still; direction is noise, not a touch
ANGLE_WEIGHT = 1.0      # degrees contributed per degree of heading change
SPEED_WEIGHT = 15.0     # degrees-equivalent contributed per 1 m/s of speed change
SCORE_THRESHOLD = 35.0  # candidate touch if angle + speed terms clear this
MIN_GAP_FRAMES = 5      # 0.2 s -- candidates closer than this merge into one touch
CONTROL_RADIUS = 3.5    # m -- beyond this nobody is called the toucher


def _shift(array, by):
    """array shifted so result[t] = array[t + by], NaN where that index does not exist."""
    out = np.full_like(array, np.nan)
    if by > 0:
        out[:-by] = array[by:]
    elif by < 0:
        out[-by:] = array[:by]
    else:
        out[:] = array
    return out


def _touch_scores(ball_xy, status, fps, k=K_FRAMES):
    """(score, speed_in, speed_out, valid) per frame -- vectorised over the whole match."""
    n = len(ball_xy)
    valid = status != 2
    valid_int = valid.astype(np.int64)
    csum = np.concatenate(([0], np.cumsum(valid_int)))

    idx = np.arange(n)
    lo = np.clip(idx - k, 0, n)
    hi = np.clip(idx + k + 1, 0, n)
    window_full = (hi - lo == 2 * k + 1) & (idx >= k) & (idx < n - k)
    window_sum = csum[hi] - csum[lo]
    valid_window = window_full & (window_sum == 2 * k + 1)

    before = _shift(ball_xy, -k)
    after = _shift(ball_xy, k)
    incoming = ball_xy - before
    outgoing = after - ball_xy

    dt = k / float(fps)
    speed_in = np.linalg.norm(incoming, axis=1) / dt
    speed_out = np.linalg.norm(outgoing, axis=1) / dt

    dot = np.einsum("ij,ij->i", incoming, outgoing)
    norms = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    cos_angle = np.divide(dot, norms, out=np.zeros(n), where=norms > 1e-9)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    score = ANGLE_WEIGHT * angle_deg + SPEED_WEIGHT * np.abs(speed_out - speed_in)
    return score, speed_in, speed_out, valid_window


def _nonmax_suppress(candidate_idx, score, min_gap):
    """Collapse a run of candidates within min_gap of each other to its single highest score."""
    touches = []
    i = 0
    while i < len(candidate_idx):
        j = i
        best = candidate_idx[i]
        while j + 1 < len(candidate_idx) and candidate_idx[j + 1] - candidate_idx[j] <= min_gap:
            j += 1
            if score[candidate_idx[j]] > score[best]:
                best = candidate_idx[j]
        touches.append(int(best))
        i = j + 1
    return touches


def _attribute_toucher(frame, ball_xy, players_xy, team_of_track, radius):
    """(team, player_slot, distance) nearest player to the ball at `frame`, or (None, None, None)."""
    positions = players_xy[frame]
    present = ~np.isnan(positions[:, 0])
    if not present.any():
        return None, None, None
    diffs = positions[present] - ball_xy[frame]
    distances = np.linalg.norm(diffs, axis=1)
    best = int(np.argmin(distances))
    distance = float(distances[best])
    if distance > radius:
        return None, None, distance
    slot = int(np.flatnonzero(present)[best])
    return int(team_of_track[slot]), slot, distance


def detect(ball_xy, status, players_xy, team_of_track, fps=25, radius=CONTROL_RADIUS):
    """[{frame, team, player, distance, score}] in frame order.

    team/player are None where a touch was clearly detected but no player was close enough to
    credit -- a bobble, a bounce off a post, or tracking noise near the goalline. Those events are
    kept (Phase 3 may still use their timing) but never drive a possession decision on their own.
    """
    score, speed_in, speed_out, valid_window = _touch_scores(ball_xy, status, fps)
    fast_enough = (speed_in >= MIN_SPEED) | (speed_out >= MIN_SPEED)
    candidate = valid_window & fast_enough & (score >= SCORE_THRESHOLD)

    frames = _nonmax_suppress(np.flatnonzero(candidate), score, MIN_GAP_FRAMES)

    touches = []
    for frame in frames:
        team, player, distance = _attribute_toucher(frame, ball_xy, players_xy, team_of_track,
                                                      radius)
        touches.append({
            "frame": frame, "team": team, "player": player,
            "distance": distance, "score": float(score[frame]),
        })
    return touches


if __name__ == "__main__":
    import parse_atd
    from gap_fill import fill_short_gaps

    data = parse_atd.parse()
    filled_xy, status = fill_short_gaps(data["ball_xy"])
    touches = detect(filled_xy, status, data["players_xy"], data["team_of_track"])

    resolved = [t for t in touches if t["team"] is not None]
    print("{} touch candidates, {} with a credited toucher ({:.0f}%)".format(
        len(touches), len(resolved), 100.0 * len(resolved) / max(1, len(touches))))
    print("\nfirst 15 touches:")
    for t in touches[:15]:
        team = {0: "H", 1: "A", None: "?"}[t["team"]]
        print("  frame {:>7}  team {}  player {}  dist {}  score {:5.1f}".format(
            t["frame"], team, t["player"],
            "{:.2f}m".format(t["distance"]) if t["distance"] is not None else "n/a",
            t["score"]))
