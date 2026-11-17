"""Phase 4a -- the ball leaving the pitch and the set piece that brings it back.

A throw-in, corner or goal kick exists only because the ball crossed a line, and it ends only
when a player strikes a still ball from the right spot. Both are facts the tracking can see, so
the dead stretch is measured from them. Smart Tagging tags are assumed complete, and each one is
used only as a label: it says a set piece of that type happened here and confirms the exit. Its
start and end frames are never a boundary, and its Team field is never read.

  1. Exit trigger: the ball is more than EXIT_MARGIN past a line for at least EXIT_MIN_FRAMES
     observed frames. Gaps in the tracking inside the run do not break it. A side line means a
     throw-in. A goal line means a corner or a goal kick, and the tag says which.

  2. Restart trigger: the ball sits still at the spot the laws require for that type, then is
     struck. A throw-in is taken from the side line, a corner from a corner arc, a goal kick
     from inside the goal area.

  3. Real-restart check: after the strike the ball travels at least LAUNCH_DISTANCE away and
     stays on the pitch until the next touch or for STAY_FRAMES. A ball tossed to the taker, a
     dummy run or a throw that goes straight back out fails this, and the search moves on.

The taker is the nearest player to the ball at the strike. When nobody is close enough, the
team that did not touch the ball last before the exit takes it, as the laws say.
"""

import numpy as np

EXIT_MARGIN = 0.5          # m past the line before the ball counts as out
EXIT_MIN_FRAMES = 3        # observed out frames, so one noisy reading is not an exit
EXIT_GAP_FRAMES = 50       # an out run survives this many frames with no out reading inside
PROJECT_FIT_FRAMES = 5     # tracked frames before a gap used to fit the ball's path
PROJECT_MAX_FRAMES = 25    # project at most 1 s into a gap: a ball lost longer is not followed
PROJECT_MIN_SPEED = 5.0    # m/s -- slower than this the ball is not heading anywhere
NEAR_LINE = 2.0            # m from a line where a ball heading out is an exit candidate
NEAR_FIT_FRAMES = 5        # frames used to tell the ball is moving towards the line
NEAR_MIN_SPEED = 1.0       # m/s towards the line
STILL_FRAMES = 10          # 0.4 s the ball must sit still before the strike
STILL_SPEED = {"throw-in": 2.0, "corner kick": 1.5, "goal kick": 1.5}  # m/s; tracking jitter on a still ball is ~1 m/s
STRIKE_SPEED = 4.0         # m/s the ball must reach within 5 frames of the strike
LAUNCH_FRAMES = 25         # 1 s after the strike to get the ball away
LAUNCH_DISTANCE = 5.0      # m the ball must travel from the spot inside LAUNCH_FRAMES
STAY_FRAMES = 50           # 2 s the ball must stay on the pitch, unless touched again sooner
STAY_MARGIN = 1.0          # m inside the line that counts as back on the pitch
TAKER_RADIUS = 2.5         # m -- the taker is standing at the ball
TAKER_SEARCH = 10          # frames either side of the strike to find the taker

BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16
BOX_CROWD = 8              # players in the penalty box while a corner is set up
BOX_STILL_SPEED = 1.0      # m/s median speed of the box crowd while they wait
BOX_STILL_FRAMES = 50      # 2 s the crowd waits, on average below BOX_STILL_SPEED
BOX_GO_SPEED = 1.5         # m/s -- the crowd runs when the ball is kicked
BOX_SMOOTH = 12            # frames of smoothing on the crowd's speed

SIX_YARD_DEPTH = 5.5
SIX_YARD_HALF_WIDTH = 9.16
SPOT_SLACK = 1.5           # m of tracking error allowed around each restart spot
CORNER_RADIUS = 3.0
THROW_BAND = 3.0           # m either side of the side line where a throw is taken


def ball_speed(ball_xy, fps):
    """Per-frame ball speed in m/s, NaN where either neighbour is untracked."""
    step = np.full(len(ball_xy), np.nan)
    step[1:] = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * fps
    return step


def _projected_exits(ball_xy, length, width, observed):
    """[{first, last, line, projected}] -- the ball heading off the pitch into a tracking gap.

    The tracker often loses the ball exactly as it goes out: it flies over the hoardings, into
    the crowd or behind the goal. The last PROJECT_FIT_FRAMES before the gap give its velocity.
    If the straight line from there crosses a line within PROJECT_MAX_FRAMES, the crossing is
    an exit. A gap that already holds an observed exit is left to that one.
    """
    x = ball_xy[:, 0]
    seen = ~np.isnan(x)
    taken = np.zeros(len(x), dtype=bool)
    for ex in observed:
        taken[max(0, ex["first"] - EXIT_GAP_FRAMES):ex["last"] + 1] = True
    edges = np.flatnonzero(seen[:-1] & ~seen[1:])  # t tracked, t + 1 lost
    found = []
    for t in edges:
        lo = t - PROJECT_FIT_FRAMES
        if lo < 0 or not seen[lo:t + 1].all() or taken[t]:
            continue
        gap_end = t + 1
        while gap_end < len(x) and not seen[gap_end]:
            gap_end += 1
        if gap_end - t < EXIT_MIN_FRAMES + 2:
            continue  # a blink, not a ball that went somewhere
        velocity = (ball_xy[t] - ball_xy[lo]) / PROJECT_FIT_FRAMES
        if np.linalg.norm(velocity) * 25 < PROJECT_MIN_SPEED:
            continue
        for k in range(1, min(PROJECT_MAX_FRAMES, gap_end - t) + 1):
            px, py = ball_xy[t] + k * velocity
            side = py < -EXIT_MARGIN or py > width + EXIT_MARGIN
            goal = px < -EXIT_MARGIN or px > length + EXIT_MARGIN
            if side or goal:
                found.append({"first": int(t + k), "last": int(gap_end - 1),
                              "line": "side" if side and not goal else "goal", "projected": True,
                              "near": False})
                break
    return found


def _near_line_exits(ball_xy, length, width, taken_exits):
    """[{first, last, line, projected, near}] -- the ball within NEAR_LINE of a line and moving
    towards it.

    The tracker often loses the ball, or stops it at the line, before it is clearly out. A ball
    heading out that close to a line may have gone out, so it is a candidate. Most candidates are
    play along the line; only the ones a Smart Tag of the right type confirms are kept, and an
    observed or projected exit is always preferred over one of these.
    """
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    seen = ~np.isnan(x)
    gaps = {"side": np.minimum(y, width - y), "goal": np.minimum(x, length - x)}
    taken = np.zeros(len(x), dtype=bool)
    for ex in taken_exits:
        taken[ex["first"]:ex["last"] + 1] = True
    found = []
    for line, gap in gaps.items():
        near = seen & (gap <= NEAR_LINE)
        towards = np.zeros(len(x), dtype=bool)
        k = NEAR_FIT_FRAMES
        towards[k:] = (gap[:-k] - gap[k:]) * 25 / k >= NEAR_MIN_SPEED
        hit = near & towards & ~taken
        t = 0
        while t < len(x):
            if not hit[t]:
                t += 1
                continue
            first = t
            while t < len(x) and (near[t] or not seen[t]) and t - first < EXIT_GAP_FRAMES * 10:
                t += 1
            found.append({"first": int(first), "last": int(max(first, t - 1)), "line": line,
                          "projected": False, "near": True})
    return found


def exits(ball_xy, length, width, project=True, near=True):
    """[{first, last, line, projected}] -- every run where the ball is off the pitch.

    Observed exits first. With `project`, a ball last seen heading over a line into a tracking
    gap is an exit too, at the frame its path crosses the line.
    """
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    seen = ~np.isnan(x)
    side = seen & ((y < -EXIT_MARGIN) | (y > width + EXIT_MARGIN))
    goal = seen & ((x < -EXIT_MARGIN) | (x > length + EXIT_MARGIN))
    out = side | goal
    found = []
    n = len(ball_xy)
    t = 0
    while t < n:
        if not out[t]:
            t += 1
            continue
        first = last = t
        count = 0
        while t < n and (out[t] or not seen[t]) and t - last < EXIT_GAP_FRAMES:
            if out[t]:
                count += 1
                last = t
            t += 1
        if count >= EXIT_MIN_FRAMES:
            found.append({"first": first, "last": last, "line": "side" if side[first] else "goal",
                          "projected": False, "near": False})
    if project:
        found = found + _projected_exits(ball_xy, length, width, found)
    if near:
        found = found + _near_line_exits(ball_xy, length, width, found)
    return sorted(found, key=lambda ex: ex["first"])


def _at_spot(pos, kind, length, width):
    x, y = pos
    if np.isnan(x):
        return False
    if kind == "throw-in":
        return min(abs(y), abs(y - width)) <= THROW_BAND and -SPOT_SLACK <= x <= length + SPOT_SLACK
    if kind == "corner kick":
        return any(np.hypot(x - cx, y - cy) <= CORNER_RADIUS
                   for cx in (0.0, length) for cy in (0.0, width))
    if kind == "goal kick":
        in_depth = x <= SIX_YARD_DEPTH + SPOT_SLACK or x >= length - SIX_YARD_DEPTH - SPOT_SLACK
        return in_depth and abs(y - width / 2.0) <= SIX_YARD_HALF_WIDTH + SPOT_SLACK
    return False


def _on_pitch(pos, length, width):
    return (not np.isnan(pos[0]) and STAY_MARGIN <= pos[0] <= length - STAY_MARGIN
            and STAY_MARGIN <= pos[1] <= width - STAY_MARGIN)


def _is_real(strike, ball_xy, touch_frames, length, width):
    """The ball gets away from the spot and stays on the pitch: the restart was taken."""
    n = len(ball_xy)
    origin = ball_xy[strike]
    launch = ball_xy[strike + 1:min(n, strike + 1 + LAUNCH_FRAMES)]
    seen = launch[~np.isnan(launch[:, 0])]
    if not len(seen) or np.linalg.norm(seen - origin, axis=1).max() < LAUNCH_DISTANCE:
        return False
    later = [f for f in touch_frames if f > strike + 2]
    stay_end = min(n, strike + STAY_FRAMES, later[0] if later else n)
    for t in range(strike + LAUNCH_FRAMES // 2, stay_end):
        pos = ball_xy[t]
        if not np.isnan(pos[0]) and not _on_pitch(pos, length, width) and \
                not _at_spot(pos, "throw-in", length, width):
            return False
        if not np.isnan(pos[0]) and (pos[0] < -EXIT_MARGIN or pos[0] > length + EXIT_MARGIN or
                                     pos[1] < -EXIT_MARGIN or pos[1] > width + EXIT_MARGIN):
            return False
    return True


def back_in_play(after, kind, ball_xy, length, width):
    """First frame after `after` that starts a STAY_FRAMES stretch of the ball on the pitch and
    away from the restart spot.

    The restart cannot be later than this: a ball that has been rolling about the pitch for
    two seconds is live, whoever put it there. A ball placed in the goal area or carried to the
    side line is still waiting for the restart, so those frames do not count.
    """
    n = len(ball_xy)
    run = 0
    for t in range(after + 1, n):
        if _on_pitch(ball_xy[t], length, width) and not _at_spot(ball_xy[t], kind, length, width):
            run += 1
            if run >= STAY_FRAMES:
                return t - STAY_FRAMES + 1
        elif not np.isnan(ball_xy[t, 0]):
            run = 0
    return n - 1


def find_strike(kind, first, limit, ball_xy, speed, touch_frames, length, width):
    """First frame in [first, limit] where a still ball at the spot is struck for real, or None."""
    still_speed = STILL_SPEED.get(kind, 1.0)
    n = len(ball_xy)
    for t in range(max(first, STILL_FRAMES), min(limit, n - LAUNCH_FRAMES - 1) + 1):
        if not _at_spot(ball_xy[t], kind, length, width):
            continue
        before = speed[t - STILL_FRAMES + 1:t + 1]
        before = before[~np.isnan(before)]
        if len(before) < STILL_FRAMES // 2 or before.mean() >= still_speed:
            continue
        after = speed[t + 1:t + 6]
        after = after[~np.isnan(after)]
        if not len(after) or after.max() < STRIKE_SPEED:
            continue
        if _is_real(t, ball_xy, touch_frames, length, width):
            return t
    return None


def box_restart(first, limit, end_x, players_xy, players_speed, length, width, not_before=None):
    """Frame a corner is kicked, read from the players in the box, or None.

    The tracker rarely sees the ball at the corner flag, but the players are always seen. While
    a corner is set up, BOX_CROWD or more of them stand in the penalty box at that end, almost
    still. When the ball is kicked they all move at once. The kick is the first frame where the
    crowd's median speed rises through BOX_GO_SPEED after BOX_STILL_FRAMES with the crowd in the
    box averaging under BOX_STILL_SPEED. With `not_before`, the waiting may start earlier but the
    kick cannot: a ball seen placed at the flag has not been kicked yet.
    """
    lo, hi = max(0, first), min(len(players_xy), limit + 1)
    if hi - lo < BOX_STILL_FRAMES:
        return None
    x = players_xy[lo:hi, :, 0]
    y = players_xy[lo:hi, :, 1]
    depth = x >= length - BOX_DEPTH if end_x > length / 2.0 else x <= BOX_DEPTH
    in_box = depth & (np.abs(y - width / 2.0) <= BOX_HALF_WIDTH)
    crowd = in_box.sum(axis=1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(in_box, players_speed[lo:hi], np.nan), axis=1)
    med = np.convolve(np.where(np.isnan(med), 0.0, med), np.ones(BOX_SMOOTH) / BOX_SMOOTH, "same")
    start = BOX_STILL_FRAMES if not_before is None else max(BOX_STILL_FRAMES, not_before - lo)
    for t in range(start, len(med)):
        if crowd[t] < BOX_CROWD or med[t] < BOX_GO_SPEED or med[t - 1] >= BOX_GO_SPEED:
            continue
        if med[t - BOX_STILL_FRAMES:t].mean() < BOX_STILL_SPEED and \
                crowd[t - BOX_STILL_FRAMES:t].min() >= BOX_CROWD:
            return lo + t
    return None


def end_x(frame, ball_xy):
    """x of the ball last seen at or before `frame`, the end the ball went out at."""
    seen = np.flatnonzero(~np.isnan(ball_xy[:frame + 1, 0]))
    return float(ball_xy[seen[-1], 0]) if len(seen) else np.nan


def corner_taker_team(frame, end, players_xy, team_of_track, length, width):
    """Team of the player nearest a corner flag at that end at the kick."""
    cx = length if end > length / 2.0 else 0.0
    pos = players_xy[frame]
    present = ~np.isnan(pos[:, 0])
    if not present.any():
        return None
    d = np.min([np.hypot(pos[present, 0] - cx, pos[present, 1] - cy) for cy in (0.0, width)], axis=0)
    k = int(np.argmin(d))
    return None if d[k] > CORNER_RADIUS * 2 else int(team_of_track[int(np.flatnonzero(present)[k])])


def taker_team(strike, ball_xy, players_xy, team_of_track):
    """Team of the player standing at the ball at the strike, or None."""
    best = None
    for t in range(max(0, strike - TAKER_SEARCH), min(len(ball_xy), strike + TAKER_SEARCH + 1)):
        if np.isnan(ball_xy[t, 0]):
            continue
        positions = players_xy[t]
        present = ~np.isnan(positions[:, 0])
        if not present.any():
            continue
        distances = np.linalg.norm(positions[present] - ball_xy[t], axis=1)
        k = int(np.argmin(distances))
        if distances[k] > TAKER_RADIUS:
            continue
        key = (abs(t - strike), float(distances[k]))
        if best is None or key < best[0]:
            best = (key, int(team_of_track[int(np.flatnonzero(present)[k])]))
    return None if best is None else best[1]


def last_toucher(frame, touches):
    """Team of the last credited touch before `frame`, or None."""
    for touch in reversed(touches or ()):
        if touch["frame"] < frame and touch["team"] is not None:
            return int(touch["team"])
    return None
