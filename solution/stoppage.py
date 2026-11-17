"""Phase 4b -- find stoppages from the players, not the ball: a whistle stops all 22 at once.

Smart Tagging misses stoppages that have no set-piece tag (a foul, an offside, an injury), and the
ball often does not leave the pitch for them, so neither of dead_ball.py's first two signals can
see them. What every stoppage does have is the players slowing to a walk together and staying
there until the restart. In live play somebody is always running, so the median player speed of
the whole pitch rarely stays low for long.

Three pieces, each with as few numbers as possible:

  1. Calm run: the median player speed (the ATD's own speed field, averaged over one second)
     stays below CALM_SPEED for at least CALM_MIN_FRAMES. Neither number is typed in. Both are
     learned per match from the Smart Tagging set pieces by calibrate(): the pair whose calm runs
     best end at the tagged restart instants, one run per restart. Smart Tagging is automatic, so these are
     labels produced by an algorithm.

  2. Whistle: walk back from the start of the calm run while the players were still slowing down.
     The deceleration starts at the whistle, so its first frame is the start of the stoppage. No
     window length to choose.

  3. Restart: the last touch (touch_detector.py) inside the LAST END_WINDOW_SECONDS of the calm
     run. The restart is the strike that ends the stoppage, and the players only speed up after
     it, so it falls just inside the run's end. Taking the last touch rather than the first skips
     the ball being rolled to the taker. A touch earlier than the window is a mid-stop touch (the
     ball placed, a player nudging it), not the restart, so it is ignored. If there is no touch
     in the window, the stoppage ends where the calm run ends. The window is fixed in advance, a
     little longer than the lag between a kick and the players' speed rising again.
"""

import warnings

import numpy as np

# Calibration search space. Wide on purpose: the answer should come from the data, and an edge
# hit here is printed rather than quietly accepted.
END_WINDOW_SECONDS = 5.0    # a restart touch must fall within this long of the end of the calm run
CALM_SPEED_GRID = np.round(np.arange(0.5, 3.01, 0.1), 2)   # m/s
CALM_SECONDS_GRID = np.arange(1, 13)                         # s


def collective_speed(players_speed, fps):
    """Median speed of every tracked player per frame, averaged over one second. NaN if nobody."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # frames with nobody tracked
        median = np.nanmedian(players_speed, axis=1)
    seen = ~np.isnan(median)
    kernel = np.ones(fps)
    total = np.convolve(np.where(seen, median, 0.0), kernel, "same")
    count = np.convolve(seen.astype(float), kernel, "same")
    return np.divide(total, count, out=np.full(len(median), np.nan), where=count > 0)


def runs(mask):
    """[(first, last)] inclusive spans where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def calm_runs(speed, in_play, calm_speed, min_frames):
    calm = in_play & (np.nan_to_num(speed, nan=np.inf) < calm_speed)
    return [(a, b) for a, b in runs(calm) if b - a + 1 >= min_frames]


def _matches(spans, restarts, slack):
    """How many calm runs end at a tagged restart, one to one.

    A run matches a restart if the restart falls inside it or within `slack` frames after its end
    (the players speed up just after the strike). Each restart is used once, so one long run that
    swallows three restarts counts once, not three times.
    """
    used = np.zeros(len(restarts), dtype=bool)
    hits = 0
    for a, b in spans:
        inside = np.flatnonzero(~used & (restarts >= a) & (restarts <= b + slack))
        if len(inside):
            used[inside[0]] = True
            hits += 1
    return hits


def calibrate(speed, in_play, restart_frames, fps):
    """(calm_speed, min_frames, f1): the calm rule that best lines up with tagged restarts.

    precision = share of calm runs that end at a tagged restart
    recall    = share of tagged restarts that a calm run ends at

    Stoppages Smart Tagging missed count against precision, so the fit leans strict. That is the
    safe direction for a signal that can also fire on slow open play.
    """
    restarts = np.sort(np.asarray(restart_frames, dtype=np.int64))
    best = None
    for calm_speed in CALM_SPEED_GRID:
        spans_all = runs(in_play & (np.nan_to_num(speed, nan=np.inf) < calm_speed))
        for seconds in CALM_SECONDS_GRID:
            spans = [(a, b) for a, b in spans_all if b - a + 1 >= seconds * fps]
            if not spans:
                continue
            hits = _matches(spans, restarts, slack=fps)
            precision, recall = hits / len(spans), hits / len(restarts)
            if hits == 0:
                continue
            f1 = 2 * precision * recall / (precision + recall)
            if best is None or f1 > best[2]:
                best = (float(calm_speed), int(seconds * fps), float(f1))
    calm_speed, min_frames, _ = best
    if calm_speed in (CALM_SPEED_GRID[0], CALM_SPEED_GRID[-1]) or \
            min_frames // fps in (CALM_SECONDS_GRID[0], CALM_SECONDS_GRID[-1]):
        print("  stoppage.calibrate: best value is on the edge of the search grid -- widen it")
    return best


def whistle_frame(speed, first):
    """Walk back from `first` while the players were slowing down; that is where the whistle went."""
    t = first
    while t > 0 and not np.isnan(speed[t - 1]) and speed[t - 1] > speed[t]:
        t -= 1
    return t


def restart_touch(touches, first, last, window_frames=None):
    """(frame, team) of the last credited touch inside [first, last], or None.

    With `window_frames`, only touches in the final window_frames of the span count: a touch
    earlier than that is not the restart.
    """
    if window_frames is not None:
        first = max(first, last - int(window_frames))
    inside = [t for t in touches if first <= t["frame"] <= last and t["team"] is not None]
    return (inside[-1]["frame"], inside[-1]["team"]) if inside else None
