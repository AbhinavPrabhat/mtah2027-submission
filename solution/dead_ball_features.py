"""Per-frame player features for the learned stoppage detector (dead_ball_ml.py).

Everything here comes from where the players are and how fast they move. No ball column, no
label, no tag. Half 2 is turned 180 degrees before anything is computed, so in both halves each
team attacks the same end.

  frame_features / extra_features   one number per frame: team shape, speeds, sprints, standing,
                                    pressing, crowding, braking, the densest group, the fastest players
  window_features                   the same numbers averaged over +-0.5 s ... +-8 s, past and future
                                    windows, and lags: a stoppage shows up over seconds
  second_round                      how LONG the players have been (and will stay) still, how long
                                    since or until a sprint, how unevenly they move, the 10 m crowd

build() returns one row per in-play frame.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)   # frames with nobody tracked

FPS = 25
SPRINT = 5.0         # m/s
NEAR_LINE = 3.0      # m
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
DENSE_RADIUS = 5.0   # m -- players within this of a player count towards that spot's crowd
STILL_SPEED = 1.0    # m/s
STILL_FRAMES = 75    # 3 s under STILL_SPEED counts as standing
WINDOWS = (12, 25, 50, 100, 200)   # half-widths in frames: +-0.5 s ... +-8 s
SIDED = (50, 100)                  # past-only / future-only windows, 2 s and 4 s
CAP = 60 * FPS


def _nan_stats(values, axis=1):
    with np.errstate(all="ignore"):
        return (np.nanmean(values, axis), np.nanstd(values, axis), np.nanmedian(values, axis),
                np.nanmax(values, axis), np.nanmin(values, axis))


def frame_features(players_xy, players_speed, team_of_track, length, width):
    """dict of per-frame arrays from player positions and speeds only."""
    x, y = players_xy[..., 0], players_xy[..., 1]
    present = ~np.isnan(x)
    vel = np.full_like(players_xy, np.nan)
    vel[1:] = (players_xy[1:] - players_xy[:-1]) * FPS
    speed = np.where(np.isnan(players_speed), np.linalg.norm(vel, axis=2), players_speed)
    f = {}
    with np.errstate(all="ignore"):
        f["n_present"] = present.sum(1)
        for name, mask in (("all", np.ones(len(team_of_track), bool)),
                           ("h", team_of_track == 0), ("a", team_of_track == 1)):
            xs, ys, sp = x[:, mask], y[:, mask], speed[:, mask]
            f[name + "_cx"], f[name + "_sx"], _, f[name + "_maxx"], f[name + "_minx"] = _nan_stats(xs)
            f[name + "_cy"], f[name + "_sy"], _, f[name + "_maxy"], f[name + "_miny"] = _nan_stats(ys)
            f[name + "_spd_mean"], f[name + "_spd_std"], f[name + "_spd_med"], f[name + "_spd_max"], _ = _nan_stats(sp)
            f[name + "_spd_p90"] = np.nanpercentile(sp, 90, axis=1)
            f[name + "_n_sprint"] = (sp > SPRINT).sum(1)
            f[name + "_n_still"] = (sp < 0.5).sum(1)
            vx, vy = vel[:, mask, 0], vel[:, mask, 1]
            f[name + "_vx"] = np.nanmean(vx, 1)
            f[name + "_vy"] = np.nanmean(vy, 1)
            # how much the players move together: |mean velocity| / mean |velocity|
            f[name + "_align"] = np.hypot(f[name + "_vx"], f[name + "_vy"]) / (np.nanmean(np.hypot(vx, vy), 1) + 0.1)
            f[name + "_x_range"] = f[name + "_maxx"] - f[name + "_minx"]
            f[name + "_y_range"] = f[name + "_maxy"] - f[name + "_miny"]
        f["team_dx"] = f["h_cx"] - f["a_cx"]
        f["team_dy"] = f["h_cy"] - f["a_cy"]
        f["team_dist"] = np.hypot(f["team_dx"], f["team_dy"])
        # convergence: mean distance of all players to the overall centre
        d_c = np.hypot(x - f["all_cx"][:, None], y - f["all_cy"][:, None])
        f["all_rad"] = np.nanmean(d_c, 1)
        # where on the pitch
        side_gap = np.minimum(y, width - y)
        goal_gap = np.minimum(x, length - x)
        f["n_near_side"] = (side_gap < NEAR_LINE).sum(1)
        f["n_near_goal"] = (goal_gap < NEAR_LINE).sum(1)
        f["n_off_pitch"] = ((x < 0) | (x > length) | (y < 0) | (y > width)).sum(1)
        in_y = np.abs(y - width / 2.0) <= BOX_HALF_WIDTH
        for end, mask in (("l", x <= BOX_DEPTH), ("r", x >= length - BOX_DEPTH)):
            box = mask & in_y
            f["box_" + end + "_n"] = box.sum(1)
            f["box_" + end + "_h"] = (box & (team_of_track == 0)).sum(1)
            f["box_" + end + "_spd"] = np.nanmedian(np.where(box, speed, np.nan), 1)
        f["box_max_n"] = np.maximum(f["box_l_n"], f["box_r_n"])
        # nearest-opponent distances: marking is tight in a set piece, loose in open play
        hx, hy = x[:, team_of_track == 0], y[:, team_of_track == 0]
        ax, ay = x[:, team_of_track == 1], y[:, team_of_track == 1]
        dist = np.hypot(hx[:, :, None] - ax[:, None, :], hy[:, :, None] - ay[:, None, :])
        nn = np.nanmin(dist, axis=2)
        f["mark_mean"] = np.nanmean(nn, 1)
        f["mark_min"] = np.nanmin(nn, 1)
        f["n_pairs_2m"] = (dist < 2.0).sum((1, 2))
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def extra_features(players_xy, players_speed, half_starts):
    """dict of g_* arrays: where the players crowd, what the fastest do, who is standing."""
    x, y = players_xy[..., 0], players_xy[..., 1]
    n, p = x.shape
    speed = players_speed.copy()
    vel = np.full_like(players_xy, np.nan)
    vel[1:] = (players_xy[1:] - players_xy[:-1]) * FPS
    speed = np.where(np.isnan(speed), np.linalg.norm(vel, axis=2), speed)
    g = {}
    with np.errstate(all="ignore"):
        # densest spot: the player with most others inside DENSE_RADIUS, and that crowd's centre
        dx = x[:, :, None] - x[:, None, :]
        dy = y[:, :, None] - y[:, None, :]
        near = np.hypot(dx, dy) <= DENSE_RADIUS
        count = near.sum(2).astype(np.float32)
        count[np.isnan(x)] = -1
        k = np.argmax(count, axis=1)
        crowd = near[np.arange(n), k]
        g["dense_n"] = count[np.arange(n), k]
        g["dense_x"] = np.nanmean(np.where(crowd, x, np.nan), 1)
        g["dense_y"] = np.nanmean(np.where(crowd, y, np.nan), 1)
        # the 3 fastest players: where they are and where they head
        order = np.argsort(-np.nan_to_num(speed, nan=-1), axis=1)[:, :3]
        rows = np.arange(n)[:, None]
        g["fast3_x"] = np.nanmean(x[rows, order], 1)
        g["fast3_y"] = np.nanmean(y[rows, order], 1)
        g["fast3_spd"] = np.nanmean(speed[rows, order], 1)
        g["fast3_vx"] = np.nanmean(vel[rows, order, 0], 1)
        g["fast3_vy"] = np.nanmean(vel[rows, order, 1], 1)
        # speed-weighted centre: the players who move pull the centre towards the ball
        wgt = np.nan_to_num(speed, nan=0.0) + 0.1
        wgt = np.where(np.isnan(x), 0.0, wgt)
        g["wcen_x"] = np.nansum(np.nan_to_num(x) * wgt, 1) / wgt.sum(1)
        g["wcen_y"] = np.nansum(np.nan_to_num(y) * wgt, 1) / wgt.sum(1)
        # acceleration: a whistle stops everyone, a restart starts everyone
        acc = np.full_like(speed, np.nan)
        acc[1:] = (speed[1:] - speed[:-1]) * FPS
        g["acc_mean"] = np.nanmean(acc, 1)
        g["n_braking"] = (acc < -2.0).sum(1)
        g["n_accel"] = (acc > 2.0).sum(1)
        # standing: players under STILL_SPEED for the last STILL_FRAMES, counted per half
        slow = ((speed < STILL_SPEED) & ~np.isnan(x)).astype(np.int32)  # absent slots never stand
        held = np.zeros_like(slow)
        bounds = list(half_starts) + [n]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            run = np.zeros(p, np.int32)
            for t in range(lo, hi):
                run = (run + 1) * slow[t]
                held[t] = run
        g["n_standing_3s"] = (held >= STILL_FRAMES).sum(1)
        g["share_standing_3s"] = g["n_standing_3s"] / np.maximum(1, (~np.isnan(x)).sum(1))
    return {k: np.asarray(v, dtype=np.float32) for k, v in g.items()}


def _centered_mean(a, k):
    s = pd.Series(a)
    return s.rolling(2 * k + 1, center=True, min_periods=1).mean().to_numpy(np.float32)


def _centered_std(a, k):
    s = pd.Series(a)
    return s.rolling(2 * k + 1, center=True, min_periods=2).std().to_numpy(np.float32)


def _past_mean(a, k):
    return pd.Series(a).rolling(k, min_periods=1).mean().to_numpy(np.float32)


def _future_mean(a, k):
    return pd.Series(a[::-1]).rolling(k, min_periods=1).mean().to_numpy(np.float32)[::-1]


WINDOWED = ("all_spd_mean", "all_spd_med", "all_spd_p90", "all_n_sprint", "all_n_still", "all_align",
            "all_sx", "all_sy", "all_rad", "team_dist", "team_dx", "h_spd_mean", "a_spd_mean",
            "n_near_side", "n_near_goal", "n_off_pitch", "box_max_n", "box_l_spd", "box_r_spd",
            "mark_mean", "n_pairs_2m", "all_vx", "all_vy", "all_cx",
            "dense_n", "fast3_spd", "acc_mean", "n_braking", "n_accel", "n_standing_3s", "share_standing_3s")


def window_features(f, half_starts):
    """Rolling features, computed per half so no window crosses halftime."""
    out = {}
    n = len(next(iter(f.values())))
    bounds = list(half_starts) + [n]
    for name in WINDOWED:
        base = f[name]
        cols = {}
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            a = np.nan_to_num(base[lo:hi], nan=np.nanmedian(base[lo:hi]))
            for k in WINDOWS:
                cols.setdefault("w%d_mean_%s" % (k, name), []).append(_centered_mean(a, k))
                cols.setdefault("w%d_std_%s" % (k, name), []).append(_centered_std(a, k))
            for k in SIDED:
                past, fut = _past_mean(a, k), _future_mean(a, k)
                cols.setdefault("p%d_%s" % (k, name), []).append(past)
                cols.setdefault("n%d_%s" % (k, name), []).append(fut)
                cols.setdefault("d%d_%s" % (k, name), []).append(fut - past)
            for lag in (-50, -25, 25, 50):
                shifted = np.roll(a, -lag)
                if lag > 0:
                    shifted[-lag:] = a[-1]
                else:
                    shifted[:-lag] = a[0]
                cols.setdefault("lag%+d_%s" % (lag, name), []).append(shifted - a)
        for key, parts in cols.items():
            out[key] = np.concatenate(parts).astype(np.float32)
    return out



def run_len(mask, half):
    """Length of the True run ending at each frame, restarted at every half."""
    out = np.zeros(len(mask), np.int64)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        m = mask[idx].astype(np.int64)
        c = np.cumsum(m)
        reset = np.maximum.accumulate(np.where(m == 0, c, 0))
        out[idx] = c - reset
    return out


def since(mask, half):
    """Frames since the last True (0 on it), per half; CAP if none yet."""
    out = np.full(len(mask), CAP, np.int64)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        pos = np.where(mask[idx], np.arange(len(idx)), -10 ** 9)
        last = np.maximum.accumulate(pos)
        out[idx] = np.minimum(CAP, np.arange(len(idx)) - last)
    return out


def roll(v, half, k, how="mean"):
    s = pd.Series(v).groupby(half).transform(
        lambda x: getattr(x.rolling(2 * k + 1, center=True, min_periods=1), how)())
    return s.values



def second_round(feat, players_xy, half, length, width):
    """h_* columns: durations, spread and crowding. `feat` is the per-frame f_* table."""
    out = {}
    med = pd.Series(feat["f_all_spd_med"]).groupby(half).transform(
        lambda x: x.rolling(25, center=True, min_periods=1).mean()).values
    low = med < 1.5
    out["h_still_past"] = np.minimum(run_len(low, half), CAP) / FPS
    out["h_still_future"] = np.minimum(run_len(low[::-1], half[::-1])[::-1], CAP) / FPS
    active = feat["f_all_n_sprint"] >= 3
    out["h_since_sprint"] = since(active, half) / FPS
    out["h_until_sprint"] = since(active[::-1], half[::-1])[::-1] / FPS
    p90, spd_med = feat["f_all_spd_p90"], feat["f_all_spd_med"]
    disp = p90 - spd_med
    for k in (50, 125):
        out["h_disp_w%d" % k] = roll(disp, half, k)
    out["h_env_past"] = pd.Series(p90).groupby(half).transform(lambda x: x.rolling(125, min_periods=1).max()).values
    out["h_env_future"] = pd.Series(p90[::-1]).groupby(half[::-1]).transform(
        lambda x: x.rolling(125, min_periods=1).max()).values[::-1]
    n = len(players_xy)
    cnt = np.zeros(n, np.float32); line = np.zeros(n, np.float32)
    for lo in range(0, n, 4000):
        x, y = players_xy[lo:lo + 4000, :, 0], players_xy[lo:lo + 4000, :, 1]
        dist = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        c = np.nansum(dist <= 10.0, axis=2).astype(np.float32)
        c[np.isnan(x)] = -1
        k = np.argmax(c, axis=1)
        r = np.arange(len(k))
        cnt[lo:lo + 4000] = c[r, k]
        px, py = x[r, k], y[r, k]
        line[lo:lo + 4000] = np.minimum.reduce([px, length - px, py, width - py])
    out["h_dense10_n"], out["h_dense10_line"] = cnt, np.clip(line, 0, 40)
    for k in (50, 125):
        out["h_dense10_n_w%d" % k] = roll(cnt, half, k)
        out["h_dense10_line_w%d" % k] = roll(out["h_dense10_line"], half, k)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def build(players_xy, players_speed, team_of_track, frames, length, width):
    """DataFrame: frame, half, and every player feature, for the in-play `frames` (sorted)."""
    frames = np.asarray(frames)
    gap = np.flatnonzero(np.diff(frames) > 1)
    half_starts = [0] + list(gap + 1)
    pxy = players_xy[frames].copy()
    half = np.zeros(len(frames), dtype=np.int8)
    for i, lo in enumerate(half_starts):
        hi = half_starts[i + 1] if i + 1 < len(half_starts) else len(frames)
        half[lo:hi] = i + 1
    h2 = half == 2
    pxy[h2, ..., 0] = length - pxy[h2, ..., 0]
    pxy[h2, ..., 1] = width - pxy[h2, ..., 1]
    speed = players_speed[frames]
    f = frame_features(pxy, speed, team_of_track, length, width)
    g = extra_features(pxy, speed, half_starts)
    w = window_features({**f, **g}, half_starts)
    feat = {**{"f_" + k: v for k, v in f.items()}, **{"g_" + k: v for k, v in g.items()}, **w}
    feat.update(second_round(feat, pxy, half, length, width))
    out = pd.DataFrame({"frame": frames.astype(np.int32), "half": half})
    return pd.concat([out, pd.DataFrame(feat)], axis=1)
