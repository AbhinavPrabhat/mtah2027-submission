"""Phase 1a -- parse the raw ATD into dense per-frame numpy arrays, cached to disk.

starter/baseline.py streams the raw ATD once and keeps only "which team is nearest the ball".
Everything downstream in this solution (touch detection, control, dead-ball) needs more than
that: the full ball and player trajectories, in metres, with speed. This module is the one place
that re-reads the 199k-line raw file and turns it into arrays; every other module loads the cached
.npz instead of touching the raw text again.

    python solution/parse_atd.py          # parses once, writes solution/cache/BAR-ATH_atd.npz

Coordinates come out of the ATD normalised 0-1 (DATA.md); this module scales them to metres by the
real pitch size immediately, so nothing downstream has to remember to do it.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names, FIELD_SEP, GROUP_SEP, MISSING  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")


def cache_path(game=GAME):
    return os.path.join(CACHE_DIR, "{}_atd.npz".format(game))


def parse(game=GAME, data_dir=None, force=False):
    """{ball_xy, ball_speed, players_xy, players_speed, team_of_track} -- all metres / m/s.

    ball_xy, players_xy: NaN where the tracker did not see the object that frame.
    team_of_track: 0 (home) or 1 (away) per track slot, same convention as starter/baseline.py.
    """
    path = cache_path(game)
    if not force and os.path.isfile(path):
        data = np.load(path)
        return {name: data[name] for name in data.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)
    n_slots = len(team_of_track)

    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float64)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float64)
    players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float64)
    players_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float64)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0]) - 1  # ATD counts from 1, arrays count from 0
            if not (0 <= frame_idx < n_frames):
                continue

            ball = parts[2]
            if ball[0] != MISSING:
                bx, by, bs = ball.split(",")
                ball_xy[frame_idx, 0] = float(bx) * length
                ball_xy[frame_idx, 1] = float(by) * width
                ball_speed[frame_idx] = float(bs)

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                gx, gy, gs = group.split(",")
                players_xy[frame_idx, slot, 0] = float(gx) * length
                players_xy[frame_idx, slot, 1] = float(gy) * width
                players_speed[frame_idx, slot] = float(gs)

    seen = int(np.sum(~np.isnan(ball_xy[:, 0])))
    print("  ball located on {} of {} frames ({:.1f}%)".format(
        seen, n_frames, 100.0 * seen / n_frames))

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        path, ball_xy=ball_xy, ball_speed=ball_speed, players_xy=players_xy,
        players_speed=players_speed, team_of_track=team_of_track)
    print("  cached -> {}".format(os.path.relpath(path, ROOT)))

    return dict(ball_xy=ball_xy, ball_speed=ball_speed, players_xy=players_xy,
                players_speed=players_speed, team_of_track=team_of_track)


if __name__ == "__main__":
    data = parse(force="--force" in sys.argv)
    print("\narrays:")
    for name, array in data.items():
        print("  {:<14} {}".format(name, array.shape))
