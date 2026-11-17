"""Kickoff trigger detector.

A kickoff is taken from the centre spot, and the taker passes to a teammate, so that
team is in possession from the kick. Inside each kickoff tag, the trigger is the frame
the ball sits closest to the centre mark, and the player nearest that ball is the taker.
When the ball is never tracked, the player standing closest to the centre mark is the taker.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
import load_smart_tagging  # noqa: E402

KICKOFF_TYPES = {"kick off (start)", "kick off (after goal)"}


def _nearest_team(positions, team_of_track, point):
    """Team of the player closest to `point`, or None when nobody is tracked."""
    if np.isnan(point[0]):
        return None
    present = ~np.isnan(positions[:, 0])
    if not present.any():
        return None
    distances = np.linalg.norm(positions[present] - point, axis=1)
    slot = int(np.flatnonzero(present)[int(np.argmin(distances))])
    return int(team_of_track[slot])


def _closest_player(players_xy, team_of_track, centre, start, end):
    """(frame, team) of the player who comes closest to the centre mark, or None."""
    best = None
    for frame in range(start, end + 1):
        positions = players_xy[frame]
        present = ~np.isnan(positions[:, 0])
        if not present.any():
            continue
        distances = np.linalg.norm(positions[present] - centre, axis=1)
        distance = float(distances.min())
        if best is None or distance < best[0]:
            slot = int(np.flatnonzero(present)[int(np.argmin(distances))])
            best = (distance, frame, int(team_of_track[slot]))
    if best is None:
        return None
    return best[1], best[2]


def detect(ball_xy, players_xy, team_of_track, length, width, game=GAME):
    """[{start_frame, frame, team}] -- one trigger per kickoff tag that has a taker.

    `frame` is the first alive frame. The tag's frames before it are still dead.
    """
    centre = np.array([length / 2.0, width / 2.0])
    n = len(ball_xy)
    found = []
    events = load_smart_tagging.load(game)
    for event in events:
        if event["code"] != "SET PIECES":
            continue
        if (event["tags"].get("Type") or "").strip().lower() not in KICKOFF_TYPES:
            continue
        start = max(0, event["start_frame"])
        end = min(n - 1, event["end_frame"])
        if start > end:
            continue

        on_spot = None
        for frame in range(start, end + 1):
            if np.isnan(ball_xy[frame, 0]):
                continue
            distance = float(np.linalg.norm(ball_xy[frame] - centre))
            if on_spot is None or distance < on_spot[0]:
                on_spot = (distance, frame)

        if on_spot is not None:
            frame = on_spot[1]
            team = _nearest_team(players_xy[frame], team_of_track, ball_xy[frame])
        else:
            stood = _closest_player(players_xy, team_of_track, centre, start, end)
            if stood is None:
                continue
            frame, team = stood
        if team is None:
            continue
        found.append({"start_frame": start, "frame": frame, "team": team})
    return found


if __name__ == "__main__":
    import parse_atd

    data = parse_atd.parse()
    length, width = GAMES[GAME]["pitch"]
    kicks = detect(data["ball_xy"], data["players_xy"], data["team_of_track"], length, width)
    print("{} kickoffs".format(len(kicks)))
    for kick in kicks:
        print("  tag {:>7}  kick frame {:>7}  team {}".format(
            kick["start_frame"], kick["frame"], {0: "H", 1: "A"}[kick["team"]]))
