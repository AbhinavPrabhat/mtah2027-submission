"""Post-processor flow: runs after the main flow and only touches the dead/alive boundary.

  Blip filter. An alive stretch shorter than BLIP_SECONDS that sits between two dead spells is
     made dead unless play is visible inside it: two or more credited touches, or the ball travels
     at least BLIP_PATH metres.

Only frames change between D and H/A. Everything is fixed here from the tracking; nothing reads a
marking. Returns the new labels and a report.
"""
import numpy as np

FPS = 25
BLIP_SECONDS = 5.0
BLIP_TOUCHES = 2
BLIP_PATH = 10.0


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))


def fill_blips(dead, in_play, touches, ball):
    dead = dead.copy()
    limit = int(BLIP_SECONDS * FPS)
    tf = np.array([t["frame"] for t in touches if t["team"] is not None])
    filled, frames = 0, 0
    for a, b in _runs(~dead & in_play):
        if a == 0 or b + 1 >= len(dead) or not dead[a - 1] or not dead[b + 1] or b - a + 1 >= limit:
            continue
        n_touch = int(((tf >= a) & (tf <= b)).sum())
        seg = ball[a:b + 1]
        step = np.hypot(*np.diff(seg, axis=0).T)
        path = float(np.nansum(step))
        if n_touch >= BLIP_TOUCHES or path >= BLIP_PATH:
            continue
        dead[a:b + 1] = True
        filled += 1
        frames += b - a + 1
    return dead, filled, frames


def run(labels, touches, ball, in_play):
    """(labels, report)."""
    out = labels.copy()
    dead = out == "D"
    dead2, filled, frames = fill_blips(dead, in_play, touches, ball)
    out[dead2 & ~dead] = "D"
    return out, dict(blips_filled=filled, blip_frames=frames)
