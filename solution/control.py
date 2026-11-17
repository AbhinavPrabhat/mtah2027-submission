"""Phase 3 -- decide which opponent touches are real possession changes.

Not every touch by the other team is a turnover. CHALLENGE.md's own definition of a possession
change requires *controlled* possession, which rules out:
  - a deflection: the opponent's touch is immediately followed by the original team touching the
    ball again, before the opponent does anything more with it
  - an aerial duel or 50/50 ball with no clear follow-up control

This module walks the touch list in order and keeps a single "current team" pointer. An opponent
touch only moves that pointer if it is confirmed, two ways:
  - chain: within 2 s the same player touches it again, or the ball travels from that
    player's slot to a teammate's slot. Another teammate who is merely near the ball
    does not confirm. The original team touching it back still cancels the change.
  - carry: no further touch happens in that window, and the SAME player who made this kink
    actually takes the ball. Standing under a ball in the air does not count: the ball's
    path would be much longer than that player's path, and the landing touch is judged
    later. The kicker is ignored; the next touch can take the ball.

If neither holds, the touch is treated as a stray, a duel, or a kick, and possession stays
with the original team. The boundary of a confirmed change is placed at the opponent's OWN
touch frame, not later when the evidence became clear.
"""

import numpy as np

LOOKAHEAD_FRAMES = 50    # 2 s -- chain and revert only. Not lengthened to catch a long pass.
CARRY_MIN_FRAMES = 10    # 0.4 s -- the window a carry is judged over, right after the touch
CARRY_HOLD_FRAMES = 10   # same 0.4 s: a player who touches and passes inside 1 s still had it
CARRY_RADIUS = 3.5       # m -- same control radius as touch attribution
CARRY_MIN_FRACTION = 0.7 # fraction of the window the touching player must be within the radius
CARRY_EXTRA_TRAVEL = 4.0 # m -- ball path may exceed the player's path by this much and still be a carry


def live_segments(dead_mask, restart_team_at):
    """Inclusive (start, end, init_team) stretches that do not cross a dead frame.

    A restart frame inside a live run starts a new stretch, so a chain or a carry
    cannot use a touch from the other side of the whistle.
    """
    n = len(dead_mask)
    restart_frames = sorted(
        int(frame) for frame in restart_team_at
        if 0 <= int(frame) < n and not dead_mask[int(frame)])
    segments = []
    t = 0
    while t < n:
        if dead_mask[t]:
            t += 1
            continue
        run_start = t
        while t < n and not dead_mask[t]:
            t += 1
        run_end = t - 1
        boundaries = [run_start]
        boundaries.extend(frame for frame in restart_frames if run_start < frame <= run_end)
        for index, start in enumerate(boundaries):
            end = boundaries[index + 1] - 1 if index + 1 < len(boundaries) else run_end
            segments.append((start, end, restart_team_at.get(start)))
    return segments


def _player_take(frame, player, ball_xy, players_xy, n_frames,
                 hold_frames=CARRY_HOLD_FRAMES, radius=CARRY_RADIUS,
                 extra_travel=CARRY_EXTRA_TRAVEL, until=None):
    """(took, close, total). The credited player takes the ball, or it flies past them.

    close/total count frames where that player is within radius, over frames where the
    ball and the player are both visible. A carry needs that player within the radius on
    CARRY_MIN_FRACTION of those frames, and the ball's path to stay with the player's path.
    A ball in the air travels much farther than the players standing under it, so those
    frames are not a carry. Only the touching player counts: a teammate who happens to be
    nearest does not carry a ball that was passed past them.
    """
    if player is None:
        return False, 0, 0
    limit = n_frames if until is None else min(n_frames, int(until) + 1)
    end = min(limit, frame + hold_frames)
    if end - frame < CARRY_MIN_FRAMES:
        return False, 0, 0

    slot = int(player)
    samples = []
    for t in range(frame, end):
        ball_pos = ball_xy[t]
        pos = players_xy[t, slot]
        if np.isnan(ball_pos[0]) or np.isnan(pos[0]):
            continue
        samples.append((t, float(np.linalg.norm(pos - ball_pos)), ball_pos, pos))
    total = len(samples)
    close = sum(1 for _, dist, _, _ in samples if dist <= radius)
    if total < CARRY_MIN_FRAMES // 2:
        return False, close, total  # ball or player mostly unseen: nothing to judge

    if close / total < CARRY_MIN_FRACTION:
        return False, close, total

    ball_travel = 0.0
    player_travel = 0.0
    for prev, nxt in zip(samples, samples[1:]):
        ball_travel += float(np.linalg.norm(nxt[2] - prev[2]))
        player_travel += float(np.linalg.norm(nxt[3] - prev[3]))
    if ball_travel > player_travel + extra_travel:
        return False, close, total
    return True, close, total


def _slot_at(players_xy, frame, slot, search=5):
    """Last visible position of this slot at or just before `frame`, or None."""
    if slot is None:
        return None
    slot = int(slot)
    for t in range(frame, max(-1, frame - search) - 1, -1):
        pos = players_xy[t, slot]
        if not np.isnan(pos[0]):
            return pos
    return None


def _received_pass(first, second, ball_xy, players_xy, radius):
    """The ball left the first slot and moved to the second slot.

    A teammate standing near a ball that is still with the passer, or a kink behind
    the pass, does not count. The receiver has to be inside `radius` of the ball.
    """
    slot_a = first.get("player")
    slot_b = second.get("player")
    if slot_a is None or slot_b is None or slot_a == slot_b:
        return False
    ball_a = ball_xy[first["frame"]]
    ball_b = ball_xy[second["frame"]]
    if np.isnan(ball_a[0]) or np.isnan(ball_b[0]):
        return False
    pos_a = _slot_at(players_xy, second["frame"], slot_a)
    pos_b = _slot_at(players_xy, second["frame"], slot_b)
    if pos_a is None or pos_b is None:
        return False
    if float(np.linalg.norm(pos_b - ball_b)) > radius:
        return False
    if float(np.linalg.norm(pos_a - ball_b)) <= radius:
        return False
    ball_delta = ball_b - ball_a
    toward_receiver = pos_b - ball_a
    return float(np.dot(ball_delta, toward_receiver)) > 0.0


def _blank():
    return {"decision": "ignore", "other": None, "carry_close": None, "carry_total": None}


def _decide_on_live(touches, ball_xy, players_xy, team_of_track, radius, dead_mask, restart_team_at):
    """Same chain, revert, and carry as decide(), one live stretch at a time.

    A touch on a dead frame is ignored. A restart sets the team for the stretch
    that follows it, including an unknown restart, which clears the old team.
    """
    n_frames = len(ball_xy)
    credited = sorted(
        (t for t in touches if t["team"] is not None and 0 <= int(t["frame"]) < n_frames),
        key=lambda touch: int(touch["frame"]))
    notes = {}
    for touch in credited:
        if dead_mask[int(touch["frame"])]:
            notes[int(touch["frame"])] = _blank()

    changes = []
    for start, end, init_team in live_segments(dead_mask, restart_team_at):
        segment = [t for t in credited if start <= int(t["frame"]) <= end]
        has_restart = start in restart_team_at or any(
            int(frame) == start for frame in restart_team_at)
        current_team = init_team if has_restart else None
        if has_restart:
            changes.append((start, current_team))
        i = 0
        if has_restart:
            while i < len(segment) and int(segment[i]["frame"]) == start:
                touch = segment[i]
                same = current_team is not None and touch["team"] == current_team
                note = _blank()
                note["decision"] = "same" if same else "ignore"
                notes[int(touch["frame"])] = note
                i += 1
        n = len(segment)
        while i < n:
            touch = segment[i]
            frame = int(touch["frame"])
            if current_team is None:
                current_team = touch["team"]
                changes.append((frame, current_team))
                note = _blank()
                note["decision"] = "open"
                notes[frame] = note
                i += 1
                continue
            if touch["team"] == current_team:
                note = _blank()
                note["decision"] = "same"
                notes[frame] = note
                i += 1
                continue

            new_team = touch["team"]
            window_end = min(frame + LOOKAHEAD_FRAMES, end)
            reverted = chained = False
            witness = None
            j = i + 1
            while j < n and int(segment[j]["frame"]) <= window_end:
                later = segment[j]
                if later["team"] == current_team:
                    reverted = True
                    witness = int(later["frame"])
                    break
                if later["team"] == new_team:
                    witness = int(later["frame"])
                    same_player = (
                        touch.get("player") is not None
                        and later.get("player") == touch.get("player"))
                    chained = same_player or _received_pass(
                        touch, later, ball_xy, players_xy, radius)
                    break
                j += 1

            took, close, total = _player_take(
                frame, touch.get("player"), ball_xy, players_xy, n_frames,
                radius=radius, until=end)
            note = {"decision": "ignore", "other": witness,
                    "carry_close": close, "carry_total": total}
            if reverted:
                note["decision"] = "revert"
            elif chained:
                note["decision"] = "chain"
                changes.append((frame, new_team))
                current_team = new_team
            elif took:
                note["decision"] = "carry"
                changes.append((frame, new_team))
                current_team = new_team
            notes[frame] = note
            i += 1
    return changes, notes


def decide(touches, ball_xy, players_xy, team_of_track, radius=CARRY_RADIUS,
           dead_mask=None, restart_team_at=None):
    """(changes, notes). One walk, shared by the labels and the viewer.

    changes is [(frame, new_team)]. notes maps a touch frame to
    {decision, other, carry_close, carry_total}. decision is open, same, revert,
    chain, carry, or ignore. carry_close and carry_total are filled for every opponent
    touch. A second touch still decides first: revert, then chain, then the carry count.
    A touch with no team is simply absent. With a dead mask, the walk is one live
    stretch at a time and a restart sets the team before any touch in that stretch.
    """
    if dead_mask is not None:
        return _decide_on_live(
            touches, ball_xy, players_xy, team_of_track, radius, dead_mask,
            {} if restart_team_at is None else restart_team_at)
    resolved = [t for t in touches if t["team"] is not None]
    if not resolved:
        return [], {}

    n_frames = len(ball_xy)
    current_team = resolved[0]["team"]
    changes = [(resolved[0]["frame"], current_team)]
    notes = {resolved[0]["frame"]: {"decision": "open", "other": None,
                                    "carry_close": None, "carry_total": None}}

    i = 1
    n = len(resolved)
    while i < n:
        touch = resolved[i]
        frame = touch["frame"]
        if touch["team"] == current_team:
            notes[frame] = {"decision": "same", "other": None,
                            "carry_close": None, "carry_total": None}
            i += 1
            continue

        new_team = touch["team"]
        window_end = frame + LOOKAHEAD_FRAMES
        reverted = chained = False
        witness = None
        j = i + 1
        while j < n and resolved[j]["frame"] <= window_end:
            later = resolved[j]
            if later["team"] == current_team:
                reverted = True
                witness = later["frame"]
                break
            if later["team"] == new_team:
                witness = later["frame"]
                same_player = (
                    touch.get("player") is not None
                    and later.get("player") == touch.get("player"))
                chained = same_player or _received_pass(
                    touch, later, ball_xy, players_xy, radius)
                break
            j += 1

        # Revert, then a real chain, then a carry by this same slot. A same-team
        # kink that the ball never reached falls through and does not change possession.
        took, close, total = _player_take(
            frame, touch.get("player"), ball_xy, players_xy, n_frames, radius=radius)
        note = {"decision": "ignore", "other": witness, "carry_close": close, "carry_total": total}
        if reverted:
            note["decision"] = "revert"
        elif chained:
            note["decision"] = "chain"
            changes.append((frame, new_team))
            current_team = new_team
        elif took:
            note["decision"] = "carry"
            changes.append((frame, new_team))
            current_team = new_team
        notes[frame] = note
        i += 1

    return changes, notes


def confirm_changes(touches, ball_xy, players_xy, team_of_track, radius=CARRY_RADIUS,
                    dead_mask=None, restart_team_at=None):
    """[(frame, new_team)] confirmed by chain, revert, and carry.

    Pass a dead mask to walk only live frames. A restart then sets the team, and a
    touch during the stoppage cannot. `radius` is how near a slot must be to still
    have the ball, for both the carry and a pass to a teammate.
    """
    changes, _notes = decide(
        touches, ball_xy, players_xy, team_of_track, radius=radius,
        dead_mask=dead_mask, restart_team_at=restart_team_at)
    return changes


if __name__ == "__main__":
    import os
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import parse_atd
    from gap_fill import fill_short_gaps
    import touch_detector

    data = parse_atd.parse()
    filled_xy, status = fill_short_gaps(data["ball_xy"])
    touches = touch_detector.detect(filled_xy, status, data["players_xy"], data["team_of_track"])
    changes = confirm_changes(touches, filled_xy, data["players_xy"], data["team_of_track"])

    print("{} touches -> {} confirmed possession changes".format(len(touches), len(changes)))
    print("\nfirst 15 confirmed changes:")
    for frame, team in changes[:15]:
        print("  frame {:>7}  -> {}".format(frame, {0: "H", 1: "A"}[team]))
