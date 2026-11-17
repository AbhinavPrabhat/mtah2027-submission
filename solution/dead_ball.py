"""Phase 4 -- find dead-ball stretches and who restarts play.

Every boundary comes from the tracking. Smart Tagging is assumed to contain every set piece and
is used only as a label: it says which set piece happened and confirms an exit. Its start and
end frames never become a boundary, and its Team field is never read.

Precedence, highest first:

  1. Not in play. Frames outside both halves, and the second half before its kickoff, are dead
     by the match clock.
  2. Kickoff. Dead from the whistle (the start of the calm run that ends at the strike) to the
     strike on the centre spot (kickoff_detector.py). The nearest player there takes it.
  3. Flow A, the ball left the pitch (throw-in, corner, goal kick; restart_triggers.py).
       start = exit trigger: the ball over a line, matched to a tag of that type. An exit no
               tag explains is tracking noise and is skipped.
       end   = restart trigger: a still ball at the spot the laws require, struck, carried
               5 m and kept on the pitch. Without one, the calm run that covers the exit ends
               it at its last touch in the final seconds of the run. Without that, the ball back on the pitch for 2 s.
       team  = the player at the ball at the strike, else the team that did not touch it last.
     A calm run that overlaps an exit is part of that stoppage, not a second one.
  4. Flow B, a calm run with no exit (foul, offside, injury, free kick; stoppage.py). Dead from
     the whistle to the last touch inside the run.

A tag with no exit and no calm run writes nothing: those frames stay alive.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, half_video_blocks  # noqa: E402

import load_smart_tagging  # noqa: E402
import kickoff_detector  # noqa: E402
import stoppage  # noqa: E402
import restart_triggers  # noqa: E402

# Set pieces that exist only because the ball crossed a line. These run on Flow A.
EXIT_TYPES = {"throw-in", "corner kick", "goal kick"}
EXIT_MATCH_FRAMES = 40 * 25  # an exit up to 40 s before the tag can be the one it labels


def _runs(mask):
    """Inclusive [start, end] spans where mask is True."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end) - 1) for start, end in zip(edges[0::2], edges[1::2])]


def _in_play(n, game):
    """Frames inside a half and after that half's kickoff."""
    in_play = np.zeros(n, dtype=bool)
    for half, first, last in half_video_blocks(game):
        kickoff = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"] if half == 2 else first
        in_play[max(0, first, kickoff):min(n - 1, last) + 1] = True
    return in_play


def _calm_run_at(calm, first, last):
    """The first calm run that overlaps [first, last], or None."""
    for run in calm:
        if run[0] <= last and run[1] >= first:
            return run
    return None


def _tag_window_note(event):
    """Why the Smart tag is long, and why that length is not the dead window."""
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    seconds = (end - start) / 25.0
    label = (event["tags"].get("Type") or "set piece").strip()
    return (
        "Smart tag '{label}' is frames {start}–{end} ({seconds:.0f} s). "
        "Throw-in, corner, goal kick and free kick tags are a fixed 35 s phase (875 frames). "
        "Kickoff tags are 5 s. That length is the build-up stored on the tag, not the time the ball was dead. "
        "The dead window does not start at the tag start and does not end at the tag end."
    ).format(label=label, start=start, end=end, seconds=seconds)


def _exit_sighting_note(ex):
    """First sighting starts the stoppage. Last out-of-bounds frame is only the last sighting."""
    return (
        "Chosen exit: the ball over the {line} line. "
        "Dead starts at frame {first}, the first frame of that run. "
        "A run counts when the ball is seen more than 0.5 m past the line on at least 3 frames. "
        "The last out-of-bounds frame is {last}: the last frame in the same run where the tracker still saw the ball outside. "
        "That is the last sighting, not the whistle and not the restart. "
        "If the ATD drops the ball for up to 2 s inside the run, those missing frames stay part of the exit. "
        "The tracking has no whistle, no ball height, and no signal that the whole ball crossed the line."
    ).format(line=ex["line"], first=int(ex["first"]), last=int(ex["last"]))


def _learned_stoppages(ball_xy, players_xy, players_speed, team_of_track, touches, segments,
                       game, length, width, n):
    """The learned stoppage detector's spells, or None to use the calm-run flow instead.

    None when its models are not installed, a library is missing, or its own checks against this
    match's anchors fail. Say why, so a fallback is never silent.
    """
    try:
        import dead_ball_ml
    except ImportError as err:
        print("  learned stoppage detector: unavailable ({}); using the calm-run flow".format(err))
        return None
    if not dead_ball_ml.available():
        print("  learned stoppage detector: no trained models in solution/models/dead_ball_ml; using the calm-run flow")
        return None
    exits = restart_triggers.exits(ball_xy, length, width)
    result = dead_ball_ml.stoppage_dead(ball_xy, players_xy, players_speed, team_of_track, touches, segments,
                                        exits, game, length, width, _in_play(n, game))
    report = result["report"]
    if not result["ok"]:
        print("  learned stoppage detector: FELL BACK to the calm-run flow: " + "; ".join(result["reasons"]))
        return None
    print("  learned stoppage detector: evidence {}, anchors {} dead / {} live, ROC {:.2f} (clean-label) {:.2f} (HMM), "
          "{} dead spells".format(report["evidence"], report["dead_anchor_frames"], report["live_anchor_frames"],
                                  report["clean_auc"], report["hmm_auc"], len(result["spells"])))
    return result


def _match_exit(event, event_type, exits, used):
    """Index of the exit this tag labels, or None.

    The tag only says a set piece of this type happened around here. Its own frames are not a
    boundary, so the match is loose: an unused exit across the right line that starts inside
    the tag, or up to EXIT_MATCH_FRAMES before it. Stronger evidence wins: an observed exit,
    then a projected one, then a ball heading out near the line. Within the same strength the
    nearest one to the tag start wins.
    """
    line = "side" if event_type == "throw-in" else "goal"
    best = None
    for k, ex in enumerate(exits):
        if k in used or ex["line"] != line:
            continue
        if event["start_frame"] - EXIT_MATCH_FRAMES <= ex["first"] <= event["end_frame"]:
            gap = (2 if ex.get("near") else 1 if ex["projected"] else 0,
                   abs(ex["first"] - event["start_frame"]))
            if best is None or gap < best[0]:
                best = (gap, k)
    return None if best is None else best[1]


EARLY_RESTART_SECONDS = 5.0     # players still stopped this long after a set-piece restart: it was too early
SET_PIECE_RULES = ("throw-in", "corner-kick", "goal-kick", "free-kick", "penalty")


def _extend_early_restarts(segments, restarts, restart_team_at, dead, ml, touches, end_window, fps):
    """A set-piece segment whose restart the learned detector disagrees with is extended.

    The detector's decoded path stays dead for EARLY_RESTART_SECONDS or more right after the
    restart the set-piece flow chose, so the players had not resumed: that restart fired early (a
    ball struck or seen back on the pitch while play was still stopped). The segment then runs to
    the end of that decoded stretch, the early restart is removed, and the new restart takes the
    last credited touch in the final window (the old team if there is none). Everything else about
    the segment is left as the rules made it.
    """
    need = int(EARLY_RESTART_SECONDS * fps)
    spells = {first: (first, last) for first, last in ml["spells"]}
    for seg in sorted((s for s in segments if s["rule"] in SET_PIECE_RULES), key=lambda s: s["start"]):
        old = seg["end"] + 1
        spell = spells.get(old)
        if spell is None or spell[1] - spell[0] + 1 < need:
            continue
        last = spell[1]
        touch = stoppage.restart_touch(touches, old, last, end_window)
        team = touch[1] if touch is not None else restart_team_at.get(old)
        restart_team_at.pop(old, None)
        restart_team_at[last + 1] = team
        dead[old:last + 1] = True
        ml["spells"].remove(spell)
        seg["end"] = int(last)
        seg["triggers"] = list(seg.get("triggers", [])) + ["extended"]
        seg["detail"] += (" The learned detector keeps the players stopped for {:.1f} s after the chosen restart, so the segment "
                          "is extended to frame {}.").format((last - old + 1) / fps, int(last))
        for r in restarts:
            if r["frame"] == old:
                r["frame"], r["team"] = int(last + 1), team
                r["how"] += " Extended: the detector keeps play stopped until frame {}.".format(int(last + 1))


def compute(ball_xy, players_xy, team_of_track, meta_path, game=GAME,
            control_changes=None, touches=None, players_speed=None, use_ml=True):
    """(dead_mask, restart_team_at, segments, restarts).

    restart_team_at maps "first alive frame" -> team or None. segments and restarts are
    the same decisions, with the sentence the viewer shows. They are built in this loop
    so the screen cannot describe a rule the labels did not use.

    Every boundary comes from the tracking: an exit trigger, a restart strike, or a calm run.
    Smart Tagging only says which set piece happened. Two flows:

      Flow A, the ball left the pitch (throw-in, corner, goal kick). Dead from the exit. Live
        from the first real restart strike at the right spot. With no strike seen, live from
        the last touch of the calm run that covers the exit, then the first touch back on the
        pitch. A calm run that overlaps the exit is part of this stoppage, not a second one.
      Flow B, a calm run with no exit (foul, offside, injury, a free kick). Dead from the
        whistle to the last touch in the final seconds of the run, else the run's end.

    Kickoffs keep their own trigger: live from the strike on the centre spot, dead back to the
    whistle of the calm run that ends at it.
    """
    del meta_path, control_changes  # the tag's team and tag-bounded resumes are not used
    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    n = len(ball_xy)
    dead = np.zeros(n, dtype=bool)
    restart_team_at = {}
    segments = []
    restarts = []
    touches = touches or []
    touch_frames = [int(t["frame"]) for t in touches]
    kickoffs = {kick["start_frame"]: kick for kick in kickoff_detector.detect(
        ball_xy, players_xy, team_of_track, length, width, game)}

    events = load_smart_tagging.load(game)
    set_pieces = sorted((e for e in events if e["code"] == "SET PIECES"),
                        key=lambda e: e["start_frame"])

    end_window = int(stoppage.END_WINDOW_SECONDS * fps)
    speed = None
    calm = []
    if players_speed is not None and touches:
        speed = stoppage.collective_speed(players_speed, fps)
        in_play = _in_play(n, game)
        restarts_at = [e["start_frame"] for e in set_pieces
                       if "kick off" not in (e["tags"].get("Type") or "").lower()]
        calm_speed, min_frames, f1 = stoppage.calibrate(speed, in_play, restarts_at, fps)
        calm = stoppage.calm_runs(speed, in_play, calm_speed, min_frames)
        print("  stoppage: calm below {:.1f} m/s for {:.0f} s (calibrated on Smart Tagging, "
              "F1 {:.2f}), {} calm runs".format(calm_speed, min_frames / fps, f1, len(calm)))

    exits = restart_triggers.exits(ball_xy, length, width)
    bspeed = restart_triggers.ball_speed(ball_xy, fps)
    used = set()
    flow_a = []  # (dead_start, restart) of every exit stoppage, so Flow B skips them

    for event in set_pieces:
        start = max(0, event["start_frame"])
        end = min(n - 1, event["end_frame"])
        if start > end:
            continue
        event_type = (event["tags"].get("Type") or "").strip().lower()

        kick = kickoffs.get(start)
        if kick is not None:
            strike = int(kick["frame"])
            run = _calm_run_at(calm, strike - fps, strike)
            dead_start = stoppage.whistle_frame(speed, run[0]) if run is not None else strike
            dead[dead_start:strike] = True
            restart_team_at[strike] = kick["team"]
            flow_a.append((dead_start, strike))
            if strike > dead_start:
                segments.append({
                    "start": int(dead_start), "end": int(strike - 1), "rule": "kickoff",
                    "tag_start": int(event["start_frame"]),
                    "detail": " ".join([
                        "Precedence: kickoff wins, so these frames are D.",
                        _tag_window_note(event),
                        "Chosen restart: frame {}, the frame the ball is closest to the centre spot. The nearest player there takes it. The tag names no team.".format(strike),
                        "Dead runs back from that frame to {} because the players were slowing; that is the inferred whistle. It is not the tag start.".format(dead_start),
                    ]),
                })
            restarts.append({
                "frame": strike, "team": kick["team"], "type": event_type or "kick off",
                "how": "The player nearest the ball on the centre spot takes the kickoff, and passes to a teammate.",
            })
            continue

        if event_type not in EXIT_TYPES:
            continue  # free kicks have no exit: Flow B owns them

        k = _match_exit(event, event_type, exits, used)
        if k is None:
            continue  # the tracker never saw the ball leave: Flow B, if the players stopped
        used.add(k)
        ex = exits[k]
        live = restart_triggers.back_in_play(ex["last"], event_type, ball_xy, length, width)
        strike = restart_triggers.find_strike(event_type, ex["last"], live, ball_xy, bspeed,
                                              touch_frames, length, width)
        run = None
        box = None
        if strike is None and event_type == "corner kick" and players_speed is not None:
            ex_end = restart_triggers.end_x(ex["first"], ball_xy)
            # A ball seen placed at the flag means the kick is no earlier than that.
            placed = next((t for t in range(ex["last"], live + 1)
                           if restart_triggers._at_spot(ball_xy[t], event_type, length, width)), ex["first"])
            # The crowd may start running a moment after the ball is first seen back in play.
            box = restart_triggers.box_restart(ex["first"], live + restart_triggers.BOX_STILL_FRAMES,
                                               ex_end, players_xy, players_speed, length, width,
                                               not_before=placed)
        if strike is not None:
            restart = strike
            team = restart_triggers.taker_team(strike, ball_xy, players_xy, team_of_track)
            how = "Chosen restart: a still ball at the {} spot is struck at frame {} and stays on the pitch. Dead ends on the frame before it.".format(
                event_type, strike)
            team_note = "the nearest player at the strike, inside 2.5 m"
        elif box is not None:
            restart = box
            team = restart_triggers.corner_taker_team(box, ex_end, players_xy, team_of_track,
                                                      length, width)
            how = "Chosen restart: the ball is not seen at the flag, but {}+ players stood still in the box and all started moving at frame {}. That is the corner kick.".format(
                restart_triggers.BOX_CROWD, box)
            team_note = "the player nearest the corner flag at the kick"
        else:
            # No clean strike. The calm run that covers the exit is the stoppage: dead until
            # its last touch, or its end. With no calm run, the ball back on the pitch for
            # good is the only evidence left.
            run = _calm_run_at(calm, ex["first"], live)
            if run is not None:
                touch = stoppage.restart_touch(touches, max(run[0], ex["last"] + 1), run[1], end_window)
                restart, team = touch if touch is not None else (run[1] + 1, None)
                how = "No strike passed the still-ball and stay-on-pitch checks. Chosen end is frame {}: {}.".format(
                    restart, "the last touch in the final {:.0f} s of the calm run".format(stoppage.END_WINDOW_SECONDS) if touch is not None else "the end of the calm run (no touch in its final {:.0f} s)".format(stoppage.END_WINDOW_SECONDS))
            else:
                touch = None
                restart, team = live, None
                how = "No strike passed the still-ball and stay-on-pitch checks, and no calm run. Chosen end is frame {}: the ball back in play.".format(restart)
            if team is None:
                team = restart_triggers.taker_team(restart, ball_xy, players_xy, team_of_track)
            team_note = "the last credited touch in the calm run" if touch is not None else "the nearest player at that frame, inside 2.5 m"
        if team is None:
            last = restart_triggers.last_toucher(ex["first"], touches)
            team = None if last is None else 1 - last
            team_note = "the team that did not touch the ball last before the exit, because nobody was close enough"
        how += " The restart team is {}. The tag's Team field is not read.".format(team_note)
        dead[ex["first"]:restart] = True
        restart_team_at[restart] = team
        flow_a.append((ex["first"], restart))
        segments.append({
            "start": int(ex["first"]), "end": int(restart - 1), "rule": event_type.replace(" ", "-"),
            "tag_start": int(event["start_frame"]),
            "exit_first": int(ex["first"]),
            "exit_last": int(ex["last"]),
            "triggers": ["exit", "strike" if strike is not None else "box" if box is not None else ("stoppage" if run is not None else "back-in-play")],
            "detail": " ".join([
                "Precedence: {} wins, so this frame is D even when a touch was accepted.".format(event_type),
                _tag_window_note(event),
                _exit_sighting_note(ex),
                how,
            ]),
        })
        restarts.append({"frame": int(restart), "team": team, "type": event_type, "how": how})

    ml = None
    if use_ml and speed is not None:
        ml = _learned_stoppages(ball_xy, players_xy, players_speed, team_of_track, touches, segments,
                                game, length, width, n)
    if ml is not None:
        _extend_early_restarts(segments, restarts, restart_team_at, dead, ml, touches, end_window, fps)
        for first, last in ml["spells"]:
            touch = stoppage.restart_touch(touches, first, last, end_window)
            team = touch[1] if touch is not None else None
            restart = last + 1
            dead[first:restart] = True
            restart_team_at[restart] = team
            segments.append({
                "start": int(first), "end": int(last), "rule": "stoppage", "triggers": ["ml-decode"],
                "detail": (
                    "Learned stoppage detector. Frames the ball and touches prove dead or live are fixed; the rest are "
                    "scored by a clean-label model and an unsupervised player-movement HMM, and one continuous "
                    "dead/live path is decoded. This stretch is dead from frame {} to {}. The restart team is the last "
                    "touch in its final {} s, if any. Set-piece segments are not touched by this detector."
                ).format(int(first), int(last), int(stoppage.END_WINDOW_SECONDS)),
            })
            restarts.append({
                "frame": int(restart), "team": team, "type": "stoppage",
                "how": "Learned stoppage detector: the decoded dead stretch ends here.",
            })
    for run_first, run_last in (calm if ml is None else []):
        if any(a <= run_last and b >= run_first for a, b in flow_a):
            continue
        touch = stoppage.restart_touch(touches, run_first, run_last, end_window)
        restart, stop_team = touch if touch is not None else (run_last + 1, None)
        whistle = stoppage.whistle_frame(speed, run_first)
        dead[whistle:restart] = True
        restart_team_at[restart] = stop_team
        if whistle < restart:
            segments.append({
                "start": int(whistle), "end": int(restart - 1), "rule": "stoppage",
                "detail": (
                    "Precedence: stoppage wins. The ball was not seen leaving the pitch, and the players slowed to a walk. "
                    "Dead runs from frame {}, the inferred whistle, until the last touch in the final {} s of that calm run, or the end of the run when there is none. "
                    "No set-piece tag is required. A calm run that overlaps an exit belongs to that exit instead. "
                    "There is no whistle in the ATD, so the whistle is the frame the slowdown starts."
                ).format(int(whistle), int(stoppage.END_WINDOW_SECONDS)),
            })
        restarts.append({
            "frame": int(restart), "team": stop_team, "type": "stoppage",
            "how": "The players slowed to a walk. The last touch in the final {:.0f} s of that stretch restarts play, else its end.".format(stoppage.END_WINDOW_SECONDS),
        })

    within_play = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        within_play[max(0, first):min(n - 1, last) + 1] = True
    for span_start, span_end in _runs(~within_play):
        segments.append({
            "start": span_start, "end": span_end, "rule": "outside",
            "detail": "This frame is outside both halves: before kickoff, halftime, or after full time.",
        })
    dead |= ~within_play  # pre-kickoff, halftime, post-full-time: not being played, so dead

    # FAQ.md: the second half is graded from its first frame, but it does not kick off until
    # frame 113742. The 2,934 frames in between are graded and are all dead. They sit inside
    # the half, so the line above does not cover them, and the first-half team would otherwise
    # be carried straight through.
    for half, first, last in half_video_blocks(game):
        if half == 2:
            kickoff = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]
            end = min(n - 1, last, kickoff - 1)
            dead[first:min(n, last + 1, kickoff)] = True
            if first <= end:
                segments.append({
                    "start": int(first), "end": int(end), "rule": "pre-kickoff",
                    "detail": "The second half is dead until the kickoff at frame {}.".format(kickoff),
                })

    segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    return dead, restart_team_at, segments, restarts


if __name__ == "__main__":
    import parse_atd
    from gap_fill import fill_short_gaps
    import touch_detector
    import control

    data = parse_atd.parse()
    filled_xy, status = fill_short_gaps(data["ball_xy"])
    touches = touch_detector.detect(filled_xy, status, data["players_xy"], data["team_of_track"])
    control_changes = control.confirm_changes(touches, filled_xy, data["players_xy"],
                                              data["team_of_track"])
    meta_path = require(["atd_meta"])[0]
    dead, restart_team_at, _segments, _restarts = compute(
        data["ball_xy"], data["players_xy"], data["team_of_track"],
        meta_path, control_changes=control_changes, touches=touches,
        players_speed=data["players_speed"])

    n = len(dead)
    print("dead-ball frames: {} / {} ({:.1f}%)".format(int(dead.sum()), n, 100.0 * dead.sum() / n))
    print("restarts found: {}  ({} with a resolved team, {} unresolved)".format(
        len(restart_team_at), sum(1 for t in restart_team_at.values() if t is not None),
        sum(1 for t in restart_team_at.values() if t is None)))

    print("\nfirst 10 restarts:")
    for frame in sorted(restart_team_at)[:10]:
        team = restart_team_at[frame]
        print("  frame {:>7}  restart team {}".format(frame, {0: "H", 1: "A", None: "?"}[team]))
