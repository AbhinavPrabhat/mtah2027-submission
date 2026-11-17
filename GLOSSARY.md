# Glossary

Short, plain explanations of the terms used in [TECHNICAL_DOC.md](TECHNICAL_DOC.md). Terms in the document link here.

## Data and labels

## ATD
Automatic Tracking Data: the position and speed of the ball and of all 22 players on every frame (25 frames per
second, 199,319 frames). See [DATA.md](DATA.md).

## Smart Tagging
A file that lists events of the match, such as throw-ins, corners, goal kicks and kickoffs. We use it only to learn
which set piece happened. We never use its start and end times or its team field. See [DATA.md](DATA.md).

## H, A, D, X
The label of one frame: **H** Home has the ball, **A** Away has the ball, **D** dead ball (play is stopped),
**X** not graded.

## Dead ball
A frame where play is stopped: a throw-in, corner, goal kick, kickoff, foul, offside, injury and so on.

## Dead mask
The list of frames the dead-ball flow calls dead. The possession flow reads it and never labels those frames
as Home or Away.

## Possession spell
One stretch of frames in which the same team has the ball.

## Touch and touches

## Touch
A moment where a player kicks, heads or otherwise changes the ball's motion. We find it as a sudden change of
the ball's direction or speed.

## Toucher and credited touch
The toucher is the nearest player to the ball at the moment of the touch, but only if that player is within
3.5 m. A touch with a toucher has a team, and we call it a credited touch. A touch with no player near enough has
no team and never changes possession.

## Clear touch and weak touch
A touch is **clear** when the toucher is within 1.5 m of the ball and no opponent is within 3 m. Anything else is
**weak**: the toucher was far, or an opponent was close, so the team that touched it is uncertain.

## Touch fix
The step where the model checks each weak touch. If the model is at least 82% sure the other team had the ball, the
touch is flipped to that team. The rule-based logic then runs again on the corrected touches.

## Chain, carry, deflection
How the rule-based logic decides that an opponent's touch really changes possession.
- **Chain**: within 2 s the same player touches the ball again, or the ball goes to a teammate of that player.
- **Carry**: no other touch follows in a short window and the same player stays with the ball.
- **Deflection**: the original team touches the ball again before the opponent does anything with it, so
  nothing changes.

## Restart team
The team that takes the restart after a stoppage, for example the team that takes the throw-in. Possession starts
with that team.

## Dead-ball terms

## Exit trigger
The ball is seen more than 0.5 m past a line for at least three frames. This starts a throw-in, corner or goal
kick.

## Restart strike
The ball sits still at the spot the laws require and is then struck, travels at least 5 m and stays on the pitch.
This ends the stoppage.

## Calm run
A stretch in which the players' median speed stays low together. It is the sign of a stoppage with no ball exit,
such as a foul. It is also used as the rule-based fallback.

## Anchors
Frames that the ball track and the touches prove to be dead or live, for example the wait between an exit and its
restart strike, or four seconds of a ball being passed around. They are the training labels for the dead-ball
models.

## Clean labels
Frames where the tracking itself makes clear who has the ball: near a clear touch, or between two clear touches
of the same team. About 21% of live frames get a clean label. They are the training labels for the possession
models.

## Models and training

## LightGBM
A fast model made of many small decision trees. It reads a row of numbers (the features of one frame) and gives a
probability.

## HMM
A hidden Markov model: a small model with two hidden states (stopped, moving) that it finds on its own from how
the players move, with no labels at all.

## GRU
A small neural network that reads a sequence of frames (here 48 seconds) instead of one frame, so it can use what
happened before and after.

## Window features
The same player numbers (speed, spread and so on) averaged over a few seconds before and after a frame. A single
frame says little, and the pattern over seconds says much more.

## Self-training
After the first round, the models are trained a second time on the frames they were most confident about and that
agree with the rules, as well as on the clean labels.

## Cross-fitting and out-of-fold scores
The match is cut into 2-minute blocks. Each block is scored by a model that was trained on the other blocks, with
a gap of 10 s around the held-out block. Every frame therefore gets a score from a model that never saw it. See
section 3.4 of the document.

## ROC AUC
A number between 0.5 (random) and 1.0 (perfect) that says how well a model's scores separate the two classes. In
this document it is always measured against the algorithm's own labels.

## Log-odds
The probability written on a scale that can be added up. Evidence from different models is combined by adding or
averaging log-odds.

## Decode
A search for the single best sequence of labels over a whole half, instead of choosing each frame on its own. Every
change of label costs something (the **switch cost**), so a stoppage stays one block and one noisy frame cannot
split it. The method is the Viterbi algorithm.

## Guard
A check that the learned part runs against this match's own anchors or clean labels. If it fails, the rule-based
logic is used instead.

## Blip filter
The final step. A short live stretch (under 5 s) between two dead spells is turned into dead, unless play is
clearly visible inside it.

## Scores

See [EVALUATION.md](EVALUATION.md) for the full definitions.

## Spell F1
How well the possession spells match the reference: the right team, for about the right time.

## Macro F1
The average F1 over the labels H, A and D, frame by frame.

## Team accuracy
Of the frames where possession is live, the share given to the correct team.

## Dead-ball F1
The F1 of the D label alone.
