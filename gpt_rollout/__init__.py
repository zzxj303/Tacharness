"""GPT-6 x TacClaw rollout scaffold for RoboLab and Tabero (see ../README.md)."""

# README section 4: re-call with the same observation at most this many times.
MAX_RECALLS = 3
# Decisions per episode before giving up: the default of Show-Harness's LIBERO runner.
MAX_DECISIONS = 200  # only when the time limit is off (README 0.1); otherwise no cap
# Collection lengthens each task's own episode time limit by this factor (README 0.1).
TIME_LIMIT_SCALE = 2.0
# Moves shown back to GPT, newest first, as in Show-Harness's meta-action prompt.
RECENT_MOVES = 5
