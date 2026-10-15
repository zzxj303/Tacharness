"""GPT-6 x TacClaw rollout scaffold for RoboLab and Tabero (see ../README.md)."""

# README section 4: re-call with the same observation at most this many times.
MAX_RECALLS = 3
# Decisions per episode before giving up: the default of Show-Harness's LIBERO runner.
MAX_DECISIONS = 200
# Moves shown back to GPT, newest first, as in Show-Harness's meta-action prompt.
RECENT_MOVES = 5
