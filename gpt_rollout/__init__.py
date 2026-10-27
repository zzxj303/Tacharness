"""GPT-6 x TacClaw rollout scaffold for RoboLab and Tabero (see ../README.md)."""

# README section 4: re-call with the same observation at most this many times.
MAX_RECALLS = 3
# Decisions per episode before giving up: the default of Show-Harness's LIBERO runner.
MAX_DECISIONS = 200  # only when the time limit is off (README 0.1); otherwise no cap
# Collection lengthens each task's own episode time limit by this factor (README 0.1).
TIME_LIMIT_SCALE = 4.0
# Actions shown back to the policy, newest first. Show-Harness's meta-action prompt lists
# moves only; GRASP and RELEASE are listed too (README 1).
RECENT_MOVES = 5
