"""Melee stage and character constants for observation normalization and reward shaping.

Final Destination blastzones and Fox/Puff velocity limits (from Melee data / physics).
"""

# Final Destination (default stage) blastzones — world units
FD_LEFT_BLASTZONE = -224.0
FD_RIGHT_BLASTZONE = 224.0
FD_TOP_BLASTZONE = 180.0
FD_BOTTOM_BLASTZONE = -109.0

FD_MID_X = (FD_LEFT_BLASTZONE + FD_RIGHT_BLASTZONE) / 2.0
FD_HALF_WIDTH_X = (FD_RIGHT_BLASTZONE - FD_LEFT_BLASTZONE) / 2.0
FD_MID_Y = (FD_TOP_BLASTZONE + FD_BOTTOM_BLASTZONE) / 2.0
FD_HALF_HEIGHT_Y = (FD_TOP_BLASTZONE - FD_BOTTOM_BLASTZONE) / 2.0

# Stage ledge (for off_stage / center stage)
STAGE_LEFT_EDGE = -68.4
STAGE_RIGHT_EDGE = 68.4

# Character velocity limits for normalizing the 5-speed system
# Fall speed (terminal) and max upward jump velocity
FOX_TERMINAL_VELOCITY = 2.8
FOX_MAX_JUMP_VELOCITY = 3.2
FOX_MAX_HORIZONTAL_SPEED = 2.2  # run / air

PUFF_TERMINAL_VELOCITY = 1.3
PUFF_MAX_JUMP_VELOCITY = 2.4
PUFF_MAX_HORIZONTAL_SPEED = 1.35

# Reward shaping constants
DISTANCE_REWARD_SCALE = -0.001
CENTER_STAGE_REWARD = 0.005
CENTER_STAGE_X_THRESHOLD = 25.0  # |x| < this counts as "center"
NOISE_PENALTY_HITLAG_JUMPSQUAT = -0.02  # penalty for noisy inputs during hitlag/jumpsquat
