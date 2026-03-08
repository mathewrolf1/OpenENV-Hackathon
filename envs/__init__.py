from envs.melee_sim_env import MeleeSimEnv

__all__ = ["MeleeSimEnv", "MeleeTorchEnv"]

try:
    from envs.melee_torchrl_env import MeleeTorchEnv
except ImportError:
    MeleeTorchEnv = None  # libmelee may not be installed
