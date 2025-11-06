"""
Environment utilities and wrappers for reinforcement learning.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from gymnasium.wrappers import RecordEpisodeStatistics, NormalizeObservation


class ContinuousControlEnv(Env):
    """
    A simple continuous control environment for testing RL algorithms.
    
    The agent controls a 2D point that must reach a target position.
    State: [x, y, target_x, target_y]
    Action: [force_x, force_y] in range [-1, 1]
    Reward: -distance_to_target - action_penalty
    """
    
    def __init__(self, max_steps: int = 200):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        
        # State space: [x, y, target_x, target_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Action space: [force_x, force_y]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Random initial position
        self.agent_pos = np.random.uniform(-2, 2, 2)
        
        # Random target position
        self.target_pos = np.random.uniform(-2, 2, 2)
        
        self.step_count = 0
        
        state = np.concatenate([self.agent_pos, self.target_pos]).astype(np.float32)
        info = {}
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.step_count += 1
        
        # Apply action (force) with some damping
        self.agent_pos += action * 0.1
        
        # Calculate reward
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        action_penalty = 0.01 * np.sum(action**2)
        reward = -distance - action_penalty
        
        # Check if done
        terminated = distance < 0.1  # Reached target
        truncated = self.step_count >= self.max_steps
        
        # Next state
        next_state = np.concatenate([self.agent_pos, self.target_pos]).astype(np.float32)
        info = {"distance": distance}
        
        return next_state, reward, terminated, truncated, info


def make_env(env_name: str, **kwargs) -> gym.Env:
    """
    Create and configure an environment.
    
    Args:
        env_name: Name of the environment to create
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Configured gymnasium environment
    """
    if env_name == "ContinuousControl":
        env = ContinuousControlEnv(**kwargs)
    elif env_name == "CartPole-v1":
        env = gym.make("CartPole-v1", **kwargs)
    elif env_name == "MountainCarContinuous-v0":
        env = gym.make("MountainCarContinuous-v0", **kwargs)
    elif env_name == "Pendulum-v1":
        env = gym.make("Pendulum-v1", **kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Add episode statistics recording
    env = RecordEpisodeStatistics(env)
    
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """
    Get environment information for algorithm configuration.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        Dictionary containing environment information
    """
    info = {
        "state_dim": env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None,
        "action_dim": env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
        "is_discrete": isinstance(env.action_space, spaces.Discrete),
        "is_continuous": isinstance(env.action_space, spaces.Box),
        "action_space": env.action_space,
        "observation_space": env.observation_space,
    }
    
    if isinstance(env.action_space, spaces.Box):
        info["action_low"] = env.action_space.low
        info["action_high"] = env.action_space.high
        info["max_action"] = float(env.action_space.high[0])
    
    return info
