#!/usr/bin/env python3
"""
Debug action type issue.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import PPOAgent
from envs import make_env, get_env_info

def debug_action():
    """Debug action type."""
    print("Debugging action type...")
    
    # Create environment
    env = make_env("ContinuousControl")
    env_info = get_env_info(env)
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=env_info['state_dim'],
        action_dim=env_info['action_dim'],
        max_action=env_info['max_action']
    )
    
    # Test action selection
    state, _ = env.reset()
    print(f"State type: {type(state)}")
    print(f"State shape: {state.shape}")
    
    action, log_prob, value = agent.select_action(state)
    print(f"Action type: {type(action)}")
    print(f"Action value: {action}")
    print(f"Action shape: {action.shape if hasattr(action, 'shape') else 'No shape'}")
    
    # Try to step
    try:
        next_state, reward, terminated, truncated, _ = env.step(action)
        print("Step successful!")
    except Exception as e:
        print(f"Step failed: {e}")
        print(f"Action is: {action}")
        print(f"Action type: {type(action)}")
        
        # Try converting to numpy array
        if isinstance(action, (list, tuple)):
            action = np.array(action)
            print(f"Converted to numpy: {action}")
            print(f"New type: {type(action)}")
            
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                print("Step successful after conversion!")
            except Exception as e2:
                print(f"Still failed: {e2}")
    
    env.close()

if __name__ == "__main__":
    debug_action()
