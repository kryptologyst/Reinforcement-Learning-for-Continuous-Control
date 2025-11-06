#!/usr/bin/env python3
"""
Simple test script to debug PPO issues.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import PPOAgent
from envs import make_env, get_env_info

def test_ppo():
    """Test PPO agent with minimal setup."""
    print("Testing PPO agent...")
    
    # Create environment
    env = make_env("ContinuousControl")
    env_info = get_env_info(env)
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=env_info['state_dim'],
        action_dim=env_info['action_dim'],
        max_action=env_info['max_action']
    )
    
    print(f"Agent created successfully!")
    print(f"State dim: {env_info['state_dim']}")
    print(f"Action dim: {env_info['action_dim']}")
    
    # Test action selection
    state, _ = env.reset()
    print(f"Initial state: {state}")
    
    try:
        action, log_prob, value = agent.select_action(state)
        print(f"Action: {action}")
        print(f"Log prob: {log_prob}")
        print(f"Value: {value}")
        
        # Test transition storage
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        
        agent.store_transition(state, action, reward, log_prob, value, done)
        print(f"Stored transition successfully!")
        
        # Test update
        losses = agent.update()
        print(f"Update successful! Losses: {losses}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()

if __name__ == "__main__":
    test_ppo()
