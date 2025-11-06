#!/usr/bin/env python3
"""
Test just PPO agent.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import PPOAgent
from envs import make_env, get_env_info

def test_ppo_training():
    """Test PPO agent training."""
    print("Testing PPO agent training...")
    
    # Create environment
    env = make_env("ContinuousControl")
    env_info = get_env_info(env)
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=env_info['state_dim'],
        action_dim=env_info['action_dim'],
        max_action=env_info['max_action']
    )
    
    # Training loop
    rewards = []
    
    for episode in range(50):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Max 100 steps per episode
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, log_prob, value, done)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update PPO agent
        agent.update()
        
        rewards.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode:3d}: Reward = {episode_reward:8.2f}, Avg = {avg_reward:8.2f}")
    
    # Final evaluation
    final_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        final_rewards.append(episode_reward)
    
    avg_final = np.mean(final_rewards)
    std_final = np.std(final_rewards)
    
    print(f"\nFinal Performance:")
    print(f"  Average reward: {avg_final:.2f} ± {std_final:.2f}")
    print(f"  Best episode: {max(rewards):.2f}")
    print(f"  Worst episode: {min(rewards):.2f}")
    
    env.close()
    return rewards, avg_final

if __name__ == "__main__":
    test_ppo_training()
