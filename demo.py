#!/usr/bin/env python3
"""
Quick demo script to test the RL implementation.
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import DDPGAgent, PPOAgent, SACAgent
from envs import make_env, get_env_info


def test_agent(agent_class, agent_name, env_name="ContinuousControl", episodes=50):
    """Test an agent on a given environment."""
    print(f"\n{'='*50}")
    print(f"Testing {agent_name} on {env_name}")
    print(f"{'='*50}")
    
    # Create environment
    env = make_env(env_name)
    env_info = get_env_info(env)
    
    # Create agent
    if agent_name == "DDPG":
        agent = agent_class(
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            max_action=env_info['max_action']
        )
    else:
        agent = agent_class(
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            max_action=env_info['max_action']
        )
    
    # Training loop
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Max 100 steps per episode
            if agent_name == "PPO":
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, log_prob, value, done)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            else:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                
                # Update agent
                if len(agent.replay_buffer) > 32:
                    agent.update(32)
                
                if done:
                    break
        
        # Update PPO agent
        if agent_name == "PPO":
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


def main():
    """Main demo function."""
    print("🤖 RL Continuous Control Demo")
    print("=" * 50)
    
    # Test different agents
    agents_to_test = [
        (DDPGAgent, "DDPG"),
        (PPOAgent, "PPO"),
        (SACAgent, "SAC")
    ]
    
    results = {}
    
    for agent_class, agent_name in agents_to_test:
        try:
            rewards, avg_final = test_agent(agent_class, agent_name)
            results[agent_name] = {
                'rewards': rewards,
                'avg_final': avg_final
            }
        except Exception as e:
            print(f"Error testing {agent_name}: {e}")
            results[agent_name] = None
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    for agent_name, result in results.items():
        if result is not None:
            print(f"{agent_name:4s}: {result['avg_final']:8.2f}")
        else:
            print(f"{agent_name:4s}: FAILED")
    
    print("\nDemo completed! 🎉")


if __name__ == "__main__":
    main()
