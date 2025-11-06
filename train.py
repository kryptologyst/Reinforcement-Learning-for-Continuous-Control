"""
Main training script for reinforcement learning agents.
"""

import argparse
import os
import sys
import yaml
from typing import Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gymnasium as gym

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import DDPGAgent, PPOAgent, SACAgent
from envs import make_env, get_env_info


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_agent(agent, env, config: Dict[str, Any]) -> Dict[str, list]:
    """
    Train an RL agent.
    
    Args:
        agent: RL agent to train
        env: Environment to train on
        config: Training configuration
        
    Returns:
        Dictionary containing training metrics
    """
    episodes = config['training']['episodes']
    max_steps = config['training']['max_steps']
    eval_freq = config['training']['eval_freq']
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_rewards': [],
        'losses': []
    }
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            if isinstance(agent, PPOAgent):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, log_prob, value, done)
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            else:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                # Update agent
                if len(agent.replay_buffer) > config['training']['batch_size']:
                    losses = agent.update(config['training']['batch_size'])
                    metrics['losses'].append(losses)
                
                if done:
                    break
        
        # Update PPO agent
        if isinstance(agent, PPOAgent):
            losses = agent.update()
            metrics['losses'].append(losses)
        
        # Store metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, config['training']['eval_episodes'])
            metrics['eval_rewards'].append(eval_reward)
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Eval Reward: {eval_reward:.2f}")
        else:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    return metrics


def evaluate_agent(agent, env, num_episodes: int = 5) -> float:
    """
    Evaluate an agent's performance.
    
    Args:
        agent: RL agent to evaluate
        env: Environment to evaluate on
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average reward over evaluation episodes
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def plot_training_results(metrics: Dict[str, list], save_path: str = None):
    """
    Plot training results.
    
    Args:
        metrics: Training metrics dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Evaluation rewards
    if metrics['eval_rewards']:
        eval_episodes = np.arange(0, len(metrics['episode_rewards']), 
                                 len(metrics['episode_rewards']) // len(metrics['eval_rewards']))
        axes[0, 1].plot(eval_episodes, metrics['eval_rewards'])
        axes[0, 1].set_title('Evaluation Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
    
    # Episode lengths
    axes[1, 0].plot(metrics['episode_lengths'])
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Length')
    axes[1, 0].grid(True)
    
    # Losses
    if metrics['losses']:
        # Plot different loss types
        loss_keys = list(metrics['losses'][0].keys())
        for i, key in enumerate(loss_keys):
            losses = [loss[key] for loss in metrics['losses']]
            axes[1, 1].plot(losses, label=key)
        
        axes[1, 1].set_title('Training Losses')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RL agents')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--agent', type=str, default='ddpg',
                       choices=['ddpg', 'ppo', 'sac'],
                       help='Agent type to train')
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0',
                       help='Environment name')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create environment
    env = make_env(args.env)
    env_info = get_env_info(env)
    
    # Create agent
    if args.agent == 'ddpg':
        agent = DDPGAgent(
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            max_action=env_info['max_action'],
            device=device,
            **config['agent']['ddpg']
        )
    elif args.agent == 'ppo':
        agent = PPOAgent(
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            max_action=env_info['max_action'],
            device=device,
            **config['agent']['ppo']
        )
    elif args.agent == 'sac':
        agent = SACAgent(
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            max_action=env_info['max_action'],
            device=device,
            **config['agent']['sac']
        )
    
    # Train agent
    print(f"Training {args.agent.upper()} agent on {args.env}")
    metrics = train_agent(agent, env, config)
    
    # Plot results
    plot_training_results(metrics, f'logs/{args.agent}_training_results.png')
    
    # Save agent
    os.makedirs('checkpoints', exist_ok=True)
    agent.save(f'checkpoints/{args.agent}_final.pth')
    
    print("Training completed!")


if __name__ == '__main__':
    main()
