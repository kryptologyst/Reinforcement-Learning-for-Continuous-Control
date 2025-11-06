"""
Streamlit UI for RL Continuous Control project.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import os
import sys
import torch
import gymnasium as gym

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import DDPGAgent, PPOAgent, SACAgent
from envs import make_env, get_env_info


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agent(agent_type: str, env_info: dict, config: dict, device: str = "cpu"):
    """Create an RL agent."""
    if agent_type == 'ddpg':
        return DDPGAgent(
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            max_action=env_info['max_action'],
            device=device,
            **config['agent']['ddpg']
        )
    elif agent_type == 'ppo':
        return PPOAgent(
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            max_action=env_info['max_action'],
            device=device,
            **config['agent']['ppo']
        )
    elif agent_type == 'sac':
        return SACAgent(
            state_dim=env_info['state_dim'],
            action_dim=env_info['max_action'],
            device=device,
            **config['agent']['sac']
        )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Continuous Control",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Reinforcement Learning for Continuous Control")
    st.markdown("Train and evaluate RL agents on continuous control tasks")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Agent selection
    agent_type = st.sidebar.selectbox(
        "Select Agent",
        ["ddpg", "ppo", "sac"],
        help="Choose the RL algorithm to use"
    )
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Select Environment",
        ["MountainCarContinuous-v0", "Pendulum-v1", "ContinuousControl"],
        help="Choose the environment to train on"
    )
    
    # Device selection
    device = st.sidebar.selectbox(
        "Device",
        ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        help="Choose the device to run on"
    )
    
    # Training parameters
    st.sidebar.header("Training Parameters")
    episodes = st.sidebar.slider("Episodes", 100, 2000, 500)
    max_steps = st.sidebar.slider("Max Steps per Episode", 100, 2000, 1000)
    batch_size = st.sidebar.slider("Batch Size", 32, 256, 64)
    
    # Load configuration
    config = load_config('configs/train_config.yaml')
    config['training']['episodes'] = episodes
    config['training']['max_steps'] = max_steps
    config['training']['batch_size'] = batch_size
    
    # Create environment and agent
    env = make_env(env_name)
    env_info = get_env_info(env)
    agent = create_agent(agent_type, env_info, config, device)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Training", "Evaluation", "Visualization", "About"])
    
    with tab1:
        st.header("🚀 Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Start Training", type="primary"):
                with st.spinner("Training in progress..."):
                    # Training metrics
                    episode_rewards = []
                    episode_lengths = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for episode in range(episodes):
                        state, _ = env.reset()
                        episode_reward = 0
                        episode_length = 0
                        
                        for step in range(max_steps):
                            if agent_type == 'ppo':
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
                                if len(agent.replay_buffer) > batch_size:
                                    agent.update(batch_size)
                                
                                if done:
                                    break
                        
                        # Update PPO agent
                        if agent_type == 'ppo':
                            agent.update()
                        
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                        
                        # Update progress
                        progress = (episode + 1) / episodes
                        progress_bar.progress(progress)
                        status_text.text(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f}")
                    
                    st.success("Training completed!")
                    
                    # Store metrics in session state
                    st.session_state.episode_rewards = episode_rewards
                    st.session_state.episode_lengths = episode_lengths
                    st.session_state.agent = agent
        
        with col2:
            st.subheader("Training Info")
            st.write(f"**Agent:** {agent_type.upper()}")
            st.write(f"**Environment:** {env_name}")
            st.write(f"**Device:** {device}")
            st.write(f"**State Dim:** {env_info['state_dim']}")
            st.write(f"**Action Dim:** {env_info['action_dim']}")
            st.write(f"**Max Action:** {env_info['max_action']}")
    
    with tab2:
        st.header("📊 Evaluation")
        
        if 'agent' in st.session_state:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                num_eval_episodes = st.slider("Evaluation Episodes", 1, 20, 5)
                
                if st.button("Evaluate Agent"):
                    with st.spinner("Evaluating..."):
                        eval_rewards = []
                        
                        for _ in range(num_eval_episodes):
                            state, _ = env.reset()
                            episode_reward = 0
                            
                            while True:
                                action = st.session_state.agent.select_action(state, deterministic=True)
                                next_state, reward, terminated, truncated, _ = env.step(action)
                                episode_reward += reward
                                state = next_state
                                
                                if terminated or truncated:
                                    break
                            
                            eval_rewards.append(episode_reward)
                        
                        st.session_state.eval_rewards = eval_rewards
            
            with col2:
                if 'eval_rewards' in st.session_state:
                    st.subheader("Evaluation Results")
                    st.write(f"**Average Reward:** {np.mean(st.session_state.eval_rewards):.2f}")
                    st.write(f"**Std Reward:** {np.std(st.session_state.eval_rewards):.2f}")
                    st.write(f"**Min Reward:** {np.min(st.session_state.eval_rewards):.2f}")
                    st.write(f"**Max Reward:** {np.max(st.session_state.eval_rewards):.2f}")
                    
                    # Plot evaluation rewards
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(st.session_state.eval_rewards, 'o-')
                    ax.set_title("Evaluation Rewards")
                    ax.set_xlabel("Episode")
                    ax.set_ylabel("Reward")
                    ax.grid(True)
                    st.pyplot(fig)
        else:
            st.info("Please train an agent first!")
    
    with tab3:
        st.header("📈 Visualization")
        
        if 'episode_rewards' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                # Episode rewards plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(st.session_state.episode_rewards)
                ax.set_title("Episode Rewards")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.grid(True)
                st.pyplot(fig)
            
            with col2:
                # Episode lengths plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(st.session_state.episode_lengths)
                ax.set_title("Episode Lengths")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Length")
                ax.grid(True)
                st.pyplot(fig)
            
            # Moving average plot
            window_size = st.slider("Moving Average Window", 10, 100, 50)
            if window_size < len(st.session_state.episode_rewards):
                moving_avg = pd.Series(st.session_state.episode_rewards).rolling(window=window_size).mean()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(st.session_state.episode_rewards, alpha=0.3, label='Raw Rewards')
                ax.plot(moving_avg, label=f'Moving Average (window={window_size})')
                ax.set_title("Episode Rewards with Moving Average")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
        else:
            st.info("Please train an agent first!")
    
    with tab4:
        st.header("ℹ️ About")
        
        st.markdown("""
        ## Reinforcement Learning for Continuous Control
        
        This project implements state-of-the-art reinforcement learning algorithms for continuous control tasks:
        
        ### Algorithms Implemented:
        - **DDPG** (Deep Deterministic Policy Gradient): Actor-critic method for continuous control
        - **PPO** (Proximal Policy Optimization): Policy gradient method with clipping
        - **SAC** (Soft Actor-Critic): Maximum entropy reinforcement learning
        
        ### Environments Supported:
        - **MountainCarContinuous-v0**: Classic continuous control benchmark
        - **Pendulum-v1**: Inverted pendulum control task
        - **ContinuousControl**: Custom 2D point-to-point navigation task
        
        ### Features:
        - Modern PyTorch implementation with type hints
        - Comprehensive training and evaluation metrics
        - Interactive Streamlit interface
        - Support for both CPU and GPU training
        - Configurable hyperparameters
        - Real-time visualization of training progress
        
        ### Usage:
        1. Select an agent type and environment
        2. Configure training parameters
        3. Start training and monitor progress
        4. Evaluate the trained agent
        5. Visualize training results
        
        The project follows modern RL best practices and is designed for both research and educational purposes.
        """)


if __name__ == '__main__':
    main()
