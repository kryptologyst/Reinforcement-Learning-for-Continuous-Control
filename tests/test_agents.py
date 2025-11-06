"""
Unit tests for RL agents.
"""

import pytest
import torch
import numpy as np
import gymnasium as gym
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import DDPGAgent, PPOAgent, SACAgent
from envs import make_env, get_env_info, ContinuousControlEnv


class TestDDPGAgent:
    """Test cases for DDPG agent."""
    
    def test_agent_creation(self):
        """Test DDPG agent creation."""
        agent = DDPGAgent(state_dim=4, action_dim=2, max_action=1.0)
        assert agent is not None
        assert agent.device == "cpu"
    
    def test_action_selection(self):
        """Test action selection."""
        agent = DDPGAgent(state_dim=4, action_dim=2, max_action=1.0)
        state = np.random.randn(4)
        
        action = agent.select_action(state)
        assert action.shape == (2,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def test_deterministic_action(self):
        """Test deterministic action selection."""
        agent = DDPGAgent(state_dim=4, action_dim=2, max_action=1.0)
        state = np.random.randn(4)
        
        action1 = agent.select_action(state, add_noise=False)
        action2 = agent.select_action(state, add_noise=False)
        
        np.testing.assert_array_almost_equal(action1, action2)
    
    def test_transition_storage(self):
        """Test transition storage."""
        agent = DDPGAgent(state_dim=4, action_dim=2, max_action=1.0)
        
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        assert len(agent.replay_buffer) == 1
    
    def test_update(self):
        """Test agent update."""
        agent = DDPGAgent(state_dim=4, action_dim=2, max_action=1.0)
        
        # Add some transitions
        for _ in range(100):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = np.random.choice([True, False])
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Update agent
        losses = agent.update(batch_size=32)
        assert "actor_loss" in losses
        assert "critic_loss" in losses


class TestPPOAgent:
    """Test cases for PPO agent."""
    
    def test_agent_creation(self):
        """Test PPO agent creation."""
        agent = PPOAgent(state_dim=4, action_dim=2, max_action=1.0)
        assert agent is not None
        assert agent.device == "cpu"
    
    def test_action_selection(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=4, action_dim=2, max_action=1.0)
        state = np.random.randn(4)
        
        action, log_prob, value = agent.select_action(state)
        assert action.shape == (2,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_transition_storage(self):
        """Test transition storage."""
        agent = PPOAgent(state_dim=4, action_dim=2, max_action=1.0)
        
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = 1.0
        log_prob = 0.5
        value = 1.0
        done = False
        
        agent.store_transition(state, action, reward, log_prob, value, done)
        assert len(agent.states) == 1
    
    def test_update(self):
        """Test agent update."""
        agent = PPOAgent(state_dim=4, action_dim=2, max_action=1.0)
        
        # Add some transitions
        for _ in range(100):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = np.random.randn()
            log_prob = np.random.randn()
            value = np.random.randn()
            done = np.random.choice([True, False])
            
            agent.store_transition(state, action, reward, log_prob, value, done)
        
        # Update agent
        losses = agent.update()
        assert "policy_loss" in losses
        assert "value_loss" in losses


class TestSACAgent:
    """Test cases for SAC agent."""
    
    def test_agent_creation(self):
        """Test SAC agent creation."""
        agent = SACAgent(state_dim=4, action_dim=2, max_action=1.0)
        assert agent is not None
        assert agent.device == "cpu"
    
    def test_action_selection(self):
        """Test action selection."""
        agent = SACAgent(state_dim=4, action_dim=2, max_action=1.0)
        state = np.random.randn(4)
        
        action = agent.select_action(state)
        assert action.shape == (2,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def test_deterministic_action(self):
        """Test deterministic action selection."""
        agent = SACAgent(state_dim=4, action_dim=2, max_action=1.0)
        state = np.random.randn(4)
        
        action1 = agent.select_action(state, deterministic=True)
        action2 = agent.select_action(state, deterministic=True)
        
        np.testing.assert_array_almost_equal(action1, action2)
    
    def test_transition_storage(self):
        """Test transition storage."""
        agent = SACAgent(state_dim=4, action_dim=2, max_action=1.0)
        
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        assert len(agent.replay_buffer) == 1
    
    def test_update(self):
        """Test agent update."""
        agent = SACAgent(state_dim=4, action_dim=2, max_action=1.0)
        
        # Add some transitions
        for _ in range(100):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = np.random.choice([True, False])
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Update agent
        losses = agent.update(batch_size=32)
        assert "policy_loss" in losses
        assert "critic1_loss" in losses
        assert "critic2_loss" in losses


class TestEnvironments:
    """Test cases for environments."""
    
    def test_continuous_control_env(self):
        """Test custom continuous control environment."""
        env = ContinuousControlEnv()
        
        # Test reset
        state, info = env.reset()
        assert state.shape == (4,)
        assert isinstance(info, dict)
        
        # Test step
        action = np.random.uniform(-1, 1, 2)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        assert next_state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_make_env(self):
        """Test environment creation function."""
        env = make_env("ContinuousControl")
        assert env is not None
        
        env = make_env("CartPole-v1")
        assert env is not None
    
    def test_get_env_info(self):
        """Test environment info extraction."""
        env = make_env("ContinuousControl")
        info = get_env_info(env)
        
        assert "state_dim" in info
        assert "action_dim" in info
        assert "is_continuous" in info
        assert "max_action" in info


if __name__ == "__main__":
    pytest.main([__file__])
