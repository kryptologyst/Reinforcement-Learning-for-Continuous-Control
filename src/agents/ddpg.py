"""
Deep Deterministic Policy Gradient (DDPG) implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

from src.utils import ReplayBuffer


class Actor(nn.Module):
    """
    Actor network for DDPG.
    
    Maps states to continuous actions.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        max_action: Maximum action value
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.max_action = max_action
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        return self.max_action * self.network(state)


class Critic(nn.Module):
    """
    Critic network for DDPG.
    
    Maps (state, action) pairs to Q-values.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        max_action: Maximum action value
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        gamma: Discount factor
        tau: Soft update parameter
        device: Device to run on
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 lr_actor: float = 1e-3, lr_critic: float = 1e-3, gamma: float = 0.99,
                 tau: float = 0.005, device: str = "cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(device=device)
        
        # Noise for exploration
        self.noise_std = 0.1
    
    def select_action(self, state: np.ndarray, add_noise: bool = True, deterministic: bool = False) -> np.ndarray:
        """
        Select an action using the actor network.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            deterministic: Whether to use deterministic action (overrides add_noise)
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            
            if not deterministic and add_noise:
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                action += noise
                action = np.clip(action, -1.0, 1.0)
            
            return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Update the actor and critic networks.
        
        Args:
            batch_size: Size of batch to sample from replay buffer
            
        Returns:
            Dictionary containing loss values
        """
        if len(self.replay_buffer) < batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
