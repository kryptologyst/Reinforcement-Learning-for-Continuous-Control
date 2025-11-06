"""
Proximal Policy Optimization (PPO) implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque


class PPOPolicy(nn.Module):
    """
    PPO Policy network with actor and critic.
    
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
        
        # Shared layers
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor head
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(prev_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        shared_features = self.shared(state)
        
        action_mean = self.max_action * torch.tanh(self.actor_mean(shared_features))
        action_log_std = self.actor_log_std.expand_as(action_mean)
        value = self.critic(shared_features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_mean, action_log_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action_mean.shape[0], 1)
        else:
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        max_action: Maximum action value
        lr: Learning rate
        gamma: Discount factor
        eps_clip: PPO clipping parameter
        k_epochs: Number of epochs for policy update
        device: Device to run on
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 lr: float = 3e-4, gamma: float = 0.99, eps_clip: float = 0.2,
                 k_epochs: int = 4, device: str = "cpu"):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Policy network
        self.policy = PPOPolicy(state_dim, action_dim, max_action).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select an action using the policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic=deterministic)
            
            action_np = action.cpu().numpy()[0]
            if isinstance(action_np, tuple):
                action_np = np.array(action_np)
            
            return action_np, log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        log_prob: float, value: float, done: bool) -> None:
        """Store a transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary containing loss values
        """
        if len(self.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Calculate returns and advantages
        returns = self._calculate_returns()
        advantages = returns - old_values.squeeze()
        
        # Normalize advantages (handle single sample case)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.k_epochs):
            # Get current policy outputs
            _, new_log_probs, new_values = self.policy.get_action(states)
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            if new_values.dim() > 1:
                value_loss = F.mse_loss(new_values.squeeze(), returns)
            else:
                value_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Clear experience buffer
        self._clear_buffer()
        
        return {
            "policy_loss": total_policy_loss / self.k_epochs,
            "value_loss": total_value_loss / self.k_epochs
        }
    
    def _calculate_returns(self) -> torch.Tensor:
        """Calculate discounted returns."""
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = float(reward) + self.gamma * float(discounted_reward)
            returns.insert(0, discounted_reward)
        
        return torch.FloatTensor(returns).to(self.device)
    
    def _clear_buffer(self) -> None:
        """Clear the experience buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
