"""
Soft Actor-Critic (SAC) implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.utils import ReplayBuffer


class SACPolicy(nn.Module):
    """
    SAC Policy network with stochastic actor.
    
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
        
        self.shared = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Returns:
            Tuple of (action_mean, action_log_std)
        """
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x_t) * self.max_action
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) / self.max_action**2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class SACCritic(nn.Module):
    """
    SAC Critic network.
    
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


class SACAgent:
    """
    Soft Actor-Critic agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        max_action: Maximum action value
        lr: Learning rate
        gamma: Discount factor
        tau: Soft update parameter
        alpha: Temperature parameter
        device: Device to run on
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, device: str = "cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Networks
        self.policy = SACPolicy(state_dim, action_dim, max_action).to(device)
        self.policy_target = SACPolicy(state_dim, action_dim, max_action).to(device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        
        self.critic1 = SACCritic(state_dim, action_dim).to(device)
        self.critic1_target = SACCritic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = SACCritic(state_dim, action_dim).to(device)
        self.critic2_target = SACCritic(state_dim, action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(device=device)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action using the policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                mean, _ = self.policy.forward(state_tensor)
                action = torch.tanh(mean) * self.policy.max_action
            else:
                action, _ = self.policy.sample(state_tensor)
            
            return action.cpu().numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Update the policy and critic networks.
        
        Args:
            batch_size: Size of batch to sample from replay buffer
            
        Returns:
            Dictionary containing loss values
        """
        if len(self.replay_buffer) < batch_size:
            return {"policy_loss": 0.0, "critic1_loss": 0.0, "critic2_loss": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_target.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.policy_target, self.policy)
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        
        return {
            "policy_loss": policy_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item()
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
            'policy_state_dict': self.policy.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
