"""
Replay buffer implementations for reinforcement learning.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences.
    
    Args:
        capacity: Maximum number of experiences to store
        device: Device to store tensors on
    """
    
    def __init__(self, capacity: int = 100000, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add an experience to the buffer."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        )
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Experiences are sampled with probability proportional to their TD error.
    
    Args:
        capacity: Maximum number of experiences to store
        alpha: Prioritization exponent (0 = uniform sampling)
        beta: Importance sampling exponent
        device: Device to store tensors on
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, 
                 beta: float = 0.4, device: str = "cpu"):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, priority: Optional[float] = None) -> None:
        """Add an experience to the buffer."""
        if priority is None:
            priority = 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences using prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Sample experiences
        experiences = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device),
            torch.LongTensor(indices).to(self.device),
            torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        )
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """Update priorities for given indices."""
        for idx, priority in zip(indices.cpu().numpy(), priorities.cpu().numpy()):
            self.priorities[idx] = priority ** self.alpha
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
