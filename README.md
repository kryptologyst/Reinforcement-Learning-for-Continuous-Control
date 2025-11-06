# Reinforcement Learning for Continuous Control

A comprehensive implementation of state-of-the-art reinforcement learning algorithms for continuous control tasks. This project provides clean, well-documented code with interactive visualization tools and supports multiple RL algorithms and environments.

## Features

- **Multiple RL Algorithms**: DDPG, PPO, SAC implementations with modern PyTorch
- **Various Environments**: MountainCarContinuous, Pendulum, custom continuous control tasks
- **Interactive UI**: Streamlit-based web interface for training and evaluation
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Type Safety**: Full type hints and documentation
- **GPU Support**: CUDA acceleration for faster training
- **Configurable**: YAML-based configuration system
- **Reproducible**: Deterministic training with proper seeding

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Reinforcement-Learning-for-Continuous-Control.git
   cd Reinforcement-Learning-for-Continuous-Control
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; import gymnasium; print('Installation successful!')"
   ```

## Quick Start

### Command Line Training

Train a DDPG agent on MountainCarContinuous:

```bash
python train.py --agent ddpg --env MountainCarContinuous-v0 --config configs/train_config.yaml
```

Train a PPO agent on Pendulum:

```bash
python train.py --agent ppo --env Pendulum-v1 --device cuda
```

### Interactive Web Interface

Launch the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and:
1. Select an agent type (DDPG, PPO, SAC)
2. Choose an environment
3. Configure training parameters
4. Start training and monitor progress
5. Evaluate your trained agent

## Algorithms

### Deep Deterministic Policy Gradient (DDPG)
- **Type**: Actor-Critic, Off-policy
- **Best for**: Continuous control with deterministic policies
- **Key features**: Target networks, experience replay, exploration noise

### Proximal Policy Optimization (PPO)
- **Type**: Policy Gradient, On-policy
- **Best for**: Stable policy learning with clipping
- **Key features**: Clipped objective, multiple epochs per update

### Soft Actor-Critic (SAC)
- **Type**: Actor-Critic, Off-policy, Maximum Entropy
- **Best for**: Sample-efficient continuous control
- **Key features**: Twin critics, automatic temperature tuning

## Environments

### MountainCarContinuous-v0
- **Goal**: Apply continuous force to reach the flag
- **State**: [position, velocity]
- **Action**: [force] ∈ [-1, 1]
- **Reward**: -100 for reaching goal, -action² otherwise

### Pendulum-v1
- **Goal**: Swing up and balance the pendulum
- **State**: [cos(θ), sin(θ), θ̇]
- **Action**: [torque] ∈ [-2, 2]
- **Reward**: -(θ² + 0.1*θ̇² + 0.001*action²)

### ContinuousControl (Custom)
- **Goal**: Navigate a 2D point to target position
- **State**: [x, y, target_x, target_y]
- **Action**: [force_x, force_y] ∈ [-1, 1]
- **Reward**: -distance_to_target - action_penalty

## Usage Examples

### Basic Training

```python
from src.agents import DDPGAgent
from src.envs import make_env, get_env_info

# Create environment
env = make_env("MountainCarContinuous-v0")
env_info = get_env_info(env)

# Create agent
agent = DDPGAgent(
    state_dim=env_info['state_dim'],
    action_dim=env_info['action_dim'],
    max_action=env_info['max_action']
)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if len(agent.replay_buffer) > 64:
            agent.update(64)
        
        episode_reward += reward
        state = next_state
        
        if done or truncated:
            break
    
    print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

### Evaluation

```python
# Evaluate trained agent
def evaluate_agent(agent, env, num_episodes=10):
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)

avg_reward = evaluate_agent(agent, env)
print(f"Average reward: {avg_reward:.2f}")
```

## Configuration

The project uses YAML configuration files for easy parameter tuning:

```yaml
# configs/train_config.yaml
training:
  episodes: 1000
  max_steps: 1000
  batch_size: 64
  eval_freq: 50

agent:
  ddpg:
    lr_actor: 1e-3
    lr_critic: 1e-3
    gamma: 0.99
    tau: 0.005
```

## Monitoring and Logging

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir logs/

# View training metrics at http://localhost:6006
```

### Weights & Biases Integration

```python
import wandb

# Initialize W&B
wandb.init(project="rl-continuous-control")

# Log metrics during training
wandb.log({
    "episode_reward": episode_reward,
    "actor_loss": actor_loss,
    "critic_loss": critic_loss
})
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_agents.py::test_ddpg_agent
```

## 📁 Project Structure

```
rl-continuous-control/
├── src/
│   ├── agents/           # RL algorithm implementations
│   │   ├── ddpg.py       # DDPG agent
│   │   ├── ppo.py        # PPO agent
│   │   └── sac.py        # SAC agent
│   ├── envs/             # Environment definitions
│   │   └── __init__.py   # Environment utilities
│   └── utils/            # Utility functions
│       └── __init__.py   # Replay buffers, etc.
├── configs/              # Configuration files
│   └── train_config.yaml
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── logs/                  # Training logs
├── checkpoints/           # Model checkpoints
├── train.py              # Main training script
├── app.py                # Streamlit web app
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## 🔧 Development

### Code Style

The project follows PEP 8 style guidelines. Format code with:

```bash
# Format with black
black src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Adding New Algorithms

1. Create a new agent class in `src/agents/`
2. Implement the required methods: `select_action`, `store_transition`, `update`
3. Add to `src/agents/__init__.py`
4. Update configuration schema
5. Add tests in `tests/test_agents.py`

### Adding New Environments

1. Create environment class in `src/envs/`
2. Implement `reset`, `step`, and `render` methods
3. Add to `make_env` function
4. Update environment selection in UI

## References

- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the original Gym environments
- The RL community for algorithm implementations
- PyTorch team for the excellent deep learning framework
- Streamlit team for the amazing web app framework

 
# Reinforcement-Learning-for-Continuous-Control
