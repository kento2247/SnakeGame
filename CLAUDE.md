# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Snake Game implementation with Deep Q-Learning (DQN) AI capabilities. The project uses PyTorch for the neural network implementation and Pygame for the game UI.

## Common Development Commands

### Setup
```bash
# Create virtual environment (if not exists)
pyenv virtualenv 3.10.0 snakeagent
pyenv local snakeagent

# Install dependencies
poetry install
```

### Training the AI
```bash
# Basic training (headless mode)
poetry run python -m src.train

# Training with GUI (slower but visual)
poetry run python -m src.train --gui

# Resume training from checkpoint
poetry run python -m src.train --ckpt_path model/model.pth

# Custom hyperparameters example
poetry run python -m src.train --number_episodes 5000 --learning_rate 0.001 --minibatch_size 64
```

### Running the Game
Currently, there's no standalone game runner. The game runs during training or needs to be integrated with a trained model manually.

## Architecture Overview

### Core Components

1. **Game Engine** (src/game_base.py)
   - Base class defining game constants and reward function
   - Reward system: +10 for eating apple, -10 for game over, distance-based rewards

2. **Game Implementations**
   - `src/game.py`: Full game with Pygame UI rendering
   - `src/game_no_ui.py`: Headless version for faster training
   - Both implement state extraction for the neural network

3. **Neural Network** (src/model.py)
   - 4-layer fully connected network: input → 128 → 64 → 32 → output
   - ReLU activation with 0.2 dropout
   - CUDA support for GPU acceleration

4. **Training System** (src/train.py)
   - Deep Q-Network with experience replay
   - Double DQN architecture (local and target networks)
   - Epsilon-greedy exploration strategy
   - Automatic model checkpointing

### State Representation
The game state is represented as a 16-dimensional vector containing:
- Danger straight/right/left (3 values)
- Direction indicators (4 values)
- Food location relative to head (4 values)
- Additional game state information

### Action Space
4 discrete actions: UP, DOWN, LEFT, RIGHT

## Key Training Parameters

- `--number_episodes`: Total training episodes (default: 1000)
- `--maximum_number_steps_per_episode`: Max steps per game (default: 5000)
- `--epsilon_starting_value`: Initial exploration rate (default: 1.0)
- `--epsilon_ending_value`: Final exploration rate (default: 0.001)
- `--epsilon_decay_value`: Decay rate for epsilon (default: 0.995)
- `--learning_rate`: Neural network learning rate (default: 0.00025)
- `--minibatch_size`: Batch size for training (default: 32)
- `--gamma`: Discount factor for future rewards (default: 0.99)
- `--replay_buffer_size`: Experience replay buffer size (default: 5000)
- `--interpolation_parameter`: Target network update rate (default: 0.05)

## Model and Data Storage

- Trained models are saved to `model/model.pth`
- Training data/metrics are saved to `model/data.json`
- The project uses Poetry for dependency management (see pyproject.toml)

## Development Notes

- No test suite currently exists
- No linting or formatting configuration
- Modified files in git suggest active development - check git status before committing
- The game requires Python 3.10+ and uses specific pinned dependency versions