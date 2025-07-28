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
# Basic training (headless mode) - fastest training
poetry run python -m src.train

# Training with GUI (slower but visual)
poetry run python -m src.train --gui

# Resume training from checkpoint
poetry run python -m src.train --ckpt_path model/model.pth

# Long training run with custom checkpoint
poetry run python -m src.train --number_episodes 5000 --ckpt_path model/model-1.pth

# Custom hyperparameters example
poetry run python -m src.train --number_episodes 5000 --learning_rate 0.01 --minibatch_size 100 --gamma 0.95
```

### Running the Game
```bash
# Run game with UI (requires trained model in model/model.pth)
poetry run python src/game.py

# Alternative: use the eval script
bash scripts/eval_with_gui.sh
```

## Architecture Overview

### Core Components

1. **Game Engine** (src/game_base.py)
   - Base class defining game constants and reward function
   - Reward system:
     - +10 for eating apple
     - -100 for collision/game over
     - -0.1 base penalty per step
     - Distance-based rewards: +5 (very close), +2 (moderately close), -2 (far)

2. **Game Implementations**
   - `src/game.py`: Full game with Pygame UI rendering, supports AI model loading
   - `src/game_no_ui.py`: Headless version for faster training
   - Both implement `is_danger()` for collision detection and state extraction

3. **Neural Network** (src/model.py)
   - 4-layer fully connected network: 16 → 128 → 64 → 32 → 4
   - ReLU activation with 0.2 dropout between layers
   - Automatic CUDA/CPU device selection

4. **Training System** (src/train.py)
   - Deep Q-Network with experience replay buffer
   - Double DQN architecture (local and target networks)
   - Epsilon-greedy exploration with decay
   - Soft target network updates
   - Saves best model based on score
   - Training data persistence (epsilon, record score)

### State Representation
The game state is a 16-dimensional binary vector:
- Danger in 8 directions (left, right, up, down, diagonals) - 8 values
- Current snake direction (one-hot encoded) - 4 values  
- Food relative position (left, right, up, down of head) - 4 values

### Action Space
4 discrete actions represented as one-hot vectors: [1,0,0,0] = UP, [0,1,0,0] = DOWN, etc.

## Key Training Parameters

- `--number_episodes`: Total training episodes (default: 100000)
- `--maximum_number_steps_per_episode`: Max steps per game (default: 200000)
- `--epsilon_starting_value`: Initial exploration rate (default: 1.0)
- `--epsilon_ending_value`: Final exploration rate (default: 0.001)
- `--epsilon_decay_value`: Decay rate for epsilon (default: 0.99)
- `--learning_rate`: Neural network learning rate (default: 0.01)
- `--minibatch_size`: Batch size for training (default: 100)
- `--gamma`: Discount factor for future rewards (default: 0.95)
- `--replay_buffer_size`: Experience replay buffer size (default: 100000)
- `--interpolation_parameter`: Target network soft update rate (default: 0.01)

## Model and Data Storage

- Trained models saved to: `model/model.pth` or custom path via `--ckpt_path`
- Training metrics saved to: `model/data.json` (record score, epsilon value)
- Progress logged every 50 episodes with current/max/average scores
- Best model automatically saved when new high score achieved

## Development Notes

- No test suite or linting configuration exists
- The game requires Python 3.10.0 (specified in .python-version)
- Uses Poetry for dependency management with pinned versions
- PyTorch with CUDA support enabled if GPU available
- Experience replay updates every 4 steps
- Model includes both player-controlled and AI-controlled snake support