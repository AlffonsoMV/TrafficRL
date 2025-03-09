# Traffic RL: Reinforcement Learning for Traffic Light Control

This package provides a comprehensive framework for training and evaluating reinforcement learning agents for traffic light control optimization.

## Features

- **Multiple Agent Types**: DQN, Dueling DQN, Fixed-Timing, Adaptive-Timing, and Random agents
- **Realistic Traffic Simulation**: Simulates various traffic patterns and densities
- **Comprehensive Evaluation**: Tools for benchmarking and analyzing agent performance
- **Visualization**: Extensive visualization capabilities for training metrics and traffic simulations

## Installation

### From Source

To install the package from source:

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-rl.git

# Install the package in development mode
pip install -e .
```

This will install the `traffic_rl` command-line tool and all required dependencies.

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gymnasium
- Pygame
- Pandas
- Seaborn

## CLI Usage

After installation, you can use the `traffic_rl` command-line tool:

### Training

```bash
# Basic training
traffic_rl train --output results/training
```

```bash
# Basic training with real-time vizualisation (WARNING: this slows down training time)
traffic_rl train --output results/training --visualization
```

### Evaluation

```bash
# Evaluate a model on multiple traffic patterns
traffic_rl evaluate --model results/training/best_model.pth --episodes 20 --output results/evaluation
```

### Visualization

```bash
# Record a video of the environment
traffic_rl visualize --type environment --duration 5 --model results/training/best_model.pth --output results/visualizations
```

### Analysis

The analysis command runs both benchmarking and analysis in one step:

```bash
# Comprehensive analysis (includes benchmarking)
traffic_rl analyze --model results/training/best_model.pth --output results/analysis --episodes 10
```

You can also use existing benchmark results if available:

```bash
# Analysis using existing benchmark results
traffic_rl analyze --model results/training/best_model.pth --benchmark-dir results/benchmark --output results/analysis
```

## Advanced Features

The DQN implementation includes several advanced features that can be enabled in the configuration:

- **Prioritized Experience Replay**: Samples more important transitions more frequently
- **Dueling Network Architecture**: Separates state value and action advantage estimation
- **Double DQN**: Reduces overestimation of Q-values
- **Early Stopping**: Stops training when performance plateaus
