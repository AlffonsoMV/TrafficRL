# Traffic RL: Reinforcement Learning for Traffic Light Control

This package provides a comprehensive framework for training and evaluating reinforcement learning agents for traffic light control optimization.

## Features

- **Multiple Agent Types**: DQN, Fixed-Timing, and other configurable agent types
- **Realistic Traffic Patterns**: Simulates various traffic patterns including uniform, rush hour, and weekend scenarios
- **Comprehensive Evaluation**: Tools for benchmarking and analyzing agent performance
- **Visualization**: Extensive visualization capabilities for training metrics and traffic simulations

## Installation

### Quick Installation

For a quick and easy installation, you can use the provided install script which handles the package installation automatically:

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-rl.git

# Navigate to the project directory
cd traffic-rl

# Run the install script
./install.sh

# Optionally, create a virtual environment
./install.sh --venv
```

After installation, you can verify everything is working correctly with the test script that checks all dependencies and module imports:

```bash
# Test the installation
python test_installation.py

# Check the CLI
traffic_rl --help
```

### Manual Installation

To install the package manually:

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-rl.git

# Navigate to the project directory
cd traffic-rl

# Install the package
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

The package provides a unified command-line interface for all operations:

```bash
# Get help on available commands
traffic_rl --help
```

### Common Options

All commands support these common options:

```bash
# Specify a configuration file
--config PATH           Path to configuration file

# Set logging level
--log-level LEVEL       Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Set random seed for reproducibility
--seed SEED             Random seed for reproducibility
```

### Training

```bash
# Basic training
traffic_rl train --output results/training

# Training with specific configuration
traffic_rl train --config config.json --episodes 1000 --output results/training

# Training with visualization enabled
traffic_rl train --visualization --output results/training

# Training with no results plots
traffic_rl train --no-plots --output results/training
```

#### Training Options
- `--config PATH`: Path to configuration file
- `--episodes NUM`: Number of training episodes
- `--output DIR`: Output directory (default: results/training)
- `--visualization`: Enable visualization during training
- `--no-visualization`: Disable visualization during training
- `--no-plots`: Disable plotting of training results

### Evaluation

```bash
# Evaluate a model on a specific traffic pattern
traffic_rl evaluate --model results/training/best_model.pth --episodes 20 --output results/evaluation

# Evaluate on multiple traffic patterns
traffic_rl evaluate --model results/training/best_model.pth --patterns uniform,rush_hour,weekend --output results/evaluation

# Evaluate with visualization
traffic_rl evaluate --model results/training/best_model.pth --visualization --output results/evaluation
```

#### Evaluation Options
- `--model PATH`: Path to model file (required)
- `--output DIR`: Directory to save evaluation results (default: results/evaluation)
- `--episodes NUM`: Number of evaluation episodes (default: 10)
- `--patterns LIST`: Comma-separated list of traffic patterns to evaluate (default: uniform)
- `--visualization`: Enable visualization during evaluation
- `--no-visualization`: Disable visualization during evaluation

### Visualization

```bash
# Visualize environment with trained model
traffic_rl visualize --type environment --model results/training/best_model.pth --pattern rush_hour --duration 60 --output results/visualizations

# Visualize training metrics
traffic_rl visualize --type metrics --metrics results/training/training_metrics.json --output results/visualizations

# Visualize traffic patterns
traffic_rl visualize --type patterns --output results/visualizations

# Specify custom output filename
traffic_rl visualize --type environment --model results/training/best_model.pth --filename traffic_video.mp4 --output results/visualizations
```

#### Visualization Options
- `--type TYPE`: Type of visualization (environment, metrics, patterns)
- `--output DIR`: Directory to save visualizations (default: results/visualizations)
- `--model PATH`: Path to model file (for environment visualization)
- `--pattern PATTERN`: Traffic pattern to visualize (default: uniform)
- `--duration SECONDS`: Duration of video in seconds (default: 30)
- `--fps NUM`: Frames per second for video (default: 30)
- `--filename NAME`: Output filename
- `--metrics PATH`: Path to metrics file (for metrics visualization)

### Benchmarking

```bash
# Benchmark multiple agents
traffic_rl benchmark --model results/training/best_model.pth --episodes 15 --output results/benchmark --patterns uniform,rush_hour,weekend

# Benchmark without visualizations
traffic_rl benchmark --model results/training/best_model.pth --output results/benchmark --no-visualization
```

#### Benchmarking Options
- `--model PATH`: Path to trained model (for DQN agent)
- `--output DIR`: Directory to save benchmark results (default: results/benchmark)
- `--episodes NUM`: Number of episodes per benchmark (default: 10)
- `--patterns LIST`: Comma-separated list of traffic patterns to test (default: uniform,rush_hour,weekend)
- `--no-visualization`: Disable benchmark visualizations

### Analysis

```bash
# Comprehensive analysis
traffic_rl analyze --model results/training/best_model.pth --metrics results/training/training_metrics.json --benchmark-dir results/benchmark --output results/analysis --episodes 10

# Analysis without opening browser
traffic_rl analyze --model results/training/best_model.pth --metrics results/training/training_metrics.json --output results/analysis --no-browser
```

#### Analysis Options
- `--model PATH`: Path to trained model
- `--metrics PATH`: Path to training metrics file
- `--benchmark-dir DIR`: Directory with existing benchmark results
- `--output DIR`: Directory to save analysis results (default: results/analysis)
- `--patterns LIST`: Comma-separated list of traffic patterns to analyze
- `--episodes NUM`: Number of episodes for new benchmarks (default: 10)
- `--no-browser`: Don't open the analysis report in a browser

## Traffic Patterns

The framework supports several traffic patterns to simulate real-world conditions:

- **Uniform**: Constant traffic flow with slight random variations
- **Rush Hour**: Simulates morning and evening traffic peaks
- **Weekend**: Simulates a midday peak typical of weekend traffic

## Model Selection

During training, the system periodically evaluates the agent and saves the best-performing model as `best_model.pth`. This model is selected based on evaluation performance, not necessarily the final model from training.

When you evaluate or visualize a model, you can specify:

```bash
# Use the best model saved during training
traffic_rl evaluate --model results/training/best_model.pth

# Use a specific checkpoint from training
traffic_rl evaluate --model results/training/model_episode_450.pth

# Use the final model from training
traffic_rl evaluate --model results/training/final_model.pth
```

## Advanced Features

The DQN implementation includes several advanced features that are enabled by default in the configuration:

- **Prioritized Experience Replay**: Samples more important transitions more frequently
- **Dueling Network Architecture**: Separates state value and action advantage estimation
- **Double DQN**: Reduces overestimation of Q-values
- **Early Stopping**: Stops training when performance plateaus

## Project Structure

The project follows a modular architecture for easy extensibility:

```
traffic_rl/                 # Main package
├── agents/                 # Agent implementations
│   ├── base.py             # Base agent class
│   ├── dqn_agent.py        # DQN agent implementation
│   └── fixed_timing_agent.py  # Fixed timing agent
├── environment/            # Simulation environment
│   └── traffic_simulation.py  # Traffic simulation environment
├── memory/                 # Experience replay implementations
│   ├── replay_buffer.py    # Standard replay buffer
│   └── prioritized_buffer.py  # Prioritized experience replay
├── models/                 # Neural network architectures
│   ├── dqn.py              # Standard DQN network
│   └── dueling_dqn.py      # Dueling DQN architecture
├── utils/                  # Utility functions and tools
│   ├── logging.py          # Logging utilities
│   └── visualization.py    # Visualization tools
├── analysis/               # Analysis tools
│   ├── traffic_patterns.py # Traffic pattern analysis
│   ├── training.py         # Training metrics analysis
│   └── comparative.py      # Comparative analysis
├── benchmark/              # Benchmarking tools
│   ├── core.py            # Core benchmarking functionality
│   ├── visualization.py   # Benchmark visualization
│   └── agents.py          # Benchmark agent creation
├── cli.py                  # Command-line interface
├── config.py               # Configuration handling
├── train.py                # Training functionality
├── evaluate.py             # Evaluation functionality
└── results/                # Default directory for results

# Root level
├── setup.py                # Package setup script
├── requirements.txt        # Dependencies
└── custom_config.py        # Example custom configuration
```

### Key Components

- **Agents**: Different agent implementations following a common interface
- **Environment**: Traffic simulation based on Gymnasium
- **Memory**: Experience replay buffers for DQN training
- **Models**: Neural network architectures for value estimation
- **Utils**: Tools for logging and visualization
- **Analysis**: Tools for analyzing traffic patterns, training metrics, and comparative analysis
- **Benchmark**: Tools for benchmarking different agents and configurations
- **CLI**: Unified command-line interface for all operations