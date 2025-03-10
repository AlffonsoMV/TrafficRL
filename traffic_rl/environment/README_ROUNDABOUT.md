# Roundabout Traffic Simulation Environment

This module provides a roundabout traffic simulation environment for reinforcement learning. It extends the grid-based traffic simulation environment to model a roundabout with multiple entry/exit points.

## Features

- Realistic roundabout traffic simulation with multiple entry/exit points
- Traffic lights at each entry point to control flow into the roundabout
- Support for different traffic patterns (uniform, rush hour, weekend)
- Visualization of the roundabout and traffic flow
- Compatible with the existing RL agents in the Traffic RL package

## Usage

### Command Line Interface

You can use the roundabout environment with the existing CLI by specifying the `--env-type` parameter:

```bash
# Train an agent on the roundabout environment
traffic_rl train --env-type roundabout --output results/roundabout_training

# Evaluate a trained agent on the roundabout environment
traffic_rl evaluate --env-type roundabout --model results/roundabout_training/best_model.pth --output results/roundabout_evaluation
```

### Example Script

You can also use the provided example script to test the roundabout environment:

```bash
# Run the roundabout example with visualization
python -m traffic_rl.examples.roundabout_example --visualization

# Run with 6 entry points instead of the default 4
python -m traffic_rl.examples.roundabout_example --entry-points 6 --visualization
```

## Configuration

The roundabout environment uses the same configuration structure as the grid environment, with a few additional parameters:

- `num_entry_points`: Number of entry/exit points in the roundabout (default: 4)
- `roundabout_capacity`: Maximum traffic density allowed in the roundabout (default: 0.8)

You can set these parameters in your configuration file or via the command line.

## Environment Details

### State Space

For each entry point, the observation includes:
- Entry traffic density (normalized)
- Roundabout traffic density (normalized)
- Traffic light state (0 for Entry green, 1 for Roundabout green)
- Entry waiting time (normalized)
- Roundabout waiting time (normalized)

### Action Space

The action space is discrete with 2 possible actions:
- 0: Entry Green (Roundabout Red) - Allow cars to enter the roundabout
- 1: Roundabout Green (Entry Red) - Prioritize cars already in the roundabout

### Reward Function

The reward function considers:
- Negative reward for waiting cars (weighted by density)
- Positive reward for cars passing through
- Penalty for switching lights too frequently
- Fairness component to prevent uneven queues

## Visualization

The visualization shows:
- The roundabout with entry/exit roads
- Traffic lights at each entry point
- Traffic density at each entry and exit point
- Overall roundabout density

## Implementation

The roundabout environment is implemented as a subclass of the `TrafficSimulation` class, overriding the necessary methods to model roundabout traffic flow while maintaining compatibility with the existing codebase. 