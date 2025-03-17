"""
Configuration Module
==================
Default configuration and configuration loading utilities.
"""

import os
import json
import logging
import yaml

logger = logging.getLogger("TrafficRL.Config")

# Default configuration
CONFIG = {
    "sim_time": 3600,           # Simulation time in seconds
    "num_episodes": 500,        # Number of training episodes
    "max_steps": 1000,          # Maximum steps per episode
    "learning_rate": 0.0003,    # Learning rate for the optimizer
    "gamma": 0.99,              # Discount factor
    "epsilon_start": 1.0,       # Starting epsilon for exploration
    "epsilon_end": 0.01,        # Ending epsilon for exploration
    "epsilon_decay": 0.995,     # Epsilon decay rate
    "buffer_size": 100000,      # Replay buffer size
    "batch_size": 64,           # Batch size for training
    "target_update": 5,         # Target network update frequency
    "eval_frequency": 20,       # Evaluation frequency (episodes)
    "save_frequency": 25,       # Model saving frequency (episodes)
    "grid_size": 4,             # Size of the traffic grid (4x4)
    "max_cars": 30,             # Maximum number of cars per lane
    "green_duration": 10,       # Default green light duration (seconds)
    "yellow_duration": 3,       # Default yellow light duration (seconds)
    "visualization": False,     # Enable visualization during training
    "device": "auto",           # Auto-detect device (CUDA, CPU, MPS)
    "early_stopping_reward": 9999,  # Reward threshold for early stopping
    "early_stopping_patience": 100, # Number of evaluations without improvement before stopping
    "checkpoint_dir": "checkpoints",  # Directory for checkpoints
    "hidden_dim": 256,          # Hidden dimension for neural networks
    "weight_decay": 0.0001,     # L2 regularization parameter
    "grad_clip": 1.0,           # Gradient clipping value
    "use_lr_scheduler": True,   # Use learning rate scheduler
    "lr_step_size": 200,        # LR scheduler step size
    "lr_decay": 0.95,            # LR decay factor
    "clip_rewards": True,       # Whether to clip rewards
    "reward_scale": 0.1,        # Reward scaling factor
    
    # PPO-specific parameters
    "ppo_learning_rate": 3e-4,  # PPO learning rate
    "ppo_gamma": 0.99,          # PPO discount factor
    "ppo_gae_lambda": 0.95,     # GAE lambda parameter
    "ppo_clip_epsilon": 0.2,    # PPO clipping parameter
    "ppo_c1": 1.0,              # Value loss coefficient
    "ppo_c2": 0.01,             # Entropy coefficient
    "ppo_batch_size": 64,       # PPO batch size
    "ppo_n_epochs": 10,         # Number of epochs for PPO updates
    
    "traffic_patterns": {
        "uniform": {
            "arrival_rate": 0.03,
            "variability": 0.01
        },
        "rush_hour": {
            "morning_peak": 0.33,
            "evening_peak": 0.71,
            "peak_intensity": 2.0,
            "base_arrival": 0.03
        },
        "weekend": {
            "midday_peak": 0.5,
            "peak_intensity": 1.5,
            "base_arrival": 0.02
        },
        "natural": {
            "base_arrival": 0.02,
            "peak_intensity": 2.0,
            "weekend_intensity": 1.5,
            "morning_peak": 0.33,
            "evening_peak": 0.71,
            "weekend_peak": 0.5
        }
    },
    "advanced_options": {
        "prioritized_replay": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "dueling_network": True,
        "double_dqn": True
    },
    "random_seed": 42           # Random seed for reproducibility
}

def load_config(config_path=None):
    """
    Load configuration from a file (JSON or YAML), falling back to defaults if file not found.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary of configuration values
    """
    config = CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            file_extension = os.path.splitext(config_path)[1].lower()
            
            with open(config_path, 'r') as f:
                if file_extension == '.yaml' or file_extension == '.yml':
                    loaded_config = yaml.safe_load(f)
                    logger.info(f"Loaded YAML configuration from {config_path}")
                else:
                    # Default to JSON
                    loaded_config = json.load(f)
                    logger.info(f"Loaded JSON configuration from {config_path}")
                
                # Update default config with loaded values
                if loaded_config:
                    config.update(loaded_config)
                    logger.info(f"Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Falling back to default configuration")
    elif config_path:
        logger.warning(f"Configuration file {config_path} not found, using defaults")
    
    return config

def save_config(config, config_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        return False


def override_config_with_args(config, args):
    """
    Override configuration with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Parsed command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Override with any non-None values from args
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None and key in updated_config:
            updated_config[key] = value
            logger.info(f"Overriding config[{key}] with value: {value}")
    
    return updated_config


if __name__ == "__main__":
    # Example usage for saving default config
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic RL Configuration Manager")
    parser.add_argument("--save", type=str, help="Save default config to specified path")
    parser.add_argument("--load", type=str, help="Load and display config from specified path")
    args = parser.parse_args()
    
    if args.save:
        save_config(CONFIG, args.save)
        logger.info(f"Default configuration saved to {args.save}")
    
    if args.load:
        loaded_config = load_config(args.load)
        logger.info(f"Loaded configuration:")
        for key, value in loaded_config.items():
            if isinstance(value, dict):
                logger.info(f"{key}: <complex value>")
            else:
                logger.info(f"{key}: {value}")