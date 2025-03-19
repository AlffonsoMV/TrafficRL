"""
Benchmark Agents
==============
Functions for creating and managing benchmark agents.
"""

import logging
import numpy as np
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.agents.fixed_timing_agent import FixedTimingAgent, AdaptiveTimingAgent
from traffic_rl.agents.base import BaseAgent, RandomAgent

logger = logging.getLogger("TrafficRL.BenchmarkAgents")

def create_benchmark_agents(config, model_path=None):
    """
    Create a set of agents for benchmarking.
    
    Args:
        config: Configuration dictionary
        model_path: Optional path to a trained model
        
    Returns:
        Dictionary mapping agent names to agent objects
    """
    agents = {}
    
    try:
        # Get action size from config
        action_size = config.get('action_size', 4)
        
        # Create DQN agent if model path is provided
        if model_path:
            # Create a config for DQN agent with the correct hidden_dim
            # The original model uses hidden_dim=128, not 256
            # The original model also uses action_size=2, not 4
            dqn_config = {
                "learning_rate": config.get('learning_rate', 0.0005),
                "gamma": config.get('gamma', 0.99),
                "epsilon_start": 0.0,  # No exploration during evaluation
                "epsilon_end": 0.0,
                "epsilon_decay": 0.0,
                "buffer_size": 1,  # Minimal size for evaluation
                "batch_size": 1,
                "target_update": 1,
                "device": "cpu",
                "grid_size": int(np.sqrt(config.get('state_size', 180) / 5)),  # Estimate grid size
                "independent_control": True,
                "hidden_dim": 128  # Match the model's hidden_dim
            }
            
            # For DQN, use action_size=2 since that's what the model was trained with
            agents['dqn'] = DQNAgent(
                state_size=config.get('state_size', 180),
                action_size=2,  # Use 2 instead of 4 to match the saved model
                config=dqn_config
            )
            
            # Load the model
            try:
                agents['dqn'].load(model_path)
                logger.info(f"Loaded DQN model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # Continue with other agents even if DQN fails
                del agents['dqn']
        
        # Create baseline agents
        agents['fixed'] = FixedTimingAgent(
            action_size=action_size,
            phase_duration=config.get('green_duration', 30)
        )
        
        agents['adaptive'] = AdaptiveTimingAgent(
            action_size=action_size,
            min_phase_duration=config.get('green_duration', 10),
            max_phase_duration=60
        )
        
        agents['random'] = RandomAgent(
            state_size=config.get('state_size', 180),
            action_size=action_size
        )
        
        logger.info(f"Created {len(agents)} benchmark agents")
        return agents
        
    except Exception as e:
        logger.error(f"Error creating benchmark agents: {e}")
        raise 