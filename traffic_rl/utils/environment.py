"""
Environment Utilities
===================
Utility functions for working with environments.
"""

import logging
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.environment.roundabout_simulation import RoundaboutSimulation

logger = logging.getLogger("TrafficRL.Utils.Environment")

def create_environment(config, visualization=False, random_seed=None, env_type="grid"):
    """
    Create the appropriate environment based on the specified type.
    
    Args:
        config: Configuration dictionary
        visualization: Whether to enable visualization
        random_seed: Random seed for reproducibility
        env_type: Type of environment ('grid' or 'roundabout')
        
    Returns:
        The created environment
    """
    if env_type.lower() == "roundabout":
        logger.info(f"Creating roundabout environment with {config.get('num_entry_points', 4)} entry points")
        return RoundaboutSimulation(
            config=config,
            visualization=visualization,
            random_seed=random_seed
        )
    else:  # Default to grid
        logger.info(f"Creating grid environment with size {config.get('grid_size', 4)}x{config.get('grid_size', 4)}")
        return TrafficSimulation(
            config=config,
            visualization=visualization,
            random_seed=random_seed
        ) 