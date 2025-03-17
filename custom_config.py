"""
Custom Configuration
==================
Custom configuration for training with modified parameters.
"""

from traffic_rl.config import CONFIG

# Create a copy of the default configuration
custom_config = CONFIG.copy()

# Modify the parameters
custom_config["num_episodes"] = 50  # Set to 50 episodes
custom_config["visualization"] = True  # Enable visualization
custom_config["eval_frequency"] = 10  # Evaluate more frequently

if __name__ == "__main__":
    # Import here to avoid circular imports
    import os
    import logging
    from traffic_rl.train import train
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("custom_train.log")
        ]
    )
    
    logger = logging.getLogger("CustomTraining")
    logger.info("Starting training with custom configuration...")
    logger.info(f"Number of episodes: {custom_config['num_episodes']}")
    
    # Run training with custom config
    results = train(custom_config, model_dir="custom_models")
    
    logger.info("Training completed!") 