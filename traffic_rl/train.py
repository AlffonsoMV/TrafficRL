"""
Training Module
=============
Training functions for reinforcement learning agents.
"""

import os
import time
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygame  # Add pygame import

# Import environment and agent
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.environment.roundabout_simulation import RoundaboutSimulation
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.utils.visualization import visualize_results
from traffic_rl.utils.environment import create_environment
from traffic_rl.evaluate import evaluate

logger = logging.getLogger("TrafficRL.Train")

def train(config, model_dir="models", env_type="grid"):
    """
    Train the agent with improved monitoring and stability features.
    
    Args:
        config: Configuration dict
        model_dir: Directory to save models
        env_type: Type of environment ('grid' or 'roundabout')
    
    Returns:
        Dictionary of training history and metrics
    """
    # Create model directory if not exists
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except Exception as e:
        logger.error(f"Failed to create model directory: {e}")
        # Fallback to current directory
        model_dir = "."
    
    try:
        # Initialize environment
        env = create_environment(
            config=config,
            visualization=config["visualization"],
            random_seed=config.get("random_seed", 42),
            env_type=env_type
        )
        
        # Get state and action sizes
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_size = env.action_space.n
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, config)
        
        # Initialize training metrics
        metrics = {
            "rewards": [],
            "avg_rewards": [],
            "eval_rewards": [],
            "loss_values": [],
            "epsilon_values": [],
            "learning_rates": [],
            "waiting_times": [],
            "throughput": [],
            "training_time": 0
        }
        
        # Initialize early stopping variables
        best_eval_reward = -float('inf')
        patience = config.get("early_stopping_patience", 100)
        patience_counter = 0
        
        # Initialize dynamic traffic pattern
        current_pattern = "uniform"  # Start with uniform pattern
        pattern_schedule = {
            0: "uniform",         # Start with uniform
            100: "rush_hour",     # Switch to rush hour after 100 episodes
            200: "weekend",       # Switch to weekend after 200 episodes
            300: "uniform"        # Back to uniform after 300 episodes
        }
        
        # Training progress tracking
        progress_bar = None
        try:
            progress_bar = tqdm(total=config["num_episodes"], desc="Training Progress", ncols=100)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not initialize progress bar: {e}")
        
        # Record training start time
        start_time = time.time()
        
        # Training loop
        for episode in range(1, config["num_episodes"] + 1):
            # Check if we need to switch traffic pattern
            if episode in pattern_schedule:
                current_pattern = pattern_schedule[episode]
                logger.info(f"Switching to {current_pattern} traffic pattern at episode {episode}")
                env.traffic_pattern = current_pattern
                env.traffic_config = config["traffic_patterns"][current_pattern]
            
            # Reset environment
            state, _ = env.reset()
            state = state.flatten()  # Flatten for NN input
            
            # Track episode metrics
            total_reward = 0
            losses = []
            
            # Set current episode for visualization
            if hasattr(env, 'current_episode'):
                env.current_episode = episode
            
            # Episode loop
            for step in range(config["max_steps"]):
                # Set current step for visualization
                if hasattr(env, 'current_step'):
                    env.current_step = step
                
                # Handle pygame events to keep the window responsive
                if config["visualization"]:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            config["visualization"] = False
                            logger.info("Visualization disabled by user")
                
                # Choose action
                action = agent.act(state)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state.flatten()  # Flatten for NN input
                
                # Store experience in replay buffer
                agent.step(state, action, reward, next_state, terminated)
                
                # Learn from experiences
                if len(agent.memory) > config["batch_size"]:
                    # Get experiences from the agent's memory
                    experiences = agent.memory.sample()
                    if experiences is not None:
                        loss = agent.learn(experiences)
                        if loss is not None:
                            losses.append(loss)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Update metrics
            metrics["rewards"].append(total_reward)
            metrics["epsilon_values"].append(agent.epsilon)
            metrics["waiting_times"].append(float(np.mean(env.waiting_time)))
            metrics["throughput"].append(float(np.sum(env.cars_passed)))
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(metrics["rewards"][-100:])
            metrics["avg_rewards"].append(avg_reward)
            
            # Add loss if available
            if losses:
                metrics["loss_values"].append(float(np.mean(losses)))
            else:
                metrics["loss_values"].append(None)
            
            # Add learning rate
            if hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'param_groups'):
                metrics["learning_rates"].append(agent.optimizer.param_groups[0]['lr'])
            else:
                metrics["learning_rates"].append(None)
            
            # Evaluate periodically
            if episode % config["eval_frequency"] == 0:
                eval_reward = evaluate(agent, env, num_episodes=5)
                metrics["eval_rewards"].append(eval_reward)
                
                # Check for early stopping
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    patience_counter = 0
                    
                    # Save best model
                    agent.save(os.path.join(model_dir, "best_model.pth"))
                    logger.info(f"New best model saved with eval reward: {best_eval_reward:.2f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience and config.get("early_stopping_reward", float('inf')) <= best_eval_reward:
                        logger.info(f"Early stopping triggered after {patience} evaluations without improvement")
                        break
            
            # Save model periodically
            if episode % config["save_frequency"] == 0:
                agent.save(os.path.join(model_dir, f"model_episode_{episode}.pth"))
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'reward': f'{total_reward:.2f}',
                    'avg_reward': f'{avg_reward:.2f}',
                    'epsilon': f'{agent.epsilon:.2f}'
                })
            else:
                # Log progress without progress bar
                if episode % 10 == 0:
                    logger.info(f"Episode {episode}/{config['num_episodes']}, "
                               f"Reward: {total_reward:.2f}, "
                               f"Avg Reward: {avg_reward:.2f}, "
                               f"Epsilon: {agent.epsilon:.2f}")
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        # Record training end time
        end_time = time.time()
        metrics["training_time"] = end_time - start_time
        
        # Save final model
        agent.save(os.path.join(model_dir, "final_model.pth"))
        
        # Close environment
        env.close()
        
        # Create visualization of training progress
        try:
            visualize_results(
                metrics["rewards"],
                metrics["avg_rewards"],
                save_path=os.path.join(model_dir, "training_progress.png")
            )
            logger.info(f"Training visualization saved to {os.path.join(model_dir, 'training_progress.png')}")
        except Exception as e:
            logger.error(f"Failed to create training visualization: {e}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "rewards": [],
            "avg_rewards": [],
            "error": str(e)
        }


if __name__ == "__main__":
    from config import CONFIG
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Train agent
    metrics = train(CONFIG)
    
    # Visualize results
    visualize_results(metrics["rewards"], metrics["avg_rewards"], save_path="results/training_progress.png")
