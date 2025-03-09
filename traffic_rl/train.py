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
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.utils.visualization import visualize_results
from traffic_rl.evaluate import evaluate

logger = logging.getLogger("TrafficRL.Train")

def train(config, model_dir="models"):
    """
    Train the agent with improved monitoring and stability features.
    
    Args:
        config: Configuration dict
        model_dir: Directory to save models
    
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
        env = TrafficSimulation(
            config=config,
            visualization=config["visualization"],
            random_seed=config.get("random_seed", 42)
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
                pattern_config = config["traffic_patterns"].get(current_pattern, config["traffic_patterns"]["uniform"])
                logger.info(f"Switching to {current_pattern} traffic pattern at episode {episode}")
                env.traffic_pattern = current_pattern
                env.traffic_config = pattern_config
            
            # Reset environment
            state, _ = env.reset()
            state = state.flatten()  # Flatten for NN input
            
            # Initialize episode variables
            total_reward = 0
            episode_steps = 0
            waiting_time = 0
            throughput = 0
            
            # Episode loop
            for step in range(config["max_steps"]):
                # Handle pygame events to keep the window responsive
                if config["visualization"]:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            env.visualization = False
                            config["visualization"] = False
                            logger.info("Visualization disabled by user")
                
                # Update episode and step information for visualization
                if config["visualization"]:
                    env.current_episode = episode
                    env.current_step = step
                
                # Select action
                action = agent.act(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state.flatten()  # Flatten for NN input
                
                # Render the environment if visualization is enabled
                if config["visualization"]:
                    env.render()
                
                # Apply reward clipping if enabled
                if config.get("clip_rewards", False):
                    reward = np.clip(reward, -10.0, 10.0)
                
                # Apply reward scaling if specified
                if "reward_scale" in config:
                    reward *= config["reward_scale"]
                
                # Store experience
                agent.step(state, action, reward, next_state, terminated)
                
                # Update state and stats
                state = next_state
                total_reward += reward
                episode_steps += 1
                waiting_time += info.get('average_waiting_time', 0)
                throughput += info.get('total_cars_passed', 0)
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Store rewards and compute averages
            metrics["rewards"].append(total_reward)
            
            # Calculate average reward over last 100 episodes (or fewer if we don't have 100 yet)
            window_size = min(100, len(metrics["rewards"]))
            avg_reward = np.mean(metrics["rewards"][-window_size:])
            metrics["avg_rewards"].append(avg_reward)
            
            # Calculate average waiting time and throughput for this episode
            avg_waiting_time = waiting_time / episode_steps if episode_steps > 0 else 0
            avg_throughput = throughput / episode_steps if episode_steps > 0 else 0
            metrics["waiting_times"].append(avg_waiting_time)
            metrics["throughput"].append(avg_throughput)
            
            # Record epsilon and learning rate
            metrics["epsilon_values"].append(agent.epsilon)
            current_lr = agent.optimizer.param_groups[0]['lr']
            metrics["learning_rates"].append(current_lr)
            
            # Record loss values if available
            if hasattr(agent, 'loss_history') and agent.loss_history:
                # Get average loss over this episode
                metrics["loss_values"].append(np.mean(agent.loss_history[-episode_steps:]) if episode_steps > 0 else 0)
            
            # Log progress
            logger.info(f"Episode {episode}/{config['num_episodes']} - "
                       f"Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
                       f"Epsilon: {agent.epsilon:.4f}, LR: {current_lr:.6f}, "
                       f"Traffic: {current_pattern}")
            
            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'reward': f"{total_reward:.2f}",
                    'avg': f"{avg_reward:.2f}",
                    'eps': f"{agent.epsilon:.2f}",
                    'pattern': current_pattern
                })
            
            # Evaluate the agent periodically
            if episode % config["eval_frequency"] == 0:
                logger.info(f"Evaluating agent at episode {episode}...")
                eval_reward = evaluate(agent, env, num_episodes=5)
                metrics["eval_rewards"].append(eval_reward)
                logger.info(f"Evaluation - Avg Reward: {eval_reward:.2f}")
                
                # Check for improvement and save model if improved
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    patience_counter = 0
                    try:
                        model_path = os.path.join(model_dir, "best_model.pth")
                        agent.save(model_path)
                        logger.info(f"New best model saved with reward: {best_eval_reward:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to save best model: {e}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} evaluations")
                    
                    # Apply early stopping if patience is exceeded
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {patience} evaluations without improvement")
                        break
            
            # Save model periodically
            if episode % config["save_frequency"] == 0:
                try:
                    model_path = os.path.join(model_dir, f"model_episode_{episode}.pth")
                    agent.save(model_path)
                    logger.info(f"Model checkpoint saved at episode {episode}")
                except Exception as e:
                    logger.error(f"Failed to save model checkpoint: {e}")
            
            # Early stopping if we've reached a good performance
            if avg_reward > config.get("early_stopping_reward", float('inf')):
                logger.info(f"Early stopping at episode {episode} - Reached target performance")
                break
        
        # Record training end time
        end_time = time.time()
        metrics["training_time"] = end_time - start_time
        
        # Close the environment
        env.close()
        
        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()
        
        # Save final model
        try:
            model_path = os.path.join(model_dir, "final_model.pth")
            agent.save(model_path)
            logger.info("Final model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        
        return metrics
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save the model before exiting
        try:
            model_path = os.path.join(model_dir, "interrupted_model.pth")
            agent.save(model_path)
            logger.info(f"Interrupted model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save interrupted model: {e}")
        
        # Close the environment
        env.close()
        
        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()
            
        return metrics
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Close the environment
        try:
            env.close()
        except Exception:
            pass
        
        # Close progress bar
        if progress_bar is not None:
            try:
                progress_bar.close()
            except Exception:
                pass
                
        raise


if __name__ == "__main__":
    from config import CONFIG
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Train agent
    metrics = train(CONFIG)
    
    # Visualize results
    visualize_results(metrics["rewards"], metrics["avg_rewards"], save_path="results/training_progress.png")
    
    
    
    