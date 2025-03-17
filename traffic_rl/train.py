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
import pygame
import argparse
import yaml

# Import environment and agents
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.agents.ppo_agent import PPOAgent
from traffic_rl.agents.simple_dqn_agent import SimpleDQNAgent
from traffic_rl.utils.visualization import visualize_results
from traffic_rl.evaluate import evaluate

logger = logging.getLogger("TrafficRL.Train")

def create_agent(agent_type, state_size, action_size, config):
    """
    Create an agent based on the specified type.
    
    Args:
        agent_type: Type of agent to create ('dqn', 'simple_dqn', or 'ppo')
        state_size: Size of the state space
        action_size: Size of the action space
        config: Configuration dictionary
    
    Returns:
        Agent instance
    """
    if agent_type.lower() == 'dqn':
        return DQNAgent(state_size, action_size, config)
    elif agent_type.lower() == 'simple_dqn':
        return SimpleDQNAgent(state_size, action_size, config)
    elif agent_type.lower() == 'ppo':
        # Extract PPO-specific config
        ppo_config = {
            "learning_rate": config.get("ppo_learning_rate", 3e-4),
            "gamma": config.get("ppo_gamma", 0.99),
            "gae_lambda": config.get("ppo_gae_lambda", 0.95),
            "clip_epsilon": config.get("ppo_clip_epsilon", 0.2),
            "c1": config.get("ppo_c1", 1.0),
            "c2": config.get("ppo_c2", 0.01),
            "batch_size": config.get("ppo_batch_size", 64),
            "n_epochs": config.get("ppo_n_epochs", 10),
            "hidden_dim": config.get("hidden_dim", 256),
            "device": config.get("device", "auto")
        }
        return PPOAgent(state_size, action_size, ppo_config)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def train(config, model_dir="models", agent_type="dqn"):
    """
    Train the agent with improved monitoring and stability features.
    
    Args:
        config: Configuration dict
        model_dir: Directory to save models
        agent_type: Type of agent to train ('dqn' or 'ppo')
    
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
        agent = create_agent(agent_type, state_size, action_size, config)
        
        # Initialize training metrics
        metrics = {
            "rewards": [],
            "avg_rewards": [],
            "eval_rewards": [],
            "loss_values": [],
            "learning_rates": [],
            "waiting_times": [],
            "throughput": [],
            "training_time": 0
        }
        
        # Add agent-specific metrics
        if agent_type.lower() == 'dqn':
            metrics["epsilon_values"] = []
        elif agent_type.lower() == 'ppo':
            metrics["policy_loss"] = []
            metrics["value_loss"] = []
            metrics["entropy_loss"] = []
        
        # Initialize early stopping variables
        best_eval_reward = -float('inf')
        patience = config.get("early_stopping_patience", 100)
        patience_counter = 0
        
        # Initialize traffic pattern to natural only
        current_pattern = "natural"  # Use only the natural pattern
        pattern_config = config["traffic_patterns"].get(current_pattern, config["traffic_patterns"]["uniform"])
        logger.info(f"Using only the natural traffic pattern for the entire training")
        env.traffic_pattern = current_pattern
        env.traffic_config = pattern_config
        
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
            
            # Record agent-specific metrics
            if agent_type.lower() == 'dqn':
                metrics["epsilon_values"].append(agent.epsilon)
            elif agent_type.lower() == 'ppo':
                # Record PPO-specific metrics if available
                if hasattr(agent, 'policy_loss_history'):
                    metrics["policy_loss"].append(np.mean(agent.policy_loss_history[-episode_steps:]) if episode_steps > 0 else 0)
                if hasattr(agent, 'value_loss_history'):
                    metrics["value_loss"].append(np.mean(agent.value_loss_history[-episode_steps:]) if episode_steps > 0 else 0)
                if hasattr(agent, 'entropy_loss_history'):
                    metrics["entropy_loss"].append(np.mean(agent.entropy_loss_history[-episode_steps:]) if episode_steps > 0 else 0)
            
            # Record learning rate
            current_lr = agent.optimizer.param_groups[0]['lr']
            metrics["learning_rates"].append(current_lr)
            
            # Log progress
            log_msg = f"Episode {episode}/{config['num_episodes']} - "
            log_msg += f"Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
            if agent_type.lower() == 'dqn':
                log_msg += f"Epsilon: {agent.epsilon:.4f}, "
            log_msg += f"LR: {current_lr:.6f}, Traffic: {current_pattern}"
            logger.info(log_msg)
            
            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'reward': f"{total_reward:.2f}",
                    'avg': f"{avg_reward:.2f}",
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
        
        return metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agent for traffic control')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--agent-type', type=str, default='dqn', choices=['dqn', 'simple_dqn', 'ppo'], 
                        help='Type of agent to train (dqn, simple_dqn, or ppo)')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        exit(1)
    
    # Create output directory structure
    model_dir = os.path.join(args.output_dir, args.agent_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Train agent
    logger.info(f"Starting training for {args.agent_type} agent...")
    metrics = train(config, model_dir=model_dir, agent_type=args.agent_type)
    
    # Visualize results
    os.makedirs(os.path.join(args.output_dir, args.agent_type, 'plots'), exist_ok=True)
    visualize_results(
        metrics["rewards"], 
        metrics["avg_rewards"], 
        save_path=os.path.join(args.output_dir, args.agent_type, 'plots', 'training_progress.png')
    )
    
    logger.info(f"Training completed. Models saved to {model_dir}")
    
    
    
    