"""
Evaluation Module
===============
Functions for evaluating trained reinforcement learning agents.
"""

import os
import numpy as np
import torch
import logging
import pygame  # Add pygame import

# Import environment and agent
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.environment.roundabout_simulation import RoundaboutSimulation
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.utils.environment import create_environment

logger = logging.getLogger("Evaluate")

def evaluate(agent, env, num_episodes=10):
    """
    Evaluate the agent without exploration.
    
    Args:
        agent: The agent to evaluate
        env: The environment
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Average reward over episodes
    """
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten for NN input
        total_reward = 0
        
        for step in range(1000):  # Max steps
            # Handle pygame events to keep the window responsive
            if hasattr(env, 'visualization') and env.visualization:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        env.visualization = False
                        logger.info("Visualization disabled by user")
            
            # Update episode and step information for visualization
            if hasattr(env, 'visualization') and env.visualization:
                env.current_episode = episode + 1
                env.current_step = step
            
            # Select action (using evaluation mode)
            action = agent.act(state, eval_mode=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()  # Flatten for NN input
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
    
    # Calculate average reward
    avg_reward = np.mean(rewards)
    
    return avg_reward

def evaluate_agent(config, model_path, traffic_pattern="uniform", num_episodes=10, env_type="grid"):
    """
    Evaluate a trained agent on a specific traffic pattern.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model
        traffic_pattern: Traffic pattern to evaluate on
        num_episodes: Number of episodes to evaluate
        env_type: Type of environment ('grid' or 'roundabout')
    
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Set traffic pattern in config
        config["traffic_pattern"] = traffic_pattern
        
        # Initialize environment
        env = create_environment(
            config=config,
            visualization=config.get("visualization", False),
            random_seed=config.get("random_seed"),
            env_type=env_type
        )
        
        # Get state and action sizes
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_size = env.action_space.n
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, config)
        
        # Load trained model
        if not agent.load(model_path):
            logger.error(f"Failed to load model from {model_path}")
            return {
                "avg_reward": 0.0,
                "std_reward": 0.0,
                "avg_waiting_time": 0.0,
                "avg_throughput": 0.0,
                "error": "Failed to load model"
            }
        
        # Evaluate agent
        rewards = []
        waiting_times = []
        throughputs = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = state.flatten()  # Flatten for NN input
            total_reward = 0
            episode_waiting_time = 0
            episode_throughput = 0
            
            for step in range(1000):  # Max steps
                # Handle pygame events to keep the window responsive
                if hasattr(env, 'visualization') and env.visualization:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            env.visualization = False
                            logger.info("Visualization disabled by user")
                
                # Update episode and step information for visualization
                if hasattr(env, 'visualization') and env.visualization:
                    env.current_episode = episode + 1
                    env.current_step = step
                
                # Select action (using evaluation mode)
                action = agent.act(state, eval_mode=True)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state.flatten()  # Flatten for NN input
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Track metrics
                episode_waiting_time += info.get('average_waiting_time', 0)
                episode_throughput += info.get('total_cars_passed', 0)
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Calculate average metrics for this episode
            avg_episode_waiting_time = episode_waiting_time / (step + 1)
            avg_episode_throughput = episode_throughput / (step + 1)
            
            # Store metrics
            rewards.append(total_reward)
            waiting_times.append(avg_episode_waiting_time)
            throughputs.append(avg_episode_throughput)
            
            logger.info(f"Episode {episode+1}/{num_episodes} - Reward: {total_reward:.2f}")
        
        # Calculate overall metrics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_waiting_time = np.mean(waiting_times)
        avg_throughput = np.mean(throughputs)
        
        # Close environment
        env.close()
        
        return {
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "avg_waiting_time": float(avg_waiting_time),
            "avg_throughput": float(avg_throughput),
            "rewards": rewards,
            "waiting_times": waiting_times,
            "throughputs": throughputs
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "avg_reward": 0.0,
            "std_reward": 0.0,
            "avg_waiting_time": 0.0,
            "avg_throughput": 0.0,
            "error": str(e)
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    from config import CONFIG
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--pattern", type=str, default="uniform", help="Traffic pattern to evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--output", type=str, default="results/evaluation.json", help="Output file for results")
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_agent(CONFIG, args.model, args.pattern, args.episodes)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {args.output}")
