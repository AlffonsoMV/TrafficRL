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
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import environment and agent
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.utils.visualization import visualize_results
from traffic_rl.analysis.comparative import comparative_analysis

logger = logging.getLogger("TrafficRL.Evaluate")

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
            
            # Get action from agent (eval_mode=True for no exploration)
            # This works with both single actions and per-intersection actions
            action = agent.act(state, eval_mode=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.flatten()  # Flatten for NN input
            
            # Render if visualization is enabled
            if hasattr(env, 'visualization') and env.visualization:
                env.render()
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Record total reward
        rewards.append(total_reward)
        logger.info(f"Evaluation episode {episode+1}/{num_episodes} - Reward: {total_reward:.2f}")
    
    # Calculate average reward
    avg_reward = np.mean(rewards)
    logger.info(f"Evaluation complete - Average reward: {avg_reward:.2f}")
    
    return avg_reward

def evaluate_agent(config, model_path, traffic_pattern="uniform", num_episodes=10):
    """
    Evaluate a trained agent from a model file.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the model file
        traffic_pattern: Traffic pattern to use for evaluation
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of evaluation results
    """
    try:
        # Initialize environment
        env = TrafficSimulation(
            config=config,
            visualization=config["visualization"],
            random_seed=config.get("random_seed", 42)
        )
        
        # Set traffic pattern
        if traffic_pattern in config["traffic_patterns"]:
            pattern_config = config["traffic_patterns"][traffic_pattern]
            env.traffic_pattern = traffic_pattern
            env.traffic_config = pattern_config
            logger.info(f"Using traffic pattern: {traffic_pattern}")
        else:
            logger.warning(f"Traffic pattern {traffic_pattern} not found, using uniform")
            env.traffic_pattern = "uniform"
            env.traffic_config = config["traffic_patterns"]["uniform"]
        
        # Get state and action sizes
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_size = env.action_space.n
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, config)
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        success = agent.load(model_path)
        if not success:
            logger.error(f"Failed to load model from {model_path}")
            return {
                "error": f"Failed to load model from {model_path}",
                "status": "error",
                "model_path": model_path
            }
        
        logger.info(f"Evaluating agent from {model_path} with {traffic_pattern} traffic pattern...")
        
        # Run evaluation
        rewards = []
        waiting_times = []
        throughputs = []
        densities = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = state.flatten()
            total_reward = 0
            episode_waiting = 0
            episode_throughput = 0
            episode_density = []
            
            for step in range(config.get("max_steps", 1000)):
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
                    env.current_episode = episode + 1
                    env.current_step = step
                
                action = agent.act(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state.flatten()
                
                # Render the environment if visualization is enabled
                if config["visualization"]:
                    env.render()
                
                state = next_state
                total_reward += reward
                
                # Track metrics
                episode_waiting += info.get('average_waiting_time', 0)
                episode_throughput += info.get('total_cars_passed', 0)
                episode_density.append(info.get('traffic_density', 0))
                
                if terminated or truncated:
                    break
            
            # Store episode results
            rewards.append(total_reward)
            waiting_times.append(episode_waiting / (step + 1))
            throughputs.append(episode_throughput)
            densities.append(np.mean(episode_density))
            
            logger.info(f"Episode {episode+1}/{num_episodes} - Reward: {total_reward:.2f}")
        
        # Calculate statistics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_waiting = np.mean(waiting_times)
        avg_throughput = np.mean(throughputs)
        avg_density = np.mean(densities)
        
        logger.info(f"Evaluation complete - Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Avg Waiting Time: {avg_waiting:.2f}, Avg Throughput: {avg_throughput:.2f}")
        
        # Close environment
        env.close()
        
        # Return results
        return {
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "avg_waiting_time": float(avg_waiting),
            "avg_throughput": float(avg_throughput),
            "avg_density": float(avg_density),
            "traffic_pattern": traffic_pattern,
            "model_path": model_path,
            "num_episodes": num_episodes
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
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
