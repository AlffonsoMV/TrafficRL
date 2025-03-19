"""
Core Benchmark Functionality
==========================
Core functions for running benchmarks and evaluating agents.
"""

import os
import json
import numpy as np
import logging
import time
from datetime import datetime
from tqdm import tqdm

from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.agents.base import BaseAgent

logger = logging.getLogger("TrafficRL.Benchmark")

class NumpyEncoder(json.JSONEncoder):
    """Encoder to handle numpy arrays in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def benchmark_agents(config, agents_to_benchmark, traffic_patterns, num_episodes=10, 
                     output_dir="results/benchmark", create_visualizations=True):
    """
    Benchmark multiple agents on multiple traffic patterns.
    
    Args:
        config: Configuration dictionary
        agents_to_benchmark: Dictionary mapping agent names to agent objects or model paths
        traffic_patterns: List of traffic pattern names to test
        num_episodes: Number of episodes to evaluate each agent on each pattern
        output_dir: Directory to save benchmark results
        create_visualizations: Whether to create visualizations of results
        
    Returns:
        Dictionary of benchmark results
    """
    try:
        # Create timestamp for this benchmark run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_id = f"benchmark_{timestamp}"
        
        # Create output directory
        benchmark_dir = os.path.join(output_dir, benchmark_id)
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Initialize results storage
        benchmark_results = {
            'config': config,
            'timestamp': timestamp,
            'results': {}
        }
        
        # Run benchmarks for each agent
        for agent_name, agent in agents_to_benchmark.items():
            logger.info(f"Benchmarking agent: {agent_name}")
            agent_results = {
                'patterns': {}
            }
            
            # Run benchmarks for each traffic pattern
            for pattern in traffic_patterns:
                logger.info(f"Testing pattern: {pattern}")
                pattern_results = []
                
                # Run multiple episodes
                for episode in tqdm(range(num_episodes), desc=f"{agent_name} - {pattern}"):
                    episode_result = run_benchmark_episode(
                        config, agent, pattern, episode
                    )
                    pattern_results.append(episode_result)
                
                # Aggregate results for this pattern
                agent_results['patterns'][pattern] = {
                    'episodes': pattern_results,
                    'summary': aggregate_episode_results(pattern_results)
                }
            
            benchmark_results['results'][agent_name] = agent_results
        
        # Save results
        results_file = os.path.join(benchmark_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=4, cls=NumpyEncoder)
        
        # Create visualizations if requested
        if create_visualizations:
            from .visualization import create_benchmark_visualizations
            create_benchmark_visualizations(benchmark_results, benchmark_dir)
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        raise

def run_benchmark_episode(config, agent, pattern, episode_num):
    """
    Run a single benchmark episode.
    
    Args:
        config: Configuration dictionary
        agent: Agent to benchmark
        pattern: Traffic pattern to test
        episode_num: Episode number for logging
        
    Returns:
        Dictionary of episode results
    """
    try:
        # Initialize environment with the specified traffic pattern
        env = TrafficSimulation(
            config=config,
            visualization=False,  # No visualization during benchmarking
            random_seed=episode_num,
            traffic_pattern=pattern
        )
        
        # Reset environment
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_metrics = []
        
        # Set a maximum number of steps to prevent infinite loops
        max_steps = config.get("max_steps", 500)
        
        # Check if this is the DQN agent (which uses 2 actions) vs others (which use 4)
        is_dqn_agent = agent.__class__.__name__ == "DQNAgent"
        
        # Run episode
        while not done and episode_steps < max_steps:
            try:
                # Process state for DQN agent if needed
                if is_dqn_agent:
                    # Flatten the state for DQN agent - original model expects a flat array
                    dqn_state = state.flatten()
                    action = agent.act(dqn_state, eval_mode=True)
                else:
                    # Use state as is for other agents
                    action = agent.act(state, eval_mode=True)
                
                # Handle new Gymnasium API which returns 5 values
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_steps += 1
                episode_metrics.append(info)
                
                state = next_state
            except Exception as e:
                logger.error(f"Error in episode step: {e}")
                done = True  # Break the loop if there's an error
        
        # If no metrics were collected, return a default result
        if not episode_metrics:
            logger.warning(f"No metrics collected for episode {episode_num}")
            return {
                'episode': episode_num,
                'total_reward': 0,
                'steps': episode_steps,
                'avg_waiting_time': 0,
                'avg_queue_length': 0,
                'throughput': 0,
                'delay_distribution': [0]
            }
        
        # Calculate episode summary
        episode_summary = {
            'episode': episode_num,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'avg_waiting_time': np.mean([m.get('average_waiting_time', 0) for m in episode_metrics]),
            'avg_queue_length': np.mean([m.get('traffic_density', 0) for m in episode_metrics]),
            'throughput': np.sum([m.get('total_cars_passed', 0) for m in episode_metrics]) if episode_metrics else 0,
            'delay_distribution': [m.get('average_waiting_time', 0) for m in episode_metrics]
        }
        
        return episode_summary
        
    except Exception as e:
        logger.error(f"Error in benchmark episode: {e}")
        # Return a default result instead of None to prevent errors in aggregate_episode_results
        return {
            'episode': episode_num,
            'total_reward': 0,
            'steps': 0,
            'avg_waiting_time': 0,
            'avg_queue_length': 0,
            'throughput': 0,
            'delay_distribution': [0]
        }

def aggregate_episode_results(episode_results):
    """
    Aggregate results from multiple episodes.
    
    Args:
        episode_results: List of episode result dictionaries
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not episode_results:
        return None
    
    return {
        'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
        'std_reward': np.std([r['total_reward'] for r in episode_results]),
        'avg_waiting_time': np.mean([r['avg_waiting_time'] for r in episode_results]),
        'std_waiting_time': np.std([r['avg_waiting_time'] for r in episode_results]),
        'avg_queue_length': np.mean([r['avg_queue_length'] for r in episode_results]),
        'avg_throughput': np.mean([r['throughput'] for r in episode_results]),
        'delay_distribution': np.concatenate([r['delay_distribution'] for r in episode_results])
    } 