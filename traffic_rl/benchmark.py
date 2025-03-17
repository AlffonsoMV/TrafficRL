"""
Benchmark Module
=============
Comprehensive benchmarking tools for comparing different agents and configurations.
"""

import os
import json
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from datetime import datetime

# Import environment and agents
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.agents.simple_dqn_agent import SimpleDQNAgent
from traffic_rl.agents.ppo_agent import PPOAgent
from traffic_rl.agents.fixed_timing_agent import FixedTimingAgent, AdaptiveTimingAgent
from traffic_rl.agents.base import BaseAgent, RandomAgent
from traffic_rl.utils.analysis import comparative_analysis
from traffic_rl.config import load_config

logger = logging.getLogger("TrafficRL.Benchmark")

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
        
        # Save configuration used for this benchmark
        with open(os.path.join(benchmark_dir, "benchmark_config.json"), 'w') as f:
            # Filter out any non-serializable objects from config
            config_copy = {k: v for k, v in config.items() if isinstance(v, (dict, list, str, int, float, bool, type(None)))}
            json.dump(config_copy, f, indent=4)
        
        # Initialize results dictionary
        benchmark_results = {
            "benchmark_id": benchmark_id,
            "config": config_copy,
            "timestamp": timestamp,
            "num_episodes": num_episodes,
            "results": {}
        }
        
        # Initialize progress bar
        total_runs = len(agents_to_benchmark) * len(traffic_patterns)
        progress_bar = tqdm(total=total_runs, desc="Benchmarking Progress")
        
        # Dictionary to store all episode data for detailed analysis
        episode_data = []
        
        # Run benchmark for each agent on each traffic pattern
        for agent_name, agent_info in agents_to_benchmark.items():
            for pattern in traffic_patterns:
                # Update progress bar
                progress_bar.set_description(f"Testing {agent_name} on {pattern}")
                
                # Initialize environment
                env = TrafficSimulation(
                    config=config,
                    visualization=False,
                    random_seed=config.get("random_seed", 42)
                )
                
                # Set traffic pattern
                if pattern in config["traffic_patterns"]:
                    env.traffic_pattern = pattern
                    env.traffic_config = config["traffic_patterns"][pattern]
                else:
                    logger.warning(f"Traffic pattern '{pattern}' not found in config. Using uniform.")
                    env.traffic_pattern = "uniform"
                    env.traffic_config = config["traffic_patterns"]["uniform"]
                
                # Get state and action sizes
                state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                action_size = env.action_space.n
                
                # Initialize agent
                agent = None
                
                if isinstance(agent_info, str):
                    # Agent info is a model path, so load the appropriate agent
                    if "simple_dqn" in agent_info.lower():
                        agent = SimpleDQNAgent(state_size, action_size, config.get("agents", {}).get("simple_dqn", {}))
                        if not agent.load(agent_info):
                            logger.error(f"Failed to load SimpleDQN model for agent '{agent_name}' from {agent_info}")
                            continue
                    elif "ppo" in agent_info.lower():
                        agent = PPOAgent(state_size, action_size, config.get("agents", {}).get("ppo", {}))
                        if not agent.load(agent_info):
                            logger.error(f"Failed to load PPO model for agent '{agent_name}' from {agent_info}")
                            continue
                    else:  # Default to DQN
                        agent = DQNAgent(state_size, action_size, config.get("agents", {}).get("dqn", {}))
                        if not agent.load(agent_info):
                            logger.error(f"Failed to load DQN model for agent '{agent_name}' from {agent_info}")
                            continue
                elif callable(getattr(agent_info, 'act', None)):
                    # Agent info is already an agent object
                    agent = agent_info
                else:
                    logger.error(f"Invalid agent info for '{agent_name}': {agent_info}")
                    continue
                
                # Run evaluation episodes
                rewards = []
                waiting_times = []
                throughputs = []
                densities = []
                step_counts = []
                congestion_levels = []
                action_counts = {0: 0, 1: 0}  # Count of NS vs EW actions
                
                # Track time
                start_time = time.time()
                
                for episode in range(num_episodes):
                    # Reset environment and agent if it has a reset method
                    state, _ = env.reset()
                    state_flat = state.flatten()
                    
                    if hasattr(agent, 'reset') and callable(agent.reset):
                        agent.reset()
                    
                    total_reward = 0
                    episode_waiting = 0
                    episode_throughput = 0
                    episode_densities = []
                    steps = 0
                    episode_actions = []
                    
                    # Run episode
                    for step in range(config.get("max_steps", 1000)):
                        action = agent.act(state_flat, eval_mode=True)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        next_state_flat = next_state.flatten()
                        
                        state_flat = next_state_flat
                        total_reward += reward
                        
                        # Track metrics
                        episode_waiting += info.get('average_waiting_time', 0)
                        episode_throughput += info.get('total_cars_passed', 0)
                        episode_densities.append(info.get('traffic_density', 0))
                        episode_actions.append(int(action))
                        
                        # Count actions 
                        if isinstance(action, (int, np.integer)):
                            action_counts[action] += 1
                        
                        steps += 1
                        
                        if terminated or truncated:
                            break
                    
                    # Calculate average congestion level for this episode
                    avg_density = np.mean(episode_densities) if episode_densities else 0
                    congestion_level = "High" if avg_density > 0.7 else "Medium" if avg_density > 0.4 else "Low"
                    
                    # Store episode results
                    rewards.append(total_reward)
                    waiting_times.append(episode_waiting / steps if steps > 0 else 0)
                    throughputs.append(episode_throughput)
                    densities.append(avg_density)
                    step_counts.append(steps)
                    congestion_levels.append(congestion_level)
                    
                    # Add detailed episode data for further analysis
                    episode_data.append({
                        "agent": agent_name,
                        "pattern": pattern,
                        "episode": episode,
                        "reward": total_reward,
                        "waiting_time": episode_waiting / steps if steps > 0 else 0,
                        "throughput": episode_throughput,
                        "avg_density": avg_density,
                        "steps": steps,
                        "congestion_level": congestion_level,
                        "ns_actions": episode_actions.count(0),
                        "ew_actions": episode_actions.count(1)
                    })
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Calculate action distribution
                total_actions = sum(action_counts.values())
                action_distribution = {
                    "NS_Green": action_counts[0] / total_actions * 100 if total_actions > 0 else 0,
                    "EW_Green": action_counts[1] / total_actions * 100 if total_actions > 0 else 0
                }
                
                # Calculate statistics
                benchmark_results["results"][f"{agent_name}_{pattern}"] = {
                    "agent": agent_name,
                    "traffic_pattern": pattern,
                    "avg_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "avg_waiting_time": float(np.mean(waiting_times)),
                    "avg_throughput": float(np.mean(throughputs)),
                    "avg_density": float(np.mean(densities)),
                    "avg_steps": float(np.mean(step_counts)),
                    "elapsed_time": float(elapsed_time),
                    "episodes": num_episodes,
                    "action_distribution": action_distribution
                }
                
                # Log results
                logger.info(f"Benchmark results for {agent_name} on {pattern}:")
                logger.info(f"  Average Reward: {benchmark_results['results'][f'{agent_name}_{pattern}']['avg_reward']:.2f} ± "
                           f"{benchmark_results['results'][f'{agent_name}_{pattern}']['std_reward']:.2f}")
                logger.info(f"  Average Waiting Time: {benchmark_results['results'][f'{agent_name}_{pattern}']['avg_waiting_time']:.2f}")
                logger.info(f"  Average Throughput: {benchmark_results['results'][f'{agent_name}_{pattern}']['avg_throughput']:.2f}")
                logger.info(f"  Action Distribution: NS={action_distribution['NS_Green']:.1f}%, EW={action_distribution['EW_Green']:.1f}%")
                
                # Close environment
                env.close()
                
                # Update progress bar
                progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Convert episode data to DataFrame for easier analysis
        episode_df = pd.DataFrame(episode_data)
        episode_df.to_csv(os.path.join(benchmark_dir, "episode_data.csv"), index=False)
        
        # Save benchmark results
        benchmark_file = os.path.join(benchmark_dir, "benchmark_results.json")
        with open(benchmark_file, 'w') as f:
            # Filter out non-serializable objects
            json.dump(benchmark_results, f, indent=4)
        
        logger.info(f"Benchmark results saved to {benchmark_file}")
        
        # Create detailed visualizations if requested
        if create_visualizations:
            create_benchmark_visualizations(benchmark_results, episode_df, benchmark_dir)
        
        return benchmark_results
    
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def create_benchmark_visualizations(benchmark_results, episode_df, output_dir):
    """
    Create detailed visualizations of benchmark results.
    
    Args:
        benchmark_results: Dictionary containing benchmark results
        episode_df: DataFrame with detailed episode data
        output_dir: Directory to save visualizations
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Extract results for easier plotting
        results = benchmark_results["results"]
        
        # Get unique agents and patterns
        agents = sorted(set([r.split('_')[0] for r in results.keys()]))
        patterns = sorted(set([r.split('_', 1)[1] for r in results.keys()]))
        
        # 1. Reward Comparison Across Agents and Patterns
        plt.figure(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        x = np.arange(len(patterns))
        width = 0.8 / len(agents)
        
        for i, agent in enumerate(agents):
            rewards = [results.get(f"{agent}_{pattern}", {}).get("avg_reward", 0) for pattern in patterns]
            errors = [results.get(f"{agent}_{pattern}", {}).get("std_reward", 0) for pattern in patterns]
            
            offset = (i - len(agents)/2 + 0.5) * width
            plt.bar(x + offset, rewards, width, label=agent, yerr=errors, capsize=5)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel('Average Reward')
        plt.title('Reward Comparison Across Agents and Traffic Patterns')
        plt.xticks(x, patterns)
        plt.legend(title="Agent")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, agent in enumerate(agents):
            rewards = [results.get(f"{agent}_{pattern}", {}).get("avg_reward", 0) for pattern in patterns]
            offset = (i - len(agents)/2 + 0.5) * width
            
            for j, reward in enumerate(rewards):
                plt.text(x[j] + offset, reward + 1, f"{reward:.1f}", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "reward_comparison.png"), dpi=300)
        plt.close()
        
        # 2. Waiting Time Comparison
        plt.figure(figsize=(12, 8))
        
        for i, agent in enumerate(agents):
            waiting_times = [results.get(f"{agent}_{pattern}", {}).get("avg_waiting_time", 0) for pattern in patterns]
            
            offset = (i - len(agents)/2 + 0.5) * width
            plt.bar(x + offset, waiting_times, width, label=agent)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel('Average Waiting Time')
        plt.title('Waiting Time Comparison Across Agents and Traffic Patterns')
        plt.xticks(x, patterns)
        plt.legend(title="Agent")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, agent in enumerate(agents):
            waiting_times = [results.get(f"{agent}_{pattern}", {}).get("avg_waiting_time", 0) for pattern in patterns]
            offset = (i - len(agents)/2 + 0.5) * width
            
            for j, waiting_time in enumerate(waiting_times):
                plt.text(x[j] + offset, waiting_time + 0.1, f"{waiting_time:.1f}", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "waiting_time_comparison.png"), dpi=300)
        plt.close()
        
        # 3. Throughput Comparison
        plt.figure(figsize=(12, 8))
        
        for i, agent in enumerate(agents):
            throughputs = [results.get(f"{agent}_{pattern}", {}).get("avg_throughput", 0) for pattern in patterns]
            
            offset = (i - len(agents)/2 + 0.5) * width
            plt.bar(x + offset, throughputs, width, label=agent)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel('Average Throughput (vehicles)')
        plt.title('Throughput Comparison Across Agents and Traffic Patterns')
        plt.xticks(x, patterns)
        plt.legend(title="Agent")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "throughput_comparison.png"), dpi=300)
        plt.close()
        
        # 4. Action Distribution Comparison
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each traffic pattern
        fig, axes = plt.subplots(1, len(patterns), figsize=(18, 6))
        if len(patterns) == 1:
            axes = [axes]  # Make it iterable if only one pattern
        
        for i, pattern in enumerate(patterns):
            ax = axes[i]
            
            # Prepare data for pie charts
            labels = ['NS Green', 'EW Green']
            
            for j, agent in enumerate(agents):
                result_key = f"{agent}_{pattern}"
                if result_key in results:
                    action_dist = results[result_key].get("action_distribution", {})
                    sizes = [action_dist.get("NS_Green", 0), action_dist.get("EW_Green", 0)]
                    
                    # Create a small subplot within the main subplot
                    if len(agents) <= 2:
                        # For 1-2 agents, use a simpler layout
                        size = 0.5
                        position = [0.25 + j*0.5, 0.5]
                    else:
                        # For 3+ agents, arrange in a grid
                        cols = min(len(agents), 3)
                        rows = (len(agents) + cols - 1) // cols
                        size = 0.9 / max(cols, rows)
                        col = j % cols
                        row = j // cols
                        position = [0.1 + col * (0.8/cols) + size/2, 
                                   0.9 - row * (0.8/rows) - size/2]
                    
                    # Create wedges
                    wedges, texts, autotexts = ax.pie(
                        sizes, 
                        labels=None,
                        autopct='%1.1f%%',
                        startangle=90,
                        radius=size/2,
                        center=position,
                        wedgeprops=dict(width=size/5, edgecolor='w')
                    )
                    
                    # Customize text
                    for autotext in autotexts:
                        autotext.set_fontsize(8)
                    
                    # Add agent name as title for this pie
                    ax.text(position[0], position[1] + 0.05 + size/2, agent,
                           horizontalalignment='center', verticalalignment='bottom',
                           fontsize=10, fontweight='bold')
            
            # Set subplot title
            ax.set_title(f"Traffic Pattern: {pattern}")
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add a common legend
        fig.legend(wedges, labels, title="Action", loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2)
        
        plt.suptitle('Action Distribution by Agent and Traffic Pattern', fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0.1, 1, 0.9])  # Adjust for suptitle and legend
        plt.savefig(os.path.join(plots_dir, "action_distribution.png"), dpi=300)
        plt.close()
        
        # 5. Performance under different congestion levels (using episode data)
        plt.figure(figsize=(15, 8))
        
        # Create pivot table for average reward by agent and congestion level
        congestion_pivot = episode_df.pivot_table(
            values='reward', 
            index='agent', 
            columns='congestion_level', 
            aggfunc='mean'
        ).fillna(0)
        
        # Plot heatmap
        sns.heatmap(congestion_pivot, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5)
        plt.title('Average Reward by Agent and Congestion Level')
        plt.ylabel('Agent')
        plt.xlabel('Congestion Level')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "congestion_performance.png"), dpi=300)
        plt.close()
        
        # 6. Performance metrics radar chart
        # Prepare data for radar chart
        categories = ['Reward', 'Waiting Time', 'Throughput', 'Traffic Density']
        
        # Normalize metrics for radar chart
        metrics_data = {}
        for agent in agents:
            agent_data = []
            for pattern in patterns:
                result_key = f"{agent}_{pattern}"
                if result_key in results:
                    # Get metrics
                    reward = results[result_key].get("avg_reward", 0)
                    waiting_time = results[result_key].get("avg_waiting_time", 0)
                    throughput = results[result_key].get("avg_throughput", 0)
                    density = results[result_key].get("avg_density", 0)
                    
                    # Store metrics
                    metrics_data[(agent, pattern)] = [reward, waiting_time, throughput, density]
        
        # Normalize metrics across all agents (min-max scaling)
        normalized_data = {}
        for i, category in enumerate(categories):
            # Extract all values for this category
            values = [data[i] for data in metrics_data.values()]
            if values:
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val if max_val > min_val else 1
                
                # Normalize each value (higher is better)
                if category == 'Waiting Time' or category == 'Traffic Density':
                    # Inverse for metrics where lower is better
                    for key in metrics_data:
                        if key in normalized_data:
                            normalized_data[key][i] = 1 - ((metrics_data[key][i] - min_val) / range_val)
                        else:
                            normalized_data[key] = [0] * len(categories)
                            normalized_data[key][i] = 1 - ((metrics_data[key][i] - min_val) / range_val)
                else:
                    # Normal for metrics where higher is better
                    for key in metrics_data:
                        if key in normalized_data:
                            normalized_data[key][i] = (metrics_data[key][i] - min_val) / range_val
                        else:
                            normalized_data[key] = [0] * len(categories)
                            normalized_data[key][i] = (metrics_data[key][i] - min_val) / range_val
        
        # Create radar charts for each traffic pattern
        for pattern in patterns:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of categories
            N = len(categories)
            
            # Set ticks and labels
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            plt.xticks(angles[:-1], categories)
            
            # Draw one axis per variable and add labels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
            plt.ylim(0, 1)
            
            # Plot each agent
            for agent in agents:
                if (agent, pattern) in normalized_data:
                    values = normalized_data[(agent, pattern)]
                    values += values[:1]  # Close the loop
                    
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent)
                    ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title(f'Performance Metrics Comparison - {pattern}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"radar_chart_{pattern}.png"), dpi=300)
            plt.close()
            
        # 7. Time series analysis of episode rewards
        plt.figure(figsize=(15, 8))
        
        # Plot reward by episode for each agent
        for agent in agents:
            agent_data = episode_df[episode_df['agent'] == agent]
            for pattern in patterns:
                pattern_data = agent_data[agent_data['pattern'] == pattern]
                if not pattern_data.empty:
                    plt.plot(pattern_data['episode'], pattern_data['reward'], 
                            marker='o', markersize=4, linestyle='-', alpha=0.7,
                            label=f"{agent} - {pattern}")
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward by Episode for Different Agents and Traffic Patterns')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "episode_rewards.png"), dpi=300)
        plt.close()
        
        logger.info(f"Benchmark visualizations created in {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error creating benchmark visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())


def create_benchmark_agents(config, model_paths=None):
    """
    Create a dictionary of agents for benchmarking.
    
    Args:
        config: Configuration dictionary
        model_paths: List of paths to trained models 
                     (if None, only baseline agents are created)
        
    Returns:
        Dictionary mapping agent names to agent objects or model paths
    """
    agents = {}
    
    # Create a sample environment to get state and action sizes
    env = TrafficSimulation(
        config=config,
        visualization=False,
        random_seed=config.get("random_seed", 42)
    )
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    env.close()
    
    # Add baseline agents
    agents["FixedTiming"] = FixedTimingAgent(
        action_size=2,  # Binary action space [NS_GREEN, EW_GREEN]
        phase_duration=config.get("green_duration", 10) * 3  # 3x longer than default green duration
    )
    
    agents["AdaptiveTiming"] = AdaptiveTimingAgent(
        action_size=2,
        min_phase_duration=config.get("green_duration", 10),
        max_phase_duration=config.get("green_duration", 10) * 6
    )
    
    agents["Random"] = RandomAgent(
        state_size=state_size,
        action_size=2,
        seed=config.get("random_seed", 42)
    )
    
    # Add trained agents if model paths are provided
    if model_paths:
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                # Create agent based on file name pattern
                if "simple_dqn" in model_path.lower():
                    agent_config = {
                        "hidden_dim": 64,
                        "device": "auto",
                        "learning_rate": 0.0005,
                        "gamma": 0.99,
                        "epsilon_start": 1.0,
                        "epsilon_end": 0.01,
                        "epsilon_decay": 0.995,
                        "buffer_size": 10000,
                        "batch_size": 64,
                    }
                    agent = SimpleDQNAgent(state_size, action_size, agent_config)
                    if agent.load(model_path):
                        agents["SimpleDQN"] = agent
                        logger.info("SimpleDQN agent loaded successfully")
                    else:
                        logger.error(f"Failed to load SimpleDQN model from {model_path}")
                
                elif "ppo" in model_path.lower():
                    agent_config = {
                        "hidden_dim": 256,
                        "device": "auto",
                        "learning_rate": 0.0003,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_ratio": 0.2,
                        "value_coef": 1.0,
                        "entropy_coef": 0.01,
                        "batch_size": 64,
                        "update_epochs": 10,
                        "max_grad_norm": 0.5
                    }
                    agent = PPOAgent(state_size, action_size, agent_config)
                    if agent.load(model_path):
                        agents["PPO"] = agent
                        logger.info("PPO agent loaded successfully")
                    else:
                        logger.error(f"Failed to load PPO model from {model_path}")
                
                else:  # Assume DQN
                    agent_config = {
                        "hidden_dim": 64,
                        "device": "auto",
                        "learning_rate": 0.001,
                        "gamma": 0.99,
                        "epsilon_start": 1.0,
                        "epsilon_end": 0.01,
                        "epsilon_decay": 0.995,
                        "buffer_size": 10000,
                        "batch_size": 64,
                        "update_every": 4
                    }
                    agent = DQNAgent(state_size, action_size, agent_config)
                    if agent.load(model_path):
                        agents["DQN"] = agent
                        logger.info("DQN agent loaded successfully")
                    else:
                        logger.error(f"Failed to load DQN model from {model_path}")
    
    return agents


if __name__ == "__main__":
    # Example usage
    import argparse
    from traffic_rl.utils.logging import setup_logging
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark traffic light control agents")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--model", type=str, nargs="+", default=None, help="Path(s) to trained model(s)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per benchmark")
    parser.add_argument("--output", type=str, default="results/benchmark", help="Output directory")
    parser.add_argument("--patterns", type=str, default="uniform,rush_hour,weekend", 
                        help="Comma-separated list of traffic patterns to test")
    parser.add_argument("--no-visualizations", action="store_true", 
                        help="Disable creation of visualizations (useful for headless environments)")
    
    args = parser.parse_args()
    
    # Setup logging
    from traffic_rl.utils.logging import setup_logging
    logger = setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse traffic patterns
    patterns = args.patterns.split(',')
    
    # Create benchmark agents
    agents = create_benchmark_agents(config, args.model)
    
    # Run benchmark
    results = benchmark_agents(
        config=config,
        agents_to_benchmark=agents,
        traffic_patterns=patterns,
        num_episodes=args.episodes,
        output_dir=args.output,
        create_visualizations=not args.no_visualizations
    )
    
    # Print summary
    logger.info("Benchmark Summary:")
    for key, value in results["results"].items():
        if isinstance(value, dict) and "avg_reward" in value:
            logger.info(f"  {key}: Reward={value['avg_reward']:.2f}, Waiting={value['avg_waiting_time']:.2f}")
