"""
Benchmark Utilities
================
Benchmarking tools for comparing different agents and configurations.
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
from traffic_rl.environment.roundabout_simulation import RoundaboutSimulation
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.agents.fixed_timing_agent import FixedTimingAgent, AdaptiveTimingAgent
from traffic_rl.agents.base import BaseAgent, RandomAgent
from traffic_rl.config import load_config
from traffic_rl.utils.environment import create_environment

logger = logging.getLogger("TrafficRL.Utils.Benchmark")

def benchmark_agents(agents, config, patterns, episodes=10, env_type="grid"):
    """
    Benchmark multiple agents on multiple traffic patterns.
    
    Args:
        agents: Dictionary mapping agent names to agent objects
        config: Configuration dictionary
        patterns: List of traffic pattern names to test
        episodes: Number of episodes to evaluate each agent on each pattern
        env_type: Type of environment ('grid' or 'roundabout')
        
    Returns:
        Dictionary of benchmark results
    """
    try:
        # Create timestamp for this benchmark run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_id = f"benchmark_{timestamp}"
        
        # Initialize results dictionary
        results = {
            "timestamp": timestamp,
            "config": config,
            "patterns": patterns,
            "episodes": episodes,
            "env_type": env_type,
            "results": {}
        }
        
        # Initialize episode data for visualization
        episode_data = []
        
        # Benchmark each agent on each traffic pattern
        for pattern in patterns:
            logger.info(f"Benchmarking on {pattern} traffic pattern...")
            
            # Set traffic pattern in config
            config["traffic_pattern"] = pattern
            
            # Initialize environment
            env = create_environment(
                config=config,
                visualization=False,
                random_seed=config.get("random_seed"),
                env_type=env_type
            )
            
            # Benchmark each agent
            for agent_name, agent in agents.items():
                logger.info(f"Evaluating {agent_name} agent...")
                
                # Initialize metrics
                rewards = []
                waiting_times = []
                throughputs = []
                densities = []
                
                # Run episodes
                for episode in range(episodes):
                    state, _ = env.reset()
                    state = state.flatten()  # Flatten for NN input
                    total_reward = 0
                    episode_waiting_time = 0
                    episode_throughput = 0
                    episode_density = []
                    
                    for step in range(1000):  # Max steps
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
                        episode_density.append(info.get('traffic_density', 0))
                        
                        # Check if episode is done
                        if terminated or truncated:
                            break
                    
                    # Calculate average metrics for this episode
                    avg_episode_waiting_time = episode_waiting_time / (step + 1)
                    avg_episode_throughput = episode_throughput / (step + 1)
                    avg_episode_density = np.mean(episode_density)
                    
                    # Store metrics
                    rewards.append(total_reward)
                    waiting_times.append(avg_episode_waiting_time)
                    throughputs.append(avg_episode_throughput)
                    densities.append(avg_episode_density)
                    
                    # Add to episode data for visualization
                    episode_data.append({
                        "agent": agent_name,
                        "pattern": pattern,
                        "episode": episode,
                        "reward": total_reward,
                        "waiting_time": avg_episode_waiting_time,
                        "throughput": avg_episode_throughput,
                        "density": avg_episode_density
                    })
                
                # Calculate overall metrics
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                avg_waiting_time = np.mean(waiting_times)
                avg_throughput = np.mean(throughputs)
                avg_density = np.mean(densities)
                
                # Store results
                key = f"{agent_name}_{pattern}"
                results["results"][key] = {
                    "agent": agent_name,
                    "pattern": pattern,
                    "avg_reward": float(avg_reward),
                    "std_reward": float(std_reward),
                    "avg_waiting_time": float(avg_waiting_time),
                    "avg_throughput": float(avg_throughput),
                    "avg_density": float(avg_density),
                    "rewards": rewards,
                    "waiting_times": waiting_times,
                    "throughputs": throughputs,
                    "densities": densities
                }
                
                logger.info(f"{agent_name} on {pattern}: Reward={avg_reward:.2f}±{std_reward:.2f}, Waiting={avg_waiting_time:.2f}")
            
            # Close environment
            env.close()
        
        # Convert episode data to DataFrame
        episode_df = pd.DataFrame(episode_data)
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }


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
            columns='density', 
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


def create_benchmark_agents(agent_types, config, model_path=None):
    """
    Create a dictionary of agents for benchmarking.
    
    Args:
        agent_types: List of agent types to create
        config: Configuration dictionary
        model_path: Path to trained model for DQN agent
        
    Returns:
        Dictionary mapping agent names to agent objects
    """
    # Create a dummy environment to get state and action sizes
    env = create_environment(
        config=config,
        visualization=False,
        random_seed=config.get("random_seed"),
        env_type=config.get("env_type", "grid")
    )
    
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    
    # Close the environment
    env.close()
    
    # Create agents dictionary
    agents = {}
    
    for agent_type in agent_types:
        if agent_type.lower() == "dqn":
            # Create DQN agent
            agent = DQNAgent(state_size, action_size, config)
            
            # Load model if provided
            if model_path and os.path.exists(model_path):
                if agent.load(model_path):
                    logger.info(f"Loaded DQN model from {model_path}")
                    agents["DQN"] = agent
                else:
                    logger.error(f"Failed to load DQN model from {model_path}")
            else:
                logger.warning("No model path provided for DQN agent, using untrained agent")
                agents["DQN_Untrained"] = agent
        
        elif agent_type.lower() == "fixed":
            # Create fixed timing agent
            agents["Fixed_Timing"] = FixedTimingAgent(
                action_size=action_size,
                phase_duration=config.get("green_duration", 10)
            )
        
        elif agent_type.lower() == "adaptive":
            # Create adaptive timing agent
            agents["Adaptive_Timing"] = AdaptiveTimingAgent(
                action_size=action_size,
                min_phase_duration=5,
                max_phase_duration=20
            )
        
        elif agent_type.lower() == "random":
            # Create random agent
            agents["Random"] = RandomAgent(
                state_size=state_size, 
                action_size=action_size,
                seed=config.get("random_seed")
            )
    
    return agents
