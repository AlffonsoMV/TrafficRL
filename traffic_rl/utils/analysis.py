"""
Analysis Utilities
================
Comprehensive tools for analyzing agent behavior and performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger("TrafficRL.Utils.Analysis")

def analyze_training_metrics(metrics, save_dir="results"):
    """
    Analyze and visualize training metrics.
    
    Args:
        metrics: Dictionary of training metrics
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract metrics
        rewards = metrics.get("rewards", [])
        avg_rewards = metrics.get("avg_rewards", [])
        eval_rewards = metrics.get("eval_rewards", [])
        losses = metrics.get("loss_values", [])
        epsilons = metrics.get("epsilon_values", [])
        learning_rates = metrics.get("learning_rates", [])
        waiting_times = metrics.get("waiting_times", [])
        throughputs = metrics.get("throughput", [])
        
        # Calculate summary statistics
        summary = {
            "reward": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)) if rewards else None,
                "max": float(np.max(rewards)) if rewards else None,
                "final_avg": float(np.mean(rewards[-100:])) if len(rewards) > 100 else float(np.mean(rewards))
            },
            "waiting_time": {
                "mean": float(np.mean(waiting_times)) if waiting_times else None,
                "min": float(np.min(waiting_times)) if waiting_times else None,
                "max": float(np.max(waiting_times)) if waiting_times else None
            },
            "throughput": {
                "mean": float(np.mean(throughputs)) if throughputs else None,
                "min": float(np.min(throughputs)) if throughputs else None,
                "max": float(np.max(throughputs)) if throughputs else None
            },
            "training_time": metrics.get("training_time", 0)
        }
        
        # Plot rewards
        if rewards:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards, alpha=0.6, label='Episode Reward', color='#3498db')
            plt.plot(avg_rewards, label='Avg Reward (100 episodes)', color='#2c3e50', linewidth=2)
            if eval_rewards:
                # Plot evaluation rewards at their corresponding episodes
                eval_episodes = [i * metrics.get("eval_frequency", 20) for i in range(len(eval_rewards))]
                plt.plot(eval_episodes, eval_rewards, 'ro-', label='Evaluation Reward', color='#e74c3c', markersize=5)
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards Over Time', fontweight='bold')
            plt.legend(frameon=True, fancybox=True, shadow=True)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "reward_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot losses
        if losses:
            plt.figure(figsize=(12, 6))
            plt.plot(losses, color='#e74c3c', linewidth=1.5)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time', fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "loss_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot epsilon decay
        if epsilons:
            plt.figure(figsize=(12, 6))
            plt.plot(epsilons, color='#9b59b6', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate (Epsilon) Over Time', fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "epsilon_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot learning rate
        if learning_rates:
            plt.figure(figsize=(12, 6))
            plt.plot(learning_rates, color='#2ecc71', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Over Time', fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "learning_rate_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot waiting times and throughput
        if waiting_times and throughputs:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1.plot(waiting_times, color='#e67e22', linewidth=1.5)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Average Waiting Time')
            ax1.set_title('Average Waiting Time Per Episode', fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            ax2.plot(throughputs, color='#27ae60', linewidth=1.5)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Throughput')
            ax2.set_title('Average Throughput Per Episode', fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "performance_metrics.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save summary statistics
        with open(os.path.join(save_dir, "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Training analysis completed and saved to {save_dir}")
        
        return summary
    
    except Exception as e:
        logger.error(f"Error analyzing training metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def comparative_analysis(results, save_dir="results"):
    """
    Analyze and visualize comparative benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract agent types and traffic patterns
        agent_types = set()
        traffic_patterns = set()
        
        for key in results.keys():
            if key == "summary":
                continue
                
            parts = key.split('_')
            if len(parts) >= 2:
                agent_type = parts[0]
                pattern = '_'.join(parts[1:])
                
                agent_types.add(agent_type)
                traffic_patterns.add(pattern)
        
        agent_types = sorted(list(agent_types))
        traffic_patterns = sorted(list(traffic_patterns))
        
        # Prepare data for bar plots
        rewards_data = {}
        waiting_data = {}
        throughput_data = {}
        
        for pattern in traffic_patterns:
            rewards_data[pattern] = []
            waiting_data[pattern] = []
            throughput_data[pattern] = []
            
            for agent in agent_types:
                key = f"{agent}_{pattern}"
                if key in results:
                    result = results[key]
                    rewards_data[pattern].append(result.get("avg_reward", 0))
                    waiting_data[pattern].append(result.get("avg_waiting_time", 0))
                    throughput_data[pattern].append(result.get("avg_throughput", 0))
                else:
                    rewards_data[pattern].append(0)
                    waiting_data[pattern].append(0)
                    throughput_data[pattern].append(0)
        
        # Create comparative bar plots - FIX GROUPING BY PATTERN
        bar_width = 0.2  # Narrow bars to fit more in each group
        index = np.arange(len(traffic_patterns))  # X locations for patterns
        
        # Plot reward comparison
        plt.figure(figsize=(12, 8))
        
        for i, agent in enumerate(agent_types):
            agent_rewards = []
            for pattern in traffic_patterns:
                agent_idx = agent_types.index(agent)
                agent_rewards.append(rewards_data[pattern][agent_idx])
            
            offset = (i - len(agent_types)/2 + 0.5) * bar_width
            plt.bar(index + offset, agent_rewards, bar_width, label=agent)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel('Average Reward')
        plt.title('Reward Comparison by Agent and Traffic Pattern')
        plt.xticks(index, traffic_patterns)
        plt.legend(title="Agent")
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, "reward_comparison.png"))
        plt.close()
        
        # Plot waiting time comparison
        plt.figure(figsize=(12, 8))
        
        for i, agent in enumerate(agent_types):
            agent_waiting_times = []
            for pattern in traffic_patterns:
                agent_idx = agent_types.index(agent)
                agent_waiting_times.append(waiting_data[pattern][agent_idx])
            
            offset = (i - len(agent_types)/2 + 0.5) * bar_width
            plt.bar(index + offset, agent_waiting_times, bar_width, label=agent)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel('Average Waiting Time')
        plt.title('Waiting Time Comparison Across Agents and Traffic Patterns')
        plt.xticks(index, traffic_patterns)
        plt.legend(title="Agent")
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, "waiting_time_comparison.png"))
        plt.close()
        
        # Plot throughput comparison
        plt.figure(figsize=(12, 8))
        
        for i, agent in enumerate(agent_types):
            agent_throughputs = []
            for pattern in traffic_patterns:
                agent_idx = agent_types.index(agent)
                agent_throughputs.append(throughput_data[pattern][agent_idx])
            
            offset = (i - len(agent_types)/2 + 0.5) * bar_width
            plt.bar(index + offset, agent_throughputs, bar_width, label=agent)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel('Average Throughput')
        plt.title('Throughput Comparison Across Agents and Traffic Patterns')
        plt.xticks(index, traffic_patterns)
        plt.legend(title="Agent")
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, "throughput_comparison.png"))
        plt.close()
        
        # Calculate improvement percentages
        improvements = {}
        for pattern in traffic_patterns:
            if len(agent_types) >= 2:  # Need at least two agents to compare
                baseline_idx = agent_types.index("FixedTiming") if "FixedTiming" in agent_types else 0
                trained_idx = agent_types.index("TrainedDQN") if "TrainedDQN" in agent_types else 1
                
                baseline_reward = rewards_data[pattern][baseline_idx]
                trained_reward = rewards_data[pattern][trained_idx]
                
                if baseline_reward != 0:
                    reward_improvement = (trained_reward - baseline_reward) / abs(baseline_reward) * 100
                else:
                    reward_improvement = float('inf') if trained_reward > 0 else float('-inf')
                
                baseline_waiting = waiting_data[pattern][baseline_idx]
                trained_waiting = waiting_data[pattern][trained_idx]
                
                if baseline_waiting != 0:
                    waiting_improvement = (baseline_waiting - trained_waiting) / baseline_waiting * 100
                else:
                    waiting_improvement = float('inf') if trained_waiting < baseline_waiting else float('-inf')
                
                baseline_throughput = throughput_data[pattern][baseline_idx]
                trained_throughput = throughput_data[pattern][trained_idx]
                
                if baseline_throughput != 0:
                    throughput_improvement = (trained_throughput - baseline_throughput) / baseline_throughput * 100
                else:
                    throughput_improvement = float('inf') if trained_throughput > 0 else float('-inf')
                
                improvements[pattern] = {
                    "reward_improvement": float(reward_improvement),
                    "waiting_time_improvement": float(waiting_improvement),
                    "throughput_improvement": float(throughput_improvement)
                }
        
        # Create additional recommended visualizations
        
        # 0. Agent Performance Radar Chart - Compare agents across multiple metrics
        plt.figure(figsize=(12, 10))
        
        # Prepare data for radar chart
        metrics = ['Reward', 'Waiting Time', 'Throughput', 'Density']
        
        # For each agent, calculate average metrics across all patterns
        agent_metrics = {}
        for agent in agent_types:
            rewards = []
            waiting_times = []
            throughputs = []
            densities = []
            
            for pattern in traffic_patterns:
                key = f"{agent}_{pattern}"
                if key in results:
                    rewards.append(results[key].get("avg_reward", 0))
                    waiting_times.append(results[key].get("avg_waiting_time", 0))
                    throughputs.append(results[key].get("avg_throughput", 0))
                    densities.append(results[key].get("avg_density", 0))
            
            if rewards:  # Only add if we have data
                agent_metrics[agent] = [
                    np.mean(rewards),
                    np.mean(waiting_times),
                    np.mean(throughputs),
                    np.mean(densities)
                ]
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_metrics = {}
        for i, metric in enumerate(metrics):
            values = [metrics_list[i] for metrics_list in agent_metrics.values()]
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val > min_val else 1
            
            # For waiting time and density, lower is better, so invert
            if metric in ['Waiting Time', 'Density']:
                for agent in agent_metrics:
                    if agent not in normalized_metrics:
                        normalized_metrics[agent] = [0] * len(metrics)
                    normalized_metrics[agent][i] = 1 - ((agent_metrics[agent][i] - min_val) / range_val)
            else:
                for agent in agent_metrics:
                    if agent not in normalized_metrics:
                        normalized_metrics[agent] = [0] * len(metrics)
                    normalized_metrics[agent][i] = (agent_metrics[agent][i] - min_val) / range_val
        
        # Create radar chart
        # Number of variables
        N = len(metrics)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], metrics)
        
        # Draw the agent performance
        for agent, values in normalized_metrics.items():
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Agent Performance Comparison Across Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "agent_performance_radar.png"))
        plt.close()
        
        # 1. Traffic Pattern Impact Analysis - Simple Bar Chart
        plt.figure(figsize=(12, 8))
        
        # Create a direct bar chart using matplotlib directly
        bar_width = 0.8 / len(agent_types)
        
        for i, agent in enumerate(agent_types):
            agent_rewards = []
            x_positions = []
            
            for j, pattern in enumerate(traffic_patterns):
                key = f"{agent}_{pattern}"
                if key in results:
                    reward = float(results[key].get("avg_reward", 0))
                    agent_rewards.append(reward)
                    x_positions.append(j + (i - len(agent_types)/2 + 0.5) * bar_width)
            
            plt.bar(x_positions, agent_rewards, width=bar_width, label=agent)
        plt.title("Traffic Pattern Impact on Agent Performance (Reward)")
        plt.ylabel("Agent")
        plt.xlabel("Traffic Pattern")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "traffic_pattern_impact.png"))
        plt.close()
        
        # 2. Correlation between traffic density and waiting time
        plt.figure(figsize=(10, 6))
        
        # Extract density and waiting time data
        density_data = []
        waiting_time_data = []
        agent_labels = []
        pattern_labels = []
        
        for agent in agent_types:
            for pattern in traffic_patterns:
                key = f"{agent}_{pattern}"
                if key in results:
                    density_data.append(results[key].get("avg_density", 0))
                    waiting_time_data.append(results[key].get("avg_waiting_time", 0))
                    agent_labels.append(agent)
                    pattern_labels.append(pattern)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(density_data, waiting_time_data, c=[agent_types.index(a) for a in agent_labels], 
                             s=100, alpha=0.7, cmap='viridis')
        
        # Add labels for each point
        for i, (x, y, agent, pattern) in enumerate(zip(density_data, waiting_time_data, agent_labels, pattern_labels)):
            plt.annotate(f"{agent}-{pattern}", (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, label='Agent Type')
        plt.xlabel('Traffic Density')
        plt.ylabel('Average Waiting Time')
        plt.title('Correlation Between Traffic Density and Waiting Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "density_waiting_correlation.png"))
        plt.close()
        
        # 3. Decision Analysis - Action Distribution by Traffic Pattern
        plt.figure(figsize=(15, 10))
        
        # Create a grouped bar chart for action distribution
        action_data = pd.DataFrame(columns=['Agent', 'Pattern', 'NS_Green_Pct', 'EW_Green_Pct'])
        
        idx = 0
        for agent in agent_types:
            for pattern in traffic_patterns:
                key = f"{agent}_{pattern}"
                if key in results and "action_distribution" in results[key]:
                    action_dist = results[key]["action_distribution"]
                    action_data.loc[idx] = [
                        agent, 
                        pattern, 
                        action_dist.get("NS_Green", 0),
                        action_dist.get("EW_Green", 0)
                    ]
                    idx += 1
        
        if not action_data.empty:
            # Reshape data for plotting
            action_data_melted = pd.melt(
                action_data, 
                id_vars=['Agent', 'Pattern'], 
                value_vars=['NS_Green_Pct', 'EW_Green_Pct'],
                var_name='Action',
                value_name='Percentage'
            )
            
            # Create grouped bar chart
            plt.figure(figsize=(14, 8))
            sns.barplot(x='Pattern', y='Percentage', hue='Action', data=action_data_melted, 
                       palette=['green', 'red'])
            
            plt.title('Action Distribution by Traffic Pattern')
            plt.xlabel('Traffic Pattern')
            plt.ylabel('Percentage (%)')
            plt.legend(title='Action')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "action_distribution_by_pattern.png"))
            plt.close()
        
        # Save comparative analysis results
        comparative_results = {
            "agent_types": agent_types,
            "traffic_patterns": traffic_patterns,
            "rewards_data": rewards_data,
            "waiting_data": waiting_data,
            "throughput_data": throughput_data,
            "improvements": improvements
        }
        
        with open(os.path.join(save_dir, "comparative_analysis.json"), 'w') as f:
            json.dump(comparative_results, f, indent=4)
        
        logger.info(f"Comparative analysis completed and saved to {save_dir}")
        
        return comparative_results
    
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def analyze_decision_boundaries(agent_analysis, save_dir="results"):
    """
    Analyze and visualize agent decision boundaries.
    
    Args:
        agent_analysis: Dictionary of agent analysis results
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have decision boundaries data
        if "decision_boundaries" not in agent_analysis:
            logger.warning("No decision boundaries data found in agent analysis")
            return {}
        
        # Extract decision boundaries data
        boundaries = agent_analysis["decision_boundaries"]
        
        # Create visualizations for each light state
        for light_state, data in boundaries.items():
            if not data:
                continue
                
            # Extract data for visualization
            ns_densities = np.array([point["ns_density"] for point in data])
            ew_densities = np.array([point["ew_density"] for point in data])
            actions = np.array([point["action"] for point in data])
            
            # Create density grid
            grid_size = int(np.sqrt(len(data)))
            action_grid = actions.reshape(grid_size, grid_size)
            
            # Plot decision boundary
            plt.figure(figsize=(10, 8))
            
            # Custom colormap for the two actions: Green for NS=0, Red for EW=1
            cmap = ListedColormap(['green', 'red'])
            
            plt.imshow(
                action_grid, 
                origin='lower', 
                cmap=cmap,
                extent=[0, 1, 0, 1],
                aspect='auto'
            )
            
            # Add labels and title
            plt.xlabel('East-West Density')
            plt.ylabel('North-South Density')
            plt.title(f'Traffic Light Decision Policy (Current Light: {light_state})')
            
            # Add colorbar with labels
            cbar = plt.colorbar(ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['NS Green', 'EW Green'])
            
            # Add grid
            plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(save_dir, f"decision_boundary_{light_state}.png"))
            plt.close()
            
            # Calculate statistics
            ns_greater = np.sum(ns_densities > ew_densities)
            ew_greater = np.sum(ew_densities > ns_densities)
            ns_actions = np.sum(actions == 0)  # NS Green
            ew_actions = np.sum(actions == 1)  # EW Green
            
            correct_decisions = (
                np.sum((ns_densities > ew_densities) & (actions == 0)) +  # NS density higher and NS green
                np.sum((ew_densities > ns_densities) & (actions == 1))    # EW density higher and EW green
            )
            
            total_decisions = len(data)
            decision_accuracy = correct_decisions / total_decisions * 100 if total_decisions > 0 else 0
            
            # Save statistics
            stats = {
                "light_state": light_state,
                "ns_density_higher_count": int(ns_greater),
                "ew_density_higher_count": int(ew_greater),
                "ns_green_actions": int(ns_actions),
                "ew_green_actions": int(ew_actions),
                "correct_decisions": int(correct_decisions),
                "total_decisions": int(total_decisions),
                "decision_accuracy": float(decision_accuracy)
            }
            
            with open(os.path.join(save_dir, f"decision_stats_{light_state}.json"), 'w') as f:
                json.dump(stats, f, indent=4)
        
        logger.info(f"Decision boundary analysis completed and saved to {save_dir}")
        
        return {"decision_analysis_complete": True}
    
    except Exception as e:
        logger.error(f"Error analyzing decision boundaries: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def visualize_queue_length_dynamics(episode_data, save_dir="results"):
    """
    Visualize queue length dynamics over time.
    
    Args:
        episode_data: DataFrame containing episode data with queue lengths
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of visualization paths
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have the necessary data
        if not isinstance(episode_data, pd.DataFrame) or episode_data.empty:
            logger.warning("No episode data provided for queue length visualization")
            return {}
        
        # Extract unique agents and patterns
        agents = episode_data['agent'].unique()
        patterns = episode_data['pattern'].unique()
        
        # Create visualizations
        visualization_paths = {}
        
        # 1. Queue Length Time Series
        plt.figure(figsize=(14, 8))
        
        # Group by agent and pattern
        for agent in agents:
            agent_data = episode_data[episode_data['agent'] == agent]
            for pattern in patterns:
                pattern_data = agent_data[agent_data['pattern'] == pattern]
                if 'avg_queue_length' in pattern_data.columns and not pattern_data.empty:
                    plt.plot(
                        pattern_data['episode'], 
                        pattern_data['avg_queue_length'], 
                        marker='o', 
                        linestyle='-', 
                        label=f"{agent} - {pattern}"
                    )
        
        plt.xlabel('Episode')
        plt.ylabel('Average Queue Length')
        plt.title('Queue Length Dynamics Over Episodes')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        queue_ts_path = os.path.join(save_dir, "queue_length_time_series.png")
        plt.savefig(queue_ts_path)
        plt.close()
        visualization_paths["queue_time_series"] = queue_ts_path
        
        # 2. Maximum Queue Length Comparison
        plt.figure(figsize=(12, 8))
        
        # Calculate max queue length for each agent and pattern
        max_queue_data = []
        for agent in agents:
            for pattern in patterns:
                subset = episode_data[(episode_data['agent'] == agent) & 
                                     (episode_data['pattern'] == pattern)]
                if 'max_queue_length' in subset.columns and not subset.empty:
                    max_queue = subset['max_queue_length'].max()
                    max_queue_data.append({
                        'agent': agent,
                        'pattern': pattern,
                        'max_queue_length': max_queue
                    })
        
        if max_queue_data:
            max_queue_df = pd.DataFrame(max_queue_data)
            
            # Create grouped bar chart
            sns.barplot(x='agent', y='max_queue_length', hue='pattern', data=max_queue_df)
            plt.xlabel('Agent')
            plt.ylabel('Maximum Queue Length')
            plt.title('Maximum Queue Length Comparison')
            plt.legend(title='Traffic Pattern')
            plt.grid(True, axis='y')
            
            # Save figure
            max_queue_path = os.path.join(save_dir, "max_queue_length_comparison.png")
            plt.savefig(max_queue_path)
            plt.close()
            visualization_paths["max_queue_comparison"] = max_queue_path
        
        # 3. Queue Length Distribution
        plt.figure(figsize=(14, 10))
        
        # Create subplot grid based on number of agents and patterns
        n_agents = len(agents)
        n_patterns = len(patterns)
        fig, axes = plt.subplots(n_agents, n_patterns, figsize=(n_patterns*5, n_agents*4), 
                                squeeze=False, sharex=True, sharey=True)
        
        # Plot queue length distribution for each agent and pattern
        for i, agent in enumerate(agents):
            for j, pattern in enumerate(patterns):
                subset = episode_data[(episode_data['agent'] == agent) & 
                                     (episode_data['pattern'] == pattern)]
                
                if 'avg_queue_length' in subset.columns and not subset.empty:
                    ax = axes[i, j]
                    sns.histplot(subset['avg_queue_length'], kde=True, ax=ax)
                    ax.set_title(f"{agent} - {pattern}")
                    ax.set_xlabel('Average Queue Length')
                    ax.set_ylabel('Frequency')
                    ax.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        queue_dist_path = os.path.join(save_dir, "queue_length_distribution.png")
        plt.savefig(queue_dist_path)
        plt.close()
        visualization_paths["queue_distribution"] = queue_dist_path
        
        logger.info(f"Queue length dynamics visualizations saved to {save_dir}")
        return visualization_paths
    
    except Exception as e:
        logger.error(f"Error visualizing queue length dynamics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def visualize_spatial_traffic_flow(traffic_data, grid_size, save_dir="results"):
    """
    Create spatial visualizations of traffic flow and density.
    
    Args:
        traffic_data: Dictionary or DataFrame containing traffic density data
        grid_size: Size of the traffic grid
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of visualization paths
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have the necessary data
        if not traffic_data:
            logger.warning("No traffic data provided for spatial visualization")
            return {}
        
        visualization_paths = {}
        
        # 1. Traffic Density Heatmap
        plt.figure(figsize=(10, 8))
        
        # Create a grid for the heatmap
        density_grid = np.zeros((grid_size, grid_size))
        
        # Fill the grid with density values
        if isinstance(traffic_data, dict):
            for idx, density in traffic_data.items():
                if isinstance(idx, int):
                    row = idx // grid_size
                    col = idx % grid_size
                    if 0 <= row < grid_size and 0 <= col < grid_size:
                        density_grid[row, col] = density
        elif isinstance(traffic_data, pd.DataFrame) and 'intersection_id' in traffic_data.columns:
            for _, row in traffic_data.iterrows():
                idx = row['intersection_id']
                grid_row = idx // grid_size
                grid_col = idx % grid_size
                if 0 <= grid_row < grid_size and 0 <= grid_col < grid_size:
                    density_grid[grid_row, grid_col] = row.get('density', 0)
        
        # Plot heatmap
        sns.heatmap(density_grid, annot=True, cmap="YlOrRd", fmt=".2f", 
                   cbar_kws={'label': 'Traffic Density'})
        plt.title('Spatial Traffic Density Distribution')
        plt.xlabel('East-West Position')
        plt.ylabel('North-South Position')
        
        # Save figure
        density_heatmap_path = os.path.join(save_dir, "traffic_density_heatmap.png")
        plt.savefig(density_heatmap_path)
        plt.close()
        visualization_paths["density_heatmap"] = density_heatmap_path
        
        # 2. Flow Direction Visualization
        plt.figure(figsize=(12, 10))
        
        # Create grids for NS and EW flow
        ns_flow = np.zeros((grid_size, grid_size))
        ew_flow = np.zeros((grid_size, grid_size))
        
        # Fill the grids with flow values
        if isinstance(traffic_data, dict):
            for idx, data in traffic_data.items():
                if isinstance(idx, int):
                    row = idx // grid_size
                    col = idx % grid_size
                    if 0 <= row < grid_size and 0 <= col < grid_size:
                        if isinstance(data, dict):
                            ns_flow[row, col] = data.get('ns_flow', 0)
                            ew_flow[row, col] = data.get('ew_flow', 0)
        elif isinstance(traffic_data, pd.DataFrame) and 'intersection_id' in traffic_data.columns:
            for _, row in traffic_data.iterrows():
                idx = row['intersection_id']
                grid_row = idx // grid_size
                grid_col = idx % grid_size
                if 0 <= grid_row < grid_size and 0 <= grid_col < grid_size:
                    ns_flow[grid_row, grid_col] = row.get('ns_flow', 0)
                    ew_flow[grid_row, grid_col] = row.get('ew_flow', 0)
        
        # Create a grid of coordinates
        y, x = np.mgrid[0:grid_size, 0:grid_size]
        
        # Plot flow arrows
        plt.quiver(x, y, ew_flow, ns_flow, scale=50, width=0.002, 
                  color='blue', alpha=0.7)
        
        # Add background heatmap for total flow
        total_flow = ns_flow + ew_flow
        plt.imshow(total_flow, cmap='YlOrRd', alpha=0.3, origin='lower')
        plt.colorbar(label='Total Flow Volume')
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Set labels and title
        plt.title('Traffic Flow Direction and Volume')
        plt.xlabel('East-West Position')
        plt.ylabel('North-South Position')
        
        # Adjust axes
        plt.xlim(-0.5, grid_size-0.5)
        plt.ylim(-0.5, grid_size-0.5)
        
        # Save figure
        flow_viz_path = os.path.join(save_dir, "traffic_flow_visualization.png")
        plt.savefig(flow_viz_path)
        plt.close()
        visualization_paths["flow_visualization"] = flow_viz_path
        
        logger.info(f"Spatial traffic flow visualizations saved to {save_dir}")
        return visualization_paths
    
    except Exception as e:
        logger.error(f"Error visualizing spatial traffic flow: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def visualize_decision_timing(episode_data, save_dir="results"):
    """
    Analyze and visualize decision timing patterns.
    
    Args:
        episode_data: DataFrame containing episode data with action timing
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of visualization paths
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have the necessary data
        if not isinstance(episode_data, pd.DataFrame) or episode_data.empty:
            logger.warning("No episode data provided for decision timing visualization")
            return {}
        
        visualization_paths = {}
        
        # 1. Signal Change Timing Histogram
        if 'time_between_changes' in episode_data.columns:
            plt.figure(figsize=(12, 8))
            
            # Group by agent
            agents = episode_data['agent'].unique()
            
            for agent in agents:
                agent_data = episode_data[episode_data['agent'] == agent]
                
                # Create histogram of time between signal changes
                sns.histplot(agent_data['time_between_changes'], kde=True, 
                           label=agent, alpha=0.6)
            
            plt.xlabel('Time Between Signal Changes (seconds)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Time Between Signal Changes by Agent')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            timing_hist_path = os.path.join(save_dir, "signal_change_timing_histogram.png")
            plt.savefig(timing_hist_path)
            plt.close()
            visualization_paths["timing_histogram"] = timing_hist_path
        
        # 2. Density Change vs. Signal Timing Correlation
        if all(col in episode_data.columns for col in ['density_change', 'time_to_next_change']):
            plt.figure(figsize=(12, 8))
            
            # Group by agent
            agents = episode_data['agent'].unique()
            
            for agent in agents:
                agent_data = episode_data[episode_data['agent'] == agent]
                
                # Create scatter plot
                plt.scatter(
                    agent_data['density_change'], 
                    agent_data['time_to_next_change'],
                    alpha=0.6,
                    label=agent
                )
            
            plt.xlabel('Change in Traffic Density')
            plt.ylabel('Time to Next Signal Change (seconds)')
            plt.title('Correlation Between Traffic Density Changes and Signal Timing')
            plt.legend()
            plt.grid(True)
            
            # Add trend line
            if len(episode_data) > 1:
                x = episode_data['density_change']
                y = episode_data['time_to_next_change']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
                
                # Calculate correlation
                corr = np.corrcoef(x, y)[0, 1]
                plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), 
                           xycoords='axes fraction', fontsize=12)
            
            # Save figure
            density_timing_path = os.path.join(save_dir, "density_signal_timing_correlation.png")
            plt.savefig(density_timing_path)
            plt.close()
            visualization_paths["density_timing_correlation"] = density_timing_path
        
        # 3. Signal Change Patterns by Traffic Pattern
        if all(col in episode_data.columns for col in ['pattern', 'signal_changes_per_minute']):
            plt.figure(figsize=(14, 8))
            
            # Create grouped bar chart
            sns.barplot(x='agent', y='signal_changes_per_minute', hue='pattern', 
                       data=episode_data)
            
            plt.xlabel('Agent')
            plt.ylabel('Signal Changes per Minute')
            plt.title('Signal Change Frequency by Agent and Traffic Pattern')
            plt.legend(title='Traffic Pattern')
            plt.grid(True, axis='y')
            
            # Save figure
            change_pattern_path = os.path.join(save_dir, "signal_change_patterns.png")
            plt.savefig(change_pattern_path)
            plt.close()
            visualization_paths["change_patterns"] = change_pattern_path
        
        logger.info(f"Decision timing visualizations saved to {save_dir}")
        return visualization_paths
    
    except Exception as e:
        logger.error(f"Error visualizing decision timing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


# Define a consistent style for all plots
def set_plot_style():
    """Set a consistent, modern style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Custom style settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Modern, minimal grid
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # Improved figure aesthetics
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

def visualize_learning_progress(q_values_history, states_history, save_dir="results"):
    """
    Visualize the learning progress of the agent.
    
    Args:
        q_values_history: List of Q-value arrays at different training stages
        states_history: List of state arrays at different training stages
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of visualization paths
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have the necessary data
        if not q_values_history or not states_history:
            logger.warning("No Q-values or states history provided for learning progress visualization")
            return {}
        
        # Set consistent plot style
        set_plot_style()
        
        visualization_paths = {}
        
        # Create a figure for Q-value visualization
        # We'll create a 2x2 grid of plots:
        # - Top row: Q-values for NS Green action (0)
        # - Bottom row: Q-values for EW Green action (1)
        # - Left column: Current light state is NS Green (0)
        # - Right column: Current light state is EW Green (1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get the latest stage
        latest_q_values = q_values_history[-1]
        
        # Plot titles and labels
        light_state_labels = ["Current: NS Green", "Current: EW Green"]
        action_labels = ["Action: Keep/Set NS Green", "Action: Keep/Set EW Green"]
        
        # Use a consistent colormap
        cmap = 'viridis'
        
        # Create heatmaps for each combination of current light state and action
        for light_idx, light_state in enumerate(light_state_labels):
            for action_idx, action_label in enumerate(action_labels):
                ax = axes[action_idx, light_idx]
                
                # Get Q-values for this light state and action
                q_values = latest_q_values[light_idx][:, :, action_idx]
                
                # Create heatmap with improved aesthetics
                im = ax.imshow(q_values, cmap=cmap, origin='lower', 
                              extent=[0, 1, 0, 1], aspect='auto')
                
                # Add colorbar with consistent styling
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Q-Value', fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                
                # Add labels and title with consistent styling
                ax.set_xlabel('East-West Density')
                ax.set_ylabel('North-South Density')
                ax.set_title(f"{light_state}\n{action_label}", fontweight='bold')
                
                # Add subtle grid
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # Add decision boundary
                # Find where action 0 has higher Q-value than action 1 (or vice versa)
                if action_idx == 0:
                    # For NS Green action, highlight where it's the better action
                    other_q_values = latest_q_values[light_idx][:, :, 1]
                    better_action = q_values > other_q_values
                    # Draw contour around regions where this action is better
                    ax.contour(np.linspace(0, 1, q_values.shape[1]), 
                              np.linspace(0, 1, q_values.shape[0]),
                              better_action, levels=[0.5], colors='white', linewidths=2)
        
        # Add a main title to the figure
        fig.suptitle('Q-Value Analysis by Light State and Action', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save figure with high quality
        q_surface_path = os.path.join(save_dir, "q_value_surface_plots.png")
        plt.savefig(q_surface_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths["q_surface"] = q_surface_path
        
        # Create a second visualization: Q-value difference plot
        # This shows which action is preferred in which state
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Use a consistent diverging colormap
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        
        for light_idx, light_state in enumerate(light_state_labels):
            ax = axes[light_idx]
            
            # Calculate Q-value difference (Action 0 - Action 1)
            # Positive values mean NS Green is preferred, negative mean EW Green is preferred
            q_diff = latest_q_values[light_idx][:, :, 0] - latest_q_values[light_idx][:, :, 1]
            
            # Find max absolute value for symmetric color scaling
            max_abs_diff = np.max(np.abs(q_diff))
            
            # Create heatmap with improved aesthetics
            im = ax.imshow(q_diff, cmap=cmap, origin='lower', 
                          extent=[0, 1, 0, 1], aspect='auto',
                          vmin=-max_abs_diff, vmax=max_abs_diff)
            
            # Add colorbar with consistent styling
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Q-Value Difference\n(NS Green - EW Green)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            
            # Add labels and title with consistent styling
            ax.set_xlabel('East-West Density')
            ax.set_ylabel('North-South Density')
            ax.set_title(f"Action Preference with {light_state}", fontweight='bold')
            
            # Add subtle grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add decision boundary at Q-diff = 0
            ax.contour(np.linspace(0, 1, q_diff.shape[1]), 
                      np.linspace(0, 1, q_diff.shape[0]),
                      q_diff, levels=[0], colors='black', linewidths=2)
            
            # Add diagonal line (where NS density = EW density)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            # Annotate regions with improved styling
            ax.text(0.25, 0.75, "NS Green\nPreferred", ha='center', va='center', 
                   color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
            ax.text(0.75, 0.25, "EW Green\nPreferred", ha='center', va='center', 
                   color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Add a main title to the figure
        fig.suptitle('Action Preference Analysis by Light State', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save figure with high quality
        q_diff_path = os.path.join(save_dir, "q_value_difference_plots.png")
        plt.savefig(q_diff_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths["q_diff"] = q_diff_path
        
        # 2. State Representation Visualization using t-SNE or PCA
        if states_history and len(states_history) > 0:
            # Select states from different stages
            n_stages = min(len(states_history), 3)
            selected_states = []
            stage_labels = []
            
            for i in range(n_stages):
                stage_idx = i * (len(states_history) - 1) // (n_stages - 1) if n_stages > 1 else 0
                states = states_history[stage_idx]
                
                # Take a sample of states if there are too many
                max_samples = 1000
                if len(states) > max_samples:
                    indices = np.random.choice(len(states), max_samples, replace=False)
                    states_sample = states[indices]
                else:
                    states_sample = states
                
                selected_states.append(states_sample)
                stage_labels.extend([f"Stage {stage_idx}"] * len(states_sample))
            
            # Combine states from all stages
            combined_states = np.vstack(selected_states)
            
            # Reshape states if they have more than 2 dimensions
            if combined_states.ndim > 2:
                # Flatten all dimensions except the first one (samples)
                orig_shape = combined_states.shape
                combined_states = combined_states.reshape(orig_shape[0], -1)
                logger.info(f"Reshaped states from {orig_shape} to {combined_states.shape} for PCA")
            
            # Apply dimensionality reduction
            plt.figure(figsize=(12, 10))
            
            # PCA visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined_states)
            
            plt.subplot(1, 2, 1)
            for i in range(n_stages):
                start_idx = sum(len(s) for s in selected_states[:i])
                end_idx = start_idx + len(selected_states[i])
                plt.scatter(
                    pca_result[start_idx:end_idx, 0],
                    pca_result[start_idx:end_idx, 1],
                    alpha=0.6,
                    label=f"Stage {i * (len(states_history) - 1) // (n_stages - 1) if n_stages > 1 else 0}"
                )
            
            plt.title('PCA of State Representations')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            
            # t-SNE visualization
            try:
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(combined_states)
                
                plt.subplot(1, 2, 2)
                for i in range(n_stages):
                    start_idx = sum(len(s) for s in selected_states[:i])
                    end_idx = start_idx + len(selected_states[i])
                    plt.scatter(
                        tsne_result[start_idx:end_idx, 0],
                        tsne_result[start_idx:end_idx, 1],
                        alpha=0.6,
                        label=f"Stage {i * (len(states_history) - 1) // (n_stages - 1) if n_stages > 1 else 0}"
                    )
                
                plt.title('t-SNE of State Representations')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.legend()
            except Exception as e:
                logger.warning(f"t-SNE visualization failed: {e}")
            
            plt.tight_layout()
            
            # Save figure
            state_viz_path = os.path.join(save_dir, "state_representation_visualization.png")
            plt.savefig(state_viz_path)
            plt.close()
            visualization_paths["state_visualization"] = state_viz_path
        
        logger.info(f"Learning progress visualizations saved to {save_dir}")
        return visualization_paths
    
    except Exception as e:
        logger.error(f"Error visualizing learning progress: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def visualize_failure_modes(episode_data, save_dir="results"):
    """
    Visualize and analyze failure modes of the agent.
    
    Args:
        episode_data: DataFrame containing episode data with performance metrics
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of visualization paths
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have the necessary data
        if not isinstance(episode_data, pd.DataFrame) or episode_data.empty:
            logger.warning("No episode data provided for failure mode visualization")
            return {}
        
        visualization_paths = {}
        
        # 1. Performance Distribution and Failure Threshold
        if 'reward' in episode_data.columns:
            plt.figure(figsize=(12, 8))
            
            # Group by agent
            agents = episode_data['agent'].unique()
            
            for agent in agents:
                agent_data = episode_data[episode_data['agent'] == agent]
                
                # Create histogram of rewards
                sns.histplot(agent_data['reward'], kde=True, label=agent, alpha=0.6)
            
            # Calculate failure threshold (e.g., bottom 10% of rewards)
            all_rewards = episode_data['reward']
            failure_threshold = np.percentile(all_rewards, 10)
            
            # Add vertical line for failure threshold
            plt.axvline(x=failure_threshold, color='r', linestyle='--', 
                      label=f'Failure Threshold ({failure_threshold:.2f})')
            
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution and Failure Threshold')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            failure_dist_path = os.path.join(save_dir, "failure_threshold_distribution.png")
            plt.savefig(failure_dist_path)
            plt.close()
            visualization_paths["failure_distribution"] = failure_dist_path
            
            # 2. Failure Mode Analysis - State Comparison
            plt.figure(figsize=(14, 10))
            
            # Identify successful and failed episodes
            failed_episodes = episode_data[episode_data['reward'] < failure_threshold]
            successful_episodes = episode_data[episode_data['reward'] > np.percentile(all_rewards, 90)]
            
            # Compare metrics between successful and failed episodes
            metrics = ['avg_density', 'avg_waiting_time', 'avg_throughput', 'avg_queue_length']
            available_metrics = [m for m in metrics if m in episode_data.columns]
            
            if available_metrics:
                # Create subplot for each metric
                for i, metric in enumerate(available_metrics):
                    plt.subplot(2, 2, i+1)
                    
                    # Calculate average values for failed and successful episodes
                    failed_avg = failed_episodes.groupby('agent')[metric].mean()
                    success_avg = successful_episodes.groupby('agent')[metric].mean()
                    
                    # Create DataFrame for plotting
                    compare_df = pd.DataFrame({
                        'Failed': failed_avg,
                        'Successful': success_avg
                    })
                    
                    # Plot grouped bar chart
                    compare_df.plot(kind='bar', ax=plt.gca())
                    plt.title(f'{metric} Comparison')
                    plt.ylabel(metric)
                    plt.grid(True, axis='y')
                
                plt.tight_layout()
                plt.suptitle('Comparison Between Successful and Failed Episodes', fontsize=16, y=1.02)
                
                # Save figure
                failure_compare_path = os.path.join(save_dir, "success_failure_comparison.png")
                plt.savefig(failure_compare_path)
                plt.close()
                visualization_paths["failure_comparison"] = failure_compare_path
            
            # 3. Failure Correlation with Traffic Patterns
            if 'pattern' in episode_data.columns:
                plt.figure(figsize=(12, 8))
                
                # Calculate failure rate by pattern and agent
                failure_rates = []
                
                for agent in agents:
                    for pattern in episode_data['pattern'].unique():
                        subset = episode_data[(episode_data['agent'] == agent) & 
                                            (episode_data['pattern'] == pattern)]
                        
                        if not subset.empty:
                            total_episodes = len(subset)
                            failed_episodes = sum(subset['reward'] < failure_threshold)
                            failure_rate = failed_episodes / total_episodes * 100
                            
                            failure_rates.append({
                                'agent': agent,
                                'pattern': pattern,
                                'failure_rate': failure_rate
                            })
                
                if failure_rates:
                    failure_df = pd.DataFrame(failure_rates)
                    
                    # Create heatmap
                    failure_pivot = failure_df.pivot(index='agent', columns='pattern', values='failure_rate')
                    sns.heatmap(failure_pivot, annot=True, cmap='YlOrRd', fmt='.1f')
                    plt.title('Failure Rate (%) by Agent and Traffic Pattern')
                    
                    # Save figure
                    failure_pattern_path = os.path.join(save_dir, "failure_rate_by_pattern.png")
                    plt.savefig(failure_pattern_path)
                    plt.close()
                    visualization_paths["failure_pattern"] = failure_pattern_path
        
        logger.info(f"Failure mode visualizations saved to {save_dir}")
        return visualization_paths
    
    except Exception as e:
        logger.error(f"Error visualizing failure modes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def visualize_fairness_metrics(episode_data, save_dir="results"):
    """
    Visualize fairness metrics across different directions and agents.
    
    Args:
        episode_data: DataFrame containing episode data with directional metrics
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of visualization paths
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have the necessary data
        if not isinstance(episode_data, pd.DataFrame) or episode_data.empty:
            logger.warning("No episode data provided for fairness metrics visualization")
            return {}
        
        visualization_paths = {}
        
        # 1. Waiting Time Distribution by Direction
        if all(col in episode_data.columns for col in ['ns_waiting_time', 'ew_waiting_time']):
            plt.figure(figsize=(14, 8))
            
            # Group by agent
            agents = episode_data['agent'].unique()
            
            # Create subplot for each agent
            fig, axes = plt.subplots(1, len(agents), figsize=(6*len(agents), 6), sharey=True)
            if len(agents) == 1:
                axes = [axes]  # Make it iterable if only one agent
            
            for i, agent in enumerate(agents):
                agent_data = episode_data[episode_data['agent'] == agent]
                
                # Create box plot for waiting times by direction
                waiting_data = pd.DataFrame({
                    'North-South': agent_data['ns_waiting_time'],
                    'East-West': agent_data['ew_waiting_time']
                })
                
                waiting_data.plot.box(ax=axes[i])
                axes[i].set_title(agent)
                axes[i].set_ylabel('Waiting Time')
                axes[i].grid(True, axis='y')
            
            plt.tight_layout()
            plt.suptitle('Waiting Time Distribution by Direction', fontsize=16, y=1.02)
            
            # Save figure
            waiting_dist_path = os.path.join(save_dir, "waiting_time_distribution_by_direction.png")
            plt.savefig(waiting_dist_path)
            plt.close()
            visualization_paths["waiting_distribution"] = waiting_dist_path
        
        # 2. Gini Coefficient for Waiting Time Inequality
        if all(col in episode_data.columns for col in ['ns_waiting_time', 'ew_waiting_time']):
            plt.figure(figsize=(12, 8))
            
            # Calculate Gini coefficient for each episode
            def gini(x):
                # Mean absolute difference
                mad = np.abs(np.subtract.outer(x, x)).mean()
                # Relative mean absolute difference
                rmad = mad / np.mean(x)
                # Gini coefficient
                return rmad / 2
            
            # Calculate Gini coefficients for each agent
            gini_data = []
            
            for agent in episode_data['agent'].unique():
                agent_data = episode_data[episode_data['agent'] == agent]
                
                for pattern in agent_data['pattern'].unique():
                    pattern_data = agent_data[agent_data['pattern'] == pattern]
                    
                    for _, row in pattern_data.iterrows():
                        # Calculate Gini coefficient for this episode
                        waiting_times = [row['ns_waiting_time'], row['ew_waiting_time']]
                        gini_coef = gini(waiting_times)
                        
                        gini_data.append({
                            'agent': agent,
                            'pattern': pattern,
                            'gini_coefficient': gini_coef
                        })
            
            if gini_data:
                gini_df = pd.DataFrame(gini_data)
                
                # Create box plot of Gini coefficients by agent
                sns.boxplot(x='agent', y='gini_coefficient', hue='pattern', data=gini_df)
                plt.xlabel('Agent')
                plt.ylabel('Gini Coefficient')
                plt.title('Fairness (Gini Coefficient) by Agent and Traffic Pattern')
                plt.legend(title='Traffic Pattern')
                plt.grid(True, axis='y')
                
                # Save figure
                gini_path = os.path.join(save_dir, "fairness_gini_coefficient.png")
                plt.savefig(gini_path)
                plt.close()
                visualization_paths["gini_coefficient"] = gini_path
        
        # 3. Direction Bias Analysis
        if all(col in episode_data.columns for col in ['ns_waiting_time', 'ew_waiting_time']):
            plt.figure(figsize=(12, 8))
            
            # Calculate direction bias (NS waiting time - EW waiting time)
            bias_data = []
            
            for agent in episode_data['agent'].unique():
                agent_data = episode_data[episode_data['agent'] == agent]
                
                for pattern in agent_data['pattern'].unique():
                    pattern_data = agent_data[agent_data['pattern'] == pattern]
                    
                    # Calculate average bias for this agent and pattern
                    avg_ns_waiting = pattern_data['ns_waiting_time'].mean()
                    avg_ew_waiting = pattern_data['ew_waiting_time'].mean()
                    bias = avg_ns_waiting - avg_ew_waiting
                    
                    bias_data.append({
                        'agent': agent,
                        'pattern': pattern,
                        'direction_bias': bias
                    })
            
            if bias_data:
                bias_df = pd.DataFrame(bias_data)
                
                # Create bar chart of direction bias
                sns.barplot(x='agent', y='direction_bias', hue='pattern', data=bias_df)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.xlabel('Agent')
                plt.ylabel('Direction Bias (NS - EW Waiting Time)')
                plt.title('Direction Bias by Agent and Traffic Pattern')
                plt.legend(title='Traffic Pattern')
                plt.grid(True, axis='y')
                
                # Save figure
                bias_path = os.path.join(save_dir, "direction_bias.png")
                plt.savefig(bias_path)
                plt.close()
                visualization_paths["direction_bias"] = bias_path
        
        logger.info(f"Fairness metrics visualizations saved to {save_dir}")
        return visualization_paths
    
    except Exception as e:
        logger.error(f"Error visualizing fairness metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def create_comprehensive_report(config, model_path, benchmark_results, output_dir, env_type="grid"):
    """
    Create a comprehensive analysis report.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model
        benchmark_results: Benchmark results dictionary
        output_dir: Directory to save the report
        env_type: Type of environment ('grid' or 'roundabout')
        
    Returns:
        Dictionary with report information
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize report data
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "env_type": env_type,
            "config": config,
            "benchmark_results": benchmark_results
        }
        
        # Save report data
        with open(os.path.join(output_dir, "report_data.json"), 'w') as f:
            # Convert numpy values to native types
            json.dump(report, f, indent=4)
        
        logger.info(f"Comprehensive report created in {output_dir}")
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to create comprehensive report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
