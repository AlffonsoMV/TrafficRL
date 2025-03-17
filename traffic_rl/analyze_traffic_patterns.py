"""
Traffic Pattern Analysis Script
==============================
Analyze how agent performance varies across different traffic patterns.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import argparse
from matplotlib.gridspec import GridSpec

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TrafficRL.PatternAnalysis')

def load_experiment_data(experiment_dir):
    """
    Load data from different traffic pattern experiments.
    
    Args:
        experiment_dir: Base directory containing experiment subdirectories
        
    Returns:
        Dictionary mapping pattern names to metrics
    """
    patterns = {}
    
    # Look for pattern-specific subdirectories
    if os.path.exists(experiment_dir):
        # Try to find pattern directories like "uniform", "rush_hour", etc.
        potential_patterns = [d for d in os.listdir(experiment_dir) 
                             if os.path.isdir(os.path.join(experiment_dir, d))]
        
        for pattern in potential_patterns:
            metrics_path = os.path.join(experiment_dir, pattern, "training_metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        patterns[pattern] = json.load(f)
                        logger.info(f"Loaded data for pattern: {pattern}")
                except Exception as e:
                    logger.error(f"Error loading data for pattern {pattern}: {e}")
    
    if not patterns:
        logger.warning(f"No pattern data found in {experiment_dir}")
        # Try to load any metrics file from the base directory
        metrics_path = os.path.join(experiment_dir, "training_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    patterns["default"] = json.load(f)
                    logger.info("Loaded default metrics file")
            except Exception as e:
                logger.error(f"Error loading default metrics: {e}")
    
    return patterns

def detect_pattern_transitions(metrics):
    """
    Try to detect traffic pattern transitions from the metrics.
    
    Args:
        metrics: The metrics data dictionary
        
    Returns:
        List of (episode, pattern) tuples marking transitions
    """
    transitions = []
    
    # Check if we have any rewards data
    if 'rewards' not in metrics or not metrics['rewards']:
        return transitions
    
    # Look for significant changes in reward that might indicate pattern changes
    rewards = np.array(metrics['rewards'])
    episodes = len(rewards)
    
    # Simple detection: significant drops in rolling average might be pattern changes
    if episodes > 20:
        window = 5
        rolling_avg = pd.Series(rewards).rolling(window=window).mean().iloc[window-1:].values
        diffs = np.diff(rolling_avg)
        
        # Find large negative drops (potential pattern transitions)
        threshold = -np.std(diffs) * 2  # Adjust threshold for sensitivity
        potential_transitions = np.where(diffs < threshold)[0] + window
        
        # Filter to transitions that are at least 20 episodes apart
        filtered_transitions = []
        for i, ep in enumerate(potential_transitions):
            if i == 0 or ep - filtered_transitions[-1] >= 20:
                filtered_transitions.append(ep)
        
        # Add to transitions list with "Unknown" pattern label
        for ep in filtered_transitions:
            if 1 <= ep < episodes:  # Ensure episode is in valid range
                transitions.append((int(ep), "Unknown"))
    
    # Add the first episode as the initial pattern
    transitions.insert(0, (1, "Initial"))
    
    return transitions

def plot_pattern_comparison(patterns, save_path=None):
    """
    Create a comparison plot of agent performance across different traffic patterns.
    
    Args:
        patterns: Dictionary mapping pattern names to metrics
        save_path: Path to save the figure (optional)
    """
    if not patterns:
        logger.error("No pattern data to compare")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    metrics_to_plot = [
        ('rewards', 'Reward', 0),
        ('waiting_times', 'Average Waiting Time', 1),
        ('throughput', 'Average Throughput', 2),
        ('loss_values', 'Training Loss', 3)
    ]
    
    for i, (metric_key, metric_label, ax_idx) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        
        for j, (pattern_name, metrics) in enumerate(patterns.items()):
            color = colors[j % len(colors)]
            
            if metric_key in metrics and metrics[metric_key]:
                data = metrics[metric_key]
                episodes = range(1, len(data) + 1)
                
                # Plot raw data with low alpha
                ax.plot(episodes, data, alpha=0.3, color=color)
                
                # Plot smoothed data for clarity
                if len(data) >= 5:
                    window = min(10, len(data) // 5)
                    smoothed = pd.Series(data).rolling(window=window).mean()
                    ax.plot(episodes, smoothed, linewidth=2, color=color, label=f"{pattern_name}")
                else:
                    ax.plot(episodes, data, linewidth=2, color=color, label=f"{pattern_name}")
        
        ax.set_title(f"{metric_label} Comparison", fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for loss values
        if metric_key == 'loss_values':
            ax.set_yscale('log')
    
    # Add a common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(patterns), 
                bbox_to_anchor=(0.5, 0.98), fontsize=12)
    
    plt.suptitle('Performance Comparison Across Traffic Patterns', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved pattern comparison to {save_path}")
    
    plt.show()

def analyze_single_pattern_run(metrics, save_dir=None):
    """
    Analyze a single run that may contain multiple traffic patterns.
    
    Args:
        metrics: Metrics dictionary from a single run
        save_dir: Directory to save output figures
    """
    # Detect potential pattern transitions
    transitions = detect_pattern_transitions(metrics)
    
    if len(transitions) <= 1:
        logger.info("No significant pattern transitions detected")
        return
    
    # Plot rewards with marked transitions
    plt.figure(figsize=(14, 7))
    
    # Plot rewards
    rewards = metrics.get('rewards', [])
    episodes = range(1, len(rewards) + 1)
    plt.plot(episodes, rewards, alpha=0.4, color='blue')
    
    # Add smoothed curve
    if len(rewards) >= 5:
        window = min(10, len(rewards) // 5)
        smoothed = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(episodes, smoothed, linewidth=2, color='red', 
                 label=f"{window}-Episode Moving Average")
    
    # Add vertical lines for transitions
    prev_ep = 1
    for i, (ep, pattern) in enumerate(transitions[1:], 1):
        plt.axvline(x=ep, color='green', linestyle='--', alpha=0.7)
        
        # Add text annotation for the pattern
        mid_ep = (prev_ep + ep) // 2
        y_pos = max(rewards) * 0.9
        plt.text(mid_ep, y_pos, transitions[i-1][1], 
                 horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
        
        prev_ep = ep
    
    # Add text for the last segment
    if prev_ep < len(rewards):
        mid_ep = (prev_ep + len(rewards)) // 2
        y_pos = max(rewards) * 0.9
        plt.text(mid_ep, y_pos, transitions[-1][1], 
                 horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('Reward Timeline with Potential Pattern Transitions', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_dir:
        save_path = os.path.join(save_dir, "pattern_transitions.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved transitions plot to {save_path}")
    
    plt.show()
    
    # Create segments for each detected pattern
    segments = []
    for i in range(len(transitions)):
        start_ep = transitions[i][0]
        end_ep = transitions[i+1][0] - 1 if i < len(transitions) - 1 else len(rewards)
        pattern = transitions[i][1]
        
        segments.append({
            'pattern': pattern,
            'start': start_ep,
            'end': end_ep,
            'rewards': rewards[start_ep-1:end_ep],
            'avg_reward': np.mean(rewards[start_ep-1:end_ep])
        })
    
    # Compare metrics between segments
    plt.figure(figsize=(10, 6))
    
    # Plot average reward for each segment
    patterns = [s['pattern'] for s in segments]
    avg_rewards = [s['avg_reward'] for s in segments]
    
    plt.bar(patterns, avg_rewards, color='skyblue')
    plt.title('Average Reward by Detected Traffic Pattern', fontsize=16)
    plt.xlabel('Pattern', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    for i, v in enumerate(avg_rewards):
        plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "pattern_comparison.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved pattern comparison to {save_path}")
    
    plt.show()

def compare_pattern_statistics(patterns, save_path=None):
    """
    Create a statistical comparison of performance metrics across patterns.
    
    Args:
        patterns: Dictionary mapping pattern names to metrics
        save_path: Path to save the figure (optional)
    """
    if not patterns:
        logger.error("No pattern data to compare")
        return
    
    # Extract key statistics for each pattern
    stats = {}
    for pattern, metrics in patterns.items():
        pattern_stats = {}
        
        # Get rewards
        if 'rewards' in metrics and metrics['rewards']:
            rewards = metrics['rewards']
            pattern_stats['avg_reward'] = np.mean(rewards)
            pattern_stats['max_reward'] = np.max(rewards)
            pattern_stats['min_reward'] = np.min(rewards)
            pattern_stats['std_reward'] = np.std(rewards)
            
            # Get final performance (last 10 episodes or all if fewer)
            final_window = min(10, len(rewards))
            if final_window > 0:
                pattern_stats['final_reward'] = np.mean(rewards[-final_window:])
        
        # Get waiting times
        if 'waiting_times' in metrics and metrics['waiting_times']:
            waiting = metrics['waiting_times']
            pattern_stats['avg_waiting'] = np.mean(waiting)
            
            # Get final waiting time
            final_window = min(10, len(waiting))
            if final_window > 0:
                pattern_stats['final_waiting'] = np.mean(waiting[-final_window:])
        
        # Get throughput
        if 'throughput' in metrics and metrics['throughput']:
            throughput = metrics['throughput']
            pattern_stats['avg_throughput'] = np.mean(throughput)
            
            # Get final throughput
            final_window = min(10, len(throughput))
            if final_window > 0:
                pattern_stats['final_throughput'] = np.mean(throughput[-final_window:])
        
        stats[pattern] = pattern_stats
    
    # Convert to dataframe for easier plotting
    df = pd.DataFrame(stats).T
    
    # Create a visual dashboard comparing stats
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot average reward
    ax1 = fig.add_subplot(gs[0, 0])
    if 'avg_reward' in df.columns:
        df['avg_reward'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Average Reward', fontsize=14)
        ax1.set_ylabel('Reward')
        ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot final reward
    ax2 = fig.add_subplot(gs[0, 1])
    if 'final_reward' in df.columns:
        df['final_reward'].plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Final Reward (Last 10 Episodes)', fontsize=14)
        ax2.grid(True, axis='y', alpha=0.3)
    
    # Plot reward variability
    ax3 = fig.add_subplot(gs[0, 2])
    if 'std_reward' in df.columns:
        df['std_reward'].plot(kind='bar', ax=ax3, color='salmon')
        ax3.set_title('Reward Variability (Std Dev)', fontsize=14)
        ax3.grid(True, axis='y', alpha=0.3)
    
    # Plot waiting time
    ax4 = fig.add_subplot(gs[1, 0])
    if 'avg_waiting' in df.columns:
        df['avg_waiting'].plot(kind='bar', ax=ax4, color='orchid')
        ax4.set_title('Average Waiting Time', fontsize=14)
        ax4.set_ylabel('Waiting Time')
        ax4.grid(True, axis='y', alpha=0.3)
    
    # Plot final waiting time
    ax5 = fig.add_subplot(gs[1, 1])
    if 'final_waiting' in df.columns:
        df['final_waiting'].plot(kind='bar', ax=ax5, color='lightblue')
        ax5.set_title('Final Waiting Time', fontsize=14)
        ax5.grid(True, axis='y', alpha=0.3)
    
    # Plot final throughput
    ax6 = fig.add_subplot(gs[1, 2])
    if 'final_throughput' in df.columns:
        df['final_throughput'].plot(kind='bar', ax=ax6, color='gold')
        ax6.set_title('Final Throughput', fontsize=14)
        ax6.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Performance Statistics Across Traffic Patterns', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved statistics comparison to {save_path}")
    
    plt.show()
    
    # Print statistics table
    print("\n" + "="*50)
    print(" TRAFFIC PATTERN COMPARISON ")
    print("="*50 + "\n")
    
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n" + "="*50)
    
    return df

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze Traffic RL performance across patterns")
    parser.add_argument('--dir', type=str, default='results',
                        help='Base directory containing experiment results')
    parser.add_argument('--output', type=str, default='results/pattern_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--single', type=str, default=None,
                        help='Path to metrics file from a single run with multiple patterns')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Analyze a single run with multiple patterns if specified
    if args.single and os.path.exists(args.single):
        try:
            with open(args.single, 'r') as f:
                metrics = json.load(f)
                analyze_single_pattern_run(metrics, save_dir=args.output)
        except Exception as e:
            logger.error(f"Error analyzing single run: {e}")
    
    # Load pattern data from experiment directory
    patterns = load_experiment_data(args.dir)
    
    if patterns:
        # Create comparison plots
        plot_pattern_comparison(patterns, save_path=os.path.join(args.output, 'pattern_comparison.png'))
        
        # Compare pattern statistics
        compare_pattern_statistics(patterns, save_path=os.path.join(args.output, 'pattern_statistics.png'))
        
        logger.info(f"Analysis complete. Results saved to {args.output}")
    else:
        logger.warning("No pattern data found for analysis.")

if __name__ == "__main__":
    main() 