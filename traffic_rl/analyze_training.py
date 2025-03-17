"""
Training Analysis Script
=======================
Analyze and visualize the training results from RL experiments.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TrafficRL.Analysis')

def load_metrics(metrics_file):
    """
    Load the training metrics from a JSON file.
    
    Args:
        metrics_file: Path to the metrics JSON file
        
    Returns:
        Dictionary of metrics or None if file couldn't be loaded
    """
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Error loading metrics from {metrics_file}: {e}")
        return None
        
def plot_rewards(metrics, save_path=None):
    """
    Plot the episode rewards and moving average.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the figure (optional)
    """
    if 'rewards' not in metrics:
        logger.warning("Rewards data not found in metrics")
        return
        
    rewards = metrics['rewards']
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot individual episode rewards
    episodes = range(1, len(rewards) + 1)
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average if we have enough data
    if len(rewards) >= 5:
        window_size = min(10, len(rewards) // 2)
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        plt.plot(episodes, moving_avg, color='red', linewidth=2, 
                 label=f'{window_size}-Episode Moving Average')
    
    # If we have evaluation rewards, plot them too
    if 'eval_rewards' in metrics and len(metrics['eval_rewards']) > 0:
        eval_episodes = [metrics['eval_frequency'] * (i+1) for i in range(len(metrics['eval_rewards']))]
        plt.plot(eval_episodes, metrics['eval_rewards'], 'go--', linewidth=2, markersize=8,
                 label='Evaluation Reward')
    
    plt.title('Training Rewards Over Time', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved rewards plot to {save_path}")
    
    plt.show()

def plot_training_metrics(metrics, save_path=None):
    """
    Create a comprehensive dashboard of training metrics.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the figure (optional)
    """
    # Check for required metrics
    required = ['rewards', 'epsilon_values', 'waiting_times', 'throughput']
    missing = [m for m in required if m not in metrics or not metrics[m]]
    
    if missing:
        logger.warning(f"Missing required metrics: {missing}")
        return
    
    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # 1. Rewards plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = metrics['rewards']
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, alpha=0.4, color='blue', label='Episode Reward')
    
    # Add moving average
    if len(rewards) >= 5:
        window_size = min(10, len(rewards) // 2)
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        ax1.plot(episodes, moving_avg, color='red', linewidth=2, 
                 label=f'{window_size}-Episode Moving Average')
    
    ax1.set_title('Training Rewards', fontsize=14)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Exploration (epsilon) plot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    epsilon_values = metrics['epsilon_values']
    ax2.plot(episodes, epsilon_values, color='green', linewidth=2)
    ax2.set_title('Exploration Rate (Epsilon)', fontsize=14)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True, alpha=0.3)
    
    # 3. Waiting times plot (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    waiting_times = metrics['waiting_times']
    ax3.plot(episodes, waiting_times, color='orange', linewidth=2)
    ax3.set_title('Average Waiting Time', fontsize=14)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Waiting Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Throughput plot (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    throughput = metrics['throughput']
    ax4.plot(episodes, throughput, color='purple', linewidth=2)
    ax4.set_title('Average Throughput', fontsize=14)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cars Passed Per Step')
    ax4.grid(True, alpha=0.3)
    
    # 5. Loss values plot (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    if 'loss_values' in metrics and metrics['loss_values']:
        loss_values = metrics['loss_values']
        ax5.plot(episodes, loss_values, color='brown', linewidth=2)
        ax5.set_title('Training Loss', fontsize=14)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Average Loss')
        ax5.set_yscale('log')  # Log scale for loss values
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Loss data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 6. Learning rate plot (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    if 'learning_rates' in metrics and metrics['learning_rates']:
        learning_rates = metrics['learning_rates']
        ax6.plot(episodes, learning_rates, color='blue', linewidth=2)
        ax6.set_title('Learning Rate', fontsize=14)
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Learning Rate')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Learning rate data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # Add general title and tighten layout
    plt.suptitle('Traffic RL Training Metrics Dashboard', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved dashboard to {save_path}")
    
    plt.show()

def analyze_reward_breakdown(metrics_file, save_dir=None):
    """
    Load detailed reward breakdown if available and visualize components.
    
    Args:
        metrics_file: Path to the metrics JSON file
        save_dir: Directory to save figures (optional)
    """
    # Get directory of metrics file to check for detailed logs
    metrics_dir = os.path.dirname(metrics_file)
    detailed_log = os.path.join(metrics_dir, "reward_components.csv")
    
    if not os.path.exists(detailed_log):
        logger.warning(f"Detailed reward breakdown not found at {detailed_log}")
        return
    
    try:
        # Load detailed reward components
        df = pd.read_csv(detailed_log)
        
        # Create visualization of reward components
        plt.figure(figsize=(14, 8))
        
        # Plot each component
        components = [col for col in df.columns if col != 'episode' and col != 'step' and col != 'total_reward']
        
        for component in components:
            plt.plot(df['step'], df[component], label=component)
        
        plt.plot(df['step'], df['total_reward'], 'k-', linewidth=2, label='Total Reward')
        
        plt.title('Reward Component Breakdown', fontsize=16)
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('Reward Value', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        if save_dir:
            save_path = os.path.join(save_dir, "reward_components.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            logger.info(f"Saved reward components plot to {save_path}")
        
        plt.show()
        
        # Create correlation heatmap of reward components
        plt.figure(figsize=(10, 8))
        corr = df[components + ['total_reward']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation Between Reward Components', fontsize=16)
        
        if save_dir:
            save_path = os.path.join(save_dir, "reward_correlation.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            logger.info(f"Saved reward correlation plot to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error analyzing reward breakdown: {e}")

def summarize_training(metrics):
    """
    Print a summary of the training statistics.
    
    Args:
        metrics: Dictionary of training metrics
    """
    if not metrics:
        logger.error("No metrics data to summarize")
        return
    
    print("\n" + "="*50)
    print(" TRAFFIC RL TRAINING SUMMARY ")
    print("="*50)
    
    # Calculate statistics
    num_episodes = len(metrics.get('rewards', []))
    if num_episodes == 0:
        print("No episode data found")
        return
    
    final_avg_reward = np.mean(metrics.get('rewards', [])[-10:]) if num_episodes >= 10 else np.mean(metrics.get('rewards', []))
    max_reward = np.max(metrics.get('rewards', []))
    max_reward_episode = np.argmax(metrics.get('rewards', [])) + 1
    
    training_time = metrics.get('training_time', 0)
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    # Find best evaluation score if available
    if 'eval_rewards' in metrics and metrics['eval_rewards']:
        best_eval = np.max(metrics['eval_rewards'])
        best_eval_idx = np.argmax(metrics['eval_rewards'])
        best_eval_episode = (best_eval_idx + 1) * metrics.get('eval_frequency', 1)
    else:
        best_eval = "N/A"
        best_eval_episode = "N/A"
    
    # Print summary
    print(f"\nNumber of Episodes: {num_episodes}")
    print(f"Training Time: {hours}h {minutes}m {seconds}s")
    print(f"\nFinal Average Reward (last 10 eps): {final_avg_reward:.2f}")
    print(f"Maximum Episode Reward: {max_reward:.2f} (Episode {max_reward_episode})")
    print(f"Best Evaluation Reward: {best_eval} (Episode {best_eval_episode})")
    
    # Print traffic metrics if available
    if 'waiting_times' in metrics and metrics['waiting_times']:
        avg_waiting = np.mean(metrics['waiting_times'][-10:])
        print(f"\nFinal Avg Waiting Time: {avg_waiting:.2f}")
    
    if 'throughput' in metrics and metrics['throughput']:
        avg_throughput = np.mean(metrics['throughput'][-10:])
        print(f"Final Avg Throughput: {avg_throughput:.2f} cars/step")
    
    print("\n" + "="*50)

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze Traffic RL training results")
    parser.add_argument('--metrics', type=str, default='results/training/training_metrics.json',
                        help='Path to the training metrics JSON file')
    parser.add_argument('--output', type=str, default='results/analysis',
                        help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load metrics
    metrics = load_metrics(args.metrics)
    if not metrics:
        return
    
    # Summarize training
    summarize_training(metrics)
    
    # Generate plots
    plot_rewards(metrics, save_path=os.path.join(args.output, 'rewards.png'))
    plot_training_metrics(metrics, save_path=os.path.join(args.output, 'metrics_dashboard.png'))
    
    # Analyze reward breakdown
    analyze_reward_breakdown(args.metrics, save_dir=args.output)
    
    logger.info(f"Analysis complete. Results saved to {args.output}")

if __name__ == "__main__":
    main() 