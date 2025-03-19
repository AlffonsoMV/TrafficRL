"""
Training Analysis
================
Analyze training metrics and performance.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging

logger = logging.getLogger('TrafficRL.TrainingAnalysis')

# Define colors for black background
COLORS = {
    'background': 'black',
    'text': 'white',
    'grid': '#555555',
    'line1': '#00B386',  # teal green
    'line2': '#FF5C5C',  # bright red
    'line3': '#5B9BD5',  # blue
    'line4': '#FFD166',  # gold
}

def analyze_training_metrics(training_data, output_dir=None):
    """
    Analyze training metrics and generate visualizations.
    
    Args:
        training_data: Dictionary containing training metrics
        output_dir: Optional directory to save visualizations
    """
    if not training_data:
        logger.warning("No training data to analyze")
        return
    
    # Set style for black background
    plt.rcParams.update({
        'text.color': COLORS['text'],
        'axes.labelcolor': COLORS['text'],
        'axes.edgecolor': COLORS['grid'],
        'axes.facecolor': COLORS['background'],
        'figure.facecolor': COLORS['background'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'savefig.facecolor': COLORS['background'],
        'savefig.transparent': False,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 10), facecolor=COLORS['background'])
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Plot reward over time
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = training_data.get('rewards', [])
    episodes = range(len(rewards))
    ax1.plot(episodes, rewards, color=COLORS['line1'], linewidth=2, alpha=0.9)
    
    # Add rolling average for smoother visualization
    if len(rewards) > 10:
        window_size = min(len(rewards) // 10, 20)  # Adaptive window size
        rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
        ax1.plot(episodes, rolling_avg, color='white', linewidth=1.5, alpha=0.8, 
                linestyle='--', label=f'{window_size}-Episode Avg')
    
    ax1.set_title('Training Rewards', color=COLORS['text'], fontweight='bold')
    ax1.set_xlabel('Episode', color=COLORS['text'])
    ax1.set_ylabel('Reward', color=COLORS['text'])
    ax1.tick_params(axis='both', colors=COLORS['text'])
    ax1.grid(axis='both', alpha=0.3, linestyle='--', color=COLORS['grid'])
    ax1.legend(facecolor='#333333', framealpha=0.7).get_texts()[0].set_color(COLORS['text'])
    
    # Plot loss over time
    ax2 = fig.add_subplot(gs[0, 1])
    losses = training_data.get('losses', [])
    ax2.plot(episodes, losses, color=COLORS['line2'], linewidth=2, alpha=0.9)
    
    # Add rolling average for smoother visualization
    if len(losses) > 10:
        window_size = min(len(losses) // 10, 20)  # Adaptive window size
        rolling_avg = pd.Series(losses).rolling(window=window_size).mean()
        ax2.plot(episodes, rolling_avg, color='white', linewidth=1.5, alpha=0.8, 
                linestyle='--', label=f'{window_size}-Episode Avg')
    
    ax2.set_title('Training Loss', color=COLORS['text'], fontweight='bold')
    ax2.set_xlabel('Episode', color=COLORS['text'])
    ax2.set_ylabel('Loss', color=COLORS['text'])
    ax2.tick_params(axis='both', colors=COLORS['text'])
    ax2.grid(axis='both', alpha=0.3, linestyle='--', color=COLORS['grid'])
    ax2.legend(facecolor='#333333', framealpha=0.7).get_texts()[0].set_color(COLORS['text'])
    
    # Plot epsilon over time
    ax3 = fig.add_subplot(gs[1, 0])
    epsilons = training_data.get('epsilons', [])
    ax3.plot(episodes, epsilons, color=COLORS['line3'], linewidth=2, alpha=0.9)
    ax3.set_title('Epsilon Decay', color=COLORS['text'], fontweight='bold')
    ax3.set_xlabel('Episode', color=COLORS['text'])
    ax3.set_ylabel('Epsilon', color=COLORS['text'])
    ax3.tick_params(axis='both', colors=COLORS['text'])
    ax3.grid(axis='both', alpha=0.3, linestyle='--', color=COLORS['grid'])
    
    # Plot average waiting time
    ax4 = fig.add_subplot(gs[1, 1])
    waiting_times = training_data.get('avg_waiting_times', [])
    ax4.plot(episodes, waiting_times, color=COLORS['line4'], linewidth=2, alpha=0.9)
    
    # Add rolling average for smoother visualization
    if len(waiting_times) > 10:
        window_size = min(len(waiting_times) // 10, 20)  # Adaptive window size
        rolling_avg = pd.Series(waiting_times).rolling(window=window_size).mean()
        ax4.plot(episodes, rolling_avg, color='white', linewidth=1.5, alpha=0.8, 
                linestyle='--', label=f'{window_size}-Episode Avg')
    
    ax4.set_title('Average Waiting Time', color=COLORS['text'], fontweight='bold')
    ax4.set_xlabel('Episode', color=COLORS['text'])
    ax4.set_ylabel('Time (s)', color=COLORS['text'])
    ax4.tick_params(axis='both', colors=COLORS['text'])
    ax4.grid(axis='both', alpha=0.3, linestyle='--', color=COLORS['grid'])
    ax4.legend(facecolor='#333333', framealpha=0.7).get_texts()[0].set_color(COLORS['text'])
    
    # Add a watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Training Analysis', 
        ha='right', va='bottom', 
        fontsize=8, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_analysis.png'), facecolor=COLORS['background'])
    plt.close()
    
    # Generate summary statistics
    summary = {
        'final_reward': rewards[-1] if rewards else None,
        'final_loss': losses[-1] if losses else None,
        'final_epsilon': epsilons[-1] if epsilons else None,
        'final_waiting_time': waiting_times[-1] if waiting_times else None,
        'avg_reward': np.mean(rewards) if rewards else None,
        'avg_loss': np.mean(losses) if losses else None,
        'avg_waiting_time': np.mean(waiting_times) if waiting_times else None
    }
    
    if output_dir:
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    
    return summary

def plot_training_curves(metrics_dict, output_dir=None):
    """
    Plot training curves for multiple metrics.
    
    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
        output_dir: Optional directory to save visualizations
    """
    # Set style for black background
    plt.rcParams.update({
        'text.color': COLORS['text'],
        'axes.labelcolor': COLORS['text'],
        'axes.edgecolor': COLORS['grid'],
        'axes.facecolor': COLORS['background'],
        'figure.facecolor': COLORS['background'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'savefig.facecolor': COLORS['background'],
        'savefig.transparent': False,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    plt.figure(figsize=(12, 6), facecolor=COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['background'])
    
    # Create color cycle for different metrics
    colors = [COLORS['line1'], COLORS['line2'], COLORS['line3'], COLORS['line4']]
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(values, label=metric_name, color=color, linewidth=2, alpha=0.9)
    
    plt.title('Training Curves', color=COLORS['text'], fontweight='bold')
    plt.xlabel('Episode', color=COLORS['text'])
    plt.ylabel('Value', color=COLORS['text'])
    
    # Style legend for black background
    legend = plt.legend(facecolor='#333333', framealpha=0.7)
    for text in legend.get_texts():
        text.set_color(COLORS['text'])
        
    plt.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.spines['left'].set_color(COLORS['grid'])
    
    # Add a watermark
    plt.gcf().text(
        0.99, 0.01, 
        'Traffic RL', 
        ha='right', va='bottom', 
        fontsize=8, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), facecolor=COLORS['background'])
    plt.close() 