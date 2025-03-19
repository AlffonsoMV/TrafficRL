"""
Comparative Analysis
==================
Compare performance across different agents and configurations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from datetime import datetime

logger = logging.getLogger('TrafficRL.ComparativeAnalysis')

# Define colors for black background
COLORS = {
    'background': 'black',
    'text': 'white',
    'grid': '#555555',
    'bar1': '#00B386',  # teal green
    'bar2': '#E63946',  # bright red
    'bar3': '#5B9BD5',  # blue
    'bar4': '#FFD166',  # gold
    'accent': '#FF5C5C',  # pink red
}

def comparative_analysis(agents_data, output_dir=None):
    """
    Compare performance across different agents.
    
    Args:
        agents_data: Dictionary mapping agent names to their performance metrics
        output_dir: Optional directory to save visualizations
    """
    if not agents_data:
        logger.warning("No agent data to compare")
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
    
    # Get agent list and assign colors
    agents = list(agents_data.keys())
    agent_colors = []
    for i, agent in enumerate(agents):
        if 'dqn' in agent.lower():
            agent_colors.append(COLORS['bar1'])  # DQN agents get teal
        elif 'fixed' in agent.lower():
            agent_colors.append(COLORS['bar2'])  # Fixed timing gets red
        elif 'adaptive' in agent.lower():
            agent_colors.append(COLORS['bar3'])  # Adaptive gets blue
        else:
            # Alternate between gold and pink for others
            agent_colors.append(COLORS['bar4'] if i % 2 == 0 else COLORS['accent'])
    
    # Plot average waiting time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    waiting_times = [data['avg_waiting_time'] for data in agents_data.values()]
    
    # Create enhanced bars with white edges
    bars1 = ax1.bar(agents, waiting_times, color=agent_colors, edgecolor='white', linewidth=1.2, alpha=0.95)
    
    # Add value labels on top of bars with white text
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.05 * max(waiting_times),
            f'{height:.1f}',
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=10,
            color='white'
        )
    
    ax1.set_title('Average Waiting Time by Agent', color=COLORS['text'], fontweight='bold')
    ax1.set_xlabel('Agent', color=COLORS['text'])
    ax1.set_ylabel('Average Waiting Time (s)', color=COLORS['text'])
    ax1.tick_params(axis='both', colors=COLORS['text'])
    ax1.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS['grid'])
    
    # Plot throughput comparison
    ax2 = fig.add_subplot(gs[0, 1])
    throughput = [data['throughput'] for data in agents_data.values()]
    
    # Create enhanced bars with white edges
    bars2 = ax2.bar(agents, throughput, color=agent_colors, edgecolor='white', linewidth=1.2, alpha=0.95)
    
    # Add value labels with white text
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.05 * max(throughput),
            f'{int(height)}',
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=10,
            color='white'
        )
    
    ax2.set_title('Throughput by Agent', color=COLORS['text'], fontweight='bold')
    ax2.set_xlabel('Agent', color=COLORS['text'])
    ax2.set_ylabel('Vehicles per Hour', color=COLORS['text'])
    ax2.tick_params(axis='both', colors=COLORS['text'])
    ax2.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS['grid'])
    
    # Plot queue length comparison
    ax3 = fig.add_subplot(gs[1, 0])
    queue_lengths = [data['avg_queue_length'] for data in agents_data.values()]
    
    # Create enhanced bars with white edges
    bars3 = ax3.bar(agents, queue_lengths, color=agent_colors, edgecolor='white', linewidth=1.2, alpha=0.95)
    
    # Add value labels with white text
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.05 * max(queue_lengths),
            f'{height:.1f}',
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=10,
            color='white'
        )
    
    ax3.set_title('Average Queue Length by Agent', color=COLORS['text'], fontweight='bold')
    ax3.set_xlabel('Agent', color=COLORS['text'])
    ax3.set_ylabel('Average Queue Length', color=COLORS['text'])
    ax3.tick_params(axis='both', colors=COLORS['text'])
    ax3.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS['grid'])
    
    # Plot delay distribution comparison
    ax4 = fig.add_subplot(gs[1, 1])
    delays = [data['delay_distribution'] for data in agents_data.values()]
    
    # Create enhanced boxplot with custom colors
    boxplot = ax4.boxplot(delays, labels=agents, patch_artist=True)
    
    # Customize boxplot colors for black background
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=agent_colors[i % len(agent_colors)], alpha=0.8)
        box.set(edgecolor='white', linewidth=1.2)
    
    for whisker in boxplot['whiskers']:
        whisker.set(color='white', linewidth=1.2)
    
    for cap in boxplot['caps']:
        cap.set(color='white', linewidth=1.2)
    
    for median in boxplot['medians']:
        median.set(color='white', linewidth=1.5)
    
    for flier in boxplot['fliers']:
        flier.set(marker='o', markerfacecolor='white', markeredgecolor='none', markersize=4, alpha=0.7)
    
    ax4.set_title('Delay Distribution by Agent', color=COLORS['text'], fontweight='bold')
    ax4.set_xlabel('Agent', color=COLORS['text'])
    ax4.set_ylabel('Delay (s)', color=COLORS['text'])
    ax4.tick_params(axis='both', colors=COLORS['text'])
    ax4.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS['grid'])
    
    # Add a watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Analysis', 
        ha='right', va='bottom', 
        fontsize=8, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'comparative_analysis.png'), facecolor=COLORS['background'])
    plt.close()
    
    # Generate summary statistics
    summary = {
        'best_agent': min(agents_data.items(), key=lambda x: x[1]['avg_waiting_time'])[0],
        'worst_agent': max(agents_data.items(), key=lambda x: x[1]['avg_waiting_time'])[0],
        'avg_waiting_time': np.mean(waiting_times),
        'std_waiting_time': np.std(waiting_times),
        'avg_throughput': np.mean(throughput),
        'avg_queue_length': np.mean(queue_lengths)
    }
    
    if output_dir:
        with open(os.path.join(output_dir, 'comparative_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    
    return summary

def create_comprehensive_report(analysis_results, output_dir=None):
    """
    Create a comprehensive report combining all analysis results.
    
    Args:
        analysis_results: Dictionary containing results from different analyses
        output_dir: Optional directory to save the report
    """
    if not analysis_results:
        logger.warning("No analysis results to report")
        return
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'best_agent': analysis_results.get('best_agent'),
            'best_pattern': analysis_results.get('best_pattern'),
            'overall_performance': {
                'avg_waiting_time': analysis_results.get('avg_waiting_time'),
                'avg_throughput': analysis_results.get('avg_throughput'),
                'avg_queue_length': analysis_results.get('avg_queue_length')
            }
        },
        'details': analysis_results
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'comprehensive_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
    
    return report 