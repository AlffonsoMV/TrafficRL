"""
Traffic Pattern Analysis
======================
Analyze how agent performance varies across different traffic patterns.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from matplotlib.gridspec import GridSpec

logger = logging.getLogger('TrafficRL.PatternAnalysis')

def load_experiment_data(experiment_dir):
    """
    Load data from different traffic pattern experiments.
    
    Args:
        experiment_dir: Base directory containing experiment subdirectories
        
    Returns:
        Dictionary mapping pattern names to metrics
    """
    pattern_data = {}
    
    try:
        for pattern_dir in os.listdir(experiment_dir):
            pattern_path = os.path.join(experiment_dir, pattern_dir)
            if os.path.isdir(pattern_path):
                metrics_file = os.path.join(pattern_path, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        pattern_data[pattern_dir] = json.load(f)
        return pattern_data
    except Exception as e:
        logger.error(f"Error loading experiment data: {e}")
        return {}

def analyze_traffic_patterns(pattern_data, output_dir=None):
    """
    Analyze and visualize performance across different traffic patterns.
    
    Args:
        pattern_data: Dictionary mapping pattern names to metrics
        output_dir: Optional directory to save visualizations
    """
    if not pattern_data:
        logger.warning("No pattern data to analyze")
        return
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot average waiting time
    ax1 = fig.add_subplot(gs[0, 0])
    waiting_times = [data['avg_waiting_time'] for data in pattern_data.values()]
    patterns = list(pattern_data.keys())
    ax1.bar(patterns, waiting_times)
    ax1.set_title('Average Waiting Time by Pattern')
    ax1.set_xlabel('Traffic Pattern')
    ax1.set_ylabel('Average Waiting Time (s)')
    
    # Plot queue lengths
    ax2 = fig.add_subplot(gs[0, 1])
    queue_lengths = [data['avg_queue_length'] for data in pattern_data.values()]
    ax2.bar(patterns, queue_lengths)
    ax2.set_title('Average Queue Length by Pattern')
    ax2.set_xlabel('Traffic Pattern')
    ax2.set_ylabel('Average Queue Length')
    
    # Plot throughput
    ax3 = fig.add_subplot(gs[1, 0])
    throughput = [data['throughput'] for data in pattern_data.values()]
    ax3.bar(patterns, throughput)
    ax3.set_title('Throughput by Pattern')
    ax3.set_xlabel('Traffic Pattern')
    ax3.set_ylabel('Vehicles per Hour')
    
    # Plot delay distribution
    ax4 = fig.add_subplot(gs[1, 1])
    delays = [data['delay_distribution'] for data in pattern_data.values()]
    ax4.boxplot(delays, labels=patterns)
    ax4.set_title('Delay Distribution by Pattern')
    ax4.set_xlabel('Traffic Pattern')
    ax4.set_ylabel('Delay (s)')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pattern_analysis.png'))
    plt.close()
    
    # Generate summary statistics
    summary = {
        'best_pattern': min(pattern_data.items(), key=lambda x: x[1]['avg_waiting_time'])[0],
        'worst_pattern': max(pattern_data.items(), key=lambda x: x[1]['avg_waiting_time'])[0],
        'avg_waiting_time': np.mean(waiting_times),
        'std_waiting_time': np.std(waiting_times),
        'avg_queue_length': np.mean(queue_lengths),
        'avg_throughput': np.mean(throughput)
    }
    
    if output_dir:
        with open(os.path.join(output_dir, 'pattern_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    
    return summary 