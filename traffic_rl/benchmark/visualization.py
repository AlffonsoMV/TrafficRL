"""
Benchmark Visualization
=====================
Visualization functions for benchmark results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from cycler import cycler

logger = logging.getLogger("TrafficRL.BenchmarkViz")

# Define a modern, professional color palette
COLORS = {
    # Highlight DQN with an elegant green
    'dqn': '#00B386',  # Bright teal green for DQN (main model to highlight)
    'entity_dqn': '#00D6A1',  # Lighter teal for entity DQN variant
    
    # Other models in coordinated red tones
    'fixed': '#E63946',  # Bright red for fixed timing
    'adaptive': '#FF8C94',  # Soft red for adaptive
    'random': '#A71D31',  # Darker red for random
    'ppo': '#FF5C5C',  # Medium red for PPO
    
    # Background and styling colors
    'background': 'black',  # Black background
    'background_element': '#333333',  # Dark gray for UI elements
    'grid': '#555555',  # Medium gray grid lines
    'text': '#FFFFFF',  # White text for high contrast
    'accent': '#5B9BD5',  # Blue accent color
    
    # Highlight colors
    'highlight': '#FFD166',  # Yellow gold highlight
    'success': '#06D6A0',  # Bright green success
    'warning': '#F95738',  # Orange warning
}

# Define consistent plotting styles
PLOT_STYLE = {
    'figure.facecolor': COLORS['background'],  # Black figure background
    'axes.facecolor': COLORS['background'],    # Black axes background
    'savefig.transparent': False,              # Opaque saved figures with black background
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 16,                     # Increased from 13
    'axes.titlesize': 20,                     # Increased from 16
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,                    # Increased from 11
    'ytick.labelsize': 14,                    # Increased from 11
    'legend.fontsize': 14,                    # Increased from 11
    'figure.titlesize': 22,                   # Increased from 18
    'figure.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': COLORS['grid'],
    'axes.axisbelow': True,
    'axes.edgecolor': COLORS['grid'],
    'axes.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.color': COLORS['text'],          # White text color
    'axes.labelcolor': COLORS['text'],     # White axis labels
    'xtick.color': COLORS['text'],         # White tick labels
    'ytick.color': COLORS['text'],         # White tick labels
    'legend.labelcolor': COLORS['text'],   # White legend text
}

# Create a modern red color cycle for non-DQN agents
red_cycle = cycler(color=['#FF5C5C', '#A71D31', '#E63946', '#FF8C94'])

def format_thousands(x, pos):
    """Format numbers with thousands separator."""
    return f'{int(x):,}'

def create_benchmark_visualizations(benchmark_results, output_dir):
    """
    Create modernized, professional visualizations for benchmark results.
    
    Args:
        benchmark_results: Dictionary containing benchmark results
        output_dir: Directory to save visualizations
    """
    try:
        # Set global style parameters
        plt.rcParams.update(PLOT_STYLE)
        
        # Create output directories for different formats
        png_dir = os.path.join(output_dir, 'png')
        svg_dir = os.path.join(output_dir, 'svg')
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(svg_dir, exist_ok=True)
        
        # Create performance comparison plots
        create_performance_comparison(benchmark_results, png_dir, svg_dir)
        
        # Create pattern-specific plots
        create_pattern_analysis(benchmark_results, png_dir, svg_dir)
        
        # Create throughput comparison
        create_throughput_comparison(benchmark_results, png_dir, svg_dir)
        
        # Create queue length comparison
        create_queue_comparison(benchmark_results, png_dir, svg_dir)
        
        # Create waiting time distribution
        create_waiting_time_distribution(benchmark_results, png_dir, svg_dir)
        
        # Create episode rewards plot
        create_reward_comparison(benchmark_results, png_dir, svg_dir)
        
        # Create comprehensive dashboard
        create_dashboard(benchmark_results, png_dir, svg_dir)
        
        # Create radar chart comparison (new)
        create_radar_chart_comparison(benchmark_results, png_dir, svg_dir)
        
        logger.info(f"Enhanced visualizations saved to {output_dir} (PNG and SVG formats)")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())

def _save_figure(fig, filename, png_dir, svg_dir, dpi=300):
    """Save figure in both PNG and SVG formats with black backgrounds."""
    # Save as PNG
    png_path = os.path.join(png_dir, f"{filename}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor=COLORS['background'], transparent=False)
    
    # Save as SVG
    svg_path = os.path.join(svg_dir, f"{filename}.svg")
    fig.savefig(svg_path, bbox_inches='tight', facecolor=COLORS['background'], transparent=False)

def _get_agent_colors(agents):
    """Get consistent colors for agents, highlighting DQN-based agents with green."""
    agent_colors = []
    for agent in agents:
        agent_lower = agent.lower()
        if 'dqn' in agent_lower:
            # For DQN agents, use the special green highlight colors
            if 'entity' in agent_lower:
                agent_colors.append(COLORS['entity_dqn'])
            else:
                agent_colors.append(COLORS['dqn'])
        else:
            # For other agents, choose from the red palette based on agent name
            color_key = agent_lower if agent_lower in COLORS else 'fixed'
            agent_colors.append(COLORS[color_key])
    
    return agent_colors

def create_performance_comparison(benchmark_results, png_dir, svg_dir):
    """
    Create modern comparison plot of agent performances.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    agents = list(benchmark_results['results'].keys())
    waiting_times = []
    
    for agent in agents:
        agent_results = benchmark_results['results'][agent]
        avg_waiting_time = np.mean([
            pattern['summary']['avg_waiting_time']
            for pattern in agent_results['patterns'].values()
        ])
        waiting_times.append(avg_waiting_time)
    
    # Use consistent colors for each agent (green for DQN, red variants for others)
    agent_colors = _get_agent_colors(agents)
    
    # Create stylish bar chart with slightly rounded edges and enhanced appearance
    bars = ax.bar(
        agents, 
        waiting_times, 
        color=agent_colors, 
        width=0.65, 
        edgecolor='white',
        linewidth=1.2,
        alpha=0.95
    )
    
    # Add bright glow effect to bars
    for bar in bars:
        x = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        
        # Add subtle highlight effect
        rect = plt.Rectangle(
            (x, 0), width, height,
            alpha=0.15,
            color='white',
            linewidth=0
        )
        ax.add_patch(rect)
    
    # Add value labels on top of each bar with improved styling - with white text
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.5,
            f'{height:.1f}',
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=12,
            color='white'
        )
    
    # Add elegant title and labels
    ax.set_title('Average Waiting Time by Agent Type', pad=20, color=COLORS['text'])
    ax.set_xlabel('Agent Type', labelpad=10, color=COLORS['text'])
    ax.set_ylabel('Average Waiting Time (seconds)', labelpad=10, color=COLORS['text'])
    ax.tick_params(axis='x', labelrotation=0, colors=COLORS['text'])
    ax.tick_params(axis='y', colors=COLORS['text'])
    
    # Add subtle horizontal grid lines only
    ax.grid(axis='y', linestyle='--', alpha=0.4, color=COLORS['grid'])
    
    # Beautify the figure with enhanced styling for black background
    fig.tight_layout(pad=3)
    
    # Add a watermark/credit
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        color=COLORS['text'], 
        alpha=0.5,
        fontsize=10
    )
    
    # Save in both formats
    _save_figure(fig, 'performance_comparison', png_dir, svg_dir)
    plt.close(fig)

def create_pattern_analysis(benchmark_results, png_dir, svg_dir):
    """
    Create modern analysis plots for different traffic patterns.
    """
    patterns = list(next(iter(benchmark_results['results'].values()))['patterns'].keys())
    agents = list(benchmark_results['results'].keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract waiting times for each agent and pattern
    data = []
    for agent in agents:
        for pattern in patterns:
            waiting_time = benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_waiting_time']
            data.append({
                'Agent': agent,
                'Pattern': pattern.replace('_', ' ').title(),
                'Waiting Time': waiting_time
            })
    
    df = pd.DataFrame(data)
    
    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index='Pattern', columns='Agent', values='Waiting Time')
    
    # Create custom color palette with DQN highlighted in green
    agent_colors = {agent: color for agent, color in zip(agents, _get_agent_colors(agents))}
    
    # Plot grouped bar chart with custom colors and improved styling
    ax = pivot_df.plot(
        kind='bar', 
        ax=ax, 
        width=0.8, 
        color=agent_colors, 
        edgecolor='white',
        linewidth=1.2,
        alpha=0.95
    )
    
    # Add value labels on bars with improved styling - with white text
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.1f', 
            fontweight='bold', 
            fontsize=10,
            padding=3,
            color='white'
        )
    
    # Set elegant title and labels with white text
    ax.set_title('Average Waiting Time by Traffic Pattern and Agent Type', pad=20, color=COLORS['text'])
    ax.set_xlabel('Traffic Pattern', labelpad=15, color=COLORS['text'])
    ax.set_ylabel('Average Waiting Time (seconds)', labelpad=15, color=COLORS['text'])
    ax.tick_params(axis='both', colors=COLORS['text'])
    
    # Improve x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', color=COLORS['text'])
    
    # Add a light grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color=COLORS['grid'])
    
    # Customize legend with improved styling for black background
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles, 
        labels, 
        title='Agent Type', 
        title_fontsize=15,
        loc='upper right', 
        frameon=True, 
        framealpha=0.6, 
        edgecolor=COLORS['grid'],
        facecolor=COLORS['background_element'],
        fancybox=True
    )
    plt.setp(lgd.get_title(), fontweight='bold', color=COLORS['text'])
    plt.setp(lgd.get_texts(), color=COLORS['text'])
    
    fig.tight_layout(pad=3)
    
    # Add a subtle watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        fontsize=10, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    # Save in both formats
    _save_figure(fig, 'pattern_analysis', png_dir, svg_dir)
    plt.close(fig)

def create_throughput_comparison(benchmark_results, png_dir, svg_dir):
    """
    Create modern comparison of throughput across agents and patterns.
    """
    patterns = list(next(iter(benchmark_results['results'].values()))['patterns'].keys())
    agents = list(benchmark_results['results'].keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract throughput for each agent and pattern
    data = []
    for agent in agents:
        for pattern in patterns:
            throughput = benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_throughput']
            data.append({
                'Agent': agent,
                'Pattern': pattern.replace('_', ' ').title(),
                'Throughput': throughput
            })
    
    df = pd.DataFrame(data)
    
    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index='Pattern', columns='Agent', values='Throughput')
    
    # Create custom color palette highlighting DQN
    agent_colors = {agent: color for agent, color in zip(agents, _get_agent_colors(agents))}
    
    # Create a gradient fill function to make bars more visually appealing
    def gradient_fill(x, y, fill_color, ax):
        """Create a gradient-filled bar for enhanced visuals"""
        import matplotlib.patches as patches
        
        # Bar width is typically 0.8 / number of categories
        width = 0.8 / len(agents)
        
        # Draw a rectangle with gradient
        for i, (xi, yi) in enumerate(zip(x, y)):
            rect = patches.Rectangle((xi - width/2, 0), width, yi, linewidth=1.2, 
                                   edgecolor='white', facecolor=fill_color, alpha=0.95)
            ax.add_patch(rect)
            
            # Add highlight effect
            highlight = patches.Rectangle((xi - width/2, 0), width, yi, linewidth=0,
                                       facecolor='white', alpha=0.1)
            ax.add_patch(highlight)
    
    # Plot grouped bar chart with custom colors and modern styling
    ax = pivot_df.plot(
        kind='bar', 
        ax=ax, 
        width=0.8, 
        color=agent_colors, 
        edgecolor='white',
        linewidth=1.2,
        alpha=0.95
    )
    
    # Add thousands formatter to y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    
    # Add value labels on bars with improved formatting - with white text
    for container in ax.containers:
        labels = [f'{int(v/1000)}k' if v > 1000 else str(int(v)) for v in container.datavalues]
        ax.bar_label(
            container, 
            labels=labels, 
            fontweight='bold', 
            fontsize=10,
            padding=3,
            color='white'
        )
    
    # Set elegant title and labels with white text
    ax.set_title('Average Vehicle Throughput by Traffic Pattern and Agent Type', pad=20, color=COLORS['text'])
    ax.set_xlabel('Traffic Pattern', labelpad=15, color=COLORS['text'])
    ax.set_ylabel('Average Throughput (vehicles)', labelpad=15, color=COLORS['text'])
    ax.tick_params(axis='both', colors=COLORS['text'])
    
    # Improve x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', color=COLORS['text'])
    
    # Add a light grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color=COLORS['grid'])
    
    # Customize legend with improved styling for black background
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles, 
        labels, 
        title='Agent Type', 
        title_fontsize=15,
        loc='upper right', 
        frameon=True, 
        framealpha=0.6, 
        edgecolor=COLORS['grid'],
        facecolor=COLORS['background_element'],
        fancybox=True
    )
    plt.setp(lgd.get_title(), fontweight='bold', color=COLORS['text'])
    plt.setp(lgd.get_texts(), color=COLORS['text'])
    
    fig.tight_layout(pad=3)
    
    # Add a subtle watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        fontsize=10, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    # Save in both formats
    _save_figure(fig, 'throughput_comparison', png_dir, svg_dir)
    plt.close(fig)

def create_queue_comparison(benchmark_results, png_dir, svg_dir):
    """
    Create modern comparison of queue lengths across agents and patterns.
    """
    patterns = list(next(iter(benchmark_results['results'].values()))['patterns'].keys())
    agents = list(benchmark_results['results'].keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract queue lengths for each agent and pattern
    data = []
    for agent in agents:
        for pattern in patterns:
            queue_length = benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_queue_length']
            data.append({
                'Agent': agent,
                'Pattern': pattern.replace('_', ' ').title(),
                'Queue Length': queue_length
            })
    
    df = pd.DataFrame(data)
    
    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index='Pattern', columns='Agent', values='Queue Length')
    
    # Create custom color palette highlighting DQN in green
    agent_colors = {agent: color for agent, color in zip(agents, _get_agent_colors(agents))}
    
    # Plot grouped bar chart with custom colors and modern styling
    ax = pivot_df.plot(
        kind='bar', 
        ax=ax, 
        width=0.8, 
        color=agent_colors, 
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9
    )
    
    # Add value labels on bars with improved formatting - with white text
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.3f', 
            fontweight='bold', 
            fontsize=9,
            padding=3,
            color='white'
        )
    
    # Set elegant title and labels
    ax.set_title('Average Queue Length by Traffic Pattern and Agent Type', pad=20)
    ax.set_xlabel('Traffic Pattern', labelpad=15)
    ax.set_ylabel('Average Queue Length (vehicles)', labelpad=15)
    
    # Improve x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Add a light grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Customize legend with improved styling
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles, 
        labels, 
        title='Agent Type', 
        title_fontsize=12,
        loc='upper right', 
        frameon=True, 
        framealpha=0.8, 
        edgecolor=COLORS['grid'],
        fancybox=True
    )
    plt.setp(lgd.get_title(), fontweight='bold')
    
    fig.tight_layout(pad=3)
    
    # Add a subtle watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        fontsize=10, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    # Save in both formats
    _save_figure(fig, 'queue_comparison', png_dir, svg_dir)
    plt.close(fig)

def create_waiting_time_distribution(benchmark_results, png_dir, svg_dir):
    """
    Create modern distribution plots of waiting times.
    """
    patterns = list(next(iter(benchmark_results['results'].values()))['patterns'].keys())
    agents = list(benchmark_results['results'].keys())
    
    # Create a figure with a subplot for each pattern
    fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 5*len(patterns)))
    if len(patterns) == 1:
        axes = [axes]
    
    # Get agent colors
    agent_colors = {agent: color for agent, color in zip(agents, _get_agent_colors(agents))}
    
    for i, pattern in enumerate(patterns):
        ax = axes[i]
        
        # Extract waiting time distributions for each agent in this pattern
        for agent in agents:
            delay_distribution = benchmark_results['results'][agent]['patterns'][pattern]['summary']['delay_distribution']
            
            # Create density plot for the delay distribution with modern styling
            sns.kdeplot(
                delay_distribution, 
                ax=ax, 
                label=agent, 
                color=agent_colors.get(agent, '#999999'), 
                fill=True, 
                alpha=0.25, 
                linewidth=2.5,
                bw_adjust=0.8  # Smoother curves
            )
        
        # Add refined title and labels
        ax.set_title(f'Waiting Time Distribution - {pattern.replace("_", " ").title()}', pad=15)
        ax.set_xlabel('Waiting Time (seconds)', labelpad=10)
        ax.set_ylabel('Density', labelpad=10)
        
        # Improve the legend
        lgd = ax.legend(
            title='Agent Type', 
            title_fontsize=11, 
            frameon=True, 
            framealpha=0.8, 
            edgecolor=COLORS['grid'],
            fancybox=True
        )
        plt.setp(lgd.get_title(), fontweight='bold')
        
        # Add minimal grid for readability
        ax.grid(axis='both', linestyle='--', alpha=0.3)
        # put the x lim to 0 to 10
        ax.set_xlim(0, 300)
        
        
        # Beautify the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Improve tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    fig.tight_layout(pad=3)
    
    # Add a subtle watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        fontsize=10, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    # Save in both formats
    _save_figure(fig, 'waiting_time_distribution', png_dir, svg_dir)
    plt.close(fig)

def create_reward_comparison(benchmark_results, png_dir, svg_dir):
    """
    Create modern comparison of total rewards across agents and patterns.
    """
    patterns = list(next(iter(benchmark_results['results'].values()))['patterns'].keys())
    agents = list(benchmark_results['results'].keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract rewards for each agent and pattern
    data = []
    for agent in agents:
        for pattern in patterns:
            reward = benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_reward']
            data.append({
                'Agent': agent,
                'Pattern': pattern.replace('_', ' ').title(),
                'Reward': reward
            })
    
    df = pd.DataFrame(data)
    
    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index='Pattern', columns='Agent', values='Reward')
    
    # Create custom color palette highlighting DQN in green
    agent_colors = {agent: color for agent, color in zip(agents, _get_agent_colors(agents))}
    
    # Plot grouped bar chart with custom colors and modern styling
    ax = pivot_df.plot(
        kind='bar', 
        ax=ax, 
        width=0.8, 
        color=agent_colors, 
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9
    )
    
    # Add value labels on bars with improved formatting - with white text
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.1f', 
            fontweight='bold', 
            fontsize=9,
            padding=3,
            color='white'
        )
    
    # Set elegant title and labels
    ax.set_title('Average Episode Reward by Traffic Pattern and Agent Type', pad=20)
    ax.set_xlabel('Traffic Pattern', labelpad=15)
    ax.set_ylabel('Average Reward', labelpad=15)
    
    # Improve x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Add a light grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Customize legend with improved styling
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles, 
        labels, 
        title='Agent Type', 
        title_fontsize=12,
        loc='upper right', 
        frameon=True, 
        framealpha=0.8, 
        edgecolor=COLORS['grid'],
        fancybox=True
    )
    plt.setp(lgd.get_title(), fontweight='bold')
    
    fig.tight_layout(pad=3)
    
    # Add a subtle watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        fontsize=10, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    # Save in both formats
    _save_figure(fig, 'reward_comparison', png_dir, svg_dir)
    plt.close(fig)

def create_radar_chart_comparison(benchmark_results, png_dir, svg_dir):
    """
    Create a radar chart comparing all metrics across models.
    This visualizes multiple metrics at once in a radial format.
    """
    # Get agents and patterns
    agents = list(benchmark_results['results'].keys())
    patterns = list(next(iter(benchmark_results['results'].values()))['patterns'].keys())
    
    # Get agent colors
    agent_colors = {agent: color for agent, color in zip(agents, _get_agent_colors(agents))}
    
    # Metrics we want to compare (names and normalization directions)
    # For some metrics, higher is better (1), for others, lower is better (-1)
    metrics = {
        'avg_waiting_time': {'label': 'Waiting Time', 'direction': -1, 'format': '{:.1f}s'},
        'avg_queue_length': {'label': 'Queue Length', 'direction': -1, 'format': '{:.3f}'},
        'avg_throughput': {'label': 'Throughput', 'direction': 1, 'format': '{:.0f}'},
        'avg_reward': {'label': 'Reward', 'direction': 1, 'format': '{:.1f}'}
    }
    
    # Number of metrics to plot
    n_metrics = len(metrics)
    
    # Calculate average metrics across patterns for each agent
    agent_metrics = {}
    for agent in agents:
        agent_data = {}
        for metric_key, metric_info in metrics.items():
            values = []
            for pattern in patterns:
                # Extract the metric value
                value = benchmark_results['results'][agent]['patterns'][pattern]['summary'][metric_key]
                values.append(value)
            # Calculate average across patterns
            agent_data[metric_key] = np.mean(values)
        agent_metrics[agent] = agent_data
    
    # Create figure with specific styling for black background
    fig = plt.figure(figsize=(14, 12), facecolor=COLORS['background'])
    ax = fig.add_subplot(111, polar=True)
    
    # Set radar chart angles
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    # Close the radar chart by repeating the first angle
    angles += angles[:1]
    
    # Set labels for each angle
    labels = [metrics[key]['label'] for key in metrics.keys()]
    # Add the first label at the end to close the polygon
    labels += [labels[0]]
    
    # Plot angle lines
    ax.set_theta_offset(np.pi / 2)  # Start at top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Set labels with enhanced styling
    plt.xticks(angles[:-1], labels[:-1], fontsize=16, fontweight='bold', color=COLORS['text'])
    
    # Normalize metrics for radar chart
    # First, get min and max for each metric across all agents
    metric_ranges = {}
    for metric_key in metrics.keys():
        values = [agent_metrics[agent][metric_key] for agent in agents]
        metric_ranges[metric_key] = {'min': min(values), 'max': max(values)}
    
    # Function to normalize values between 0 and 1 considering if higher or lower is better
    def normalize_metric(value, metric_key):
        min_val = metric_ranges[metric_key]['min']
        max_val = metric_ranges[metric_key]['max']
        direction = metrics[metric_key]['direction']
        
        # If min and max are the same, return 0.5 to avoid division by zero
        if min_val == max_val:
            return 0.5
        
        # Normalize and adjust direction (higher is better = 1, lower is better = -1)
        if direction > 0:  # Higher is better
            return (value - min_val) / (max_val - min_val)
        else:  # Lower is better
            return (max_val - value) / (max_val - min_val)
    
    # Add ring levels with labels
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(levels)
    ax.set_yticklabels([f"{int(level*100)}%" for level in levels], color=COLORS['text'], fontsize=12)
    
    # Enhanced grid styling for black background
    ax.set_rlabel_position(0)  # Move radial labels to convenient position
    ax.grid(True, linestyle='--', color=COLORS['grid'], alpha=0.5, linewidth=0.8)
    
    # Add circular grid (concentric circles)
    for level in levels:
        ax.plot(np.linspace(0, 2*np.pi, 100), [level] * 100, 
                color=COLORS['grid'], linestyle='--', alpha=0.4, linewidth=0.8)
    
    # Add axis lines connecting center to each metric point
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1.1], color=COLORS['grid'], linestyle='--', 
                alpha=0.5, linewidth=0.8, zorder=1)
    
    # Add highlight for the axis background
    ax.set_facecolor(COLORS['background'])
    
    # Plot each agent with enhanced styling
    for agent in agents:
        # Get normalized values for this agent
        values = []
        for metric_key in metrics.keys():
            norm_value = normalize_metric(agent_metrics[agent][metric_key], metric_key)
            values.append(norm_value)
        
        # Close the polygon by repeating the first value
        values += values[:1]
        
        # Plot the agent's polygon with enhanced styling
        ax.plot(angles, values, linewidth=2.5, label=agent, color=agent_colors[agent], alpha=0.9, zorder=10)
        
        # Fill with gradient and slight glow effect
        ax.fill(angles, values, alpha=0.3, color=agent_colors[agent], zorder=5)
        
        # Add points at each metric vertex with white highlight
        for i, angle in enumerate(angles[:-1]):  # Skip the last point (duplicate)
            ax.scatter(angle, values[i], s=100, color=agent_colors[agent], 
                      edgecolor='white', linewidth=1.5, alpha=0.9, zorder=15)
            
            # Add value annotations near each point
            value = agent_metrics[agent][list(metrics.keys())[i]]
            format_str = metrics[list(metrics.keys())[i]]['format']
            formatted_val = format_str.format(value)
            
            # Position the text slightly outside the point
            ha = 'center'
            offset_factor = 1.05
            if angle == 0:  # Top point
                va = 'bottom'
                offset_factor = 1.08
            elif 0 < angle < np.pi:  # Right side
                ha = 'left'
                offset_factor = 1.05
            elif angle == np.pi:  # Bottom point
                va = 'top'
                offset_factor = 1.08
            else:  # Left side
                ha = 'right'
                offset_factor = 1.05
            
            # Add the value with white text
            ax.text(angle, values[i] * offset_factor, formatted_val,
                   color=COLORS['text'], fontsize=12, fontweight='bold',
                   ha=ha, va='center', zorder=20)
    
    # Customize the chart
    ax.set_ylim(0, 1)
    
    # Add legend with enhanced styling for black background
    legend = plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(agents), 4),  # Arrange agents in rows for better visibility
        title='Agent Type',
        title_fontsize=16,
        frameon=True,
        framealpha=0.6,
        edgecolor=COLORS['grid'],
        facecolor=COLORS['background_element'],
        fancybox=True,
        fontsize=14
    )
    
    # Style legend text and title
    plt.setp(legend.get_title(), fontweight='bold', color=COLORS['text'])
    plt.setp(legend.get_texts(), color=COLORS['text'])
    
    # Add a title with enhanced styling
    plt.title('Performance Metrics Comparison', fontweight='bold', size=22, pad=30, color=COLORS['text'])
    
    # Add contextual annotation about metrics
    context_text = "Metrics normalized to 0-100%. For waiting time and queue length, lower values are better."
    fig.text(0.5, 0.02, context_text, ha='center', va='center', 
             color=COLORS['text'], fontsize=13, alpha=0.8)
    
    # Add a watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        fontsize=10, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    # Save in both formats
    _save_figure(fig, 'radar_chart_comparison', png_dir, svg_dir)
    plt.close(fig)

def create_dashboard(benchmark_results, png_dir, svg_dir):
    """
    Create a comprehensive modern dashboard with key metrics.
    """
    patterns = list(next(iter(benchmark_results['results'].values()))['patterns'].keys())
    agents = list(benchmark_results['results'].keys())
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Traffic Control Agent Performance Dashboard', y=0.98, fontsize=24, fontweight='bold')
    
    # Define grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Get agent colors
    agent_colors = _get_agent_colors(agents)
    agent_color_dict = {agent: color for agent, color in zip(agents, agent_colors)}
    
    # 1. Overall Waiting Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    avg_waiting_times = []
    for agent in agents:
        avg_waiting_time = np.mean([
            benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_waiting_time']
            for pattern in patterns
        ])
        avg_waiting_times.append(avg_waiting_time)
    
    # Create enhanced bar chart
    bars = ax1.bar(
        agents, 
        avg_waiting_times, 
        color=agent_colors, 
        width=0.65,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9
    )
    
    # Add value labels with improved styling - with white text
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.2,
            f'{height:.1f}', 
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=12,
            color='white'
        )
    
    ax1.set_title('Average Waiting Time')
    ax1.set_xlabel('Agent Type', labelpad=10)
    ax1.set_ylabel('Seconds', labelpad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='x', labelrotation=0)
    
    # 2. Overall Throughput Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    avg_throughputs = []
    for agent in agents:
        avg_throughput = np.mean([
            benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_throughput']
            for pattern in patterns
        ])
        avg_throughputs.append(avg_throughput)
    
    # Create enhanced bar chart
    bars = ax2.bar(
        agents, 
        avg_throughputs, 
        color=agent_colors, 
        width=0.65,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9
    )
    
    # Add thousands formatter to y-axis
    ax2.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    
    # Add value labels with improved styling - with white text
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., 
            height + height*0.02,
            f'{int(height/1000)}k', 
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=10,
            color='white'
        )
    
    ax2.set_title('Average Throughput')
    ax2.set_xlabel('Agent Type', labelpad=10)
    ax2.set_ylabel('Vehicles', labelpad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.tick_params(axis='x', labelrotation=0)
    
    # 3. Pattern-specific waiting time comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    # Extract data
    data = []
    for agent in agents:
        for pattern in patterns:
            waiting_time = benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_waiting_time']
            data.append({
                'Agent': agent,
                'Pattern': pattern.replace('_', ' ').title(),
                'Waiting Time': waiting_time
            })
    
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Pattern', columns='Agent', values='Waiting Time')
    
    # Plot grouped bar chart with enhanced styling
    pivot_df.plot(
        kind='bar', 
        ax=ax3, 
        color=agent_color_dict, 
        width=0.8,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9
    )
    
    # Add value labels with improved styling - white text
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.1f', fontweight='bold', fontsize=12, padding=3, color='white')
    
    ax3.set_title('Waiting Time by Traffic Pattern')
    ax3.set_xlabel('Traffic Pattern', labelpad=10)
    ax3.set_ylabel('Average Waiting Time (seconds)', labelpad=10)
    
    # Improve legend
    lgd = ax3.legend(
        title='Agent Type', 
        title_fontsize=14,
        frameon=True, 
        framealpha=0.8, 
        edgecolor=COLORS['grid'],
        fancybox=True
    )
    plt.setp(lgd.get_title(), fontweight='bold')
    
    # Add light grid
    ax3.grid(axis='y', linestyle='--', alpha=0.4)
    ax3.tick_params(axis='x', labelrotation=0)
    
    # 4. Metrics table with modern styling
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Prepare table data
    table_data = []
    for agent in agents:
        row = [agent]
        # Calculate averages across all patterns
        avg_waiting = np.mean([
            benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_waiting_time']
            for pattern in patterns
        ])
        avg_throughput = np.mean([
            benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_throughput']
            for pattern in patterns
        ])
        avg_queue = np.mean([
            benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_queue_length']
            for pattern in patterns
        ])
        avg_reward = np.mean([
            benchmark_results['results'][agent]['patterns'][pattern]['summary']['avg_reward']
            for pattern in patterns
        ])
        
        # Format and add metrics to the row
        row.append(f"{avg_waiting:.2f}")
        row.append(f"{avg_throughput:.0f}")
        row.append(f"{avg_queue:.2f}")
        row.append(f"{avg_reward:.2f}")
        
        table_data.append(row)
    
    # Create modern table with improved styling
    column_headers = ['Agent', 'Avg Waiting Time (s)', 'Throughput', 'Avg Queue Length', 'Avg Reward']
    table = ax4.table(
        cellText=table_data,
        colLabels=column_headers, 
        loc='center',
        cellLoc='center',
        colColours=['#f8f9fa']*len(column_headers)
    )
    
    # Style table with modern look
    table.auto_set_font_size(False)
    table.set_fontsize(14)  # Increased from original 10
    table.scale(1, 1.5)
    
    # Style header row
    for j, header in enumerate(column_headers):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#e9ecef')
        
    # Color the agent cells with agent colors - with white text
    for i, agent in enumerate(agents):
        cell = table[(i+1, 0)]
        cell_color = agent_color_dict.get(agent, '#999999')
        cell.set_facecolor(cell_color)
        cell.set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Performance Metrics Summary')
    
    # Add footer with timestamp
    fig.text(
        0.5, 0.01, 
        f"Generated: {benchmark_results['timestamp']}", 
        ha='center', 
        fontsize=11,
        color=COLORS['text']
    )
    
    # Overall figure styling
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add a watermark
    fig.text(
        0.99, 0.01, 
        'Traffic RL Benchmark', 
        ha='right', va='bottom', 
        fontsize=10, 
        color=COLORS['text'], 
        alpha=0.5
    )
    
    # Save in both formats
    _save_figure(fig, 'performance_dashboard', png_dir, svg_dir, dpi=300)
    plt.close(fig)
