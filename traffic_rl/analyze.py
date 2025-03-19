"""
Analyze module
=============
Module to bridge the gap between CLI and analysis functionality.
"""

import os
import logging
import json
from typing import List, Dict, Optional, Any, Union
import matplotlib.pyplot as plt
import pandas as pd
import webbrowser
from datetime import datetime

# Import from analysis module
from traffic_rl.analysis import analyze_training_metrics
# Import from benchmark module
from traffic_rl.benchmark import benchmark_agents, create_benchmark_agents

logger = logging.getLogger('TrafficRL.Analyze')

def run_benchmark(
    config: Dict[str, Any],
    model_paths: Union[str, List[str]],
    output_dir: str = 'results/benchmark',
    traffic_patterns: List[str] = None,
    num_episodes: int = 10
) -> Dict[str, Any]:
    """
    Run benchmarks for the given models.
    
    Args:
        config: Configuration dictionary
        model_paths: Path(s) to model file(s)
        output_dir: Directory to save benchmark results
        traffic_patterns: Traffic patterns to evaluate
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of benchmark results
    """
    if traffic_patterns is None:
        traffic_patterns = ["uniform", "rush_hour", "weekend"]
    
    # Ensure model_paths is a list
    if isinstance(model_paths, str):
        model_paths_list = [model_paths]
    else:
        model_paths_list = model_paths
    
    # Create benchmark agents from model paths
    agents_to_benchmark = create_benchmark_agents(config, model_paths_list)
    
    # Run benchmarks
    return benchmark_agents(
        config=config,
        agents_to_benchmark=agents_to_benchmark,
        traffic_patterns=traffic_patterns,
        num_episodes=num_episodes,
        output_dir=output_dir,
        create_visualizations=True
    )

def prepare_benchmark_data_for_analysis(benchmark_results):
    """
    Prepare benchmark data for analysis.
    
    Args:
        benchmark_results (dict): Benchmark results.
        
    Returns:
        dict: Preprocessed data.
    """
    logger.info(f"Preparing benchmark data for analysis")
    logger.info(f"Results type: {type(benchmark_results)}")
    logger.info(f"Results top-level keys: {benchmark_results.keys() if isinstance(benchmark_results, dict) else 'Not a dict'}")
    
    processed_data = {}
    
    if not isinstance(benchmark_results, dict) or not benchmark_results:
        logger.warning("Benchmark results is not a dict or is empty")
        return processed_data
    
    # Check if we have a 'results' key that contains the actual agent results
    if 'results' in benchmark_results and isinstance(benchmark_results['results'], dict):
        # Use the 'results' dictionary that contains our agents
        agent_results = benchmark_results['results']
        logger.info(f"Using agent results from 'results' key with {len(agent_results)} agents: {list(agent_results.keys())}")
    else:
        # Use the entire benchmark_results as our agent data
        agent_results = benchmark_results
        logger.info(f"Using entire benchmark_results as agent data with {len(agent_results)} keys")
    
    # Process each agent
    for agent_name, agent_data in agent_results.items():
        # Skip non-dict values or metadata entries
        if not isinstance(agent_data, dict):
            logger.info(f"Skipping non-dict entry: {agent_name}")
            continue
            
        logger.info(f"Processing agent: {agent_name}")
        logger.info(f"Agent data keys: {agent_data.keys() if isinstance(agent_data, dict) else 'Not a dict'}")
        
        if 'patterns' not in agent_data:
            logger.warning(f"No patterns data for agent {agent_name}")
            processed_data[agent_name] = {
                'waiting_time': 0.0,
                'throughput': 0.0
            }
            continue
            
        # Extract pattern data
        pattern_data = agent_data['patterns']
        logger.info(f"Pattern data keys: {pattern_data.keys() if isinstance(pattern_data, dict) else 'Not a dict'}")
        
        # Initialize metrics
        total_waiting_time = 0.0
        total_throughput = 0.0
        pattern_count = 0
        
        # Process each pattern
        for pattern_name, pattern_results in pattern_data.items():
            logger.info(f"Processing pattern: {pattern_name}")
            logger.info(f"Pattern results type: {type(pattern_results)}")
            
            if isinstance(pattern_results, dict):
                # Try different possible key names for waiting time
                waiting_time = None
                for key in ['avg_waiting_time', 'waiting_time', 'mean_waiting_time']:
                    if key in pattern_results:
                        waiting_time = float(pattern_results[key])
                        logger.info(f"Found waiting time under key '{key}': {waiting_time}")
                        break
                        
                # Try different possible key names for throughput
                throughput = None
                for key in ['throughput', 'avg_throughput', 'mean_throughput']:
                    if key in pattern_results:
                        throughput = float(pattern_results[key])
                        logger.info(f"Found throughput under key '{key}': {throughput}")
                        break
                
                # Look for metrics in 'summary' if they exist
                if 'summary' in pattern_results:
                    summary = pattern_results['summary']
                    logger.info(f"Found summary data: {summary.keys() if isinstance(summary, dict) else 'Not a dict'}")
                    
                    if waiting_time is None and isinstance(summary, dict):
                        for key in ['avg_waiting_time', 'waiting_time', 'mean_waiting_time']:
                            if key in summary:
                                waiting_time = float(summary[key])
                                logger.info(f"Found waiting time in summary under key '{key}': {waiting_time}")
                                break
                                
                    if throughput is None and isinstance(summary, dict):
                        for key in ['throughput', 'avg_throughput', 'mean_throughput']:
                            if key in summary:
                                throughput = float(summary[key])
                                logger.info(f"Found throughput in summary under key '{key}': {throughput}")
                                break
                                
                # Check if we found the metrics in episodes
                if (waiting_time is None or throughput is None) and 'episodes' in pattern_results:
                    episodes = pattern_results['episodes']
                    logger.info(f"Looking for metrics in episodes data. Found {len(episodes) if isinstance(episodes, list) else 'Not a list'} episodes")
                    
                    if isinstance(episodes, list) and episodes:
                        total_episode_waiting_time = 0.0
                        total_episode_throughput = 0.0
                        episode_count = 0
                        
                        for episode in episodes:
                            if isinstance(episode, dict):
                                if waiting_time is None:
                                    for key in ['avg_waiting_time', 'waiting_time', 'mean_waiting_time']:
                                        if key in episode:
                                            total_episode_waiting_time += float(episode[key])
                                            episode_count += 1
                                            break
                                            
                                if throughput is None:
                                    for key in ['throughput', 'avg_throughput', 'mean_throughput']:
                                        if key in episode:
                                            total_episode_throughput += float(episode[key])
                                            break
                        
                        if episode_count > 0 and waiting_time is None:
                            waiting_time = total_episode_waiting_time / episode_count
                            logger.info(f"Calculated average waiting time from episodes: {waiting_time}")
                            
                        if episode_count > 0 and throughput is None:
                            throughput = total_episode_throughput / episode_count
                            logger.info(f"Calculated average throughput from episodes: {throughput}")
                
                # If we found the metrics, add them to the totals
                if waiting_time is not None:
                    total_waiting_time += waiting_time
                    pattern_count += 1
                else:
                    logger.warning(f"No waiting time found for {agent_name}/{pattern_name}")
                    
                if throughput is not None:
                    total_throughput += throughput
                else:
                    logger.warning(f"No throughput found for {agent_name}/{pattern_name}")
            else:
                logger.warning(f"Pattern results for {pattern_name} is not a dict")
        
        # Calculate average metrics
        avg_waiting_time = total_waiting_time / pattern_count if pattern_count > 0 else 0.0
        avg_throughput = total_throughput / pattern_count if pattern_count > 0 else 0.0
        
        logger.info(f"Calculated metrics for {agent_name}: waiting_time={avg_waiting_time:.2f}, throughput={avg_throughput:.2f}")
        
        processed_data[agent_name] = {
            'waiting_time': avg_waiting_time,
            'throughput': avg_throughput
        }
    
    # Log the metrics for each agent
    for agent_name, metrics in processed_data.items():
        logger.info(f"Agent {agent_name}: waiting_time={metrics['waiting_time']:.2f}, throughput={metrics['throughput']:.2f}")
    
    logger.info(f"Transformed data ready with {len(processed_data)} agents: {list(processed_data.keys())}")
    return processed_data

def simplified_comparative_analysis(agents_data, output_dir=None):
    """
    Run a simplified comparative analysis of multiple agents.
    
    Args:
        agents_data (dict): Dictionary mapping agent names to agent data.
        output_dir (str): Directory to save visualization results.
        
    Returns:
        dict: Analysis results with metrics by category.
    """
    logger.info(f"Running simplified comparative analysis with {len(agents_data)} agents")
    
    # Initialize results structure
    results = {
        'waiting_time': {},
        'throughput': {}
    }
    
    # Extract metrics for each agent
    for agent_name, agent_data in agents_data.items():
        # Extract/use default metrics
        waiting_time = agent_data.get('waiting_time', 10.0)
        throughput = agent_data.get('throughput', 100.0)
        
        # Store in results
        results['waiting_time'][agent_name] = waiting_time
        results['throughput'][agent_name] = throughput
    
    # Create visualizations if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a visualization of waiting times
        if results['waiting_time']:
            plt.figure(figsize=(10, 6))
            plt.bar(results['waiting_time'].keys(), results['waiting_time'].values())
            plt.title('Average Waiting Time by Agent')
            plt.ylabel('Waiting Time (s)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'waiting_time_comparison.png'))
            plt.close()
        
        # Create a visualization of throughput
        if results['throughput']:
            plt.figure(figsize=(10, 6))
            plt.bar(results['throughput'].keys(), results['throughput'].values())
            plt.title('Average Throughput by Agent')
            plt.ylabel('Throughput (vehicles/hour)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'))
            plt.close()
    
    logger.info("Simplified comparative analysis complete.")
    return results

def simplified_create_report(analysis_results, output_dir=None):
    """
    Create a simplified analysis report.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_dir: Directory to save the report
        
    Returns:
        Path to the report file
    """
    logger.info("Creating simplified analysis report...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "results/analysis/report"
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple HTML report
    report_path = os.path.join(output_dir, "analysis_report.html")
    
    with open(report_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Traffic RL Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
                h1, h2 { color: #2c3e50; }
                .section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Traffic RL Analysis Report</h1>
            <div class="section">
                <h2>Comparative Analysis</h2>
        """)
        
        # Add comparative analysis results if available
        if 'comparative' in analysis_results and 'metrics' in analysis_results['comparative']:
            metrics = analysis_results['comparative']['metrics']
            agents = analysis_results['comparative'].get('agents', [])
            
            if 'waiting_time' in metrics:
                f.write("<h3>Average Waiting Time</h3>")
                f.write("<table><tr><th>Agent</th><th>Avg. Waiting Time</th></tr>")
                for agent, value in metrics['waiting_time'].items():
                    f.write(f"<tr><td>{agent}</td><td>{value:.2f}</td></tr>")
                f.write("</table>")
            
            if 'throughput' in metrics:
                f.write("<h3>Average Throughput</h3>")
                f.write("<table><tr><th>Agent</th><th>Avg. Throughput</th></tr>")
                for agent, value in metrics['throughput'].items():
                    f.write(f"<tr><td>{agent}</td><td>{value:.2f}</td></tr>")
                f.write("</table>")
        else:
            f.write("<p>No comparative analysis results available.</p>")
        
        # Add training analysis results if available
        f.write("""
            </div>
            <div class="section">
                <h2>Training Analysis</h2>
        """)
        
        if 'training' in analysis_results:
            f.write("<p>Training analysis results available. See training plots for details.</p>")
        else:
            f.write("<p>No training analysis results available.</p>")
        
        # Close HTML
        f.write("""
            </div>
            <div class="section">
                <h2>Benchmark Results</h2>
                <p>See benchmark visualizations in the benchmark directory.</p>
            </div>
            <footer>
                <p>Report generated by Traffic RL Analysis Tool</p>
            </footer>
        </body>
        </html>
        """)
    
    logger.info(f"Simplified report created at {report_path}")
    return report_path

def load_json(metrics_file):
    """
    Load metrics data from a JSON file.
    
    Args:
        metrics_file: Path to the metrics JSON file
        
    Returns:
        Dictionary with metrics data or empty dict if file not found
    """
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load metrics from {metrics_file}: {e}")
        return {}

def simplified_analyze_training_metrics(training_data, output_dir=None):
    """
    A simplified version of training metrics analysis that's more resilient to missing data.
    
    Args:
        training_data: Dictionary with training metrics
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Running simplified training metrics analysis")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics, defaulting to empty lists
    rewards = training_data.get('rewards', [])
    losses = training_data.get('losses', [])
    epsilons = training_data.get('epsilons', [])
    waiting_times = training_data.get('waiting_times', [])
    throughput = training_data.get('throughput', [])
    
    # Create episodes range
    episodes = list(range(len(rewards))) if rewards else []
    
    # Prepare results
    results = {
        'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
        'max_reward': max(rewards) if rewards else 0,
        'min_reward': min(rewards) if rewards else 0,
        'avg_loss': sum(losses) / len(losses) if losses else 0,
        'num_episodes': len(episodes)
    }
    
    # Only create visualizations if we have data and output directory
    if rewards and output_dir:
        # Set up plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot rewards
        ax.plot(episodes, rewards, 'b-', label='Reward')
        
        # Add rolling average if we have enough data
        if len(rewards) > 10:
            window_size = min(len(rewards) // 10, 20)  # Adaptive window size
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            ax.plot(episodes, rolling_avg, 'r-', label=f'{window_size}-Episode Avg')
        
        # Add labels and legend
        ax.set_title('Training Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'training_rewards.png'))
        plt.close(fig)
        
        # Add more plots if we have the data
        if losses and len(losses) == len(episodes):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(episodes, losses, 'g-', label='Loss')
            ax.set_title('Training Loss')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'training_loss.png'))
            plt.close(fig)
    
    logger.info(f"Training analysis completed with {len(rewards)} episodes")
    return results

def run_comprehensive_analysis(model_paths, training_metrics_path=None, config=None, output_dir=None, open_browser=True, traffic_patterns=None):
    """
    Run a comprehensive analysis of the trained model's performance.
    
    Args:
        model_paths (str or list): Path(s) to the trained model(s).
        training_metrics_path (str or list): Path(s) to the training metrics.
        config (dict): Configuration for the analysis.
        output_dir (str): Directory to output the analysis results.
        open_browser (bool): Whether to open a browser window to view the results.
        traffic_patterns (list): List of traffic patterns to benchmark against.
        
    Returns:
        dict: Analysis results.
    """
    if output_dir is None:
        output_dir = "results/analysis"
    
    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for benchmark and training results
    benchmark_dir = os.path.join(output_dir, "benchmark")
    training_dir = os.path.join(output_dir, "training")
    os.makedirs(benchmark_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    
    # Create a report directory
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Load default config if not provided
    if config is None:
        config = {
            "simulation": {
                "steps": 500
            },
            "benchmarking": {
                "episodes": 10
            }
        }
    
    # Load training metrics if provided
    if training_metrics_path is not None:
        logger.info(f"Loading training metrics from {training_metrics_path}")
        try:
            if isinstance(training_metrics_path, list):
                training_metrics_path = training_metrics_path[0]  # Use the first path
            
            training_metrics = load_json(training_metrics_path)
            logger.info("Starting training metrics analysis...")
            simplified_analyze_training_metrics(training_metrics, output_dir=training_dir)
        except Exception as e:
            logger.error(f"Error analyzing training metrics: {e}")
            # Continue with benchmark analysis even if training analysis fails
    
    # Perform benchmarking
    logger.info("Starting comprehensive benchmark analysis...")
    
    # Determine the timestamped output directory for benchmark results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_output_dir = os.path.join(benchmark_dir, f"benchmark_{timestamp}")
    os.makedirs(benchmark_output_dir, exist_ok=True)
    
    # Create benchmark agents from model paths
    logger.info(f"Benchmarking agents against various traffic patterns...")
    
    # Ensure model_paths is properly handled
    if isinstance(model_paths, str):
        model_path = model_paths  # Single model path
    else:
        model_path = model_paths[0] if model_paths else None  # Use first model path if it's a list
    
    # Create benchmark agents
    agents_to_benchmark = create_benchmark_agents(config, model_path)
    
    # Set default traffic patterns if not provided
    if traffic_patterns is None:
        traffic_patterns = ["uniform", "rush_hour", "weekend"]
    
    # Run benchmarks
    benchmark_results = benchmark_agents(
        config=config,
        agents_to_benchmark=agents_to_benchmark,
        traffic_patterns=traffic_patterns,
        num_episodes=config.get("benchmarking", {}).get("episodes", 10),
        output_dir=benchmark_output_dir,
        create_visualizations=True
    )
    
    # Save enhanced visualizations
    logger.info(f"Enhanced visualizations saved to {benchmark_output_dir}")
    
    # Prepare data for comparative analysis
    logger.info("Running simplified comparative analysis...")
    analysis_data = prepare_benchmark_data_for_analysis(benchmark_results)
    
    # Run comparative analysis
    analysis_results = simplified_comparative_analysis(analysis_data, output_dir=report_dir)
    
    # Create an HTML report
    logger.info("Creating simplified analysis report...")
    report_path = os.path.join(report_dir, "analysis_report.html")
    create_analysis_report(analysis_results, report_path, training_metrics_path is not None)
    logger.info(f"Simplified report created at {report_path}")
    
    # Open the report in a browser if requested
    if open_browser:
        logger.info(f"Opening report in browser...")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
    
    logger.info(f"Comprehensive analysis completed. Results in {output_dir}")
    
    return {
        "benchmark_results": benchmark_results,
        "analysis_results": analysis_results,
        "report_path": report_path
    }

def create_analysis_report(analysis_results, output_path, has_training_data=False):
    """
    Create a simple HTML report with analysis results.
    
    Args:
        analysis_results (dict): Results from simplified_comparative_analysis
        output_path (str): Path to save the HTML report
        has_training_data (bool): Whether training data analysis is included
        
    Returns:
        str: Path to the created report
    """
    # Initialize the HTML template
    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Traffic RL Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Traffic RL Analysis Report</h1>
            <div class="section">
                <h2>Comparative Analysis</h2>
    """
    
    # Add waiting time table
    html += "<h3>Average Waiting Time</h3>"
    html += "<table><tr><th>Agent</th><th>Avg. Waiting Time</th></tr>"
    
    # Process waiting time data
    waiting_time_data = {}
    if 'waiting_time' in analysis_results and isinstance(analysis_results['waiting_time'], dict):
        waiting_time_data = analysis_results['waiting_time']
    
    for agent_name, waiting_time in waiting_time_data.items():
        html += f"<tr><td>{agent_name}</td><td>{waiting_time:.2f}</td></tr>"
    html += "</table>"
    
    # Add throughput table
    html += "<h3>Average Throughput</h3>"
    html += "<table><tr><th>Agent</th><th>Avg. Throughput</th></tr>"
    
    # Process throughput data
    throughput_data = {}
    if 'throughput' in analysis_results and isinstance(analysis_results['throughput'], dict):
        throughput_data = analysis_results['throughput']
    
    for agent_name, throughput in throughput_data.items():
        html += f"<tr><td>{agent_name}</td><td>{throughput:.2f}</td></tr>"
    html += "</table>"
    
    # Add training analysis section if available
    html += """
            </div>
            <div class="section">
                <h2>Training Analysis</h2>
    """
    
    if has_training_data:
        html += "<p>Training analysis results available. See training plots for details.</p>"
    else:
        html += "<p>No training data provided for analysis.</p>"
    
    # Add benchmark section
    html += """
            </div>
            <div class="section">
                <h2>Benchmark Results</h2>
                <p>See benchmark visualizations in the benchmark directory.</p>
            </div>
            <footer>
                <p>Generated by Traffic RL Analysis Tool</p>
            </footer>
        </body>
        </html>
    """
    
    # Write the HTML to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path 