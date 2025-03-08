"""
Analysis Script
=============
Script for generating comprehensive analysis and comparison of different agents.
"""

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import modules
from traffic_rl.config import load_config
from traffic_rl.utils.logging import setup_logging
from traffic_rl.utils.analysis import (
    analyze_training_metrics, 
    comparative_analysis, 
    analyze_decision_boundaries,
    create_comprehensive_report
)
from traffic_rl.utils.benchmark import benchmark_agents, create_benchmark_agents

logger = logging.getLogger("Analyze")

def run_comprehensive_analysis(
    config, 
    model_paths, 
    training_metrics=None, 
    benchmark_dir=None,
    output_dir="results/analysis",
    traffic_patterns=None,
    num_episodes=50
):
    """
    Run comprehensive analysis across multiple models and baselines.
    
    Args:
        config: Configuration dictionary
        model_paths: List of paths to trained models
        training_metrics: List of paths to training metrics (optional)
        benchmark_dir: Directory with existing benchmark results (optional)
        output_dir: Directory to save analysis results
        traffic_patterns: List of traffic patterns to analyze
        num_episodes: Number of episodes for any new benchmarks
        
    Returns:
        Path to the comprehensive report
    """
    try:
        # Create timestamp for this analysis run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_id = f"analysis_{timestamp}"
        
        # Create output directory
        analysis_dir = os.path.join(output_dir, analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Initialize report data
        report_data = {
            "training": {},
            "benchmark": {},
            "agent_analysis": {},
            "evaluation": {}
        }
        
        # Set default traffic patterns if not provided
        if traffic_patterns is None:
            traffic_patterns = ["uniform", "rush_hour", "weekend"]
        
        # 1. Analyze training metrics if provided
        if training_metrics:
            logger.info("Analyzing training metrics...")
            for i, metrics_file in enumerate(training_metrics):
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Get model label from file path
                    model_label = os.path.basename(os.path.dirname(metrics_file))
                    
                    # Analyze metrics
                    training_dir = os.path.join(analysis_dir, f"training_{model_label}")
                    os.makedirs(training_dir, exist_ok=True)
                    
                    training_summary = analyze_training_metrics(metrics, save_dir=training_dir)
                    report_data["training"][model_label] = {
                        "metrics_file": metrics_file,
                        "summary": training_summary,
                        "output_dir": training_dir
                    }
                    
                    logger.info(f"Training analysis for {model_label} completed.")
        
        # 2. Run benchmark or use existing benchmark results
        if benchmark_dir and os.path.exists(benchmark_dir):
            # Use existing benchmark results
            logger.info(f"Using existing benchmark results from {benchmark_dir}...")
            
            # Find benchmark results file
            results_file = os.path.join(benchmark_dir, "benchmark_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    benchmark_results = json.load(f)
                
                # Copy results to our analysis directory
                benchmark_output = os.path.join(analysis_dir, "benchmark")
                os.makedirs(benchmark_output, exist_ok=True)
                
                # Save results
                with open(os.path.join(benchmark_output, "benchmark_results.json"), 'w') as f:
                    json.dump(benchmark_results, f, indent=4)
                
                # Copy any existing plots
                plot_files = [f for f in os.listdir(benchmark_dir) if f.endswith(('.png', '.jpg'))]
                for plot_file in plot_files:
                    import shutil
                    src_path = os.path.join(benchmark_dir, plot_file)
                    dest_path = os.path.join(benchmark_output, plot_file)
                    shutil.copy2(src_path, dest_path)
                
                # Add to report data
                report_data["benchmark"]["existing"] = {
                    "results_file": results_file,
                    "output_dir": benchmark_output
                }
                
                logger.info("Existing benchmark results processed.")
            else:
                logger.warning(f"No benchmark results found in {benchmark_dir}")
                benchmark_results = None
        else:
            # Run new benchmark
            logger.info("Running new benchmark across models and baselines...")
            
            # Prepare benchmark output directory
            benchmark_output = os.path.join(analysis_dir, "benchmark")
            os.makedirs(benchmark_output, exist_ok=True)
            
            # Create agents dictionary for benchmarking
            agents = {}
            
            # Add baseline agents
            baseline_agents = create_benchmark_agents(config)
            agents.update(baseline_agents)
            
            # Add trained models
            for i, model_path in enumerate(model_paths):
                if os.path.exists(model_path):
                    # Extract model name from path
                    agents[f"TrainedModel"] = model_path
            
            # Run benchmark
            if agents:
                benchmark_results = benchmark_agents(
                    config=config,
                    agents_to_benchmark=agents,
                    traffic_patterns=traffic_patterns,
                    num_episodes=num_episodes,
                    output_dir=benchmark_output,
                    create_visualizations=True
                )
                
                # Add to report data
                report_data["benchmark"]["new"] = {
                    "results": benchmark_results,
                    "output_dir": benchmark_output
                }
                
                logger.info("Benchmark completed.")
            else:
                logger.warning("No agents available for benchmarking")
                benchmark_results = None
        
        # 3. Handle benchmark results - avoid recreating visualizations
        if benchmark_results:
            logger.info("Processing benchmark results...")
            
            # Get results section from benchmark results
            if "results" in benchmark_results:
                benchmark_data = benchmark_results["results"]
            else:
                benchmark_data = benchmark_results
            
            # Check if we're using existing benchmark results
            using_existing_benchmark = benchmark_dir and os.path.exists(benchmark_dir)
            
            # Always run comparative analysis for consistency
            logger.info("Running comparative analysis...")
            
            # Run comparative analysis
            comparative_output = os.path.join(analysis_dir, "comparative")
            os.makedirs(comparative_output, exist_ok=True)
            
            comparative_results = comparative_analysis(benchmark_data, save_dir=comparative_output)
            
            # Add to report data
            report_data["comparative"] = {
                "results": comparative_results,
                "output_dir": comparative_output
            }
            
            logger.info("Comparative analysis completed.")
        
        # 4. Analyze agent behavior for each model
        if model_paths:
            logger.info("Analyzing agent behavior...")
            
            for i, model_path in enumerate(model_paths):
                if os.path.exists(model_path):
                    # Extract model name from path
                    model_name = os.path.basename(os.path.dirname(model_path))
                    print("Model name: ", model_name)
                    
                    # Create output directory for agent analysis
                    agent_output = os.path.join(analysis_dir, f"agent_{model_name}")
                    os.makedirs(agent_output, exist_ok=True)
                    
                    # Generate agent analysis data including Q-value visualization
                    from traffic_rl.agents.dqn_agent import DQNAgent
                    from traffic_rl.environment.traffic_simulation import TrafficSimulation
                    from traffic_rl.utils.analysis import visualize_learning_progress
                    
                    # Initialize environment to get state and action sizes
                    env = TrafficSimulation(config=config, visualization=False)
                    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                    action_size = env.action_space.n
                    
                    # Load the agent
                    agent = DQNAgent(state_size, action_size, config)
                    if agent.load(model_path):
                        logger.info(f"Successfully loaded model from {model_path} for Q-value analysis")
                        
                        # Generate sample states to evaluate Q-values
                        q_values_history = []
                        states_history = []
                        
                        # Create a grid of states with different traffic densities
                        grid_size = 10
                        ns_densities = np.linspace(0, 1, grid_size)
                        ew_densities = np.linspace(0, 1, grid_size)
                        
                        # Create states for both light states (NS green and EW green)
                        light_states = [0, 1]  # 0=NS Green, 1=EW Green
                        
                        # Create a more comprehensive set of states
                        all_states = []
                        all_q_values = []
                        
                        for light_state in light_states:
                            # Create a 2D grid of states for this light state
                            states_grid = np.zeros((grid_size, grid_size, state_size))
                            q_values_grid = np.zeros((grid_size, grid_size, action_size))
                            
                            for i, ns in enumerate(ns_densities):
                                for j, ew in enumerate(ew_densities):
                                    # Create a more realistic state representation
                                    state = np.zeros(state_size)
                                    
                                    # First intersection state (we focus on one intersection)
                                    # [NS_density, EW_density, light_state, NS_waiting, EW_waiting]
                                    state[0] = ns  # North-South density
                                    state[1] = ew  # East-West density
                                    state[2] = light_state  # Current light state
                                    
                                    # Set waiting times proportional to density and light state
                                    if light_state == 0:  # NS Green
                                        state[3] = 0.1 * ns  # Low NS waiting (green light)
                                        state[4] = 0.5 * ew  # Higher EW waiting (red light)
                                    else:  # EW Green
                                        state[3] = 0.5 * ns  # Higher NS waiting (red light)
                                        state[4] = 0.1 * ew  # Low EW waiting (green light)
                                    
                                    # Store the state
                                    states_grid[i, j] = state
                                    
                                    # Get Q-values from the agent
                                    q_value = agent.get_q_values(state)
                                    q_values_grid[i, j] = q_value
                            
                            all_states.append(states_grid)
                            all_q_values.append(q_values_grid)
                        
                        # Add to history
                        states_history.append(all_states)
                        q_values_history.append(all_q_values)
                        
                        # Visualize learning progress with Q-values
                        q_viz_dir = os.path.join(agent_output, "q_values")
                        os.makedirs(q_viz_dir, exist_ok=True)
                        visualize_learning_progress(q_values_history, states_history, save_dir=q_viz_dir)
                        logger.info(f"Q-value visualization saved to {q_viz_dir}")
                    
                    # Create agent analysis data
                    agent_analysis = {
                        "model_path": model_path,
                        "model_name": model_name,
                        "decision_boundaries": {},  # Would be populated by actual analysis
                        "q_values_dir": os.path.join(agent_output, "q_values") if hasattr(agent, 'local_network') else None
                    }
                    
                    # Save agent analysis
                    with open(os.path.join(agent_output, "agent_analysis.json"), 'w') as f:
                        json.dump(agent_analysis, f, indent=4)
                    
                    # Add to report data
                    report_data["agent_analysis"][model_name] = {
                        "analysis": agent_analysis,
                        "output_dir": agent_output
                    }
                    
                    logger.info(f"Agent analysis for {model_name} completed.")
        
        # 5. Generate comprehensive report
        logger.info("Generating comprehensive report...")
        
        # Create report directory
        report_dir = os.path.join(analysis_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Select data for report
        training_data = None
        if report_data["training"]:
            # Use first training dataset
            first_key = next(iter(report_data["training"]))
            metrics_file = report_data["training"][first_key].get("metrics_file")
            if metrics_file and os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    training_data = json.load(f)
        
        benchmark_data = None
        if benchmark_results and "results" in benchmark_results:
            benchmark_data = benchmark_results["results"]
        elif benchmark_results:
            benchmark_data = benchmark_results
        else:
            benchmark_data = None
        
        agent_data = None
        if report_data["agent_analysis"]:
            # Use first agent analysis
            first_key = next(iter(report_data["agent_analysis"]))
            agent_data = report_data["agent_analysis"][first_key].get("analysis")
        
        # Generate report
        report_path = create_comprehensive_report(
            training_metrics=training_data,
            benchmark_results=benchmark_data,
            agent_analysis=agent_data,
            save_dir=report_dir
        )
        
        # Save report data
        with open(os.path.join(analysis_dir, "analysis_metadata.json"), 'w') as f:
            # Convert to serializable format
            serializable_data = {}
            for key, value in report_data.items():
                if isinstance(value, dict):
                    serializable_data[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            serializable_data[key][k] = {
                                key2: value2 for key2, value2 in v.items() 
                                if not isinstance(value2, (pd.DataFrame, np.ndarray))
                            }
                        else:
                            serializable_data[key][k] = v
                else:
                    serializable_data[key] = value
            
            json.dump(serializable_data, f, indent=4)
        
        logger.info(f"Comprehensive analysis completed. Results in {analysis_dir}")
        
        return analysis_dir
    
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic Control RL Comprehensive Analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--models", type=str, nargs='+', default=[], help="Paths to trained models")
    parser.add_argument("--training-metrics", type=str, nargs='+', default=[], help="Paths to training metrics files")
    parser.add_argument("--benchmark-dir", type=str, default=None, help="Directory with existing benchmark results")
    parser.add_argument("--output", type=str, default="results/analysis", help="Output directory")
    parser.add_argument("--patterns", type=str, default="uniform,rush_hour,weekend", 
                        help="Comma-separated list of traffic patterns to analyze")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for benchmarking")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse traffic patterns
    patterns = args.patterns.split(',')
    
    # Run comprehensive analysis
    analysis_dir = run_comprehensive_analysis(
        config=config,
        model_paths=args.models,
        training_metrics=args.training_metrics,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output,
        traffic_patterns=patterns,
        num_episodes=args.episodes
    )
    
    if analysis_dir:
        logger.info(f"Analysis completed. Results available in {analysis_dir}")
        
        # Find HTML report
        report_path = os.path.join(analysis_dir, "report", "analysis_report.html")
        if os.path.exists(report_path):
            logger.info(f"HTML report available at: {report_path}")
            
            # Try to open the report in a browser
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
            except Exception as e:
                logger.warning(f"Could not open report in browser: {e}")
    else:
        logger.error("Analysis failed.")
