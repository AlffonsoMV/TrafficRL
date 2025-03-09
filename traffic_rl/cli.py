"""
Command Line Interface
====================
Unified command-line interface for the Traffic RL package.

This module provides a consolidated CLI with subcommands for all major functions:
- train: Train a reinforcement learning agent
- evaluate: Evaluate a trained agent
- visualize: Create visualizations of the environment and agent performance
- benchmark: Compare multiple agents across different traffic patterns
- analyze: Perform in-depth analysis of agent behavior and performance
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import random
import torch
from datetime import datetime

from traffic_rl.config import load_config, save_config, override_config_with_args
from traffic_rl.train import train
from traffic_rl.evaluate import evaluate_agent
from traffic_rl.utils.benchmark import benchmark_agents, create_benchmark_agents
from traffic_rl.utils.logging import setup_logging
from traffic_rl.utils.environment import create_environment
from traffic_rl.utils.visualization import (
    visualize_results, 
    visualize_traffic_patterns,
    save_visualization
)
from traffic_rl.utils.analysis import (
    analyze_training_metrics, 
    comparative_analysis, 
    analyze_decision_boundaries,
    create_comprehensive_report
)
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.environment.roundabout_simulation import RoundaboutSimulation
from traffic_rl.agents.dqn_agent import DQNAgent


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        return True
    return False


def train_command(args, logger):
    """Run the training command."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Set random seed if provided
    if set_random_seed(config.get("random_seed")):
        logger.info(f"Random seed set to {config.get('random_seed')}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output, "config.json")
    save_config(config, config_path)
    
    # Train the agent
    metrics = train(config, model_dir=args.output, env_type=args.env_type)
    
    # Save metrics
    try:
        metrics_path = os.path.join(args.output, "training_metrics.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [float(v) for v in value]
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Training metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save training metrics: {e}")
    
    # Print summary
    logger.info("Training Summary:")
    logger.info(f"  Episodes Completed: {len(metrics['rewards'])}")
    logger.info(f"  Final Average Reward: {metrics['avg_rewards'][-1] if metrics['avg_rewards'] else 'N/A'}")
    logger.info(f"  Best Model Path: {os.path.join(args.output, 'best_model.pth')}")
    
    return True


def evaluate_command(args, logger):
    """Run the evaluation command."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Set random seed if provided
    if set_random_seed(config.get("random_seed")):
        logger.info(f"Random seed set to {config.get('random_seed')}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Validate model path
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return False
    
    # Parse traffic patterns
    patterns = [p.strip() for p in args.patterns.split(',')]
    
    # Evaluate on each pattern
    results = {}
    for pattern in patterns:
        logger.info(f"Evaluating on {pattern} traffic pattern...")
        
        result = evaluate_agent(
            config=config,
            model_path=args.model,
            traffic_pattern=pattern,
            num_episodes=args.episodes,
            env_type=args.env_type
        )
        
        results[pattern] = result
        
        # Print summary
        logger.info(f"Evaluation results for {pattern}:")
        logger.info(f"  Average Reward: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
        logger.info(f"  Average Waiting Time: {result['avg_waiting_time']:.2f}")
        logger.info(f"  Average Throughput: {result['avg_throughput']:.2f}")
    
    # Save results
    results_path = os.path.join(args.output, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    return True


def visualize_command(args, logger):
    """Run the visualization command."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Set random seed if provided
    if set_random_seed(config.get("random_seed")):
        logger.info(f"Random seed set to {config.get('random_seed')}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine what to visualize
    if args.type == "environment":
        # Validate model path if provided
        if args.model and not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            return False
        
        # Initialize environment
        env = create_environment(
            config=config,
            visualization=False,  # We'll use custom visualization
            random_seed=config.get("random_seed"),
            env_type=args.env_type
        )
        
        # Set traffic pattern
        pattern = args.pattern
        if pattern in config["traffic_patterns"]:
            env.traffic_pattern = pattern
            env.traffic_config = config["traffic_patterns"][pattern]
        else:
            logger.warning(f"Traffic pattern {pattern} not found, using uniform")
            env.traffic_pattern = "uniform"
            env.traffic_config = config["traffic_patterns"]["uniform"]
        
        # Initialize agent if model provided
        if args.model:
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize agent
            agent = DQNAgent(state_size, action_size, config)
            
            # Load model
            if agent.load(args.model):
                logger.info(f"Model loaded from {args.model}")
                env.recording_agent = agent
            else:
                logger.warning(f"Failed to load model, using random actions")
                env.recording_agent = None
        else:
            logger.info("No model provided, using random actions")
            env.recording_agent = None
        
        # Determine video filename
        if args.filename:
            video_path = os.path.join(args.output, args.filename)
        else:
            agent_type = "trained" if args.model else "random"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(
                args.output, 
                f"traffic_sim_{pattern}_{agent_type}_{timestamp}.mp4"
            )
        
        # Create visualization
        logger.info(f"Creating environment visualization: {video_path}")
        success = save_visualization(
            env=env,
            filename=video_path,
            fps=args.fps,
            duration=args.duration
        )
        
        # Close environment
        env.close()
        
        if success:
            logger.info(f"Visualization saved to {video_path}")
            return True
        else:
            logger.error("Failed to create visualization")
            return False
            
    elif args.type == "metrics":
        # Validate metrics file
        if not args.metrics:
            logger.error("No metrics file provided")
            return False
        if not os.path.exists(args.metrics):
            logger.error(f"Metrics file not found: {args.metrics}")
            return False
        
        # Load metrics
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
        
        # Visualize rewards
        if "rewards" in metrics and "avg_rewards" in metrics:
            rewards_path = os.path.join(args.output, "rewards_plot.png")
            visualize_results(
                metrics["rewards"], 
                metrics["avg_rewards"], 
                save_path=rewards_path
            )
            logger.info(f"Rewards plot saved to {rewards_path}")
        else:
            logger.warning("Metrics file does not contain reward data")
        
        return True
        
    elif args.type == "patterns":
        # Determine output path
        patterns_file = os.path.join(args.output, "traffic_patterns.png")
        
        # Use the consolidated visualization function
        result = visualize_traffic_patterns(config, save_path=patterns_file)
        
        if result:
            logger.info(f"Traffic patterns visualization saved to {patterns_file}")
            return True
        else:
            logger.error("Failed to visualize traffic patterns")
            return False
        
    else:
        logger.error(f"Unknown visualization type: {args.type}")
        return False


def benchmark_command(args, logger):
    """Run the benchmark command."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Set random seed if provided
    if set_random_seed(config.get("random_seed")):
        logger.info(f"Random seed set to {config.get('random_seed')}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Parse agent types
    agent_types = [a.strip() for a in args.agents.split(',')]
    
    # Parse traffic patterns
    patterns = [p.strip() for p in args.patterns.split(',')]
    
    # Create agents
    agents = create_benchmark_agents(
        agent_types=agent_types,
        config=config,
        model_path=args.model
    )
    
    # Run benchmark
    results = benchmark_agents(
        agents=agents,
        config=config,
        patterns=patterns,
        episodes=args.episodes,
        env_type=args.env_type
    )
    
    # Save results
    results_path = os.path.join(args.output, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Benchmark results saved to {results_path}")
    
    return True


def analyze_command(args, logger):
    """Run the analysis command."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Set random seed if provided
    if set_random_seed(config.get("random_seed")):
        logger.info(f"Random seed set to {config.get('random_seed')}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Validate model path
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return False
    
    # Parse traffic patterns
    if args.patterns:
        patterns = [p.strip() for p in args.patterns.split(',')]
    else:
        patterns = ["uniform", "rush_hour", "weekend"]
    
    # Add env_type to config
    config["env_type"] = args.env_type
    
    # Check if we need to run benchmarks first
    if args.benchmark_dir and os.path.exists(args.benchmark_dir):
        logger.info(f"Using existing benchmark results from {args.benchmark_dir}")
        benchmark_results_path = os.path.join(args.benchmark_dir, "benchmark_results.json")
        
        if not os.path.exists(benchmark_results_path):
            logger.error(f"Benchmark results file not found: {benchmark_results_path}")
            return False
        
        with open(benchmark_results_path, 'r') as f:
            benchmark_results = json.load(f)
    else:
        logger.info("Running benchmarks as part of analysis...")
        
        # Create benchmark agents
        agents = create_benchmark_agents(
            agent_types=["dqn", "fixed", "random"],
            config=config,
            model_path=args.model
        )
        
        # Run benchmark
        benchmark_results = benchmark_agents(
            agents=agents,
            config=config,
            patterns=patterns,
            episodes=args.episodes,
            env_type=args.env_type
        )
        
        # Save benchmark results
        benchmark_results_path = os.path.join(args.output, "benchmark_results.json")
        with open(benchmark_results_path, 'w') as f:
            json.dump(benchmark_results, f, indent=4)
        
        logger.info(f"Benchmark results saved to {benchmark_results_path}")
    
    # Run analysis
    logger.info("Running comprehensive analysis...")
    
    # Create comprehensive report
    report = create_comprehensive_report(
        config=config,
        model_path=args.model,
        benchmark_results=benchmark_results,
        output_dir=os.path.join(args.output, "report"),
        env_type=args.env_type
    )
    
    logger.info(f"Analysis complete. Report saved to {os.path.join(args.output, 'report')}")
    
    return True


def add_common_args(parser):
    """Add common arguments to a parser."""
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--visualization", action="store_true", help="Enable visualization")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    return parser


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Traffic RL - Reinforcement Learning for Traffic Light Control",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Only add log-level to the main parser
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a reinforcement learning agent")
    train_parser.add_argument("--config", type=str, help="Path to configuration file")
    train_parser.add_argument("--output", type=str, default="results/training", help="Output directory")
    train_parser.add_argument("--visualization", action="store_true", help="Enable visualization")
    train_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    train_parser.add_argument("--episodes", type=int, help="Number of training episodes")
    train_parser.add_argument("--no-visualization", dest="visualization", action="store_false", help="Disable visualization during training")
    train_parser.add_argument("--no-plots", action="store_true", help="Disable plotting of training results")
    train_parser.add_argument("--env-type", type=str, default="grid", choices=["grid", "roundabout"], 
                        help="Type of environment (grid or roundabout)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument("--config", type=str, help="Path to configuration file")
    eval_parser.add_argument("--output", type=str, default="results/evaluation", help="Output directory")
    eval_parser.add_argument("--visualization", action="store_true", help="Enable visualization")
    eval_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    eval_parser.add_argument("--model", type=str, required=True,
                            help="Path to model file")
    eval_parser.add_argument("--episodes", type=int, default=10,
                            help="Number of evaluation episodes")
    eval_parser.add_argument("--patterns", type=str, default="uniform",
                            help="Comma-separated list of traffic patterns to evaluate")
    eval_parser.add_argument("--no-visualization", dest="visualization", action="store_false", help="Disable visualization during evaluation")
    eval_parser.add_argument("--env-type", type=str, default="grid", choices=["grid", "roundabout"], 
                        help="Type of environment (grid or roundabout)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create visualizations")
    viz_parser.add_argument("--config", type=str, help="Path to configuration file")
    viz_parser.add_argument("--output", type=str, default="results/visualizations", help="Output directory")
    viz_parser.add_argument("--visualization", action="store_true", help="Enable visualization")
    viz_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    viz_parser.add_argument("--type", type=str, default="environment",
                            choices=["environment", "metrics", "patterns"],
                            help="Type of visualization to create")
    viz_parser.add_argument("--model", type=str, default=None,
                            help="Path to model file (for environment visualization)")
    viz_parser.add_argument("--duration", type=int, default=60,
                            help="Duration of visualization in seconds")
    viz_parser.add_argument("--pattern", type=str, default="uniform",
                            help="Traffic pattern to visualize")
    viz_parser.add_argument("--env-type", type=str, default="grid", choices=["grid", "roundabout"], 
                        help="Type of environment (grid or roundabout)")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark multiple agents")
    bench_parser.add_argument("--config", type=str, help="Path to configuration file")
    bench_parser.add_argument("--output", type=str, default="results/benchmark", help="Output directory")
    bench_parser.add_argument("--visualization", action="store_true", help="Enable visualization")
    bench_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    bench_parser.add_argument("--agents", type=str, default="dqn,fixed,random",
                               help="Comma-separated list of agents to benchmark")
    bench_parser.add_argument("--model", type=str, default=None,
                               help="Path to trained model (for DQN agent)")
    bench_parser.add_argument("--episodes", type=int, default=10,
                               help="Number of episodes per benchmark")
    bench_parser.add_argument("--patterns", type=str, default="uniform,rush_hour,weekend",
                               help="Comma-separated list of traffic patterns to benchmark")
    bench_parser.add_argument("--env-type", type=str, default="grid", choices=["grid", "roundabout"], 
                        help="Type of environment (grid or roundabout)")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze agent performance")
    analyze_parser.add_argument("--config", type=str, help="Path to configuration file")
    analyze_parser.add_argument("--output", type=str, default="results/analysis", help="Output directory")
    analyze_parser.add_argument("--visualization", action="store_true", help="Enable visualization")
    analyze_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    analyze_parser.add_argument("--model", type=str, required=True,
                                help="Path to trained model")
    analyze_parser.add_argument("--episodes", type=int, default=10,
                                help="Number of episodes for analysis")
    analyze_parser.add_argument("--benchmark-dir", type=str, default=None,
                                help="Directory with existing benchmark results")
    analyze_parser.add_argument("--patterns", type=str, default=None,
                                help="Comma-separated list of traffic patterns to analyze")
    analyze_parser.add_argument("--env-type", type=str, default="grid", choices=["grid", "roundabout"], 
                        help="Type of environment (grid or roundabout)")
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.INFO  # Default to INFO level
    if hasattr(args, 'log_level') and args.log_level:
        log_level = getattr(logging, args.log_level.upper())
    
    logger = setup_logging(log_level)
    
    # Check if a command was specified
    if not args.command:
        logger.error("No command specified. Use --help for usage information.")
        return 1
    
    # Set random seed if provided
    if hasattr(args, 'seed') and args.seed is not None:
        set_random_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Run the appropriate command
    try:
        if args.command == "train":
            success = train_command(args, logger)
        elif args.command == "evaluate":
            success = evaluate_command(args, logger)
        elif args.command == "visualize":
            success = visualize_command(args, logger)
        elif args.command == "benchmark":
            success = benchmark_command(args, logger)
        elif args.command == "analyze":
            success = analyze_command(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
