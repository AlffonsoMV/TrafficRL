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
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.agents.ppo_agent import PPOAgent
from traffic_rl.agents.simple_dqn_agent import SimpleDQNAgent
from traffic_rl.agents.entity_dqn_agent import EntityDQNAgent


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
    if args.episodes is not None:
        config["num_episodes"] = args.episodes
        logger.info(f"Setting number of episodes to {args.episodes} (from command line)")
    
    # Set visualization flag if provided
    if args.visualization is not None:
        config["visualization"] = args.visualization
        logger.info(f"Setting visualization to {args.visualization} (from command line)")
    
    # Set random seed if provided
    if set_random_seed(config.get("random_seed")):
        logger.info(f"Random seed set to {config.get('random_seed')}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output, "training_config.json")
    save_config(config, config_path)
    
    # Train agent
    logger.info(f"Starting training with {args.agent_type} agent...")
    metrics = train(config, model_dir=args.output, agent_type=args.agent_type)
    
    # Visualize results
    if not args.no_plots:
        logger.info("Training complete, visualizing results...")
        visualize_results(
            metrics["rewards"], 
            metrics["avg_rewards"], 
            save_path=os.path.join(args.output, "training_progress.png")
        )
    
    # Save metrics to file
    metrics_path = os.path.join(args.output, "training_metrics.json")
    try:
        # Convert numpy values to native types
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
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
    logger.info(f"  Agent Type: {args.agent_type}")
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
            agent_type=args.agent_type
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
        env = TrafficSimulation(
            config=config,
            visualization=False,  # We'll use custom visualization
            random_seed=config.get("random_seed")
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
            
            # Initialize agent based on agent_type
            if args.agent_type == "dqn":
                agent = DQNAgent(state_size, action_size, config)
            elif args.agent_type == "simple_dqn":
                agent = SimpleDQNAgent(state_size, action_size, config)
            elif args.agent_type == "entity_dqn":
                agent = EntityDQNAgent(state_size, action_size, config)
            elif args.agent_type == "ppo":
                agent = PPOAgent(state_size, action_size, config)
            else:
                agent = DQNAgent(state_size, action_size, config)  # Default to DQN
            
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
    
    # Parse traffic patterns
    patterns = [p.strip() for p in args.patterns.split(',')]
    
    # Parse model paths
    model_paths = []
    if args.model:
        model_paths = args.model if isinstance(args.model, list) else [args.model]
    
    # Create benchmark agents
    agents = create_benchmark_agents(config, model_paths)
    
    # Run benchmark
    results = benchmark_agents(
        config=config,
        agents_to_benchmark=agents,
        traffic_patterns=patterns,
        num_episodes=args.episodes,
        output_dir=args.output,
        create_visualizations=not args.no_visualization
    )
    
    # Print summary
    logger.info("Benchmark Summary:")
    for key, value in results["results"].items():
        if isinstance(value, dict) and "avg_reward" in value:
            logger.info(f"  {key}: Reward={value['avg_reward']:.2f}, Waiting={value['avg_waiting_time']:.2f}")
    
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
    
    # Parse model paths
    model_paths = []
    if args.model:
        model_paths = args.model if isinstance(args.model, list) else [args.model]
    
    # Parse training metrics paths
    training_metrics = []
    if args.metrics:
        training_metrics = args.metrics if isinstance(args.metrics, list) else [args.metrics]
    
    # Parse traffic patterns
    if args.patterns:
        traffic_patterns = [p.strip() for p in args.patterns.split(',')]
    else:
        traffic_patterns = ["uniform", "rush_hour", "weekend"]
    
    # Run comprehensive analysis
    from traffic_rl.analyze import run_comprehensive_analysis
    
    analysis_dir = run_comprehensive_analysis(
        config=config,
        model_paths=model_paths,
        training_metrics=training_metrics,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output,
        traffic_patterns=traffic_patterns,
        num_episodes=args.episodes
    )
    
    if analysis_dir:
        logger.info(f"Analysis completed. Results available in {analysis_dir}")
        
        # Find HTML report
        report_path = os.path.join(analysis_dir, "report", "analysis_report.html")
        if os.path.exists(report_path):
            logger.info(f"HTML report available at: {report_path}")
            
            # Try to open the report in a browser
            if not args.no_browser:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(report_path)}")
                except Exception as e:
                    logger.warning(f"Could not open report in browser: {e}")
        
        return True
    else:
        logger.error("Analysis failed.")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Traffic RL - Reinforcement Learning for Traffic Light Control",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Common arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a reinforcement learning agent")
    train_parser.add_argument("--config", type=str, help="Path to configuration file")
    train_parser.add_argument("--episodes", type=int, help="Number of training episodes")
    train_parser.add_argument("--output", type=str, default="results/training", help="Output directory")
    train_parser.add_argument("--no-visualization", dest="visualization", action="store_false", help="Disable visualization during training")
    train_parser.add_argument("--visualization", dest="visualization", action="store_true", help="Enable visualization during training")
    train_parser.set_defaults(visualization=None)  # Default to None to use config value
    train_parser.add_argument("--no-plots", action="store_true", help="Disable plotting of training results")
    train_parser.add_argument("--agent-type", type=str, 
                            choices=["dqn", "simple_dqn", "entity_dqn", "ppo"],
                            default="dqn", help="Type of agent to train (dqn, simple_dqn, entity_dqn, or ppo)")
    
    # PPO-specific arguments
    train_parser.add_argument("--ppo-learning-rate", type=float,
                            help="PPO learning rate")
    train_parser.add_argument("--ppo-gamma", type=float,
                            help="PPO discount factor")
    train_parser.add_argument("--ppo-gae-lambda", type=float,
                            help="PPO GAE lambda parameter")
    train_parser.add_argument("--ppo-clip-epsilon", type=float,
                            help="PPO clipping parameter")
    train_parser.add_argument("--ppo-c1", type=float,
                            help="PPO value loss coefficient")
    train_parser.add_argument("--ppo-c2", type=float,
                            help="PPO entropy coefficient")
    train_parser.add_argument("--ppo-batch-size", type=int,
                            help="PPO batch size")
    train_parser.add_argument("--ppo-n-epochs", type=int,
                            help="Number of epochs for PPO updates")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument("--model", type=str, required=True,
                            help="Path to model file")
    eval_parser.add_argument("--output", type=str, default="results/evaluation",
                            help="Directory to save evaluation results")
    eval_parser.add_argument("--episodes", type=int, default=10,
                            help="Number of evaluation episodes")
    eval_parser.add_argument("--patterns", type=str, default="uniform",
                            help="Comma-separated list of traffic patterns to evaluate")
    eval_parser.add_argument("--no-visualization", dest="visualization", action="store_false", help="Disable visualization during evaluation")
    eval_parser.add_argument("--visualization", dest="visualization", action="store_true", help="Enable visualization during evaluation")
    eval_parser.set_defaults(visualization=None)  # Default to None to use config value
    eval_parser.add_argument("--agent-type", type=str, 
                          choices=["dqn", "simple_dqn", "entity_dqn", "ppo"],
                          default="dqn", help="Type of agent to evaluate (dqn, simple_dqn, entity_dqn, or ppo)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create visualizations")
    viz_parser.add_argument("--type", type=str, required=True,
                           choices=["environment", "metrics", "patterns"],
                           help="Type of visualization to create")
    viz_parser.add_argument("--output", type=str, default="results/visualizations",
                           help="Directory to save visualizations")
    viz_parser.add_argument("--model", type=str, default=None,
                           help="Path to model file (for environment visualization)")
    viz_parser.add_argument("--pattern", type=str, default="uniform",
                           help="Traffic pattern to visualize")
    viz_parser.add_argument("--duration", type=int, default=30,
                           help="Duration of video in seconds")
    viz_parser.add_argument("--fps", type=int, default=30,
                           help="Frames per second for video")
    viz_parser.add_argument("--filename", type=str, default=None,
                           help="Output filename")
    viz_parser.add_argument("--metrics", type=str, default=None,
                           help="Path to metrics file (for metrics visualization)")
    viz_parser.add_argument("--agent-type", type=str, 
                           choices=["dqn", "simple_dqn", "entity_dqn", "ppo"],
                           default="dqn", help="Type of agent to visualize (dqn, simple_dqn, entity_dqn, or ppo)")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark multiple agents")
    bench_parser.add_argument("--model", type=str, default=None,
                             help="Path to trained model (for DQN agent)")
    bench_parser.add_argument("--output", type=str, default="results/benchmark",
                             help="Directory to save benchmark results")
    bench_parser.add_argument("--episodes", type=int, default=10,
                             help="Number of episodes per benchmark")
    bench_parser.add_argument("--patterns", type=str, default="uniform,rush_hour,weekend",
                             help="Comma-separated list of traffic patterns to test")
    bench_parser.add_argument("--no-visualization", action="store_true",
                             help="Disable benchmark visualizations")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze agent behavior and performance")
    analyze_parser.add_argument("--model", type=str, default=None,
                              help="Path to trained model")
    analyze_parser.add_argument("--metrics", type=str, default=None,
                              help="Path to training metrics file")
    analyze_parser.add_argument("--benchmark-dir", type=str, default=None,
                              help="Directory with existing benchmark results")
    analyze_parser.add_argument("--output", type=str, default="results/analysis",
                              help="Directory to save analysis results")
    analyze_parser.add_argument("--patterns", type=str, default=None,
                              help="Comma-separated list of traffic patterns to analyze")
    analyze_parser.add_argument("--episodes", type=int, default=10,
                              help="Number of episodes for new benchmarks")
    analyze_parser.add_argument("--no-browser", action="store_true",
                              help="Don't open the analysis report in a browser")
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(console_level=log_level)
    
    # Apply common configurations
    if args.seed is not None:
        set_random_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Execute command
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
        success = False
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
