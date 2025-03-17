import argparse
import logging
import sys
import numpy as np
import torch
import random
from traffic_rl.config import load_config
from traffic_rl.train import train
from traffic_rl.evaluate import evaluate
from traffic_rl.utils.visualization import visualize
from traffic_rl.utils.benchmark import benchmark
from traffic_rl.analyze import analyze

logger = logging.getLogger("TrafficRL.Main")

def main():
    """Main entry point for the TrafficRL CLI."""
    parser = argparse.ArgumentParser(description="TrafficRL - Reinforcement Learning for Traffic Control")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       default="INFO", help="Set the logging level")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--agent-type", type=str, required=True, 
                            choices=["dqn", "simple_dqn", "entity_dqn", "ppo"], 
                            help="Type of agent to train")
    train_parser.add_argument("--output", type=str, default="results/training",
                            help="Directory to save training results")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    
    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize traffic simulation")
    vis_parser.add_argument("--model", type=str, help="Path to trained model (optional)")
    vis_parser.add_argument("--episodes", type=int, default=1, help="Number of visualization episodes")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark comparisons")
    benchmark_parser.add_argument("--model", type=str, nargs="+", help="Paths to trained models")
    benchmark_parser.add_argument("--patterns", type=str, help="Comma-separated list of traffic patterns to test")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training results")
    analyze_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    analyze_parser.add_argument("--episodes", type=int, default=10, help="Number of analysis episodes")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    
    try:
        if args.command == "train":
            train(config, args.agent_type, args.resume, model_dir=args.output)
        elif args.command == "evaluate":
            evaluate(config, args.model, args.episodes)
        elif args.command == "visualize":
            visualize(config, args.model, args.episodes)
        elif args.command == "benchmark":
            # Parse traffic patterns
            patterns = args.patterns.split(",") if args.patterns else ["uniform"]
            benchmark(config, args.model, patterns)
        elif args.command == "analyze":
            analyze(config, args.model, args.episodes)
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 