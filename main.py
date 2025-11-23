from src.visualization import analyze_model, plot_metrics
from src.training import train_cmf, train_ncf, train_ptupcdr
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Domain Recommendation System CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Train CMF
    parser_cmf = subparsers.add_parser(
        "train-cmf", help="Train CMF (Cross-Domain Matrix Factorization) Baseline")

    # Train NCF
    parser_ncf = subparsers.add_parser(
        "train-ncf", help="Train NCF (Neural Collaborative Filtering) Baseline")

    # Train PTUPCDR
    parser_ptupcdr = subparsers.add_parser(
        "train-ptupcdr", help="Train PTUPCDR (Personalized Transfer of User Preferences) Model")

    # Visualize
    parser_viz = subparsers.add_parser(
        "visualize", help="Generate Visualizations")
    parser_viz.add_argument("--type", choices=["latent", "case-study", "ncf-failure",
                            "comparison", "all"], default="all", help="Type of visualization to generate")

    args = parser.parse_args()

    if args.command == "train-cmf":
        train_cmf.run()
    elif args.command == "train-ncf":
        train_ncf.run()
    elif args.command == "train-ptupcdr":
        train_ptupcdr.run_ptupcdr()
    elif args.command == "visualize":
        if args.type in ["latent", "all"]:
            analyze_model.viz_latent_space()
        if args.type in ["case-study", "all"]:
            analyze_model.generate_case_study()
        if args.type in ["ncf-failure", "all"]:
            plot_metrics.plot_ncf_failure()
        if args.type in ["comparison", "all"]:
            plot_metrics.plot_model_comparison()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
