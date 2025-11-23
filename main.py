from src.data.user_profile_generator import generate_user_profiles
from src.data.slice_text_features import FeatureSlicer
from src.data.feature_engineering import FeatureEngineer
from src.data.data_splitter import CrossDomainSplitter
from src.data.preprocessing import Preprocessor
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

    # Data Processing
    parser_data = subparsers.add_parser(
        "process-data", help="Run the full data processing pipeline")
    parser_data.add_argument(
        "--source", type=str, required=True, help="Path to raw source domain JSONL")
    parser_data.add_argument(
        "--target", type=str, required=True, help="Path to raw target domain JSONL")
    parser_data.add_argument("--meta-source", type=str,
                             required=True, help="Path to raw source metadata JSONL")
    parser_data.add_argument("--meta-target", type=str,
                             required=True, help="Path to raw target metadata JSONL")
    parser_data.add_argument("--output", type=str, default="data/processed",
                             help="Output directory for processed data")

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

    if args.command == "process-data":
        print("--- Starting Data Processing Pipeline ---")

        # 1. Preprocessing
        print("\n[Step 1] Preprocessing Raw Data...")
        preprocessor = Preprocessor(args.source, args.target, args.output)
        preprocessor.process()

        # 2. Splitting
        print("\n[Step 2] Splitting Data...")
        splitter = CrossDomainSplitter(args.output)
        splitter.load_data()
        splitter.perform_cold_user_split(val_ratio=0.1, test_ratio=0.1)

        # 3. Feature Engineering
        print("\n[Step 3] Generating Item Embeddings...")
        engineer = FeatureEngineer(
            args.output, model_name='Qwen/Qwen3-Embedding-0.6B')
        engineer.process_domain(
            args.meta_source, 'source_item_mapping.json', 'source_domain', batch_size=64)
        engineer.process_domain(
            args.meta_target, 'target_item_mapping.json', 'target_domain', batch_size=64)

        # 4. Slicing Features (MRL)
        print("\n[Step 4] Slicing Features (MRL)...")
        slicer = FeatureSlicer(args.output, target_dim=128)
        slicer.slice_and_save("source_domain_feat.npy", "source_domain_mrl")
        slicer.slice_and_save("target_domain_feat.npy", "target_domain_mrl")

        # 5. User Profiles
        print("\n[Step 5] Generating User Profiles...")
        generate_user_profiles(args.output, "source")
        generate_user_profiles(args.output, "target")

        print("\nData Processing Complete!")

    elif args.command == "train-cmf":
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
