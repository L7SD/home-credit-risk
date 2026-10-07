import argparse
import json
from pathlib import Path
from .demo import create_demo
from .features import build_dataset
from .modeling import predict, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Home Credit training and prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Generate synthetic CSVs, not real credit records"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--rows", type=int, default=500)
    demo.add_argument("--seed", type=int, default=42)
    train = commands.add_parser(
        "train", help="Compare on validation; evaluate one selected model on test"
    )
    train.add_argument("--applications", type=Path, required=True)
    train.add_argument("--history-dir", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--trees", type=int, default=150)
    train.add_argument(
        "--refit",
        action="store_true",
        help="Also create a full-labeled-data submission model",
    )
    train.add_argument(
        "--xgboost", action="store_true", help="Include optional XGBoost candidate"
    )
    train.add_argument(
        "--importance",
        action="store_true",
        help="Compute sampled validation permutation importance",
    )
    prediction = commands.add_parser(
        "predict", help="Load your own trusted model and write probabilities"
    )
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--applications", type=Path, required=True)
    prediction.add_argument("--history-dir", type=Path)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            create_demo(args.output, args.rows, args.seed)
            print(f"Synthetic demo files: {args.output}")
        else:
            frame = build_dataset(
                args.applications, args.history_dir, training=args.command == "train"
            )
            if args.command == "train":
                data_kind = (
                    "synthetic-demo"
                    if (args.applications.parent / "SYNTHETIC_DATA.json").exists()
                    else "user-provided"
                )
                report = run_experiment(
                    frame,
                    args.output,
                    seed=args.seed,
                    trees=args.trees,
                    use_history=args.history_dir is not None,
                    refit=args.refit,
                    use_xgboost=args.xgboost,
                    importance=args.importance,
                    data_kind=data_kind,
                )
                print(
                    json.dumps(
                        {
                            "data_kind": data_kind,
                            "selected_model": report["selected_model"],
                            "test": report["selected_model_test"],
                        },
                        indent=2,
                    )
                )
            else:
                predict(
                    args.model,
                    frame,
                    args.output,
                    use_history=args.history_dir is not None,
                )
                print(f"Predictions: {args.output}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
