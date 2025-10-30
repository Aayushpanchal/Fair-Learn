"""
Main training script with CLI entrypoint.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from utils.config import get_dataset_config, ExperimentConfig
from data_loading.loaders import load_dataset
from models.trainer import Trainer
from metrics.fairness_metrics import compute_fairness_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Train a classifier with optional bias mitigation under DP constraints"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["adult", "compas", "folktables", "celeba", "bank", "german", "heart", "default", "student_mat", "student_por"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--sensitive_attr",
        type=str,
        default=None,
        help="Sensitive attribute column name (if not specified, uses first from config)"
    )
    
    # Privacy arguments
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Privacy budget (epsilon) for differential privacy"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Failure probability (delta) for differential privacy. Defaults to 1/n^2"
    )
    
    # Mitigation arguments
    parser.add_argument(
        "--mitigation",
        type=str,
        choices=["none", "reweighing", "fairbalance"],
        default="none",
        help="Bias mitigation technique to apply"
    )
    
    # Training arguments
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Fraction of data to use for testing"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save trained model checkpoint"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.random_seed)
    
    # Load configuration
    dataset_config = get_dataset_config(args.dataset)
    sensitive_attr = args.sensitive_attr or dataset_config.sensitive_attrs[0]
    
    logger.info(f"Starting experiment: dataset={args.dataset}, epsilon={args.epsilon}, mitigation={args.mitigation}")
    
    # Load and preprocess data
    logger.info(f"Loading dataset: {args.dataset}")
    X_train, X_test, y_train, y_test = load_dataset(
        args.dataset,
        sensitive_attrs=dataset_config.sensitive_attrs,
        target_col=dataset_config.target_col,
        test_size=args.test_size,
        random_seed=args.random_seed
    )
    
    # Train model with optional mitigation and DP
    trainer = Trainer(
        model_type="logistic_regression",
        epsilon=args.epsilon,
        delta=args.delta if args.delta else (1.0 / len(X_train) ** 2),
        mitigation=args.mitigation if args.mitigation != "none" else None
    )
    
    logger.info("Training model...")
    model = trainer.fit(X_train, y_train, sensitive_attr)
    
    # Evaluate
    logger.info("Computing fairness metrics...")
    metrics = compute_fairness_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        sensitive_attr=sensitive_attr,
        preprocessor=trainer.preprocessor
    )
    
    # Print results
    logger.info("Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Equal Opportunity Difference: {metrics['equal_opportunity_diff']:.4f}")
    logger.info(f"  Average Odds Difference: {metrics['average_odds_diff']:.4f}")
    logger.info(f"  Demographic Parity Difference: {metrics['demographic_parity_diff']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"{args.dataset}_epsilon{args.epsilon}_{args.mitigation}_{timestamp}.csv"
    
    results_df = pd.DataFrame([{
        "dataset": args.dataset,
        "epsilon": args.epsilon,
        "delta": args.delta or (1.0 / len(X_train) ** 2),
        "mitigation": args.mitigation,
        "accuracy": metrics['accuracy'],
        "equal_opportunity_diff": metrics['equal_opportunity_diff'],
        "average_odds_diff": metrics['average_odds_diff'],
        "demographic_parity_diff": metrics['demographic_parity_diff']
    }])
    
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to: {results_file}")
    
    if args.save_model:
        import pickle
        model_file = output_dir / "checkpoints" / f"model_{timestamp}.pkl"
        model_file.parent.mkdir(exist_ok=True)
        with open(model_file, "wb") as f:
            pickle.dump({"model": model, "preprocessor": trainer.preprocessor}, f)
        logger.info(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()

