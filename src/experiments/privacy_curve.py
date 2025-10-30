"""
Privacy-fairness curve experiment: sweep epsilon and measure fairness metrics.
"""

import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LogisticRegression

from utils.config import get_dataset_config
from data_loading.loaders import load_dataset
from metrics.fairness_metrics import compute_fairness_metrics
import utils.seeding as seeding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep privacy budget (epsilon) and measure fairness trade-offs"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["adult", "compas", "folktables", "celeba"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--sensitive_attr",
        type=str,
        default=None,
        help="Sensitive attribute column name"
    )
    parser.add_argument(
        "--epsilon_min",
        type=float,
        default=0.001,
        help="Minimum epsilon value (more private)"
    )
    parser.add_argument(
        "--epsilon_max",
        type=float,
        default=10.0,
        help="Maximum epsilon value (less private)"
    )
    parser.add_argument(
        "--n_epsilons",
        type=int,
        default=20,
        help="Number of epsilon values to sweep"
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=100,
        help="Number of private model instances per epsilon (for bounds)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set seed
    seeding.set_seed(args.random_seed)
    rng = np.random.default_rng(seed=args.random_seed)
    
    # Load configuration
    dataset_config = get_dataset_config(args.dataset)
    sensitive_attr = args.sensitive_attr or dataset_config.sensitive_attrs[0]
    
    logger.info(f"Running privacy-fairness curve experiment on {args.dataset}")
    logger.info(f"Sweeping epsilon from {args.epsilon_min} to {args.epsilon_max} ({args.n_epsilons} points)")
    
    # Load data
    X_train, X_test, y_train, y_test = load_dataset(
        args.dataset,
        sensitive_attrs=dataset_config.sensitive_attrs,
        target_col=dataset_config.target_col,
        test_size=0.3,
        random_seed=args.random_seed
    )
    
    # Preprocess to numpy arrays for this experiment
    if hasattr(X_train, 'values'):
        sensitive_train = X_train[sensitive_attr].values
        sensitive_test = X_test[sensitive_attr].values
    else:
        sensitive_train = X_train[:, X_train.columns.get_loc(sensitive_attr)]
        sensitive_test = X_test[:, X_test.columns.get_loc(sensitive_attr)]
    
    # Convert to numpy if needed
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=[sensitive_attr]) if hasattr(X_train, 'columns') else X_train)
    X_test_scaled = scaler.transform(X_test.drop(columns=[sensitive_attr]) if hasattr(X_test, 'columns') else X_test)
    
    # Epsilon values (log scale)
    epsilons = np.logspace(
        np.log10(args.epsilon_min),
        np.log10(args.epsilon_max),
        args.n_epsilons
    )
    
    n, p = X_train_scaled.shape
    delta = 1.0 / (n ** 2)
    
    # Compute data norm bound
    L = max(np.linalg.norm(X_train_scaled, axis=1, ord=2))
    mu = 1.0  # Regularization parameter
    
    logger.info(f"n={n}, p={p}, L={L:.2f}, delta={delta:.2e}")
    
    # Train baseline (non-private) model
    clf_base = LogisticRegression(fit_intercept=False, C=n/mu, max_iter=10000, random_state=args.random_seed)
    clf_base.fit(X_train_scaled, y_train)
    
    # Store results
    results = {
        'epsilons': epsilons,
        'epsilon_min': args.epsilon_min,
        'epsilon_max': args.epsilon_max,
        'n_models': args.n_models,
        'dataset': args.dataset,
        'sensitive_attr': sensitive_attr,
        'fairness_metrics': []
    }
    
    # Loop over epsilon values
    for i, epsilon in enumerate(epsilons):
        logger.info(f"epsilon={epsilon:.4f} ({i+1}/{len(epsilons)})")
        
        # Compute noise scale for output perturbation
        noise_scale = L * np.sqrt(8 * np.log(1.25 / delta)) / (n * mu * epsilon)
        
        # Evaluate non-private model
        metrics_base = _evaluate_model(
            clf_base, X_test_scaled, y_test, sensitive_test, scaler
        )
        
        # Generate multiple private model instances for empirical bounds
        metrics_priv = []
        for j in range(args.n_models):
            # Create private model by adding noise to coefficients
            clf_priv = deepcopy(clf_base)
            noise = rng.normal(loc=0, scale=noise_scale, size=p)
            clf_priv.coef_ = clf_base.coef_ + noise
            
            # Evaluate
            metrics = _evaluate_model(
                clf_priv, X_test_scaled, y_test, sensitive_test, scaler
            )
            metrics_priv.append(metrics)
        
        # Aggregate results
        metrics_priv_mean = {
            'metric': 'avg',
            'accuracy': np.mean([m['accuracy'] for m in metrics_priv]),
            'equal_opportunity_diff': np.mean([abs(m['equal_opportunity_diff']) for m in metrics_priv]),
            'average_odds_diff': np.mean([abs(m['average_odds_diff']) for m in metrics_priv])
        }
        
        results['fairness_metrics'].append({
            'epsilon': epsilon,
            'noise_scale': noise_scale,
            'baseline': metrics_base,
            'private_mean': metrics_priv_mean,
            'private_all': metrics_priv if args.n_models <= 10 else None  # Only store if small
        })
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"privacy_curve_{args.dataset}_{timestamp}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Also save as CSV for easy plotting
    df_results = _results_to_dataframe(results)
    csv_file = output_file.with_suffix('.csv')
    df_results.to_csv(csv_file, index=False)
    logger.info(f"CSV saved to: {csv_file}")


def _evaluate_model(model, X_test, y_test, sensitive_test, scaler):
    """Evaluate a model and return fairness metrics."""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Simple fairness computation for two groups
    group_0_mask = (sensitive_test == 0)
    group_1_mask = (sensitive_test == 1)
    
    # Confusion matrices
    conf_0 = _confusion_matrix(y_test[group_0_mask], y_pred[group_0_mask])
    conf_1 = _confusion_matrix(y_test[group_1_mask], y_pred[group_1_mask])
    
    # Compute metrics
    accuracy = (y_test == y_pred).mean()
    
    tpr_0 = conf_0['tp'] / max(conf_0['tp'] + conf_0['fn'], 1)
    tpr_1 = conf_1['tp'] / max(conf_1['tp'] + conf_1['fn'], 1)
    equal_opportunity_diff = tpr_1 - tpr_0
    
    fpr_0 = conf_0['fp'] / max(conf_0['fp'] + conf_0['tn'], 1)
    fpr_1 = conf_1['fp'] / max(conf_1['fp'] + conf_1['tn'], 1)
    average_odds_diff = 0.5 * ((tpr_1 - tpr_0) + (fpr_1 - fpr_0))
    
    return {
        'accuracy': accuracy,
        'equal_opportunity_diff': equal_opportunity_diff,
        'average_odds_diff': average_odds_diff
    }


def _confusion_matrix(y_true, y_pred):
    """Compute confusion matrix entries."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def _results_to_dataframe(results):
    """Convert results dict to pandas DataFrame."""
    rows = []
    for result in results['fairness_metrics']:
        rows.append({
            'epsilon': result['epsilon'],
            'noise_scale': result['noise_scale'],
            'baseline_accuracy': result['baseline']['accuracy'],
            'baseline_eod': abs(result['baseline']['equal_opportunity_diff']),
            'baseline_aod': abs(result['baseline']['average_odds_diff']),
            'private_accuracy': result['private_mean']['accuracy'],
            'private_eod': result['private_mean']['equal_opportunity_diff'],
            'private_aod': result['private_mean']['average_odds_diff']
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

