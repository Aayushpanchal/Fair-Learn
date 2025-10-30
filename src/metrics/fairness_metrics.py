"""
Fairness metrics for evaluating classifier outputs across sensitive groups.
"""

import numpy as np
from collections import Counter
from typing import Dict, Any, Optional


def compute_fairness_metrics(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    sensitive_attr: str,
    preprocessor: Optional[Any] = None
) -> Dict[str, float]:
    """
    Compute fairness metrics for a trained model.
    
    Args:
        model: Trained classifier with .predict() and .predict_proba() methods.
        X_test: Test features (DataFrame or array-like).
        y_test: True labels (binary, {0, 1}).
        sensitive_attr: Name of the sensitive attribute column in X_test.
        preprocessor: Optional preprocessor to apply to X_test before prediction.
    
    Returns:
        Dictionary with metrics:
        - accuracy: Overall classification accuracy
        - equal_opportunity_diff: TPR difference across groups
        - average_odds_diff: Average of TPR and FPR differences
        - demographic_parity_diff: Positive prediction rate difference
    """
    # Get predictions
    if preprocessor is not None:
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_test_processed = X_test
    
    y_pred = model.predict(X_test_processed)
    
    # Convert to numpy if needed
    if hasattr(X_test, 'values'):
        sensitive_values = X_test[sensitive_attr].values
    else:
        sensitive_values = X_test[:, X_test.columns.get_loc(sensitive_attr)]
    
    # Compute confusion matrices for each group
    group_0_mask = (sensitive_values == 0)
    group_1_mask = (sensitive_values == 1)
    
    conf_0 = _compute_confusion_matrix(y_test[group_0_mask], y_pred[group_0_mask])
    conf_1 = _compute_confusion_matrix(y_test[group_1_mask], y_pred[group_1_mask])
    
    # Compute metrics
    accuracy = (y_test == y_pred).mean()
    
    tpr_0 = conf_0['tp'] / max(conf_0['tp'] + conf_0['fn'], 1)
    tpr_1 = conf_1['tp'] / max(conf_1['tp'] + conf_1['fn'], 1)
    equal_opportunity_diff = tpr_1 - tpr_0
    
    fpr_0 = conf_0['fp'] / max(conf_0['fp'] + conf_0['tn'], 1)
    fpr_1 = conf_1['fp'] / max(conf_1['fp'] + conf_1['tn'], 1)
    average_odds_diff = 0.5 * ((tpr_1 - tpr_0) + (fpr_1 - fpr_0))
    
    pr_0 = (conf_0['tp'] + conf_0['fp']) / max(len(y_test[group_0_mask]), 1)
    pr_1 = (conf_1['tp'] + conf_1['fp']) / max(len(y_test[group_1_mask]), 1)
    demographic_parity_diff = pr_1 - pr_0
    
    return {
        'accuracy': accuracy,
        'equal_opportunity_diff': equal_opportunity_diff,
        'average_odds_diff': average_odds_diff,
        'demographic_parity_diff': demographic_parity_diff,
        'tpr_group_0': tpr_0,
        'tpr_group_1': tpr_1,
        'fpr_group_0': fpr_0,
        'fpr_group_1': fpr_1
    }


def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute confusion matrix entries.
    
    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
    
    Returns:
        Dictionary with keys: 'tp', 'tn', 'fp', 'fn'.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


class EqualOpportunityDifference:
    """
    Equal Opportunity Difference (EOD).
    
    Measures fairness in True Positive Rates across sensitive groups:
    EOD = TPR(A=1) - TPR(A=0)
    
    Values closer to 0 indicate better fairness.
    """
    
    @staticmethod
    def compute(y_true, y_pred, sensitive_attr, group_0_val=0, group_1_val=1):
        """Compute EOD."""
        mask_0 = (sensitive_attr == group_0_val)
        mask_1 = (sensitive_attr == group_1_val)
        
        tpr_0 = _compute_tpr(y_true[mask_0], y_pred[mask_0])
        tpr_1 = _compute_tpr(y_true[mask_1], y_pred[mask_1])
        
        return tpr_1 - tpr_0


class AverageOddsDifference:
    """
    Average Odds Difference (AOD).
    
    Measures average fairness in both TPR and FPR:
    AOD = 0.5 × [(TPR₁ - TPR₀) + (FPR₁ - FPR₀)]
    
    Values closer to 0 indicate better fairness.
    """
    
    @staticmethod
    def compute(y_true, y_pred, sensitive_attr, group_0_val=0, group_1_val=1):
        """Compute AOD."""
        mask_0 = (sensitive_attr == group_0_val)
        mask_1 = (sensitive_attr == group_1_val)
        
        tpr_0 = _compute_tpr(y_true[mask_0], y_pred[mask_0])
        tpr_1 = _compute_tpr(y_true[mask_1], y_pred[mask_1])
        fpr_0 = _compute_fpr(y_true[mask_0], y_pred[mask_0])
        fpr_1 = _compute_fpr(y_true[mask_1], y_pred[mask_1])
        
        return 0.5 * ((tpr_1 - tpr_0) + (fpr_1 - fpr_0))


def _compute_tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute True Positive Rate."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = tp + fn
    return tp / max(denom, 1)


def _compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute False Positive Rate."""
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    denom = fp + tn
    return fp / max(denom, 1)

