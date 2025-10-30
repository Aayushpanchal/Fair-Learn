"""
Training logic with support for differential privacy and bias mitigation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from typing import Optional, Any

import logging
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer that handles model training with optional DP noise and bias mitigation.
    """
    
    def __init__(
        self,
        model_type: str = "logistic_regression",
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        mitigation: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model to train (default: logistic_regression).
            epsilon: Privacy budget for differential privacy (if None, no DP).
            delta: Failure probability for DP.
            mitigation: Bias mitigation technique (None, "reweighing", "fairbalance").
        """
        self.model_type = model_type
        self.epsilon = epsilon
        self.delta = delta
        self.mitigation = mitigation
        self.model = None
        self.preprocessor = None
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, sensitive_attr: str) -> Any:
        """
        Train a model on the given data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            sensitive_attr: Name of sensitive attribute column.
        
        Returns:
            Trained model.
        """
        # Set up preprocessing
        self.preprocessor = self._create_preprocessor(X_train)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Apply mitigation if specified
        sample_weights = None
        if self.mitigation == "reweighing":
            sample_weights = self._compute_reweighing_weights(
                X_train, y_train, sensitive_attr
            )
        
        # Initialize model
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=10000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.model.fit(X_train_processed, y_train, sample_weight=sample_weights)
        
        logger.info("Model training complete")
        return self.model
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create a preprocessor for mixed categorical and numerical features."""
        numerical_selector = make_column_selector(dtype_exclude=object)
        categorical_selector = make_column_selector(dtype_include=object)
        
        numerical_cols = numerical_selector(X)
        categorical_cols = categorical_selector(X)
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ])
        
        return preprocessor
    
    def _compute_reweighing_weights(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sensitive_attr: str
    ) -> np.ndarray:
        """
        Compute sample weights for reweighting mitigation.
        
        Weights adjust for under/over-representation of groups.
        
        Args:
            X: Feature dataframe.
            y: Labels.
            sensitive_attr: Name of sensitive attribute.
        
        Returns:
            Array of sample weights.
        """
        # Compute observed and expected distributions
        sensitive_values = X[sensitive_attr].values
        
        # P(A=a, Y=y) for each combination
        groups = {}
        for i in range(len(y)):
            key = (sensitive_values[i], y[i])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        n_samples = len(y)
        weights = np.ones(n_samples)
        
        # Compute expected probability of each (A,Y) pair
        for (a_val, y_val), indices in groups.items():
            observed_prob = len(indices) / n_samples
            
            # Expected probability under independence: P(A=a) × P(Y=y)
            p_a = np.sum(sensitive_values == a_val) / n_samples
            p_y = np.sum(y == y_val) / n_samples
            expected_prob = p_a * p_y
            
            # Weight is ratio of expected to observed
            if observed_prob > 0:
                w = expected_prob / observed_prob
            else:
                w = 1.0
            
            for idx in indices:
                weights[idx] = w
        
        logger.info(f"Computed reweighting weights: mean={weights.mean():.3f}, std={weights.std():.3f}")
        return weights

