"""
Centralized configuration for experiments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    data_path: str
    sensitive_attrs: List[str]
    target_col: str
    
@dataclass
class PrivacyConfig:
    """Differential privacy parameters."""
    epsilon: float
    delta: Optional[float] = None
    
@dataclass
class ExperimentConfig:
    """Experiment hyperparameters."""
    # Dataset
    dataset: str
    sensitive_attr: str
    target_col: str
    
    # Privacy
    epsilon: float
    delta: Optional[float] = None
    
    # Model
    model_type: str = "logistic_regression"
    random_seed: int = 42
    
    # Bias mitigation
    mitigation: Optional[str] = None  # "reweighing", "fairbalance", etc.
    
    # Training
    test_size: float = 0.3
    
    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    
def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Return dataset-specific configuration."""
    configs = {
        "adult": DatasetConfig(
            name="adult",
            data_path="data/adult.csv",
            sensitive_attrs=["sex", "race"],
            target_col="income"
        ),
        "compas": DatasetConfig(
            name="compas",
            data_path="data/compas-scores-two-years.csv",
            sensitive_attrs=["sex", "race"],
            target_col="two_year_recid"
        ),
        "folktables": DatasetConfig(
            name="folktables",
            data_path="data/data.csv",
            sensitive_attrs=["SEX", "RAC1P"],
            target_col="TARGET"
        ),
        "celeba": DatasetConfig(
            name="celeba",
            data_path="data/celebA_preprocessed.csv",
            sensitive_attrs=["Male"],
            target_col="Smiling"
        ),
        "bank": DatasetConfig(
            name="bank",
            data_path="data/bank.csv",
            sensitive_attrs=["age"],
            target_col="y"
        ),
        "german": DatasetConfig(
            name="german",
            data_path="data/german.data",
            sensitive_attrs=["age", "sex"],
            target_col="credit"
        ),
        "heart": DatasetConfig(
            name="heart",
            data_path="data/heart.csv",
            sensitive_attrs=["age"],
            target_col="y"
        ),
        "default": DatasetConfig(
            name="default",
            data_path="data/default.csv",
            sensitive_attrs=["SEX"],
            target_col="default payment next month"
        ),
        "student_mat": DatasetConfig(
            name="student_mat",
            data_path="data/student-mat.csv",
            sensitive_attrs=["sex"],
            target_col="G3"
        ),
        "student_por": DatasetConfig(
            name="student_por",
            data_path="data/student-por.csv",
            sensitive_attrs=["sex"],
            target_col="G3"
        )
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return configs[dataset_name]

