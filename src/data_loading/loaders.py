"""
Dataset loaders for various fairness datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

import logging
logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    sensitive_attrs: List[str],
    target_col: str,
    test_size: float = 0.3,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and split a dataset into train/test sets.
    
    Args:
        dataset_name: Name of the dataset.
        sensitive_attrs: List of sensitive attribute column names.
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as pandas DataFrames and numpy arrays.
    """
    loader = get_data_loader(dataset_name)
    X_train, X_test, y_train, y_test = loader(
        sensitive_attrs=sensitive_attrs,
        target_col=target_col,
        test_size=test_size,
        random_seed=random_seed
    )
    return X_train, X_test, y_train, y_test


def get_data_loader(dataset_name: str):
    """
    Get the appropriate data loader function for a dataset.
    
    Args:
        dataset_name: Name of the dataset.
    
    Returns:
        Loader function.
    """
    loaders = {
        "adult": load_adult,
        "compas": load_compas,
        "folktables": load_folktables,
        "celeba": load_celeba,
        "bank": load_bank,
        "german": load_german,
        "heart": load_heart,
        "default": load_default,
        "student_mat": load_student_mat,
        "student_por": load_student_por
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return loaders[dataset_name]


def load_adult(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load Adult Income dataset."""
    df = pd.read_csv("data/adult.csv")
    
    # Binarize sensitive attributes
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "Male" else 0)
    df['race'] = df['race'].apply(lambda x: 1 if x == "White" else 0)
    df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_compas(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load COMPAS dataset."""
    df = pd.read_csv("data/compas-scores-two-years.csv")
    
    # Filter features
    features_to_keep = ['sex', 'age', 'age_cat', 'race',
                        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                        'priors_count', 'c_charge_degree', 'c_charge_desc',
                        'two_year_recid']
    df = df[features_to_keep]
    
    # Binarize
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "Male" else 0)
    df['race'] = df['race'].apply(lambda x: 1 if x == "Caucasian" else 0)
    df['two_year_recid'] = df['two_year_recid'].apply(lambda x: 1 if x == 0 else 0)
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_folktables(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load FolkTables dataset."""
    df = pd.read_csv("data/data.csv")
    
    # Binarize target
    df["TARGET"] = df["TARGET"].apply(lambda x: 1 if x == True else 0)
    df["SEX"] = df["SEX"].apply(lambda x: 1 if x == 1 else 0)  # 1 = Male
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_celeba(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load CelebA dataset."""
    df = pd.read_csv("data/celebA_preprocessed.csv", sep=",")
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_bank(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load Bank dataset."""
    df = pd.read_csv("data/bank.csv", sep=";")
    
    # Binarize
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    df['y'] = df['y'].apply(lambda x: 1 if x == "yes" else 0)
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_german(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load German Credit dataset."""
    column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'sex',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
    
    df = pd.read_csv("data/german.data", sep=' ', header=None, names=column_names)
    
    # Binarize
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    df["sex"] = df["sex"].apply(lambda x: 1 if x in {"A91", "A93", "A94"} else 0)
    df['credit'] = df['credit'].apply(lambda x: 1 if x == 1 else 0)
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_heart(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load Heart dataset."""
    df = pd.read_csv("data/heart.csv")
    
    # Binarize
    df['age'] = df['age'].apply(lambda x: 1 if x > 60 else 0)
    df['y'] = df['y'].apply(lambda x: 1 if x == 0 else 0)
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_default(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load Default dataset."""
    df = pd.read_csv("data/default.csv")
    
    # Binarize
    df['SEX'] = df['SEX'].apply(lambda x: 0 if x == 2 else 1)
    df['default payment next month'] = df['default payment next month'].apply(lambda x: 1 if x == 0 else 0)
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_student_mat(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load Student Math dataset."""
    df = pd.read_csv("data/student-mat.csv", sep=";")
    
    # Binarize
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "M" else 0)
    df['y'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3'])
    
    target_col = 'y'  # Override since we renamed
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def load_student_por(sensitive_attrs: List[str], target_col: str, test_size: float, random_seed: int):
    """Load Student Portuguese dataset."""
    df = pd.read_csv("data/student-por.csv", sep=";")
    
    # Binarize
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "M" else 0)
    df['y'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3'])
    
    target_col = 'y'  # Override
    
    return _split_dataframe(df, sensitive_attrs, target_col, test_size, random_seed)


def _split_dataframe(
    df: pd.DataFrame,
    sensitive_attrs: List[str],
    target_col: str,
    test_size: float,
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split dataframe into train/test sets with stratified sampling across groups.
    
    Args:
        df: Full dataframe.
        sensitive_attrs: List of sensitive attribute column names.
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.
        random_seed: Random seed.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col].values
    
    # Stratified split across (sensitive_attrs, target) groups
    groups = {}
    for i in range(len(y)):
        key = tuple([X[attr].iloc[i] for attr in sensitive_attrs] + [y[i]])
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    
    train_indices = []
    test_indices = []
    
    for key, indices in groups.items():
        n_test = int(len(indices) * test_size)
        np.random.seed(random_seed)
        test_sample = np.random.choice(indices, size=n_test, replace=False)
        train_sample = [i for i in indices if i not in test_sample]
        test_indices.extend(test_sample)
        train_indices.extend(train_sample)
    
    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

