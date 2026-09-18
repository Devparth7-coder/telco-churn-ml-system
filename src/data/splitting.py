"""Dataset splitting.

The Telco dataset contains no temporal column, so a stratified random split is
the correct strategy (documented in the leakage audit). Splitting is done
*before* any preprocessing is fitted, and the identifier column is carried in
the assignment map for reproducibility auditing.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SplitConfig


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    assignments: pd.DataFrame  # customerID -> split, for auditability


def stratified_split(
    df: pd.DataFrame, target: str, id_column: str, cfg: SplitConfig, random_state: int
) -> SplitResult:
    """Split into train/validation/test with preserved class ratios.

    Args:
        df: Validated raw dataset.
        target: Target column name.
        id_column: Identifier column (kept only in the assignment map).
        cfg: Split proportions. ``validation_size`` is a fraction of the
            remainder after the test split.
        random_state: Seed for reproducibility.

    Returns:
        The three splits plus an id->split assignment frame.
    """
    if df[id_column].duplicated().any():
        raise ValueError("Duplicate identifiers found; refusing to split (group leakage risk).")

    train_val, test = train_test_split(
        df,
        test_size=cfg.test_size,
        stratify=df[target] if cfg.stratify else None,
        random_state=random_state,
        shuffle=True,
    )
    train, validation = train_test_split(
        train_val,
        test_size=cfg.validation_size,
        stratify=train_val[target] if cfg.stratify else None,
        random_state=random_state,
        shuffle=True,
    )

    assignments = pd.concat(
        [
            pd.DataFrame({id_column: train[id_column], "split": "train"}),
            pd.DataFrame({id_column: validation[id_column], "split": "validation"}),
            pd.DataFrame({id_column: test[id_column], "split": "test"}),
        ],
        ignore_index=True,
    )
    return SplitResult(train=train, validation=validation, test=test, assignments=assignments)
