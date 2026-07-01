"""
Transformation & Training Pipeline — sklearn + XGBoost.

Constructs a standard scikit-learn Pipeline pairing a ColumnTransformer
(numeric scaling + categorical one-hot encoding) with an XGBoost
classifier. Class imbalance is addressed natively through XGBoost's
``scale_pos_weight`` hyperparameter, which up-weights the minority
(default) class during gradient computation — avoiding the memory
overhead of synthetic oversampling on large (300k+) datasets.

The pipeline is compatible with ``GridSearchCV`` and MLflow's XGBoost
autologging. MLflow lifecycle management (run context, autolog
activation) is deliberately kept out of this module and handled by
the orchestrator in ``main.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def _identify_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Partition feature columns into numeric and categorical lists
    by inspecting pandas dtype families.

    ``object`` and ``category`` dtypes are treated as categorical;
    everything else (int, float) is treated as numeric.
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    logger.info(
        "Identified %d numeric and %d categorical features.",
        len(numeric_features),
        len(categorical_features),
    )
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Assemble a ColumnTransformer with two parallel branches:

    * **numeric** — ``StandardScaler`` for zero-mean / unit-variance scaling.
    * **categorical** — ``OneHotEncoder`` with ``handle_unknown='ignore'``
      so that unseen categories at inference time produce an all-zeros
      vector rather than raising an error.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_training_pipeline(
    preprocessor: ColumnTransformer,
    config: dict[str, Any],
) -> Pipeline:
    """
    Chain preprocessor → XGBClassifier inside a standard sklearn Pipeline.

    Class imbalance is handled by ``scale_pos_weight`` (tuned via
    GridSearchCV) rather than synthetic oversampling, eliminating the
    memory pressure of SMOTE's nearest-neighbour computation on 300k+
    training rows.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=config["split"]["random_state"],
                    verbosity=0,
                ),
            ),
        ]
    )


def run_hyperparameter_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> GridSearchCV:
    """
    Execute an exhaustive grid search over the XGBoost hyperparameter
    space, including ``scale_pos_weight`` for class-imbalance tuning.

    The ``param_grid`` keys are namespaced with the pipeline step name
    (``classifier__``) so GridSearchCV routes them to the correct
    estimator inside the sklearn Pipeline.

    Returns the fitted ``GridSearchCV`` object; the best pipeline is
    accessible via ``.best_estimator_``.
    """
    model_cfg = config["model"]
    raw_grid: dict[str, list] = model_cfg["hyperparameter_grid"]

    param_grid = {f"classifier__{k}": v for k, v in raw_grid.items()}

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=model_cfg["scoring_metric"],
        cv=model_cfg["cv_folds"],
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    logger.info(
        "Starting GridSearchCV — %d candidates × %d folds.",
        _grid_candidate_count(raw_grid),
        model_cfg["cv_folds"],
    )

    search.fit(X_train, y_train)

    logger.info("Best CV score (F1): %.4f", search.best_score_)
    logger.info("Best parameters: %s", search.best_params_)

    return search


def _grid_candidate_count(grid: dict[str, list]) -> int:
    """Compute the total number of hyperparameter combinations."""
    count = 1
    for values in grid.values():
        count *= len(values)
    return count
