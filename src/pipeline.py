"""
Transformation & Training Pipeline — sklearn + XGBoost + Optuna.

Constructs a standard scikit-learn Pipeline pairing a ColumnTransformer
(numeric scaling + categorical one-hot encoding) with an XGBoost
classifier. 

Bayesian Optimization via Optuna is used to efficiently search the
hyperparameter space, evaluating candidates with StratifiedKFold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def _identify_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Partition feature columns into numeric and categorical lists
    by inspecting pandas dtype families.
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
    StandardScaler (numeric) and OneHotEncoder (categorical).
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
    **xgb_kwargs: Any,
) -> Pipeline:
    """
    Chain preprocessor → XGBClassifier inside a standard sklearn Pipeline.
    Hyperparameters for XGBoost can be passed via xgb_kwargs.
    """
    base_kwargs = {
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": config["split"]["random_state"],
        "verbosity": 0,
        "device": "cuda",
    }
    base_kwargs.update(xgb_kwargs)
    
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**base_kwargs)),
        ]
    )


def run_hyperparameter_search(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
    mlflow_callback: Any,
) -> Pipeline:
    """
    Execute Optuna Bayesian Optimization over the XGBoost hyperparameter
    space. Evaluates candidates using StratifiedKFold cross-validation.
    """
    model_cfg = config["model"]
    n_trials = model_cfg.get("optuna", {}).get("n_trials", 30)
    cv_folds = model_cfg["cv_folds"]
    scoring = model_cfg["scoring_metric"]
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }
        
        pipeline = build_training_pipeline(preprocessor, config, **params)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config["split"]["random_state"])
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        return float(np.mean(scores))
    
    logger.info("Starting Optuna Bayesian Optimization — %d trials × %d folds.", n_trials, cv_folds)
    
    # Optuna logging noise reduction
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective, 
        n_trials=n_trials, 
        callbacks=[mlflow_callback],
    )
    
    logger.info("Best CV score (F1): %.4f", study.best_value)
    logger.info("Best parameters: %s", study.best_params)
    
    # Refit best pipeline on the full training set
    logger.info("Refitting best pipeline on full training set...")
    best_pipeline = build_training_pipeline(preprocessor, config, **study.best_params)
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline
