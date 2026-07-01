"""
Model Compliance Layer — SHAP-Based Global Interpretability.

Provides an isolated interpretability interface that extracts the
trained tree estimator from a fitted sklearn pipeline, computes SHAP
values on a representative sample of the test partition, and persists
a publication-quality beeswarm summary plot.

For datasets exceeding ~5,000 test observations, the module
automatically subsamples to keep SHAP computation tractable while
still producing a statistically representative feature-importance
ranking. The plot is saved locally and optionally logged to the
active MLflow run's artifact store.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

DEFAULT_MAX_SAMPLES = 5000


def _extract_booster(pipeline: Pipeline) -> XGBClassifier:
    """
    Walk the sklearn pipeline's named steps to retrieve the trained
    XGBClassifier instance.

    The pipeline is expected to follow the convention established in
    ``pipeline.build_training_pipeline``, where the classifier lives
    under the ``"classifier"`` key.
    """
    return pipeline.named_steps["classifier"]


def _resolve_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Extract post-transformation feature names from the ColumnTransformer.

    After one-hot encoding, the feature space expands beyond the raw
    column names. This function calls ``get_feature_names_out()`` on
    the fitted preprocessor to recover the full set of transformed
    column labels for SHAP plot annotation.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def compute_shap_values(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> shap.Explanation:
    """
    Transform the raw test features through the preprocessor, build
    a ``TreeExplainer`` from the fitted XGBoost model, and compute
    SHAP values.

    For large test sets, a random subsample of ``max_samples`` rows
    is used to keep the SHAP computation under a few minutes while
    still producing statistically stable feature-importance rankings.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    booster = _extract_booster(pipeline)

    if len(X_test) > max_samples:
        logger.info(
            "Subsampling test set from %d to %d for SHAP computation.",
            len(X_test),
            max_samples,
        )
        X_sample = X_test.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_test

    X_transformed: np.ndarray = preprocessor.transform(X_sample)
    feature_names = _resolve_feature_names(pipeline)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names

    logger.info(
        "Computed SHAP values for %d observations across %d features.",
        X_transformed.shape[0],
        X_transformed.shape[1],
    )
    return shap_values


def save_shap_summary(
    shap_values: shap.Explanation,
    output_path: Path,
    log_to_mlflow: bool = True,
    dpi: int = 200,
) -> None:
    """
    Render a beeswarm summary plot, save to disk as a high-resolution
    PNG, and optionally log it to the active MLflow run's artifact store.

    Uses matplotlib's ``Agg`` backend (set at module import) to avoid
    any dependency on a running display server, making this safe for
    headless CI environments.
    """
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("SHAP summary plot saved to %s", output_path)

    if log_to_mlflow:
        try:
            import mlflow
            mlflow.log_artifact(str(output_path), artifact_path="compliance")
            logger.info("SHAP plot logged to MLflow artifact store.")
        except Exception as exc:
            logger.warning("Failed to log SHAP plot to MLflow: %s", exc)
