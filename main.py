"""
Execution Core — MLflow-Integrated Training Orchestrator.

Sequences every phase of the loan default prediction lifecycle
inside a single MLflow experiment run: configuration loading,
optional Kaggle data download, Polars-based data ingestion,
pipeline construction, cross-validated hyperparameter optimisation,
model compliance (SHAP), and artifact serialisation.

All artifacts — the serialised pipeline, SHAP beeswarm plot, and
classification metrics JSON — are written to both the local
``outputs/`` directory and the MLflow artifact store for full
experiment reproducibility.

MLflow lifecycle management (tracking URI, experiment selection,
run context, autolog activation) is centralised here. Domain
modules (``data_loader``, ``pipeline``, ``interpretability``)
remain MLflow-agnostic, making them independently testable.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.download_data import ensure_dataset
from src.exceptions import MLflowConnectionError
from src.interpretability import compute_shap_values, save_shap_summary
from src.pipeline import (
    _identify_column_types,
    build_preprocessor,
    build_training_pipeline,
    run_hyperparameter_search,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    """Read the YAML configuration file and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    logger.info("Configuration loaded from %s", path)
    return config


def _setup_output_directory(config: dict[str, Any]) -> Path:
    """
    Ensure the outputs directory exists and return its resolved path.

    Creates parent directories as needed so the caller never has to
    guard against ``FileNotFoundError`` during artifact writes.
    """
    output_dir = Path(config["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready at %s", output_dir.resolve())
    return output_dir


def _configure_mlflow(config: dict[str, Any]) -> str:
    """
    Initialise the MLflow tracking backend.

    Sets the tracking URI, creates or retrieves the experiment by name,
    and returns the experiment ID. Wraps connection failures in
    ``MLflowConnectionError`` so the caller can decide whether to
    abort or degrade gracefully.
    """
    mlflow_cfg = config["mlflow"]

    try:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])
        experiment = mlflow.get_experiment_by_name(mlflow_cfg["experiment_name"])
    except Exception as exc:
        raise MLflowConnectionError(
            f"Failed to configure MLflow backend at "
            f"'{mlflow_cfg['tracking_uri']}': {exc}"
        ) from exc

    experiment_id = experiment.experiment_id
    logger.info(
        "MLflow experiment '%s' (ID: %s) active at %s",
        mlflow_cfg["experiment_name"],
        experiment_id,
        mlflow_cfg["tracking_uri"],
    )
    return experiment_id


def _export_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    """
    Build a scikit-learn classification report, augment it with run
    metadata, serialise to a structured JSON file, and return the
    report dictionary for MLflow metric logging.
    """
    report: dict[str, Any] = classification_report(
        y_true, y_pred, output_dict=True
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(CONFIG_PATH),
        "classification_report": report,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Classification metrics exported to %s", output_path)
    return report


def main() -> None:
    """
    End-to-end orchestration inside a single MLflow run.

    Execution flow:
        1. Load YAML configuration.
        2. Ensure data directory is populated (Kaggle download if needed).
        3. Configure MLflow tracking backend.
        4. Ingest Home Credit relational dataset via Polars.
        5. Perform stratified train/test split.
        6. Open ``mlflow.start_run()`` context.
        7. Enable ``mlflow.xgboost.autolog()``.
        8. Build pipeline and run GridSearchCV.
        9. Evaluate on held-out test set.
        10. Log test-set metrics to MLflow.
        11. Compute SHAP values and save beeswarm plot (logged to MLflow).
        12. Serialise best pipeline to disk and MLflow.
        13. Export metrics JSON to disk and MLflow.
        14. Set run tags from configuration.
    """
    config = _load_config(CONFIG_PATH)
    output_dir = _setup_output_directory(config)

    data_dir = Path(config["data"]["application_train"]).parent
    ensure_dataset(data_dir)

    experiment_id = _configure_mlflow(config)

    X, y = load_dataset(config)

    split_cfg = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg.get("stratify", True) else None,
    )

    logger.info(
        "Train/test split — train: %d samples, test: %d samples.",
        len(X_train),
        len(X_test),
    )

    mlflow_cfg = config["mlflow"]

    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.info("MLflow run started: %s", run.info.run_id)

        mlflow.xgboost.autolog(log_models=False)

        for tag_key, tag_value in mlflow_cfg.get("tags", {}).items():
            mlflow.set_tag(tag_key, tag_value)

        numeric_features, categorical_features = _identify_column_types(X_train)
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = build_training_pipeline(preprocessor, config)

        search = run_hyperparameter_search(pipeline, X_train, y_train, config)
        best_pipeline = search.best_estimator_

        y_pred: np.ndarray = best_pipeline.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(
            "Held-out test evaluation:\n%s",
            classification_report(y_test, y_pred),
        )

        mlflow.log_metrics({
            "test_accuracy": report["accuracy"],
            "test_f1_class_0": report["0"]["f1-score"],
            "test_f1_class_1": report["1"]["f1-score"],
            "test_precision_class_1": report["1"]["precision"],
            "test_recall_class_1": report["1"]["recall"],
        })

        shap_values = compute_shap_values(best_pipeline, X_test)
        shap_path = output_dir / config["outputs"]["shap_filename"]
        save_shap_summary(shap_values, shap_path, log_to_mlflow=True)

        model_path = output_dir / config["outputs"]["model_filename"]
        joblib.dump(best_pipeline, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        logger.info("Best pipeline serialised to %s and logged to MLflow.", model_path)

        metrics_path = output_dir / config["outputs"]["metrics_filename"]
        _export_metrics(y_test, y_pred, metrics_path)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        logger.info(
            "Training run complete. MLflow run ID: %s. "
            "Artifacts written to %s/ and MLflow store.",
            run.info.run_id,
            output_dir,
        )


if __name__ == "__main__":
    main()
