"""
Runtime Inference Utility — Single-Record CLI Prediction.

Standalone command-line script that loads a serialised Home Credit
default risk pipeline and runs inference on a single customer
observation provided as a JSON string.

Usage example::

    python predict.py --payload '{"AMT_INCOME_TOTAL": 270000, "AMT_CREDIT": 1293502, ...}'

The output is a structured JSON object printed to stdout containing:

* ``default_probability_pct`` — the model's estimated probability of
  default, expressed as a percentage.
* ``decision`` — an operational label: ``"APPROVED"`` if the default
  probability is below the decision threshold, ``"REJECTED"`` otherwise.

Design note: the threshold is deliberately set at 50 % to match the
argmax decision boundary of the classifier. In production, this would
typically be calibrated via a cost-sensitive analysis or regulatory
constraint — but that calibration belongs in a separate policy layer,
not baked into the inference script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exceptions import ModelNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("outputs/model.joblib")
DECISION_THRESHOLD = 0.50


def _load_pipeline(model_path: Path) -> Pipeline:
    """
    Deserialise the joblib artifact from disk.

    Raises ``ModelNotFoundError`` if the file does not exist, providing
    a clear diagnostic message rather than an opaque ``FileNotFoundError``.
    """
    if not model_path.exists():
        raise ModelNotFoundError(
            f"No serialised model found at {model_path.resolve()}. "
            "Run main.py to train and export the pipeline first."
        )

    pipeline: Pipeline = joblib.load(model_path)
    logger.info("Pipeline loaded from %s", model_path)
    return pipeline


def predict(
    payload: dict[str, Any],
    pipeline: Pipeline,
) -> dict[str, Any]:
    """
    Run a single observation through the fitted pipeline and return
    a structured decision payload.

    Parameters
    ----------
    payload : dict
        Raw feature dictionary matching the schema of the training data.
    pipeline : Pipeline
        Fitted sklearn pipeline (preprocessor + classifier).

    Returns
    -------
    dict
        ``{"default_probability_pct": float, "decision": str}``
    """
    observation = pd.DataFrame([payload])

    probabilities = pipeline.predict_proba(observation)[0]
    default_probability = float(probabilities[1])

    decision = "REJECTED" if default_probability >= DECISION_THRESHOLD else "APPROVED"

    return {
        "default_probability_pct": round(default_probability * 100, 2),
        "decision": decision,
    }


def main() -> None:
    """
    CLI entry-point.

    Parses a ``--payload`` argument containing a JSON string, loads
    the serialised model, runs inference, and prints the structured
    result to stdout. Diagnostic logs are routed to stderr so that
    stdout contains only machine-readable JSON output.
    """
    parser = argparse.ArgumentParser(
        description="Loan Default Prediction — single-record inference.",
    )
    parser.add_argument(
        "--payload",
        type=str,
        required=True,
        help="JSON string containing a single customer observation.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the serialised pipeline artifact.",
    )

    args = parser.parse_args()

    try:
        customer_data: dict[str, Any] = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON payload: %s", exc)
        sys.exit(1)

    pipeline = _load_pipeline(Path(args.model_path))
    result = predict(customer_data, pipeline)

    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
