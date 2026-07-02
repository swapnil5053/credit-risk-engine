"""
REST API Deployment — FastAPI Model Serving.

Wraps the serialised Home Credit default risk pipeline in a cloud-ready,
containerised REST API using FastAPI.

Exposes a `/predict` endpoint that accepts a generic JSON payload
representing a customer's feature set, and returns the model's predicted
default probability along with an approval decision.

Usage:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import RootModel

from src.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("outputs/model.joblib")
DECISION_THRESHOLD = 0.50

app = FastAPI(
    title="Home Credit Default Risk API",
    description="Enterprise Loan Default Prediction Engine",
    version="1.0.0",
)


class CustomerPayload(RootModel[dict[str, Any]]):
    """
    Dynamic dictionary structure representing a customer's feature set.
    Avoids hardcoding 122+ features, making the API robust to upstream
    feature engineering changes.
    """


def _load_pipeline() -> Any:
    """
    Deserialise the joblib artifact from disk.
    Raises an HTTP 503 if the model is not found.
    """
    if not DEFAULT_MODEL_PATH.exists():
        logger.error("Model artifact not found at %s", DEFAULT_MODEL_PATH.resolve())
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Please run the training pipeline first.",
        )
    return joblib.load(DEFAULT_MODEL_PATH)


@app.post("/predict")
def predict_endpoint(payload: CustomerPayload) -> dict[str, Any]:
    """
    Run a single observation through the fitted pipeline and return
    a structured decision payload.
    """
    try:
        pipeline = _load_pipeline()
        
        # Convert dictionary payload to pandas DataFrame (1 row)
        observation = pd.DataFrame([payload.model_dump()])

        probabilities = pipeline.predict_proba(observation)[0]
        default_probability = float(probabilities[1])

        decision = "REJECTED" if default_probability >= DECISION_THRESHOLD else "APPROVED"

        return {
            "default_probability_pct": round(default_probability * 100, 2),
            "decision": decision,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok"}
