"""
REST API Deployment — FastAPI Model Serving.

Wraps the serialised Home Credit default risk pipeline in a cloud-ready,
containerised REST API using FastAPI.

Exposes a `/predict` endpoint that accepts a generic JSON payload
representing a customer's feature set, and returns the model's predicted
default probability along with an approval decision and an automated
reasoning via the Groq GenAI Assessor.

Usage:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import RootModel

from src.exceptions import ModelNotFoundError

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("outputs/model.joblib")
DECISION_THRESHOLD = 0.50

app = FastAPI(
    title="Home Credit Default Risk API",
    description="Enterprise Loan Default Prediction Engine with GenAI Reasoning",
    version="1.1.0",
)

try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as exc:
    logger.warning("Groq client failed to initialise. Ensure GROQ_API_KEY is set: %s", exc)
    groq_client = None


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


def _generate_rejection_reasoning(payload: dict[str, Any]) -> str:
    """
    Query the Groq API (llama3-8b-8192) to generate a clinical, professional
    explanation for a loan rejection based on the provided financial metrics.
    """
    if not groq_client:
        return "Automated reasoning unavailable: GenAI Assessor is not configured."

    system_prompt = (
        "You are a Senior Credit Risk Officer at a commercial bank. "
        "Review the provided customer financial metrics and write a clinical, "
        "dry explanation of why this loan was flagged as high-risk and rejected. "
        "Focus on standard financial metrics (e.g., debt-to-income, employment length, credit). "
        "Maximum length: 2 sentences. Do not use conversational filler."
    )
    
    user_prompt = f"Customer Financial Metrics:\n{payload}"

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama3-8b-8192",
            temperature=0.0,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("GenAI reasoning failed: %s", exc)
        return "Automated reasoning temporarily unavailable."


@app.post("/predict")
def predict_endpoint(payload: CustomerPayload) -> dict[str, Any]:
    """
    Run a single observation through the fitted pipeline and return
    a structured decision payload alongside a GenAI reasoning string.
    """
    try:
        pipeline = _load_pipeline()
        
        payload_dict = payload.model_dump()
        observation = pd.DataFrame([payload_dict])

        probabilities = pipeline.predict_proba(observation)[0]
        default_probability = float(probabilities[1])

        if default_probability >= DECISION_THRESHOLD:
            decision = "REJECTED"
            reasoning = _generate_rejection_reasoning(payload_dict)
        else:
            decision = "APPROVED"
            reasoning = "Application meets automated baseline risk thresholds."

        return {
            "default_probability_pct": round(default_probability * 100, 2),
            "decision": decision,
            "reasoning": reasoning,
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
