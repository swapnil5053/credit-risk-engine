# Credit Risk Engine

An end-to-end, containerized microservice for real-time credit default prediction.

Built on the 300,000+ row Home Credit Default Risk dataset, with Bayesian hyperparameter tuning (Optuna), MLflow experiment tracking, and a containerized FastAPI inference endpoint.

> **Project Origin:** The idea for this project emerged while completing IBM's AI Fundamentals course, where I explored IBM Watson's automated machine learning workflow. That experience motivated me to build a similar credit risk pipeline from scratch, implementing the data engineering, model training, optimization, deployment, and monitoring components myself.

## System Architecture

The pipeline is entirely modular and config-driven, built with the following stack:

- **Data Engineering**: Polars for lazy-evaluated, memory-efficient relational joins across multiple banking tables.
- **Machine Learning**: XGBoost utilizing native GPU acceleration (`device: cuda`) and class-weighting (`scale_pos_weight`) to mathematically penalize false negatives.
- **Bayesian Optimization**: Optuna replaces brute-force grid search, utilizing probabilistic models to navigate the hyperparameter space efficiently.
- **MLOps Tracking**: MLflow integrated via Optuna callbacks to log trials, artifacts, and evaluation metrics to a local SQLite backend.
- **Explainability**: SHAP (SHapley Additive exPlanations) for compliance-ready, tree-based feature importance mapping.
- **Automated Risk Rationale**: LLM integration (Llama-3.1) to dynamically generate clinical, natural-language rationale for credit rejections.
- **Deployment**: FastAPI and Docker to wrap the predictive model in a highly available REST API.
- **CI/CD & DevOps**: GitHub Actions workflow for automated Docker builds and image publishing to the GitHub Container Registry (GHCR).

## Training Results & Evaluation

Hyperparameter optimization was executed using Optuna Bayesian Optimization over the XGBoost space.

**Test Set Evaluation (30,752 samples):**
- **Overall Accuracy**: 86.4%
- **Default Recall**: 38.9% (Successfully flagged ~39% of all actual defaults in a highly imbalanced environment where only ~8% of total applications default)
- **Default Precision**: 26.6%
- **F1-Score (Default Class)**: 0.316

## MLOps Dashboard

To view the tracking dashboard locally:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Local Setup & Deployment

**Prerequisites:**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Fetch data and run the training pipeline:
```bash
python src/download_data.py
python main.py
```

Serve the Model (FastAPI):
```bash
uvicorn src.api:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` to test the API via the interactive Swagger UI, or use the provided utility script to run a local inference test:
```bash
python scripts/test_api_inference.py
```

Run via Docker:
```bash
docker build -t credit-risk-engine .
docker run -p 8000:8000 credit-risk-engine
```
