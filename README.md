# Enterprise Loan Default Prediction Engine

An end-to-end risk analytics engine built to evaluate instant credit applications, optimize F1-score classification, and export production-ready model artifacts.

This project was developed to process the 300,000+ row relational Home Credit Default Risk dataset. It serves as a complete microservice, featuring automated hyperparameter tuning and a containerized REST API for real-time inference.

## System Architecture

The pipeline is entirely modular and config-driven, built with the following stack:

- **Data Engineering**: Polars for lazy-evaluated, memory-efficient relational joins across multiple banking tables.
- **Machine Learning**: XGBoost utilizing native GPU acceleration (`device: cuda`) and `scale_pos_weight` to mathematically penalize false negatives.
- **Bayesian Optimization**: Optuna replaces brute-force grid search, utilizing probabilistic models to find optimal hyperparameters efficiently.
- **MLOps Tracking**: MLflow integrated via Optuna callbacks to log trials, parameters, and model artifacts to a local SQLite backend.
- **Explainability**: SHAP (SHapley Additive exPlanations) for compliance-ready, tree-based feature importance mapping.
- **Deployment**: FastAPI and Docker to wrap the predictive model in a cloud-ready, containerized REST API.

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
Navigate to `http://127.0.0.1:8000/docs` to test the API via the interactive Swagger UI.

Run via Docker:
```bash
docker build -t credit-risk-engine .
docker run -p 8000:8000 credit-risk-engine
```
