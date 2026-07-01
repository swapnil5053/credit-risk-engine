# Loan Default Prediction Pipeline

An enterprise-grade credit default prediction system built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. The pipeline processes 300k+ relational records using Polars lazy evaluation, trains a class-balanced XGBoost classifier via cross-validated grid search, produces SHAP-based model compliance artifacts, and tracks all experiments through MLflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                   │
│  MLflow lifecycle · train/test split · artifact serialisation   │
├─────────┬──────────────┬──────────────────┬─────────────────────┤
│  config │ data_loader  │    pipeline      │  interpretability   │
│  .yaml  │  (Polars)    │ (sklearn+XGB)    │     (SHAP)          │
├─────────┼──────────────┼──────────────────┼─────────────────────┤
│         │ bureau.csv   │ StandardScaler   │  TreeExplainer      │
│         │ prev_app.csv │ OneHotEncoder    │  Beeswarm plot      │
│         │ app_train.csv│ scale_pos_weight │  MLflow artifacts   │
└─────────┴──────────────┴──────────────────┴─────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Polars lazy evaluation** | `pl.scan_csv` defers I/O, enabling query plan optimisation and reduced peak memory for 300k+ rows |
| **`scale_pos_weight` over SMOTE** | Avoids O(n²) nearest-neighbour computation on large datasets; handles imbalance natively in XGBoost's gradient computation |
| **Standard sklearn Pipeline** | Removes the `imbalanced-learn` dependency entirely; simpler serialisation and broader ecosystem compatibility |
| **MLflow tracking** | Full experiment reproducibility — hyperparameters, metrics, model artifacts, and SHAP plots are versioned per run |
| **Kaggle auto-download** | `src/download_data.py` fetches and extracts competition data on first run, eliminating manual data setup |

## Prerequisites

- Python 3.10+
- A [Kaggle API token](https://www.kaggle.com/settings) at `~/.kaggle/kaggle.json`

## Quick Start

```bash
# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run the full training pipeline
python main.py

# Launch MLflow UI to inspect experiments
mlflow ui --backend-store-uri sqlite:///mlruns.db

# Run inference on a single customer
python predict.py --payload '{"AMT_INCOME_TOTAL": 270000, "AMT_CREDIT": 1293502, ...}'
```

## Project Structure

```
loan-default-pipeline/
├── config/
│   └── settings.yaml           # Externalised configuration (paths, hyperparams, MLflow)
├── src/
│   ├── __init__.py
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── download_data.py        # Kaggle dataset auto-download
│   ├── data_loader.py          # Polars lazy eval + relational joins
│   ├── pipeline.py             # sklearn Pipeline + XGBoost (scale_pos_weight)
│   └── interpretability.py     # SHAP TreeExplainer + MLflow artifact logging
├── main.py                     # MLflow-integrated training orchestrator
├── predict.py                  # CLI single-record inference
├── requirements.txt            # Pinned dependencies
└── data/                       # Auto-populated by download_data.py (git-ignored)
    ├── application_train.csv
    ├── bureau.csv
    └── previous_application.csv
```

## Outputs

After a successful training run:

| Artifact | Location | Also logged to |
|---|---|---|
| `model.joblib` | `outputs/` | MLflow artifact store |
| `shap_summary.png` | `outputs/` | MLflow artifact store |
| `metrics.json` | `outputs/` | MLflow artifact store |

## License

This project is for educational and research purposes.
