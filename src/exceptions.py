"""
Custom exception hierarchy for the Loan Default Prediction Pipeline.

Provides granular error types so callers can react to specific failure
modes — file I/O issues, schema drift, MLflow connectivity, missing
artifacts — instead of catching opaque base exceptions.
"""


class CreditEngineError(Exception):
    """Base exception for all loan-default-pipeline failures."""


class DataFetchError(CreditEngineError):
    """
    Raised when local CSV data files cannot be read.

    Common triggers: missing files in the ``data/`` directory,
    filesystem permission errors, or corrupt CSV content that
    causes the parser to fail.
    """


class DataFormatError(CreditEngineError):
    """
    Raised when the loaded data violates expected schema invariants.

    Examples: missing target column, empty DataFrame after joins,
    unexpected column names following an upstream schema change
    in the Kaggle dataset.
    """


class MLflowConnectionError(CreditEngineError):
    """
    Raised when the MLflow tracking backend is unreachable.

    Wraps connection failures to the SQLite backend or any remote
    tracking server so the caller can decide whether to abort or
    fall back to offline logging.
    """


class ModelNotFoundError(CreditEngineError):
    """
    Raised when a serialised model artifact cannot be located on disk.

    Typically encountered at inference time when ``predict.py`` is
    invoked before a training run has produced ``outputs/model.joblib``.
    """
