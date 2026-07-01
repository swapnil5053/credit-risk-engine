"""
High-Performance Data Engineering Layer — Polars Lazy Evaluation.

Ingests the Home Credit Default Risk relational dataset using Polars'
lazy evaluation engine. Three CSVs are scanned lazily, aggregated by
``SK_ID_CURR``, joined into a single denormalised feature matrix, and
materialised to pandas exactly once for downstream sklearn compatibility.

Polars' lazy API defers all I/O and computation until ``.collect()`` is
called, allowing the query optimiser to push down predicates, prune
unused columns, and parallelise across CPU cores — critical when
processing 300k+ rows with relational joins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from src.exceptions import DataFetchError, DataFormatError

logger = logging.getLogger(__name__)


def load_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Orchestrate the full Home Credit data ingestion pipeline.

    Steps:
        1. Lazy-scan all three CSVs via ``pl.scan_csv``.
        2. Aggregate ``bureau`` and ``previous_application`` by ``SK_ID_CURR``.
        3. Left-join aggregations onto the ``application_train`` spine.
        4. Materialise to pandas for sklearn compatibility.
        5. Separate features from target, dropping the join key.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration with paths under the ``data`` key.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and binary target vector (0 = repaid, 1 = default).
    """
    data_cfg = config["data"]
    target_col: str = data_cfg["target_column"]

    application = _scan_csv_safe(Path(data_cfg["application_train"]))
    bureau = _scan_csv_safe(Path(data_cfg["bureau"]))
    prev_app = _scan_csv_safe(Path(data_cfg["previous_application"]))

    bureau_agg = _aggregate_bureau(bureau)
    prev_agg = _aggregate_previous_applications(prev_app)

    merged_df = _join_and_materialise(application, bureau_agg, prev_agg)
    pandas_df = merged_df.to_pandas()

    _validate_schema(pandas_df, target_col)

    target: pd.Series = pandas_df[target_col]
    features: pd.DataFrame = pandas_df.drop(columns=[target_col, "SK_ID_CURR"])

    logger.info("Features shape: %s", features.shape)
    logger.info(
        "Target distribution:\n%s",
        target.value_counts().to_string(),
    )

    return features, target


def _scan_csv_safe(path: Path) -> pl.LazyFrame:
    """
    Wrap ``pl.scan_csv`` with structured error handling.

    Raises ``DataFetchError`` if the file does not exist, providing
    a clear diagnostic rather than Polars' internal ``ComputeError``.
    """
    if not path.exists():
        raise DataFetchError(
            f"Required data file not found: {path.resolve()}. "
            "Run 'python -m src.download_data' to fetch the dataset from Kaggle."
        )

    try:
        return pl.scan_csv(path, infer_schema_length=10000)
    except Exception as exc:
        raise DataFetchError(
            f"Failed to scan CSV at {path}: {exc}"
        ) from exc


def _aggregate_bureau(bureau: pl.LazyFrame) -> pl.LazyFrame:
    """
    Collapse the bureau credit history table to one row per applicant.

    Computes summary statistics that capture the breadth and magnitude
    of an applicant's prior credit relationships as seen by other
    financial institutions.
    """
    return bureau.group_by("SK_ID_CURR").agg(
        pl.col("SK_ID_BUREAU").count().alias("bureau_loan_count"),
        pl.col("DAYS_CREDIT").mean().alias("bureau_days_credit_mean"),
        pl.col("DAYS_CREDIT").max().alias("bureau_days_credit_max"),
        pl.col("AMT_CREDIT_SUM").mean().alias("bureau_amt_credit_mean"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("bureau_debt_sum"),
    )


def _aggregate_previous_applications(prev: pl.LazyFrame) -> pl.LazyFrame:
    """
    Collapse previous Home Credit applications to one row per applicant.

    Captures historical application behaviour: how many times the
    applicant has applied, and the typical loan size and term.
    """
    return prev.group_by("SK_ID_CURR").agg(
        pl.col("SK_ID_PREV").count().alias("prev_app_count"),
        pl.col("AMT_APPLICATION").mean().alias("prev_amt_application_mean"),
        pl.col("AMT_CREDIT").mean().alias("prev_amt_credit_mean"),
        pl.col("CNT_PAYMENT").mean().alias("prev_cnt_payment_mean"),
    )


def _join_and_materialise(
    application: pl.LazyFrame,
    bureau_agg: pl.LazyFrame,
    prev_agg: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Left-join aggregated relational tables onto the application spine
    and collect into a materialised Polars DataFrame.

    Left joins ensure every row in ``application_train`` is preserved
    even if an applicant has no bureau or previous-application history
    (those columns will be null, handled by XGBoost natively).
    """
    joined = (
        application
        .join(bureau_agg, on="SK_ID_CURR", how="left")
        .join(prev_agg, on="SK_ID_CURR", how="left")
    )

    logger.info("Executing Polars query plan (lazy → materialised) …")
    result = joined.collect()
    logger.info("Materialised DataFrame: %d rows × %d columns.", result.height, result.width)

    return result


def _validate_schema(df: pd.DataFrame, target_col: str) -> None:
    """
    Assert basic structural invariants on the joined dataset.

    Guards against silent upstream schema changes and ensures the
    target column survived the join and materialisation process.
    """
    if df.empty:
        raise DataFormatError(
            "Materialised DataFrame is empty after relational joins."
        )

    if target_col not in df.columns:
        raise DataFormatError(
            f"Target column '{target_col}' not found in the materialised DataFrame. "
            f"Available columns: {list(df.columns[:10])} … ({len(df.columns)} total)"
        )

    unique_targets = df[target_col].nunique()
    if unique_targets != 2:
        raise DataFormatError(
            f"Expected binary target, found {unique_targets} unique values "
            f"in column '{target_col}'."
        )
