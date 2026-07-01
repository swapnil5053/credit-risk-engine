"""
Kaggle Dataset Download Utility — Home Credit Default Risk.

Automates the retrieval and extraction of the Home Credit Default Risk
competition dataset from Kaggle. Checks for the presence of required
CSV files before initiating a download, making it safe to call
repeatedly without redundant network I/O.

Prerequisites:
    A valid ``~/.kaggle/kaggle.json`` API token must exist. Generate one
    from https://www.kaggle.com/settings → "Create New Token".

Usage::

    python -m src.download_data
    # or called programmatically:
    from src.download_data import ensure_dataset
    ensure_dataset(Path("data"))
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.exceptions import DataFetchError

logger = logging.getLogger(__name__)

KAGGLE_COMPETITION = "home-credit-default-risk"

REQUIRED_FILES = [
    "application_train.csv",
    "bureau.csv",
    "previous_application.csv",
]


def ensure_dataset(data_dir: Path) -> None:
    """
    Verify that all required CSVs exist in ``data_dir``. If any are
    missing, download the full competition dataset from Kaggle and
    extract it.

    Parameters
    ----------
    data_dir : Path
        Local directory where the CSVs should reside.

    Raises
    ------
    DataFetchError
        If the Kaggle API is not configured, the download fails,
        or required files are still missing after extraction.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]

    if not missing:
        logger.info("All required data files present in %s", data_dir)
        return

    logger.info(
        "Missing %d file(s) in %s: %s — initiating Kaggle download.",
        len(missing),
        data_dir,
        missing,
    )

    _download_and_extract(data_dir)

    still_missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if still_missing:
        raise DataFetchError(
            f"Post-download verification failed. Still missing: {still_missing}. "
            f"Check that the competition '{KAGGLE_COMPETITION}' contains these files."
        )

    logger.info("Dataset download and extraction complete.")


def _download_and_extract(data_dir: Path) -> None:
    """
    Invoke the Kaggle API to download and unzip competition files.

    Defers the ``kaggle`` import to runtime so the rest of the codebase
    can be imported and tested without a Kaggle token present.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise DataFetchError(
            "The 'kaggle' package is not installed. "
            "Run: pip install kaggle"
        ) from exc

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        raise DataFetchError(
            "Kaggle authentication failed. Ensure ~/.kaggle/kaggle.json "
            f"contains a valid API token. Details: {exc}"
        ) from exc

    try:
        logger.info(
            "Downloading '%s' competition data to %s …",
            KAGGLE_COMPETITION,
            data_dir,
        )
        api.competition_download_files(
            KAGGLE_COMPETITION,
            path=str(data_dir),
            quiet=False,
        )
    except Exception as exc:
        raise DataFetchError(
            f"Failed to download competition data: {exc}"
        ) from exc

    _extract_archives(data_dir)


def _extract_archives(data_dir: Path) -> None:
    """
    Extract any ZIP archives found in the data directory and remove
    the archive files afterward to conserve disk space.
    """
    import zipfile

    for archive in data_dir.glob("*.zip"):
        logger.info("Extracting %s …", archive.name)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(data_dir)
        archive.unlink()
        logger.info("Removed archive %s after extraction.", archive.name)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    ensure_dataset(Path("data"))
