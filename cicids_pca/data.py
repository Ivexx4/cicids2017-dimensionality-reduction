"""Dataset retrieval, CICFlowMeter cleaning, and PCA feature preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CHUNK_SIZE, DATASET_SLUG, DEFAULT_DATA_DIR, EXPECTED_CSV_NAME


@dataclass
class CleaningSummary:
    """Counts of original, retained, and invalid flow records."""

    original_rows: int
    retained_rows: int
    removed_rows: int
    forward_minus_one: int
    backward_minus_one: int
    negative_min_segment: int
    class_counts_original: dict[Any, int]


@dataclass
class FeatureSummary:
    """Summary of non-finite value handling and removed predictor columns."""

    infinite_values: int
    missing_values_imputed: int
    all_missing_columns: list[str]
    constant_columns: list[str]
    excluded_columns: list[str]


def ensure_dataset(csv_path: Path | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Resolve the input CSV, downloading and extracting it from Kaggle if needed.

    Args:
        csv_path: Optional path to an already downloaded CSV.
        data_dir: Directory used to store and search for the Kaggle dataset.

    Returns:
        The resolved path to the dataset CSV.

    Raises:
        FileNotFoundError: If the provided path is missing or no unique CSV is found.
    """
    if csv_path is not None:
        csv_path = Path(csv_path).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"O CSV indicado não existe: {csv_path}")
        return csv_path

    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    expected_csv = data_dir / EXPECTED_CSV_NAME
    if not expected_csv.exists():
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_SLUG, path=str(data_dir), unzip=True, quiet=False)

    csv_files = list(data_dir.rglob("*.csv"))
    matches = [path for path in csv_files if path.name.casefold() == EXPECTED_CSV_NAME.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(csv_files) == 1:
        return csv_files[0]
    raise FileNotFoundError(
        f"Esperava encontrar {EXPECTED_CSV_NAME} ou um único CSV em {data_dir}; "
        f"foram encontrados {len(csv_files)} ficheiros CSV."
    )


def load_and_clean_data(csv_path: Path, chunksize: int = CHUNK_SIZE) -> tuple[pd.DataFrame, str, CleaningSummary]:
    """Load the dataset in chunks and remove records with CICFlowMeter error values.

    Args:
        csv_path: Path to the source dataset.
        chunksize: Number of rows to read per chunk.

    Returns:
        The cleaned rows, target-column name, and counts describing the cleaning.

    Raises:
        ValueError: If required columns or usable records are missing.
    """
    cleaned_chunks: list[pd.DataFrame] = []
    original_rows = removed_rows = 0
    forward_minus_one = backward_minus_one = negative_min_segment = 0
    class_counts_original: dict[Any, int] = {}
    target_col: str | None = None

    required = {
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "min_seg_size_forward",
    }
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if target_col is None:
            target_col = "Attack Type" if "Attack Type" in chunk.columns else "Label"
            missing = required.difference(chunk.columns)
            if missing:
                raise ValueError(f"Faltam colunas necessárias no CSV: {sorted(missing)}")
            if target_col not in chunk.columns:
                raise ValueError("O CSV tem de incluir a coluna 'Attack Type' ou 'Label'.")

        original_rows += len(chunk)
        for label, count in chunk[target_col].value_counts(dropna=False).items():
            class_counts_original[label] = class_counts_original.get(label, 0) + int(count)

        mask_forward = chunk["Init_Win_bytes_forward"].eq(-1)
        mask_backward = chunk["Init_Win_bytes_backward"].eq(-1)
        min_segment = pd.to_numeric(chunk["min_seg_size_forward"], errors="coerce")
        mask_min_segment = min_segment.lt(0)
        invalid = mask_forward | mask_backward | mask_min_segment

        forward_minus_one += int(mask_forward.sum())
        backward_minus_one += int(mask_backward.sum())
        negative_min_segment += int(mask_min_segment.sum())
        removed_rows += int(invalid.sum())
        cleaned_chunks.append(chunk.loc[~invalid].copy())

    if target_col is None or original_rows == 0:
        raise ValueError(f"O CSV não contém registos: {csv_path}")
    if not cleaned_chunks or original_rows == removed_rows:
        raise ValueError("A limpeza removeu todos os registos; não é possível continuar.")

    frame = pd.concat(cleaned_chunks, ignore_index=True)
    del cleaned_chunks
    summary = CleaningSummary(
        original_rows=original_rows,
        retained_rows=len(frame),
        removed_rows=removed_rows,
        forward_minus_one=forward_minus_one,
        backward_minus_one=backward_minus_one,
        negative_min_segment=negative_min_segment,
        class_counts_original=class_counts_original,
    )
    return frame, target_col, summary


def prepare_features(
    frame: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series, list[str], FeatureSummary]:
    """Select and clean numeric predictors for PCA while keeping the class labels.

    Infinite values are treated as missing and imputed with column medians;
    entirely missing and constant predictors are removed.

    Args:
        frame: Cleaned input records.
        target_col: Name of the class-label column to preserve separately.

    Returns:
        Predictors, target labels, retained predictor names, and a cleaning summary.

    Raises:
        ValueError: If no usable numeric predictors remain.
    """
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    excluded = [name for name in ("Destination Port", target_col) if name in numeric_columns]
    feature_columns = [name for name in numeric_columns if name not in excluded]
    if not feature_columns:
        raise ValueError("Não foram encontradas variáveis numéricas para a PCA.")

    features = frame.loc[:, feature_columns].copy()
    target = frame[target_col].copy()
    values = features.to_numpy(dtype=np.float64, copy=False)
    infinite_count = int(np.isinf(values).sum())
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    all_missing = features.columns[features.isna().all()].tolist()
    if all_missing:
        features.drop(columns=all_missing, inplace=True)

    missing_to_impute = int(features.isna().sum().sum())
    if missing_to_impute:
        features = features.fillna(features.median())

    constant = features.columns[features.nunique(dropna=False).le(1)].tolist()
    if constant:
        features.drop(columns=constant, inplace=True)
    if features.shape[1] == 0:
        raise ValueError("Não restaram variáveis válidas depois de remover colunas vazias/constantes.")

    feature_columns = features.columns.tolist()
    if not np.isfinite(features.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Persistem valores NaN ou infinitos após o tratamento.")

    summary = FeatureSummary(
        infinite_values=infinite_count,
        missing_values_imputed=missing_to_impute,
        all_missing_columns=all_missing,
        constant_columns=constant,
        excluded_columns=excluded,
    )
    return features, target, feature_columns, summary
