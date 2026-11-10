from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PCA_COMPONENTS, VARIANCE_THRESHOLD


# OopCompanion:suppressRename


def _top_anomalies(
    t_squared: np.ndarray,
    q_residual: np.ndarray,
    count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_count = len(t_squared)
    selected_count = min(count, row_count)

    def top_rows(values: np.ndarray, column: str) -> pd.DataFrame:
        positions = np.argpartition(values, row_count - selected_count)[-selected_count:]
        positions = positions[np.argsort(values[positions])[::-1]]
        return pd.DataFrame({
            "Posição": positions,
            "Observação": positions + 1,
            column: values[positions],
        })

    return top_rows(t_squared, "Hotelling T²"), top_rows(q_residual, "Q-residual")


@dataclass
class PCAResults:
    full_pca: PCA
    pca: PCA
    scores: np.ndarray
    loadings: pd.DataFrame
    variance_summary: pd.DataFrame
    kaiser_count: int
    variance_threshold_count: int
    variance_threshold: float
    analyzed_components: list[str]
    top_hotelling: pd.DataFrame
    top_q_residual: pd.DataFrame


def analyze_pca(
    features: pd.DataFrame,
    requested_components: int = PCA_COMPONENTS,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> PCAResults:
    """Standardize predictors and calculate full and retained PCA solutions."""
    if features.empty:
        raise ValueError("A matriz de atributos está vazia.")
    if features.shape[0] < 2:
        raise ValueError("A PCA necessita de pelo menos duas observações.")
    if features.shape[1] < 2:
        raise ValueError("A análise dos scores requer pelo menos duas variáveis válidas.")
    if requested_components < 2:
        raise ValueError("A análise dos scores requer pelo menos duas componentes.")
    if not 0 < variance_threshold <= 100:
        raise ValueError("O limiar de variância deve estar entre 0 e 100%.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.to_numpy(dtype=np.float64, copy=False))
    if not np.isfinite(scaled).all():
        raise ValueError("A estandardização produziu valores não finitos.")

    full_pca = PCA()
    full_pca.fit(scaled)
    eigenvalues = full_pca.explained_variance_
    ratios = full_pca.explained_variance_ratio_
    cumulative_percent = np.cumsum(ratios) * 100
    kaiser_count = int(np.count_nonzero(eigenvalues > 1.0))
    if kaiser_count == 0:
        raise ValueError("O critério de Kaiser não selecionou componentes para calcular os resíduos.")
    threshold_indices = np.flatnonzero(cumulative_percent >= variance_threshold)
    threshold_count = int(threshold_indices[0] + 1) if len(threshold_indices) else len(ratios)

    t_squared = np.empty(len(scaled), dtype=np.float64)
    q_residual = np.empty(len(scaled), dtype=np.float64)
    model_components = full_pca.components_[:kaiser_count]
    model_eigenvalues = eigenvalues[:kaiser_count]
    for start in range(0, len(scaled), 50_000):
        stop = min(start + 50_000, len(scaled))
        block = scaled[start:stop]
        model_scores = block @ model_components.T
        t_squared[start:stop] = np.sum(np.square(model_scores) / model_eigenvalues, axis=1)
        residual_sum_squares = np.einsum("ij,ij->i", block, block) - np.einsum(
            "ij,ij->i", model_scores, model_scores
        )
        q_residual[start:stop] = np.maximum(residual_sum_squares, 0.0)
    top_hotelling, top_q_residual = _top_anomalies(t_squared, q_residual)

    component_count = min(requested_components, features.shape[1], features.shape[0])
    if component_count < 1:
        raise ValueError("Não há componentes disponíveis para a PCA.")
    pca = PCA(n_components=component_count)
    scores = pca.fit_transform(scaled)
    names = [f"PC{i + 1}" for i in range(component_count)]
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=features.columns,
        columns=names,
    )
    full_names = [f"PC{i + 1}" for i in range(len(ratios))]
    variance_summary = pd.DataFrame(
        {
            "Componente": full_names,
            "Autovalor": eigenvalues,
            "Variância explicada (%)": ratios * 100,
            "Variância acumulada (%)": cumulative_percent,
        }
    ).round(3)

    return PCAResults(
        full_pca=full_pca,
        pca=pca,
        scores=scores,
        loadings=loadings,
        variance_summary=variance_summary,
        kaiser_count=kaiser_count,
        variance_threshold_count=threshold_count,
        variance_threshold=variance_threshold,
        analyzed_components=names,
        top_hotelling=top_hotelling,
        top_q_residual=top_q_residual,
    )
