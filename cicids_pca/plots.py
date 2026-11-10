from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import PCAResults
from .config import KAISER_PLOT_COMPONENTS, RANDOM_SEED


def plot_correlations(features: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, list[tuple[str, str, float]], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation = features.corr()
    upper = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1)).stack()
    strongest = upper.reindex(upper.abs().sort_values(ascending=False).head(3).index)
    pairs = [(first, second, float(value)) for (first, second), value in strongest.items()]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        correlation, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.1, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(f"Matriz de Correlação — Conjunto X ({features.shape[1]} variáveis)", fontsize=14, pad=15)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    output = output_dir / "matriz_correlacao.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return correlation, pairs, output


def plot_kaiser(results: PCAResults, output_dir: Path) -> Path:
    values = results.full_pca.explained_variance_
    count = min(KAISER_PLOT_COMPONENTS, len(values))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, count + 1), values[:count], alpha=0.7, color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Limiar de Kaiser (λ = 1)")
    ax.set_title(f"Critério de Kaiser: PC1–PC{count}", fontsize=13, pad=15)
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Autovalor (λ)")
    ax.set_xlim(0.5, count + 0.5)
    ax.set_xticks(range(1, count + 1), [f"PC{i}" for i in range(1, count + 1)], rotation=90, fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output = output_dir / "criterio_kaiser.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_scree(results: PCAResults, output_dir: Path) -> Path:
    ratios = results.full_pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios) * 100
    count = results.variance_threshold_count
    fig, ax = plt.subplots(figsize=(10, 5))
    components = np.arange(1, len(cumulative) + 1)
    ax.plot(components, cumulative, color="#ff7f0e", marker="o", linewidth=2, markersize=5)
    ax.axhline(y=results.variance_threshold, color="green", linestyle="--", linewidth=2,
               label=f"Meta de {results.variance_threshold:g}% da variância")
    ax.scatter([count], [cumulative[count - 1]], color="red", s=80, zorder=5)
    ax.annotate(
        f"{count} PCs: {cumulative[count - 1]:.2f}%", (count, cumulative[count - 1]),
        textcoords="offset points", xytext=(-30, 15), ha="center",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    ax.set_title("Scree Plot: Variância Explicada Acumulada", fontsize=13, pad=15)
    ax.set_xlabel("Número de Componentes Principais")
    ax.set_ylabel("Variância Acumulada (%)")
    ax.set_xticks(range(1, len(cumulative) + 1, 2))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    output = output_dir / "scree_plot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_loadings(results: PCAResults, output_dir: Path) -> Path:
    loadings = results.loadings
    order = loadings["PC1"].abs().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(18, 11))
    sns.heatmap(
        loadings.loc[order], cmap="vlag", center=0, vmin=-1, vmax=1,
        linewidths=0.15, linecolor="white",
        cbar_kws={"label": "Loading (correlação variável-componente)"}, ax=ax,
    )
    ax.set_title(f"Mapa de calor dos loadings — PC1 a {loadings.shape[1]}", fontsize=14, pad=12)
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Variável (ordenada pelo loading absoluto em PC1)")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    output = output_dir / f"loadings_pcs_1_{loadings.shape[1]}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_scores(results: PCAResults, target: pd.Series, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_frame = pd.DataFrame(results.scores, columns=results.analyzed_components)
    score_frame["Attack Type"] = target.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)

    sampled: list[int] = []
    for _, indices in score_frame.groupby("Attack Type", sort=False, dropna=False).groups.items():
        indexes = np.asarray(list(indices), dtype=int)
        sampled.extend(rng.choice(indexes, size=min(2_000, len(indexes)), replace=False).tolist())
    visible = score_frame.loc[sampled]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        x="PC1", y="PC2", hue="Attack Type", data=visible,
        alpha=0.5, palette="tab10", s=15, linewidth=0, ax=ax,
    )
    ratio1, ratio2 = results.pca.explained_variance_ratio_[:2] * 100
    ax.set_title(f"Scores PC1 vs PC2 — PC1: {ratio1:.2f}%; PC2: {ratio2:.2f}%", fontsize=13, pad=15)
    ax.set_xlabel(f"PC1 ({ratio1:.2f}% da variância)")
    ax.set_ylabel(f"PC2 ({ratio2:.2f}% da variância)")
    ax.grid(True, alpha=0.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Tipo de Tráfego")
    fig.tight_layout()
    first_output = output_dir / "scores_pc1_pc2.png"
    fig.savefig(first_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    component_numbers = [int(name[2:]) for name in results.analyzed_components]
    score_pairs = [(1, number) for number in component_numbers if number > 1]
    panel_sample: list[int] = []
    for _, indices in score_frame.groupby("Attack Type", sort=False, dropna=False).groups.items():
        indexes = np.asarray(list(indices), dtype=int)
        panel_sample.extend(rng.choice(indexes, size=min(500, len(indexes)), replace=False).tolist())
    panel_data = score_frame.loc[panel_sample]
    rows = int(np.ceil(len(score_pairs) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4 * rows), squeeze=False)
    axes_flat = axes.ravel()
    handles: list = []
    labels: list[str] = []
    for plot_index, (pc_x, pc_y) in enumerate(score_pairs):
        ax = axes_flat[plot_index]
        sns.scatterplot(
            x=f"PC{pc_x}", y=f"PC{pc_y}", hue="Attack Type", data=panel_data,
            alpha=0.45, palette="tab10", s=8, linewidth=0,
            legend=(plot_index == 0), ax=ax,
        )
        if plot_index == 0:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        ax.set_title(
            f"PC{pc_x} ({results.pca.explained_variance_ratio_[pc_x - 1] * 100:.1f}%) vs "
            f"PC{pc_y} ({results.pca.explained_variance_ratio_[pc_y - 1] * 100:.1f}%)",
            fontsize=10,
        )
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(f"PC{pc_x}")
        ax.set_ylabel(f"PC{pc_y}")
    for ax in axes_flat[len(score_pairs):]:
        ax.set_visible(False)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)),
                   bbox_to_anchor=(0.5, 1.01), title="Tipo de Tráfego", fontsize=8)
    final_component = component_numbers[-1]
    fig.suptitle(f"Scores: PC1 comparada com PC2–PC{final_component}", y=1.02, fontsize=14)
    fig.tight_layout()
    panel_output = output_dir / f"scores_pcs_1_{final_component}.png"
    fig.savefig(panel_output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return first_output, panel_output


def plot_biplot(results: PCAResults, target: pd.Series, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_frame = pd.DataFrame(results.scores[:, :2], columns=["PC1", "PC2"])
    score_frame["Attack Type"] = target.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    sampled: list[int] = []
    for _, indices in score_frame.groupby("Attack Type", sort=False, dropna=False).groups.items():
        indexes = np.asarray(list(indices), dtype=int)
        sampled.extend(rng.choice(indexes, size=min(1_000, len(indexes)), replace=False).tolist())
    visible = score_frame.loc[sampled]

    loadings = results.loadings[["PC1", "PC2"]]
    strongest = loadings.pow(2).sum(axis=1).nlargest(12).index
    selected_loadings = loadings.loc[strongest]
    loading_scale = 0.8 * min(
        visible["PC1"].abs().quantile(0.98) / max(selected_loadings["PC1"].abs().max(), 1e-12),
        visible["PC2"].abs().quantile(0.98) / max(selected_loadings["PC2"].abs().max(), 1e-12),
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="PC1", y="PC2", hue="Attack Type", data=visible,
        alpha=0.35, palette="tab10", s=12, linewidth=0, ax=ax,
    )
    for variable, row in selected_loadings.iterrows():
        x, y = row["PC1"] * loading_scale, row["PC2"] * loading_scale
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": "#c43c39", "lw": 1.3, "alpha": 0.85})
        ax.text(x * 1.06, y * 1.06, str(variable), color="#8f2825", fontsize=8,
                ha="center", va="center")

    ratio1, ratio2 = results.pca.explained_variance_ratio_[:2] * 100
    ax.set_title("Biplot: scores e loadings (PC1–PC2)", fontsize=14, pad=15)
    ax.set_xlabel(f"PC1 ({ratio1:.2f}% da variância)")
    ax.set_ylabel(f"PC2 ({ratio2:.2f}% da variância)")
    ax.axhline(0, color="grey", linewidth=0.7, alpha=0.6)
    ax.axvline(0, color="grey", linewidth=0.7, alpha=0.6)
    ax.grid(True, alpha=0.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Tipo de Tráfego")
    fig.tight_layout()
    output = output_dir / "biplot_pc1_pc2.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_anomalies(results: PCAResults, target: pd.Series, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    hotelling = results.top_hotelling.copy()
    residuals = results.top_q_residual.copy()
    hotelling["Classe"] = target.iloc[hotelling["Posição"]].fillna("Sem classe").astype(str).to_numpy()
    residuals["Classe"] = target.iloc[residuals["Posição"]].fillna("Sem classe").astype(str).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, data, metric, title, color in (
        (axes[0], hotelling, "Hotelling T²", "Maiores Hotelling T²", "#3977a8"),
        (axes[1], residuals, "Q-residual", "Maiores Q-residual", "#c2673a"),
    ):
        ordered = data.sort_values(metric, ascending=True)
        labels = [f"Obs. {row.Observação} · {row.Classe}" for row in ordered.itertuples()]
        ax.barh(labels, ordered[metric], color=color, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(metric)
        ax.grid(True, axis="x", alpha=0.2)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"10 observações mais extremas — modelo com {results.kaiser_count} PCs (λ > 1)", fontsize=14)
    fig.tight_layout()
    output = output_dir / "outliers_hotelling_q_residual.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output
