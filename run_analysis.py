"""Command-line entry point for the local CICIDS2017 PCA analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cicids_pca.analysis import analyze_pca
from cicids_pca.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, PCA_COMPONENTS
from cicids_pca.data import ensure_dataset, load_and_clean_data, prepare_features
from cicids_pca.plots import (
    plot_anomalies,
    plot_biplot,
    plot_correlations,
    plot_kaiser,
    plot_loadings,
    plot_scores,
    plot_scree,
)
from cicids_pca.presentation import build_presentation


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset input and analysis outputs."""
    parser = argparse.ArgumentParser(description="Análise PCA local do dataset CICIDS2017.")
    parser.add_argument("--csv", type=Path, help="Usa um CSV local; sem esta opção, descarrega-o via Kaggle se necessário.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Pasta local do dataset Kaggle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Pasta para gráficos, tabelas e apresentação.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Número de linhas lidas por bloco.")
    parser.add_argument("--components", type=int, default=PCA_COMPONENTS, help="Número de componentes a analisar (predefinição: 14).")
    parser.add_argument("--variance-threshold", type=float, default=80.0, help="Meta de variância acumulada em percentagem.")
    return parser.parse_args()


def main() -> None:
    """Run data preparation, PCA, plot/table exports, and presentation generation."""
    args = parse_args()
    if args.chunksize < 1 or args.components < 2:
        raise ValueError("--chunksize deve ser positivo e --components deve ser pelo menos 2.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ensure_dataset(args.csv, args.data_dir)
    print(f"Dataset: {csv_path}")
    frame, target_col, cleaning = load_and_clean_data(csv_path, args.chunksize)
    features, target, feature_names, feature_summary = prepare_features(frame, target_col)
    del frame

    print(
        f"Registos: {cleaning.original_rows:,} originais; "
        f"{cleaning.removed_rows:,} removidos; {cleaning.retained_rows:,} retidos."
    )
    print(
        f"Infinitos: {feature_summary.infinite_values:,}; "
        f"NaNs imputados: {feature_summary.missing_values_imputed:,}."
    )
    print(
        f"Colunas vazias removidas: {feature_summary.all_missing_columns}; "
        f"constantes removidas: {feature_summary.constant_columns}."
    )
    print(f"Variáveis utilizadas: {len(feature_names)}; classe: {target_col}.")

    _, strongest_pairs, _ = plot_correlations(features, output_dir)
    results = analyze_pca(features, args.components, args.variance_threshold)
    plot_kaiser(results, output_dir)
    plot_scree(results, output_dir)
    plot_loadings(results, output_dir)
    plot_scores(results, target, output_dir)
    plot_biplot(results, target, output_dir)
    plot_anomalies(results, target, output_dir)

    results.variance_summary.to_csv(
        output_dir / "variancia_componentes.csv", index=False, encoding="utf-8-sig"
    )
    results.loadings.to_csv(output_dir / "loadings.csv", encoding="utf-8-sig")

    top_rows = []
    for component in results.analyzed_components:
        loadings = results.loadings[component]
        for direction, selected in (
            ("positivo", loadings.nlargest(3)),
            ("negativo", loadings.nsmallest(3)),
        ):
            for variable, loading in selected.items():
                top_rows.append(
                    {
                        "Componente": component,
                        "Direção": direction,
                        "Variável": variable,
                        "Loading": loading,
                    }
                )
    pd.DataFrame(top_rows).to_csv(
        output_dir / "loadings_dominantes.csv", index=False, encoding="utf-8-sig"
    )

    for anomalies in (results.top_hotelling, results.top_q_residual):
        anomalies["Classe"] = (
            target.iloc[anomalies["Posição"]].fillna("Sem classe").astype(str).to_numpy()
        )
    results.top_hotelling.to_csv(
        output_dir / "top10_hotelling_t2.csv", index=False, encoding="utf-8-sig"
    )
    results.top_q_residual.to_csv(
        output_dir / "top10_q_residual.csv", index=False, encoding="utf-8-sig"
    )

    presentation_path = build_presentation(
        output_path=output_dir / "Análise_PCA_CICIDS2017.pptx",
        output_dir=output_dir,
        results=results,
        correlation_pairs=strongest_pairs,
        cleaning=cleaning,
        features=feature_summary,
        row_count=len(features),
        feature_count=len(feature_names),
        class_names=[str(name) for name in target.dropna().unique()],
    )
    print(f"Kaiser (lambda > 1): {results.kaiser_count} componentes.")
    explained = results.variance_summary.iloc[
        results.variance_threshold_count - 1
    ]["Variância acumulada (%)"]
    print(
        f"Meta {results.variance_threshold:g}%: "
        f"{results.variance_threshold_count} componentes ({explained:.2f}%)."
    )
    print(f"Resultados guardados em: {output_dir}")
    print(f"Apresentação: {presentation_path}")


if __name__ == "__main__":
    main()
