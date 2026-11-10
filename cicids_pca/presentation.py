from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from .analysis import PCAResults
from .data import CleaningSummary, FeatureSummary


def _rgb(r: int, g: int, b: int) -> RGBColor:
    return RGBColor(r, g, b)


BG = _rgb(248, 249, 250)
CARD = _rgb(255, 255, 255)
BORDER = _rgb(222, 226, 230)
PRIMARY = _rgb(16, 44, 87)
SECONDARY = _rgb(53, 162, 235)
TEXT = _rgb(33, 37, 41)
MUTED = _rgb(108, 117, 125)


def _background(slide, width: float = 13.333, height: float = 7.5) -> None:
    shape = slide.shapes.add_shape(1, 0, 0, Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BG
    shape.line.color.rgb = BG


def _header(slide, title: str) -> None:
    box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.95))
    frame = box.text_frame
    frame.word_wrap = True
    p = frame.paragraphs[0]
    p.text = "TRABALHO DE GRUPO I — ESTATÍSTICA PARA A CIÊNCIA DE DADOS"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = SECONDARY
    p = frame.add_paragraph()
    p.text = title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = PRIMARY


def _card(slide, left, top, width, height, title: str, paragraphs: list[str], font_size: int = 13) -> None:
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = CARD
    shape.line.color.rgb = BORDER
    shape.line.width = Pt(1)
    box = slide.shapes.add_textbox(left + Inches(0.22), top + Inches(0.18), width - Inches(0.44), height - Inches(0.36))
    frame = box.text_frame
    frame.word_wrap = True
    title_p = frame.paragraphs[0]
    title_p.text = title
    title_p.font.size = Pt(17)
    title_p.font.bold = True
    title_p.font.color.rgb = PRIMARY
    title_p.space_after = Pt(8)
    for text in paragraphs:
        p = frame.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = TEXT
        p.space_after = Pt(7)


def _text(slide, text: str, left, top, width, height, size: int = 12, color: RGBColor = TEXT, italic: bool = False) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    box.text_frame.word_wrap = True
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.italic = italic


def _picture_contain(slide, image_path: Path, left, top, max_width, max_height) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Imagem para apresentação não existe: {image_path}")
    picture = slide.shapes.add_picture(str(image_path), left, top)
    scale = min(max_width / picture.width, max_height / picture.height)
    picture.width = int(picture.width * scale)
    picture.height = int(picture.height * scale)
    picture.left = left + (max_width - picture.width) // 2
    picture.top = top + (max_height - picture.height) // 2


def _loading_card(slide, name: str, results: PCAResults, left, top, width, height) -> None:
    card = slide.shapes.add_shape(1, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD
    card.line.color.rgb = BORDER
    card.line.width = Pt(1)
    box = slide.shapes.add_textbox(left + Inches(0.28), top + Inches(0.18), width - Inches(0.56), height - Inches(0.36))
    frame = box.text_frame
    frame.word_wrap = True

    p = frame.paragraphs[0]
    p.text = f"{name}: Análise dos Loadings"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = SECONDARY
    p.space_after = Pt(7)

    row = results.variance_summary.loc[results.variance_summary["Componente"] == name].iloc[0]
    p = frame.add_paragraph()
    p.text = f"Explica {row['Variância explicada (%)']:.2f}% da variância total"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(8)

    loadings = results.loadings[name]
    positive = loadings.nlargest(3)
    negative = loadings.nsmallest(3)
    for heading, values in (("Variáveis dominantes (positivas):", positive), ("Variáveis dominantes (negativas):", negative)):
        p = frame.add_paragraph()
        p.text = heading
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = TEXT
        p.space_after = Pt(1)
        p = frame.add_paragraph()
        p.text = "\n".join(f"{variable} ({value:+.3f})" for variable, value in values.items())
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT
        p.space_after = Pt(6)

    positive_names = ", ".join(positive.head(2).index)
    negative_names = ", ".join(negative.head(2).index)
    p = frame.add_paragraph()
    p.text = "Interpretação: scores altos associam-se a valores maiores em " + positive_names + " e menores em " + negative_names + "; scores baixos indicam a direção inversa."
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT


def build_presentation(
    output_path: Path,
    output_dir: Path,
    results: PCAResults,
    correlation_pairs: list[tuple[str, str, float]],
    cleaning: CleaningSummary,
    features: FeatureSummary,
    row_count: int,
    feature_count: int,
    class_names: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _card(slide, Inches(1.5), Inches(1.8), Inches(10.333), Inches(3.8), "", [])
    box = slide.shapes.add_textbox(Inches(1.8), Inches(2.2), Inches(9.7), Inches(3.0))
    frame = box.text_frame
    frame.word_wrap = True
    p = frame.paragraphs[0]
    p.text = "Análise de Componentes Principais (PCA)"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p = frame.add_paragraph()
    p.text = "Em Dados de Tráfego de Rede e Cibersegurança (CICIDS2017)"
    p.font.size = Pt(18)
    p.font.color.rgb = SECONDARY
    p.space_after = Pt(20)
    p = frame.add_paragraph()
    p.text = "Trabalho de Grupo I — Estatística para a Ciência de Dados\nGrupo / Autor: Ivo Rosa, Nº 2024209219\nData: 2026"
    p.font.size = Pt(13)
    p.font.color.rgb = MUTED

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Descrição do Problema e Conjunto de Dados")
    _card(slide, Inches(0.8), Inches(1.7), Inches(5.6), Inches(4.9), "Descrição do Problema", [
        "Contexto: análise exploratória do CICIDS2017 com tráfego normal e diversos tipos de ataque.",
        f"Objetivo: resumir {feature_count} variáveis numéricas e interpretar as principais fontes de variabilidade através da PCA.",
        "A PCA é não supervisionada; a variável de classe é usada apenas para comparar os scores.",
    ])
    _card(slide, Inches(6.8), Inches(1.7), Inches(5.6), Inches(4.9), "Descrição dos Dados", [
        "Fonte: CICIDS2017 Cleaned & Preprocessed (Kaggle), cicids2017_cleaned.csv.",
        f"Estrutura: {cleaning.original_rows:,} registos originais e 53 colunas.",
        f"Foco: {feature_count} variáveis contínuas de duração dos fluxos, intervalos, tamanhos, contagens e cabeçalhos.",
        "Destination Port é excluída; Attack Type é usada para agrupar os scores.",
    ])

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Preparação e Análise de Correlação")
    _text(slide, "Metodologia de Preparação", Inches(0.8), Inches(1.45), Inches(5.6), Inches(0.45), 21, PRIMARY)
    preparation = [
        f"✓ Limpeza: removidas {cleaning.removed_rows:,} linhas inválidas ({cleaning.removed_rows / cleaning.original_rows * 100:.2f}%); retidos {row_count:,} fluxos.",
        f"✓ Validação: {features.infinite_values:,} infinitos e {features.missing_values_imputed:,} valores em falta tratados; {len(features.constant_columns) + len(features.all_missing_columns)} colunas vazias/constantes removidas.",
        f"✓ Estandardização Z-score aplicada às {feature_count} variáveis utilizadas na PCA.",
    ]
    y = 2.0
    for text in preparation:
        _text(slide, text, Inches(0.8), Inches(y), Inches(5.5), Inches(0.72), 12)
        y += 0.83
    _text(slide, "Análise de Correlação", Inches(0.8), Inches(4.55), Inches(5.6), Inches(0.45), 21, PRIMARY)
    examples = "; ".join(f"{a}–{b} ({v:+.2f})" for a, b, v in correlation_pairs) or "sem pares disponíveis"
    _text(slide, f"Associações lineares mais fortes: {examples}. Correlação não implica causalidade; associações fortes sugerem informação redundante.", Inches(0.8), Inches(5.12), Inches(5.55), Inches(1.25), 12)
    _picture_contain(slide, output_dir / "matriz_correlacao.png", Inches(6.8), Inches(1.55), Inches(5.6), Inches(4.9))
    _text(slide, "Matriz de correlação das variáveis usadas na PCA", Inches(6.8), Inches(6.5), Inches(5.6), Inches(0.35), 10, MUTED, True)

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Seleção de Componentes: Critério de Kaiser")
    _card(slide, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Autovalores (λ > 1)", [
        "Retêm-se as componentes cujo autovalor excede 1, isto é, que explicam mais variância do que uma variável estandardizada.",
        f"No conjunto analisado, {results.kaiser_count} componentes satisfazem o critério de Kaiser.",
        "O gráfico apresenta as primeiras 20 componentes; o cálculo do critério considera todas.",
    ])
    _picture_contain(slide, output_dir / "criterio_kaiser.png", Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2))

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Seleção de Componentes: Variância Acumulada")
    row = results.variance_summary.iloc[results.variance_threshold_count - 1]
    _card(slide, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Critério da Variância Acumulada", [
        f"Meta: pelo menos {results.variance_threshold:.0f}% da variância total.",
        f"São necessárias {results.variance_threshold_count} componentes; acumulam {row['Variância acumulada (%)']:.2f}%.",
        "O scree plot mostra a variância acumulada à medida que são adicionadas componentes.",
    ])
    _picture_contain(slide, output_dir / "scree_plot.png", Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2))

    component_names = results.analyzed_components
    for start in range(0, len(component_names), 2):
        names = component_names[start:start + 2]
        slide = prs.slides.add_slide(blank)
        _background(slide)
        _header(slide, f"PCA: Interpretação dos Loadings ({' e '.join(names)})")
        if len(names) == 2:
            positions = [Inches(0.8), Inches(6.8)]
            width = Inches(5.6)
        else:
            positions = [Inches(3.166)]
            width = Inches(7.0)
        for name, left in zip(names, positions):
            _loading_card(slide, name, results, left, Inches(1.5), width, Inches(5.2))

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, f"Mapa de Calor dos Loadings: PC1 a {len(component_names)}")
    _picture_contain(
        slide, output_dir / f"loadings_pcs_1_{len(component_names)}.png",
        Inches(2.0), Inches(1.35), Inches(9.3), Inches(5.8),
    )

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, f"PCA: Scores das {len(component_names)} Componentes")
    ratio_first_two = float(results.variance_summary.iloc[:2]["Variância explicada (%)"].sum())
    _card(slide, Inches(0.8), Inches(1.5), Inches(4.0), Inches(5.2), "Projeções por classe", [
        f"O painel compara PC1 com PC2 até PC{len(component_names)}.",
        f"PC1 + PC2 explicam {ratio_first_two:.2f}% da variância.",
        "A amostragem estratificada facilita a visualização das classes menos frequentes.",
        "A PCA é não supervisionada; a sobreposição em duas dimensões não resume toda a informação.",
    ])
    _picture_contain(
        slide, output_dir / f"scores_pcs_1_{len(component_names)}.png",
        Inches(4.95), Inches(1.55), Inches(7.9), Inches(5.2),
    )

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Biplot: Scores e Loadings (PC1–PC2)")
    _picture_contain(
        slide, output_dir / "biplot_pc1_pc2.png",
        Inches(1.0), Inches(1.4), Inches(11.3), Inches(5.1),
    )
    _text(
        slide,
        "Os pontos representam fluxos; as setas mostram as variáveis com maior contribuição conjunta para PC1 e PC2.",
        Inches(1.0), Inches(6.7), Inches(11.3), Inches(0.35), 10, MUTED, True,
    )

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Deteção de Observações Atípicas: T² e Q-residual")
    _card(slide, Inches(0.8), Inches(1.5), Inches(3.2), Inches(5.2), "Leitura dos indicadores", [
        f"O modelo usa {results.kaiser_count} PCs, selecionadas por autovalores λ > 1.",
        "Hotelling T²: distância da observação no espaço das componentes retidas.",
        "Q-residual: variação não explicada pela reconstrução com essas componentes.",
        "Os fluxos no topo são candidatos a investigação; estes indicadores, isoladamente, não confirmam ataques.",
    ], font_size=12)
    _picture_contain(
        slide, output_dir / "outliers_hotelling_q_residual.png",
        Inches(4.2), Inches(1.5), Inches(8.2), Inches(5.2),
    )

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Discussão e Conclusões")
    retained = results.variance_summary.iloc[results.variance_threshold_count - 1]
    _card(slide, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Discussão", [
        f"A PCA resume {feature_count} variáveis em componentes ortogonais; {results.variance_threshold_count} explicam {retained['Variância acumulada (%)']:.2f}% da variância.",
        f"O critério de Kaiser indica {results.kaiser_count} componentes; os loadings suportam a interpretação dos seus eixos.",
        f"A limpeza reteve {cleaning.retained_rows:,} de {cleaning.original_rows:,} fluxos; conclusões aplicam-se aos registos retidos.",
    ])
    _card(slide, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Conclusões", [
        f"Redução: {feature_count} atributos para {results.variance_threshold_count} componentes com pelo menos {results.variance_threshold:.0f}% da variância.",
        "Os scores e loadings podem apoiar análises posteriores, mas a PCA não é por si só um classificador.",
        f"Classes presentes: {', '.join(class_names[:8])}{'…' if len(class_names) > 8 else ''}.",
    ])

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _header(slide, "Referências e Declaração de Uso de IA")
    _card(slide, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Referências", [
        "Estatística para as Ciências dos Dados. (2026). Slides das aulas [Material de apoio da unidade curricular, LECD, 2.º ano].",
        "Ribeiro, E. A. (s.d.). CICIDS2017: Cleaned & Preprocessed [Conjunto de dados]. Kaggle. https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed",
        "Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a Dataset for Assessing the Activity of Intrusion Detection Systems. ICISSP.",
    ], font_size=11)
    _card(slide, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2), "Declaração de Transparência", [
        "Este trabalho foi revisto com auxílio de ferramentas de IA (Gemini e ChatGPT) para apoiar a estruturação, a verificação gramatical e a clareza expositiva.",
        "A execução do código, os cálculos estatísticos, a geração dos gráficos e a responsabilidade pelo conteúdo e conclusões pertencem ao autor.",
    ])

    slide = prs.slides.add_slide(blank)
    _background(slide)
    _card(slide, Inches(2.5), Inches(2.2), Inches(8.333), Inches(3.0), "", [])
    box = slide.shapes.add_textbox(Inches(2.7), Inches(2.8), Inches(7.933), Inches(1.8))
    frame = box.text_frame
    frame.word_wrap = True
    p = frame.paragraphs[0]
    p.text = "Obrigado!"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p = frame.add_paragraph()
    p.text = "Questões?"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(22)
    p.font.color.rgb = SECONDARY

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    return output_path
