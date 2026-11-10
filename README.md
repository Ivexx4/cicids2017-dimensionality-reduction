# cicids2017-dimensionality-reduction

Principal Component Analysis (PCA) applied to the CICIDS2017 intrusion detection dataset for dimensionality reduction, exploratory data analysis, feature interpretation, and anomaly detection.

## Overview

This project explores the application of Principal Component Analysis (PCA) to the CICIDS2017 dataset, a widely used benchmark for network intrusion detection research.

The analysis focuses on:

- Exploratory Data Analysis (EDA);
- Data cleaning and preprocessing;
- Feature standardization;
- Dimensionality reduction through PCA;
- Selection and interpretation of principal components;
- Analysis of scores and loadings;
- Detection of potentially anomalous observations using Hotelling's T² and Q-residual statistics.

The codebase is organized into modular Python components responsible for data loading, preprocessing, PCA analysis, visualization, and report generation. All outputs are stored in the `output/` directory.

---

## Dataset

### CICIDS2017

The Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset (CICIDS2017) contains realistic network traffic data including both benign activity and multiple types of cyberattacks.

Dataset source:

- Canadian Institute for Cybersecurity (CIC)
- Available through Kaggle

---

## Project Structure

```text
cicids2017-dimensionality-reduction/
│
├── data/
├── output/
├── scripts/
├── run_analysis.py
├── requirements.txt
├── README.md
└── .venv/
```

---

## Requirements

- Python 3.10+
- Windows PowerShell (recommended)
- Kaggle API credentials (optional if a local CSV is available)

---

## Automatic Validation

GitHub Actions runs on every push and pull request. It checks Python syntax and runs the unit tests on Python 3.10, 3.13, and the latest stable Python 3 release (`3.x`). The checks use synthetic test data and do not require Kaggle credentials or the CICIDS2017 dataset.

---

## Installation

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## Dataset Download (Kaggle)

If the dataset is not already available locally, the project can download it automatically using the Kaggle API.

### Step 1

Create a Kaggle API token:

1. Sign in to Kaggle.
2. Navigate to **Settings**.
3. Under **API**, select **Create New Token**.

### Step 2

Save the downloaded `kaggle.json` file to:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

### Step 3

Accept the dataset terms on Kaggle before running the script.

> Never commit or share your Kaggle API token.

---

## Running the Analysis

Execute the complete PCA pipeline:

```powershell
python run_analysis.py
```

If the dataset is not found locally, it will be downloaded and extracted automatically to:

```text
data\cicids2017-cleaned-and-preprocessed\
```

---

## Using an Existing CSV

If the dataset is already available locally, a Kaggle account is not required.

```powershell
python run_analysis.py --csv "C:\path\to\cicids2017_cleaned.csv"
```

---

## Command-Line Options

| Argument | Description |
|-----------|------------|
| `--csv` | Path to a local CSV file |
| `--data-dir` | Directory used to store or search for the dataset |
| `--output-dir` | Directory used to store generated results |
| `--chunksize` | Number of rows processed per chunk |
| `--components` | Number of principal components used for loadings and scores (default: 14) |
| `--variance-threshold` | Target cumulative explained variance percentage (default: 80) |

Example:

```powershell
python run_analysis.py ^
  --components 14 ^
  --variance-threshold 80
```

---

## Data Preprocessing

Before applying PCA, the dataset undergoes several preprocessing steps:

- Removal of invalid records according to CICFlowMeter rules;
- Conversion of infinite values into missing values;
- Median imputation of missing values;
- Removal of empty columns;
- Removal of constant features;
- Feature standardization using Z-score normalization.

Additional considerations:

- `Destination Port` is excluded from PCA predictors;
- The target class is used exclusively for visualization and interpretation;
- Class labels are not used as PCA inputs.

---

## PCA Analysis

The project performs:

### Correlation Analysis

Feature correlation matrix and heatmap generation.

### Principal Component Selection

Component retention is evaluated using:

- Kaiser Criterion (eigenvalues > 1);
- Scree Plot;
- Cumulative Explained Variance.

### Loadings Analysis

Identification of the variables with the highest contribution to each principal component.

### Scores Analysis

Projection of observations into the reduced feature space.

### Biplot

Combined visualization of:

- PCA scores;
- PCA loadings;

for the first two principal components.

---

## Outlier Detection

The PCA model retains the principal components with eigenvalues greater than 1.

Two statistics are used:

### Hotelling's T²

Measures the distance of observations within the retained PCA subspace.

### Q-Residual (Squared Prediction Error)

Measures the unexplained variation remaining after projection onto the PCA model.

These metrics identify observations worthy of further investigation, but they do not, by themselves, confirm the presence of malicious traffic.

Observation indices reported in the results correspond to the dataset ordering after preprocessing.

---

## Generated Outputs

The `output/` directory includes:

### Presentation

```text
Análise_PCA_CICIDS2017.pptx
```

A complete presentation containing:

- methodology;
- figures;
- statistical interpretation;
- PCA results.

### Visualizations

- Correlation heatmap;
- Kaiser criterion plot;
- Cumulative explained variance plot;
- Loading plots;
- Score plots;
- PCA biplot;
- Hotelling's T² plot;
- Q-residual plot.

### Tables

```text
variancia_componentes.csv
loadings.csv
loadings_dominantes.csv
top10_hotelling_t2.csv
top10_q_residual.csv
```

---

## Key Findings

This analysis demonstrates that PCA can significantly reduce the dimensionality of the CICIDS2017 dataset while preserving most of its information content.

The methodology enables:

- Identification of dominant traffic characteristics;
- Exploration of relationships between network features;
- Visualization of latent structures;
- Detection of observations exhibiting unusual behavior.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Kaggle API

---

## License

This project is distributed under the MIT License with an additional release-timing condition. Subsequent or modified versions of this project may not be published before **16 November 2026 at 23:59 (Europe/Lisbon time)**. A fork's maintainer may remove or omit that timing condition from the fork, as described in [`LICENSE`](LICENSE).

Because of this extra condition, this is a customized MIT-based license, not the standard MIT License or an OSI-approved open-source license.