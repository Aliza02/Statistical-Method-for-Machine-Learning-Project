# Clustering and Initialization: k-means vs k-means++

Author: **Aliza Aziz** (student ID 82838A), Università degli Studi di Milano.

Experimental project for **Statistical Methods for Machine Learning** (UniMi, A.Y. 2025/26),
Assignment 5, based on Arthur & Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, SODA 2007.

The final report is [`report/report.pdf`](report/report.pdf).

## What is implemented

Everything is written from scratch with numpy (`src/kmeans.py`):

- Lloyd's algorithm (assignment / update steps, exact convergence, empty-cluster handling)
- random initialization (k distinct data points chosen uniformly)
- k-means++ D² seeding
- inertia (k-means objective) and the Adjusted Rand Index used for evaluation

scikit-learn is used only to load the UCI digits dataset and, in `tests/`, to double-check our ARI.

## Datasets

| Dataset | Type | Size | Why |
|---|---|---|---|
| Norm-25 | synthetic | 5000 × 15, 25 Gaussian clusters | the paper's synthetic set; clear structure |
| Blobs-2D | synthetic | 1350 × 2, 9 clusters | to visualize local minima |
| UCI Digits | real | 1797 × 64, 10 classes | MNIST-like, high-dimensional, multiclass |
| Overlap / Dimension | synthetic | varying σ and d | extensions |

## Reproduce

**Jupyter:** open `run_project.ipynb` from this folder and run all cells.

**Terminal:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m tests.test_kmeans      # sanity checks (a few seconds)
python run_experiments.py        # all results, figures and tables (< 1 minute)
cd report && pdflatex report.tex && pdflatex report.tex
```

All randomness is controlled by explicit seeds, so the numbers in the report are reproduced exactly.

## Layout

```
src/kmeans.py          algorithms (from scratch)
src/datasets.py        data generation / loading
run_project.ipynb      notebook: runs tests, demo, all experiments, shows figures
run_experiments.py     all experiments, plots, and the LaTeX tables
tests/test_kmeans.py   correctness checks
results/               raw and summary results (csv / json)
report/                LaTeX source, figures, generated tables, final PDF
```
