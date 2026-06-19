# Credit-Card Default Prediction

> A cross-validated supervised-ML pipeline predicting next-month payment default on the UCI *Default of Credit Card Clients* dataset (**30,000 clients × 23 features**) — built around outlier-robust scaling, dimensionality reduction, recursive feature selection, and grid-searched model selection.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

## Problem

Predict whether a credit-card client will **default on next month's payment** — a **class-imbalanced** (~22% positive) and **cost-asymmetric** problem: for a lender, a missed default costs far more than a false alarm. The data is 30,000 clients described by 23 features: demographics (`X1–X5`), six months of repayment status (`X6–X11`), bill statements (`X12–X17`), and prior payments (`X18–X23`); target `Y` = default next month.

## Pipeline — and the reasoning behind each choice

| Stage | Choice | Why this choice |
|-------|--------|-----------------|
| **Scaling** | `RobustScaler` | Bill/payment amounts are heavy-tailed with extreme outliers; median/IQR scaling resists them where `StandardScaler` is distorted by the tails. |
| **Dimensionality reduction** | `PCA` → 8 components | The repayment/bill/payment block (`X6–X23`) is strongly collinear; PCA compresses it to a compact, decorrelated basis while keeping the dominant variance. |
| **Feature selection** | `RFE` / `RFECV` with `LogisticRegression` **and** `SGDClassifier` estimators | Cross-validated recursive elimination keeps only features that carry signal under two independent linear estimators — guarding against single-estimator bias. |
| **Model selection** | `GridSearchCV` over **KNN** and **Decision Tree** | Exhaustive hyper-parameter search with **5-fold cross-validation** for an honest generalization estimate, not one lucky split. |

## Results — measured, not assumed

| Model | Metric | Score |
|-------|--------|-------|
| K-Nearest Neighbors | best 5-fold CV accuracy | **≈ 81.2%** |
| Decision Tree | held-out test accuracy | **≈ 82.3%** |

> **Read these critically — this is the part that matters.** On a ~78/22 imbalanced target, *accuracy overstates performance*: a model that always predicts "no default" already scores ~78%. So the honest headline is the **≈ +4 pts over the majority baseline**, not the 82% itself. The production-grade next step is to report **ROC-AUC / PR-AUC and recall on the default class** and to test **class weighting or SMOTE** — because what a lender actually needs is to *catch defaulters*, not maximize raw accuracy.

## Tech Stack

`Python` · `Jupyter` · `scikit-learn` (`RobustScaler`, `PCA`, `RFE`/`RFECV`, `GridSearchCV`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `LogisticRegression`, `SGDClassifier`) · `pandas` · `NumPy` · `matplotlib` / `seaborn`

## Getting Started

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install jupyter scikit-learn pandas numpy matplotlib seaborn xlrd
jupyter notebook houssem.ipynb
```

> The source dataset (`.xls`) is **not committed** — download the UCI *Default of Credit Card Clients* dataset, place it in the project root, and ensure `xlrd` is installed (required for `.xls`).

## Notes

- Coursework project, but executed as a **full ML pipeline** (scaling → dimensionality reduction → cross-validated feature selection → grid-searched, cross-validated model selection) rather than a single fit — the methodology is the point.
- Reported figures come directly from the notebook's cell outputs and are seed-dependent.

---
<p align="center">Built by <b>Mohamed Habib Khattat</b> — <a href="https://github.com/MuhamedHabib">GitHub (@MuhamedHabib)</a> · <a href="https://www.linkedin.com/in/mohamed-habib-khattat-2b206a173">LinkedIn</a></p>
