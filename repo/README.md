# Conversion Prediction for Digital Marketing Campaigns

Predicting which customers will **convert** from their digital-marketing engagement data, so
marketing spend can be targeted at the people most likely to act.

> Supervised machine-learning project — BA810, Boston University (MSBA), Fall 2024.

---

## Overview

Digital marketing teams spend heavily across email, social, PPC, and other channels, but only a
small fraction of customers convert. This project builds and compares supervised classifiers that
predict the binary `Conversion` outcome from customer demographics and engagement metrics — enabling
**smarter targeting and more efficient budget allocation**.

The full analysis (EDA → preprocessing → modeling → tuning → ensembling → conclusions) lives in
**[`conversion_prediction.ipynb`](conversion_prediction.ipynb)**, which renders directly on GitHub.

## Dataset

- **8,000 customers × 20 columns** — synthetic data from Kaggle
  ([*Predict Conversion in Digital Marketing*](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset), by Rabie El Kharoua).
- Features span **demographics** (age, gender, income), **campaign** info (channel, type, ad spend),
  and **engagement** metrics (click-through rate, website visits, time on site, email opens/clicks,
  loyalty points, previous purchases).
- **Target:** `Conversion` (1 = converted, 0 = not).
- **Class imbalance:** ~**87.6% converted / 12.4% not** — so the project evaluates on **balanced
  accuracy** rather than raw accuracy, which would be misleading here.
- For data-cleaning practice, random missing values were injected into a working copy of the dataset;
  the clean base file is included at [`data/digital_marketing_campaign_dataset.csv`](data/digital_marketing_campaign_dataset.csv).

> **Data source & credit:** *Predict Conversion in Digital Marketing Dataset*, created and owned by
> **Rabie El Kharoua**, published on Kaggle for educational use —
> <https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset>.
> All credit for the original data belongs to the author.

## Approach

1. **Exploratory data analysis** — missing-value profiling, feature–target correlations, and the
   relationship between ad spend and conversion.
2. **Preprocessing pipelines** (`ColumnTransformer`) — iterative imputation, log-scaling of skewed
   numerics, one-hot encoding of categoricals, a **custom engagement "ratio" feature**, and
   class-weight balancing.
3. **Baseline models** — Logistic Regression, Decision Tree, Random Forest, XGBoost.
4. **Feature selection & tuning** — Sequential Feature Selection (forward & backward) with
   `GridSearchCV` and `RandomizedSearchCV`, optimized for balanced accuracy.
5. **Ensembles** — hard/soft **Voting** and a **Stacking** classifier with a Random Forest meta-model.
6. **Model persistence** — the final stacked model is exported with `joblib` for reuse.

## Results

Balanced accuracy across the modeling stages (higher is better):

| Model | Base | Tuned | Stacked |
|---|---|---|---|
| Logistic Regression | 0.6570 | 0.6814 | **0.8016** |
| Decision Tree | 0.6931 | 0.7084 | **0.8016** |
| Random Forest | 0.6392 | 0.6813 | **0.8016** |
| XGBoost | 0.6716 | 0.6870 | **0.8016** |

**The stacked ensemble was the best model at 0.8016 balanced accuracy** — a clear lift over any
single tuned classifier.

### Key takeaways
- Under strong class imbalance, **balanced accuracy is the metric that matters** — headline accuracy
  near 90% mostly reflects the majority class.
- **Preprocessing and stacking drove the biggest gains**; individually tuned models plateaued around
  0.68–0.71 balanced accuracy, while stacking reached 0.80.

## Repository structure

```
.
├── conversion_prediction.ipynb   # full analysis notebook (renders on GitHub)
├── data/
│   └── digital_marketing_campaign_dataset.csv
├── reports/
│   └── Team9_B_Presentation.pdf  # summary slide deck
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook conversion_prediction.ipynb   # or open in Google Colab
```

> The notebook was originally run on a null-injected copy of the dataset (for data-cleaning
> practice). To reproduce end-to-end from the clean file included here, point the load step at
> `data/digital_marketing_campaign_dataset.csv`.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · mlxtend · matplotlib · seaborn

---

*Team project (Team 9) for BA810 Supervised Machine Learning, Questrom School of Business, Boston
University.*
