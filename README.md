# PPC Ad Click Fraud Detection — MS Data Science Thesis

## Overview
Comparative analysis of machine learning models for Pay-Per-Click ad click fraud detection with strict temporal validation.

## Key Finding
Eliminating data leakage causes a **72.6% average F1-score drop**, exposing overoptimistic results in prior literature.

## Models Compared
- Decision Tree
- Random Forest  
- XGBoost (Best: F1=0.4124)
- LightGBM
- Logistic Regression

## Project Structure
\\\
ppc_fraud_project/
├── preprocessing.py          # Feature engineering + SMOTE pipeline
├── train_models.py           # Model training V1 (baseline)
├── train_models_v3.py        # Advanced training with temporal validation
├── evaluate_models.py        # Evaluation + visualizations
├── evaluate_models_v3.py     # V3 evaluation + 3-way comparison
├── final_results_summary.py  # Final results table for paper
├── extract_literature.py     # Literature review from 23 PDFs
├── gap_analysis.py           # Research gap analysis
├── write_paper.py            # Paper draft generator
├── results/                  # All charts, plots, CSVs
├── papers/                   # Literature matrix + gap analysis
└── paper/                    # Research paper draft
\\\

## Results Summary

| Model | V1 F1 (Leaky) | V3 F1 (Valid) | Drop |
|-------|--------------|---------------|------|
| Decision Tree | 0.9833 | 0.2199 | 77.6% |
| Random Forest | 0.9990 | 0.3000 | 70.0% |
| XGBoost | 0.9989 | **0.4124** | 58.7% |
| LightGBM | 0.9978 | 0.3762 | 62.3% |
| Logistic Regression | 0.9049 | 0.0284 | 96.9% |

## Dataset
TalkingData AdTracking Fraud Detection (Kaggle)
- 100,000 samples  
- 0.23% fraud prevalence (439:1 class imbalance)

## Tech Stack
Python 3.14 | scikit-learn | XGBoost | LightGBM | pandas | imbalanced-learn | matplotlib | seaborn

## Author
Saqib Khan — MS Data Science
Department of Computer Science
