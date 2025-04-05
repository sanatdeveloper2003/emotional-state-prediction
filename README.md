# Emotional State Prediction using Cardiovascular Features

## Overview
This project predicts emotional states (baseline, gratitude, neutral) using cardiovascular features extracted from the RR signal in ECG data. The dataset comes from the POPANE study, containing 425 observations across 142 subjects. The goal is to leverage machine learning to classify emotional states based on physiological responses.

## Dataset
- **Observations:** 425
- **Subjects:** 142 (3 conditions: baseline, gratitude, neutral)
- **Features:** Time-domain, frequency-domain, and non-linear metrics (e.g., SDNN, LF/HF ratio, SampEn)

## Objective
1. **Exploratory Data Analysis (EDA):** Visualize and clean data.
2. **Feature Selection:** Use methods like Random Forest and Recursive Feature Elimination.
3. **Dimensionality Reduction:** Implement PCA to simplify data.
4. **Modeling:** Train models (Logistic Regression, Decision Trees, Random Forest) and evaluate performance.

## Technologies
- **Python 3.x**
- **Libraries:** Pandas, Numpy, Matplotlib, Scikit-learn, LIME

## Setup

1. **Create Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Key Steps

- **Data Loading & Exploration:** Import and explore the dataset.
- **Preprocessing:** Clean data, handle outliers.
- **Modeling & Evaluation:** Train and evaluate models using accuracy, precision, and recall.

## Future Enhancements
- Cross-validation for better model evaluation.
- Hyperparameter tuning and ensemble methods for improved performance.

