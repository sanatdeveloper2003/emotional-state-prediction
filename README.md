# 🧠 Emotional State Prediction using Cardiovascular Features

## 📌 Overview
This project predicts **emotional states** (baseline, gratitude, neutral) using cardiovascular features extracted from ECG signals (RR intervals).  
The dataset comes from the **POPANE study**, containing 425 observations across 142 subjects.  
The goal is to leverage **machine learning** to classify emotional states based on physiological responses.

---

## 📊 Dataset
- **Observations**: 425  
- **Subjects**: 142 (3 conditions: baseline, gratitude, neutral)  
- **Features**: Time-domain, frequency-domain, and non-linear metrics (e.g., SDNN, LF/HF ratio, SampEn).  

---

## 🎯 Objectives
1. **Exploratory Data Analysis (EDA)** – Visualize and clean the dataset.  
2. **Feature Selection** – Apply methods like Random Forest feature importance and RFE.  
3. **Dimensionality Reduction** – Use PCA to simplify data.  
4. **Modeling** – Train ML models (Logistic Regression, Decision Trees, Random Forest) and evaluate.  

---

## ⚙️ Technologies
- **Python 3.11**
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, LIME  

---

## 📂 Project Structure
```
emotional-state-prediction/
│── data/                # Dataset(s)
│   └── Dataset_Study3.csv
│── notebooks/           # Jupyter notebooks (EDA, experiments)
│── reports/             # Reports, documentation
│── src/                 # Source code (main training script)
│   └── projectSA.py
│── requirements.txt     # Dependencies
│── README.md
```

---

## ▶️ Setup & Run

### 1. Clone repository
```bash
git clone https://github.com/sanatdeveloper2003/emotional-state-prediction.git
cd emotional-state-prediction
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run training script
```bash
python src/projectSA.py
```

---

## 🔑 Key Steps
- **Data Loading & Exploration** – Import and explore dataset.  
- **Preprocessing** – Handle outliers, normalize features.  
- **Modeling & Evaluation** – Train ML models and evaluate with accuracy, precision, recall.  

---

## 🚀 Future Enhancements
- Cross-validation for better model robustness.  
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV).  
- Ensemble methods for improved performance.  

---

## 📜 License
MIT License © 2025 Sanat Zhengis
