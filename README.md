# 🎓 Student Performance Analysis & ANN Prediction System

## Overview
An end-to-end data science project that uses an **Artificial Neural Network (ANN)**
to predict student pass/fail outcomes from academic and socio-demographic features.

**Test Accuracy: 96.7%  |  AUC-ROC: 0.989  |  Architecture: 4-layer deep ANN**

---

## Project Structure
```
student_performance_project/
├── app.py                          # Streamlit web application
├── Student_Performance_Analysis.ipynb  # Full analysis notebook
├── STUDENT_PERFORMANCE.csv         # Dataset (395 students, 33 features)
├── requirements.txt
├── models/
│   ├── ann_model.pkl               # Trained ANN model
│   ├── scaler.pkl                  # StandardScaler
│   ├── label_encoders.pkl          # LabelEncoders for categorical features
│   └── model_meta.json             # Model metadata & metrics
└── assets/
    ├── loss_curve.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    ├── grade_distribution.png
    ├── correlation.png
    └── eda_plots.png
```

---

## ANN Architecture
```
Input (37 features)
    ↓
Dense(256, ReLU)
    ↓
Dense(128, ReLU)
    ↓
Dense(64,  ReLU)
    ↓
Dense(32,  ReLU)
    ↓
Output (2 classes: Pass/Fail)
```

**Training:** Adam optimizer · LR=0.0005 · L2=0.0001 · Early stopping (patience=40)

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook Student_Performance_Analysis.ipynb
```
*(This trains the model and generates all plots in `assets/`)*

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## Demo Credentials

| Role    | Username | Password    | Access |
|---------|----------|-------------|--------|
| Admin   | admin    | admin123    | Full access |
| Teacher | teacher  | teacher123  | Predict, batch, analytics |
| Student | student  | student123  | Own prediction & history |

---

## Specific Objectives Achieved

| # | Objective | Status |
|---|-----------|--------|
| 1 | Data preparation & cleaning | ✅ |
| 2 | EDA & visualization | ✅ |
| 3 | Feature importance identification | ✅ |
| 4 | ANN model training (>90% accuracy) | ✅ 96.7% |
| 5 | Insights & recommendations | ✅ |
| + | Streamlit deployment UI | ✅ |
| + | User roles (admin/teacher/student) | ✅ |
| + | Registration & login system | ✅ |
| + | Batch CSV upload | ✅ |
| + | Prediction history | ✅ |

---

## Key Findings
1. **G1 & G2** (period grades) are the strongest predictors of G3
2. **Past failures** correlate negatively with final performance
3. Higher **parental education** leads to better student outcomes
4. **High absences** (>10) significantly increase fail risk
5. **Study time ≥ 3** markedly improves pass probability

## Engineered Features
- `G1G2_avg`   — average of period grades
- `G1G2_prod`  — product (interaction term)
- `G1G2_diff`  — grade trajectory (G2 - G1)
- `study_fail` — study time penalised by failure history
- `abs_study`  — absenteeism relative to study effort
