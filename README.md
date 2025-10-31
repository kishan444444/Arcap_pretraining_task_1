# 🏥 Patient Outcome Prediction using XGBoost + Streamlit

> A machine learning web application that predicts **patient outcomes** based on hospital and clinical data using an **XGBoost Classifier** and an interactive **Streamlit dashboard**.

---

## 📘 Project Overview

This project leverages hospital data (demographics, diagnosis, discharge details, etc.) to predict patient outcomes such as **recovery, readmission, or discharge type**.  
It demonstrates an end-to-end ML pipeline — from **data preprocessing → model training → deployment → interactive dashboard**.

---

## 🎯 Objectives

- Build an accurate **predictive model** for patient outcomes.  
- Deploy it via a **Streamlit dashboard** for healthcare professionals.  
- Enable real-time predictions using **cloud-hosted trained models**.

---

## 🧩 Features

✅ Preprocessed hospital data (numeric + categorical encoding)  
✅ Trained **XGBoost Classifier** with tuned hyperparameters  
✅ **Pipeline saving** (`preprocessor.pkl`, `xgb_model.pkl`) for easy reuse  
✅ **Streamlit UI** for real-time patient predictions  
✅ Ready for **cloud deployment** (Streamlit Cloud / AWS / GCP / Azure)

---

## 📊 Dataset Features

| Feature | Description |
|----------|--------------|
| `age` | Age of the patient |
| `gender` | Male / Female |
| `primary_diagnosis` | Main medical condition |
| `num_procedures` | Number of medical procedures performed |
| `days_in_hospital` | Total hospital stay duration |
| `comorbidity_score` | Health risk score (0–10) |
| `discharge_to` | Discharge destination (Home, Rehab, etc.) |

---

## 🧠 Model Architecture

| Component | Description |
|------------|--------------|
| **Preprocessing** | OneHotEncoder (for categorical features) + ColumnTransformer |
| **Model** | XGBoostClassifier |
| **Evaluation Metrics** | Accuracy, F1-Score, ROC-AUC |
| **Persistence** | Joblib serialization for model and preprocessor |

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Clean and encode dataset using `OneHotEncoder` + `ColumnTransformer`
   - Split into train/test sets

2. **Model Training**
   ```python
   from xgboost import XGBClassifier

   model = XGBClassifier(
       n_estimators=500,
       learning_rate=0.05,
       max_depth=6,
       subsample=0.8,
       colsample_bytree=0.8,
       random_state=42)



## ⚙️ System Architecture


              ┌────────────────────────────┐
             │        Hospital Data        │
             │ (EHR: Demographics, Labs)   │
             └──────────────┬──────────────┘
                            │
                            ▼
             ┌────────────────────────────┐
             │     Data Preprocessing      │
             │  • Cleaning & Encoding      │
             │  • Feature Engineering      │
             └──────────────┬──────────────┘
                            │
                            ▼
             ┌────────────────────────────┐
             │     Model Training (ML)     │
             │   XGBoost Classifier        │
             │   + ColumnTransformer(OHE)  │
             └──────────────┬──────────────┘
                            │
                            ▼
             ┌────────────────────────────┐
             │     Model Export (.pkl)     │
             │     + Preprocessor.pkl      │
             └──────────────┬──────────────┘
                            │
                            ▼
           ┌──────────────────────────────────┐
           │          Cloud Storage            │
           │ (AWS S3 / GCP / Azure / GitHub)   │
           └────────────────┬──────────────────┘
                            │
                            ▼
             ┌────────────────────────────┐
             │     Streamlit Dashboard     │
             │ • User inputs patient info  │
             │ • Applies preprocessing     │
             │ • Model predicts outcome    │
             │ • Displays risk/probability │
             └────────────────────────────┘


   
   )
   model.fit(X_train, y_train)
