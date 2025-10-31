from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error
import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

data=pd.read_csv(r"F:\Arcap REIT\Model_training_ task\Task1\archive (28)\train_df.csv")

data.drop_duplicates()

X = data.drop('readmitted', axis=1)
y = data['readmitted']

## Train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=30)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

numerical_cols=X.select_dtypes(exclude='object').columns

categorical_cols = ['gender', 'primary_diagnosis', 'discharge_to']
numerical_cols = numerical_cols

# Use OneHotEncoder, which supports handle_unknown='ignore'
ohe = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', ohe, categorical_cols)
    ]
)

# Fit on training data
X_train = preprocessor.fit_transform(X_train)

# Transform test data safely
X_test = preprocessor.transform(X_test)


model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,  # important for newer versions
    eval_metric='logloss'     # avoids warnings
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = model

st.title("🏥 Patient Outcome Prediction (XGBoost Model)")
st.write("Predict patient outcomes using demographic and hospital data.")

# --- Input fields ---
age = st.number_input("Age", min_value=0, max_value=120, value=40)
num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=50, value=1)
days_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=365, value=5)
comorbidity_score = st.slider("Comorbidity Score", 0.0, 10.0, 2.5)

gender = st.selectbox("Gender", ["Male", "Female"])
primary_diagnosis = st.selectbox(
    "Primary Diagnosis",
    ["Diabetes", "Heart Disease", "Hypertension", "Kidney Disease", "Other"]
)
discharge_to = st.selectbox(
    "Discharge To",
    ["Home Health Care", "Rehabilitation Facility", "Skilled Nursing Facility", "Other"]
)

# --- Build base numeric input ---
input_dict = {
    'age': age,
    'num_procedures': num_procedures,
    'days_in_hospital': days_in_hospital,
    'comorbidity_score': comorbidity_score,
    'gender_Male': 1 if gender == "Male" else 0,
    'primary_diagnosis_Diabetes': 1 if primary_diagnosis == "Diabetes" else 0,
    'primary_diagnosis_Heart Disease': 1 if primary_diagnosis == "Heart Disease" else 0,
    'primary_diagnosis_Hypertension': 1 if primary_diagnosis == "Hypertension" else 0,
    'primary_diagnosis_Kidney Disease': 1 if primary_diagnosis == "Kidney Disease" else 0,
    'discharge_to_Home Health Care': 1 if discharge_to == "Home Health Care" else 0,
    'discharge_to_Rehabilitation Facility': 1 if discharge_to == "Rehabilitation Facility" else 0,
    'discharge_to_Skilled Nursing Facility': 1 if discharge_to == "Skilled Nursing Facility" else 0
}

# --- Create input DataFrame ---
input_data = pd.DataFrame([input_dict])

st.subheader("🔍 Encoded Input Data")
st.dataframe(input_data)

# --- Prediction ---
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.success(f"✅ Predicted Outcome: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")