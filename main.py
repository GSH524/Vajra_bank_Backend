import os
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- GLOBAL MODELS DICTIONARY ---
MODELS = {
    "kmeans": None,
    "scaler": None,
    "label_encoder": None,
    "ohe_encoder": None,
    "loan_model": None,
    "loan_ohe": None,
    "gaussian_model": None,
    "gaussian_active_enc": None,
    "gaussian_status_enc": None
}

@app.on_event("startup")
def load_models():
    """Load all PKL files into memory on server start."""
    try:
        # 1. Profile Risk Models (KMeans)
        MODELS["kmeans"] = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
        MODELS["scaler"] = joblib.load(os.path.join(MODELS_DIR, "Scalar.pkl"))
        MODELS["label_encoder"] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        MODELS["ohe_encoder"] = joblib.load(os.path.join(MODELS_DIR, "OHE_encoding.pkl"))
        
        # 2. Loan Amount Models (XGBoost)
        MODELS["loan_model"] = joblib.load(os.path.join(MODELS_DIR, "Loan_Predictions.pkl"))
        MODELS["loan_ohe"] = joblib.load(os.path.join(MODELS_DIR, "Loanammount_Prediction_ohe.pkl"))

        # 3. Credit Health Models (Gaussian Naive Bayes)
        MODELS["gaussian_model"] = joblib.load(os.path.join(MODELS_DIR, "Gaussian_model.pkl"))
        MODELS["gaussian_active_enc"] = joblib.load(os.path.join(MODELS_DIR, "Gaussian_Label_active.pkl"))
        MODELS["gaussian_status_enc"] = joblib.load(os.path.join(MODELS_DIR, "Gaussian_label_status.pkl"))
        
        print("✅ All Machine Learning Models Loaded and Verified")
    except Exception as e:
        print(f"❌ Startup Error: {e}")

# --- 1. GAUSSIAN CREDIT RISK ENDPOINT ---

@app.post("/api/credit-risk")
async def predict_credit_gaussian(user_data: dict = Body(...)):
    if MODELS["gaussian_model"] is None:
        return {"success": False, "error": "Gaussian Model Offline"}
    try:
        df = pd.DataFrame([user_data]).fillna(0)
        if 'ActiveStatus' in df.columns:
            try:
                df['ActiveStatus'] = MODELS["gaussian_active_enc"].transform(df['ActiveStatus'].astype(str))
            except:
                df['ActiveStatus'] = 0
        features_required = MODELS["gaussian_model"].feature_names_in_
        df_final = df.reindex(columns=features_required, fill_value=0)
        prediction = MODELS["gaussian_model"].predict(df_final)[0]
        risk_label = "High" if prediction == 0 else "Low"
        return {"success": True, "prediction": int(prediction), "predictedRisk": risk_label}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- 2. UPDATED KMEANS RISK PREDICTION (PROFILE SEGMENTATION) ---

@app.post("/api/predict-risk")
async def predict_user_risk(user_data: dict = Body(...)):
    if MODELS["kmeans"] is None:
        return {"success": False, "predictedRisk": "Offline"}

    try:
        # Step 2 & 3: Selection of the 26 required fields
        features = [
            'Age', 'Gender', 'Account Type', 'Relationship_Tenure_Years',
            'Account Balance', 'Avg_Account_Balance', 'AnnualIncome',
            'Monthly_Transaction_Count', 'Avg_Transaction_Amount',
            'Digital_Transaction_Ratio', 'Days_Since_Last_Transaction',
            'Loan Amount', 'Loan Type', 'Loan Term', 'Interest Rate',
            'Active_Loan_Count', 'Credit Utilization', 'Avg_Credit_Utilization',
            'Card_Balance_to_Limit_Ratio', 'Payment Delay Days', 'CIBIL_Score',
            'Card Type', 'Credit Limit', 'Rewards Points', 
            'Reward_Points_Earned', 'ActiveStatus'
        ]

        df = pd.DataFrame([user_data])
        
        # Ensure all required columns exist, fill missing with defaults
        for col in features:
            if col not in df.columns:
                df[col] = "Unknown" if col in ['Gender', 'Account Type', 'Loan Type', 'Card Type', 'ActiveStatus'] else 0
        
        x = df[features].copy()

        # Step 4: Categorical separation
        x['ActiveStatus'] = x['ActiveStatus'].astype(str)
        x_catn = x[['ActiveStatus']].copy()
        x_catc = x[['Gender', 'Account Type', 'Loan Type', 'Card Type']].copy()

        # Step 5: Label Encoding (ActiveStatus)
        try:
            x_catn['ActiveStatus'] = MODELS["label_encoder"].transform(x_catn['ActiveStatus'])
        except:
            x_catn['ActiveStatus'] = 0

        # Step 6: One Hot Encoding (Other Categories)
        x_catc_transformed = MODELS["ohe_encoder"].transform(x_catc)
        x_catc_df = pd.DataFrame(
            x_catc_transformed.toarray(),
            columns=MODELS["ohe_encoder"].get_feature_names_out()
        )

        # Step 7: Combine categorical (OHE + label)
        x_cat_final = pd.concat([x_catc_df.reset_index(drop=True), x_catn.reset_index(drop=True)], axis=1)

        # Step 8: Numeric columns & Scaling
        x_num = x.select_dtypes(include='number')
        x_num_scaled = MODELS["scaler"].transform(x_num)
        x_num_scaled_df = pd.DataFrame(x_num_scaled, columns=x_num.columns)

        # Step 9: Combine categorical + numeric
        xr = pd.concat([x_cat_final.reset_index(drop=True), x_num_scaled_df.reset_index(drop=True)], axis=1)

        # Step 10: Align AFTER combining
        trained_columns = MODELS["kmeans"].feature_names_in_
        xr = xr.reindex(columns=trained_columns, fill_value=0)

        # Step 11: Predict
        kmeans_cluster = MODELS["kmeans"].predict(xr)[0]
        
        # Step 12: Map labels
        segment_map = {0: "Low", 1: "Medium", 2: "High"}
        risk_label = segment_map.get(kmeans_cluster, "Low")

        return {"success": True, "predictedRisk": risk_label}
        
    except Exception as e:
        print(f"Risk Prediction Error: {e}")
        return {"success": False, "predictedRisk": "Data Error"}

# --- 3. LOAN AMOUNT PREDICTION (STRICT ALIGNMENT) ---

@app.post("/api/predict-loan")
async def predict_loan_amount(user_data: dict = Body(...)):
    if MODELS["loan_model"] is None or MODELS["loan_ohe"] is None:
        return {"success": False, "predictedLoanAmount": "Offline"}
    try:
        user_features = pd.DataFrame([{
            "Age": user_data.get("Age", 0),
            "Employment Type": user_data.get("Employment Type", "Unknown"),
            "Credit Score": user_data.get("Credit Score", 0),
            "Tenure": user_data.get("Tenure", 0),
            "Years in Current City": user_data.get("Years in Current City", 0),
            "Years in Current Job": user_data.get("Years in Current Job", 0),
            "Insurance Premiums": user_data.get("Insurance Premiums", 0),
            "Residential Status": user_data.get("Residential Status", "Unknown"),
            "Residence Type": user_data.get("Residence Type", "Unknown"),
            "Loan Type": user_data.get("Loan Type", "Unknown"),
            "AnnualIncome": user_data.get("AnnualIncome", 0)
        }])
        numeric_cols = ["Age", "Credit Score", "Tenure", "Years in Current City", "Years in Current Job", "Insurance Premiums", "AnnualIncome"]
        for col in numeric_cols:
            user_features[col] = pd.to_numeric(user_features[col], errors='coerce').fillna(0)
        x_cat1 = user_features.select_dtypes(exclude='number')
        x_num1 = user_features.select_dtypes(include='number')
        x_cat1_df = pd.DataFrame(MODELS["loan_ohe"].transform(x_cat1).toarray(), columns=MODELS["loan_ohe"].get_feature_names_out())
        encoded_df = pd.concat([x_cat1_df.reset_index(drop=True), x_num1.reset_index(drop=True)], axis=1)
        encoded_df = encoded_df.reindex(columns=MODELS["loan_model"].feature_names_in_, fill_value=0)
        prediction = float(MODELS["loan_model"].predict(encoded_df)[0])
        return {"success": True, "predictedLoanAmount": max(0.0, round(prediction, 2))}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- 4. BATCH PROCESSING FOR ADMIN ---

@app.post("/api/predict-risk-batch")
async def predict_risk_batch(users_data: list = Body(...)):
    if MODELS["kmeans"] is None:
        return {"success": False, "error": "Model Offline"}
    try:
        results = []
        for user in users_data:
            res = await predict_user_risk(user)
            if res["success"]:
                results.append({"customerId": user.get("customerId"), "riskLevel": res["predictedRisk"]})
        return {"success": True, "predictions": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)