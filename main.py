import os
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, Body
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
    "gaussian_model": None
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
        
        print("✅ All Machine Learning Models Loaded and Verified")
    except Exception as e:
        print(f"❌ Startup Error: {e}")

# --- 1. GAUSSIAN CREDIT RISK ENDPOINT ---

@app.post("/api/credit-risk")
async def predict_credit_gaussian(user_data: dict = Body(...)):
    if MODELS["gaussian_model"] is None:
        return {"success": False, "error": "Gaussian Model Offline"}
    
    try:
        user_input = pd.DataFrame([user_data])

        # Handle Loan Type (One-Hot Encoding)
        loan_type_cols = ['Loan Type_Auto', 'Loan Type_Mortgage', 'Loan Type_Personal', 'Loan Type_other']
        loan_type_dummies = pd.get_dummies(user_input['Loan Type'], prefix='Loan Type')
        for col in loan_type_cols:
            if col not in loan_type_dummies.columns:
                loan_type_dummies[col] = 0
        
        loan_type_dummies = loan_type_dummies[loan_type_cols]
        user_input = user_input.drop(columns=['Loan Type'])
        user_input = pd.concat([user_input, loan_type_dummies.reset_index(drop=True)], axis=1)

        # Encode Loan Status & ActiveStatus (Mapping)
        loan_status_map = {'Approved': 0, 'Closed': 1, 'Rejected': 2, 'Default': 3}
        user_input['Loan Status'] = loan_status_map.get(user_input.get('Loan Status', ['Closed']).iloc[0], 1)

        active_map = {'Active': 1, 'Inactive': 0}
        user_input['ActiveStatus'] = active_map.get(user_input.get('ActiveStatus', ['Inactive']).iloc[0], 0)

        # Match Model Feature Order
        features_required = MODELS["gaussian_model"].feature_names_in_
        df_final = user_input.reindex(columns=features_required, fill_value=0)

        # Predict (Handles direct string output 'Low'/'High')
        prediction = MODELS["gaussian_model"].predict(df_final)[0]
        risk_label = str(prediction) 
        
        return {"success": True, "predictedRisk": risk_label}

    except Exception as e:
        print(f"❌ Gaussian Prediction Error: {str(e)}")
        return {"success": False, "error": str(e)}

# --- 2. KMEANS RISK PREDICTION (SINGLE USER) ---

@app.post("/api/predict-risk")
async def predict_user_risk(user_data: dict = Body(...)):
    if MODELS["kmeans"] is None:
        return {"success": False, "predictedRisk": "Offline"}

    try:
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
        for col in features:
            if col not in df.columns:
                df[col] = "Unknown" if col in ['Gender', 'Account Type', 'Loan Type', 'Card Type', 'ActiveStatus'] else 0
        
        x = df[features].copy()
        x['ActiveStatus'] = x['ActiveStatus'].astype(str)
        x_catn = x[['ActiveStatus']].copy()
        x_catc = x[['Gender', 'Account Type', 'Loan Type', 'Card Type']].copy()

        try:
            x_catn['ActiveStatus'] = MODELS["label_encoder"].transform(x_catn['ActiveStatus'])
        except:
            x_catn['ActiveStatus'] = 0

        x_catc_transformed = MODELS["ohe_encoder"].transform(x_catc)
        x_catc_df = pd.DataFrame(x_catc_transformed.toarray(), columns=MODELS["ohe_encoder"].get_feature_names_out())

        x_cat_final = pd.concat([x_catc_df.reset_index(drop=True), x_catn.reset_index(drop=True)], axis=1)
        x_num = x.select_dtypes(include='number')
        x_num_scaled = MODELS["scaler"].transform(x_num)
        x_num_scaled_df = pd.DataFrame(x_num_scaled, columns=x_num.columns)

        xr = pd.concat([x_cat_final.reset_index(drop=True), x_num_scaled_df.reset_index(drop=True)], axis=1)
        trained_columns = MODELS["kmeans"].feature_names_in_
        xr = xr.reindex(columns=trained_columns, fill_value=0)

        kmeans_cluster = MODELS["kmeans"].predict(xr)[0]
        segment_map = {0: "Low", 1: "Medium", 2: "High"}
        risk_label = segment_map.get(kmeans_cluster, "Low")

        return {"success": True, "predictedRisk": risk_label}
    except Exception as e:
        print(f"Risk Prediction Error: {e}")
        return {"success": False, "predictedRisk": "Data Error"}

# --- 3. BATCH PROCESSING FOR ADMIN DASHBOARD ---

@app.post("/api/predict-risk-batch")
async def predict_risk_batch(users_data: list = Body(...)):
    """
    Endpoint for CustomerTable bulk prediction.
    Processes a list of users and returns their KMeans risk predictions.
    """
    if MODELS["kmeans"] is None:
        return {"success": False, "error": "KMeans Model Offline"}
    try:
        results = []
        for user in users_data:
            # Call prediction logic for each item in the batch
            prediction_res = await predict_user_risk(user)
            if prediction_res["success"]:
                results.append({
                    "customerId": user.get("customerId"),
                    "riskLevel": prediction_res["predictedRisk"]
                })
        return {"success": True, "predictions": results}
    except Exception as e:
        print(f"❌ Batch Prediction Error: {str(e)}")
        return {"success": False, "error": str(e)}

# --- 4. LOAN AMOUNT PREDICTION ---

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)