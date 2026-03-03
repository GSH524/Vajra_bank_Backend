import os
import joblib
import warnings

# Use the same path logic as your main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# This list matches every filename mentioned in your load_models() function
model_files = [
    "kmeans.pkl",
    "Scalar.pkl",
    "label_encoder.pkl",
    "OHE_encoding.pkl",
    "Loan_Predictions.pkl",
    "Loanammount_Prediction_ohe.pkl",
    "Gaussian_model.pkl",
    "Gaussian_Label_active.pkl",
    "Gaussian_label_status.pkl"
]

def upgrade_models():
    print(f"--- Starting Model Migration to Current Scikit-Learn Version ---")
    
    for filename in model_files:
        file_path = os.path.join(MODELS_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping: {filename} (File not found)")
            continue
            
        try:
            # Silence the warning during the load process
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_obj = joblib.load(file_path)
            
            # Re-save the object. Joblib will use the current version of 
            # scikit-learn classes found in your venv to pickle the data.
            joblib.dump(model_obj, file_path)
            print(f"✅ Successfully upgraded: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to upgrade {filename}: {e}")

    print("\n--- All models processed. You can now run uvicorn! ---")

if __name__ == "__main__":
    upgrade_models()