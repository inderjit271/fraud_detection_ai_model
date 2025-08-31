from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import io
import json
import os
from typing import Dict, List, Optional
import uvicorn

from modules.data_processor import DataProcessor
from modules.model_trainer import ModelTrainer
from modules.synthetic_generator import SyntheticGenerator
from modules.visualizer import Visualizer
from modules.utils import validate_file, get_sample_data_info

app = FastAPI(title="XposeAI - Fraud Detection Dashboard", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables to store session data
session_data = {
    "uploaded_data": None,
    "column_mapping": {},
    "analysis_results": None,
    "trained_model": None,
    "training_metrics": None
}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_csv")
async def upload_csv(uploaded_file: UploadFile = File(...)):
    """Upload and validate CSV file"""
    try:
        # Validate file
        if not uploaded_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file")

        # ✅ Read CSV in chunks (no size limit)
        df_chunks = pd.read_csv(uploaded_file.file, chunksize=100000, low_memory=False)
        df = pd.concat(df_chunks, ignore_index=True)

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="File must have at least 2 columns")
        
        # Auto-detect columns
        processor = DataProcessor()
        detected_mapping = processor.auto_detect_columns(df)
        
        # Store data
        session_data["uploaded_data"] = df
        session_data["column_mapping"] = detected_mapping if detected_mapping else {}
        
        return {
            "success": True,
            "message": f"File uploaded successfully! Shape: {df.shape}",
            "shape": list(df.shape),
            "columns": list(df.columns),
            "preview": df.head(10).to_dict('records'),
            "detected_mapping": detected_mapping or {}
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@app.post("/analyze_data")
async def analyze_data():
    """Analyze uploaded data for fraud patterns"""
    try:
        if session_data["uploaded_data"] is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        if not session_data["column_mapping"]:
            raise HTTPException(status_code=400, detail="No column mapping provided")
        
        processor = DataProcessor()
        processed_data = processor.process_data(
            session_data["uploaded_data"], 
            session_data["column_mapping"]
        )
        
        if processed_data is None:
            raise HTTPException(status_code=400, detail="Data processing failed")
        
        # Analyze patterns
        analysis_results = processor.analyze_fraud_patterns(processed_data)
        session_data["analysis_results"] = analysis_results
        
        # Create visualizations
        visualizer = Visualizer()
        
        # Generate charts data
        fraud_dist_fig = visualizer.plot_fraud_distribution(processed_data)
        amount_dist_fig = visualizer.plot_amount_distribution(processed_data)
        
        fraud_count = len(analysis_results['fraud_transactions'])
        total_count = len(processed_data)
        fraud_rate = (fraud_count / total_count) * 100
        avg_fraud_amount = analysis_results['fraud_transactions']['amount'].mean() if fraud_count > 0 else 0
        
        return {
            "success": True,
            "metrics": {
                "total_transactions": total_count,
                "fraud_transactions": fraud_count,
                "fraud_rate": round(fraud_rate, 2),
                "avg_fraud_amount": round(avg_fraud_amount, 2)
            },
            "fraud_transactions": analysis_results['fraud_transactions'].head(20).to_dict('records') if fraud_count > 0 else [],
            "charts": {
                "fraud_distribution": fraud_dist_fig.to_json(),
                "amount_distribution": amount_dist_fig.to_json()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/train_model")
async def train_model(
    algorithm: str = Form(...),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    cross_validation: bool = Form(True)
):
    """Train a new fraud detection model"""
    try:
        if session_data["uploaded_data"] is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        if not session_data["column_mapping"]:
            raise HTTPException(status_code=400, detail="No column mapping provided")
        
        processor = DataProcessor()
        processed_data = processor.process_data(
            session_data["uploaded_data"], 
            session_data["column_mapping"]
        )
        
        if processed_data is None:
            raise HTTPException(status_code=400, detail="Data processing failed")
        
        # Train model
        trainer = ModelTrainer()
        model, metrics = trainer.train_model(
            processed_data,
            algorithm=algorithm.lower().replace(' ', '_'),
            test_size=test_size,
            random_state=random_state,
            cross_validation=cross_validation
        )
        
        session_data["trained_model"] = model
        session_data["training_metrics"] = metrics
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_filename = f"trained_model_{algorithm.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, f"models/{model_filename}")
        
        return {
            "success": True,
            "message": f"Model trained successfully! Saved as {model_filename}",
            "metrics": {
                "accuracy": round(metrics['accuracy'], 3),
                "precision": round(metrics['precision'], 3),
                "recall": round(metrics['recall'], 3),
                "f1": round(metrics['f1'], 3),
                "auc": round(metrics.get('auc', 0.5), 3)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/load_pretrained")
async def load_pretrained_model():
    """Load pretrained fraud detection model"""
    try:
        model_path = 'models/pretrained_model.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            session_data["trained_model"] = model
            return {"success": True, "message": "Pretrained model loaded successfully!"}
        else:
            # Create a basic model if pretrained doesn't exist
            trainer = ModelTrainer()
            model = trainer.create_basic_model()
            session_data["trained_model"] = model
            return {"success": True, "message": "Basic model created and loaded!"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.post("/generate_synthetic")
async def generate_synthetic_data(
    num_samples: int = Form(1000),
    fraud_ratio: float = Form(0.1),
    dataset_type: str = Form("credit_card"),
    amount_min: int = Form(10),
    amount_max: int = Form(5000),
    include_anomalies: bool = Form(True),
    seasonal_patterns: bool = Form(True),
    correlation_features: bool = Form(True)
):
    """Generate synthetic fraud data"""
    try:
        generator = SyntheticGenerator()
        
        synthetic_data = generator.generate_fraud_data(
            num_samples=num_samples,
            fraud_ratio=fraud_ratio,
            dataset_type=dataset_type.lower().replace(' ', '_'),
            amount_range=(amount_min, amount_max),
            include_anomalies=include_anomalies,
            seasonal_patterns=seasonal_patterns,
            correlation_features=correlation_features
        )
        
        fraud_count = sum(synthetic_data['is_fraud'])
        actual_ratio = fraud_count / num_samples
        
        return {
            "success": True,
            "message": "Synthetic data generated successfully!",
            "summary": {
                "total_samples": num_samples,
                "fraud_samples": fraud_count,
                "actual_fraud_ratio": round(actual_ratio, 3)
            },
            "preview": synthetic_data.head(10).to_dict('records'),
            "download_data": synthetic_data.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/use_synthetic_for_training")
async def use_synthetic_for_training(synthetic_data: List[Dict]):
    """Use synthetic data for model training"""
    try:
        df = pd.DataFrame(synthetic_data)
        
        # Set up column mapping for synthetic data
        column_mapping = {
            'amount': 'amount',
            'fraud_label': 'is_fraud',
            'timestamp': 'timestamp' if 'timestamp' in df.columns else None,
            'merchant': 'merchant_category' if 'merchant_category' in df.columns else None
        }
        
        session_data["uploaded_data"] = df
        session_data["column_mapping"] = {k: v for k, v in column_mapping.items() if v and v in df.columns}
        
        return {
            "success": True,
            "message": "Synthetic data loaded for training!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loading synthetic data failed: {str(e)}")

@app.post("/predict")
async def make_predictions():
    """Run predictions on current dataset"""
    try:
        if session_data["uploaded_data"] is None:
            raise HTTPException(status_code=400, detail="No data available")
        
        if session_data["trained_model"] is None:
            raise HTTPException(status_code=400, detail="No model loaded")
        
        processor = DataProcessor()
        processed_data = processor.process_data(
            session_data["uploaded_data"], 
            session_data["column_mapping"]
        )
        
        if processed_data is None:
            raise HTTPException(status_code=400, detail="Data processing failed")
        
        # Prepare features
        feature_columns = [col for col in processed_data.columns if col != 'is_fraud']
        X = processed_data[feature_columns]
        
        # Make predictions
        predictions = session_data["trained_model"].predict(X)
        probabilities = session_data["trained_model"].predict_proba(X)[:, 1]
        
        # Create results DataFrame
        results = processed_data.copy()
        results['predicted_fraud'] = predictions
        results['fraud_probability'] = probabilities
        results['prediction_correct'] = (results['is_fraud'] == results['predicted_fraud'])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if 'is_fraud' in processed_data.columns:
            y_true = processed_data['is_fraud']
            metrics = {
                "accuracy": round(accuracy_score(y_true, predictions), 3),
                "precision": round(precision_score(y_true, predictions, zero_division=0), 3),
                "recall": round(recall_score(y_true, predictions, zero_division=0), 3),
                "f1": round(f1_score(y_true, predictions, zero_division=0), 3)
            }
        else:
            metrics = {}
        
        # High-risk predictions
        high_risk = results[results['fraud_probability'] > 0.7].head(20)
        
        return {
            "success": True,
            "message": f"Predictions completed for {len(results)} transactions",
            "metrics": metrics,
            "high_risk_transactions": high_risk.to_dict('records'),
            "prediction_summary": {
                "total_transactions": len(results),
                "predicted_fraud": int(sum(predictions)),
                "high_risk_count": len(high_risk)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get current application status"""
    return {
        "data_uploaded": session_data["uploaded_data"] is not None,
        "has_mapping": bool(session_data["column_mapping"]),
        "has_analysis": session_data["analysis_results"] is not None,
        "model_loaded": session_data["trained_model"] is not None,
        "has_metrics": session_data["training_metrics"] is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)