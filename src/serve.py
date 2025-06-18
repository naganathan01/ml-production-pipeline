from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import logging
import time
import os
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('ml_prediction_errors_total', 'Total prediction errors')

# Pydantic models for request/response
class HouseFeatures(BaseModel):
    size_sqft: float = Field(..., gt=0, description="House size in square feet")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms") 
    age_years: int = Field(..., ge=0, le=100, description="Age of house in years")
    garage: int = Field(..., ge=0, le=5, description="Number of garage spaces")
    location_score: float = Field(..., ge=1, le=10, description="Location quality score")
    school_rating: float = Field(..., ge=1, le=10, description="School district rating")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: Dict[str, float]
    model_version: str
    prediction_id: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int

class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures]

class MLModelPredictor:
    def __init__(self, model_dir="models"):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = "1.0.0"
        self.load_time = time.time()
        self.prediction_count = 0
        self.load_model(model_dir)
    
    def load_model(self, model_dir):
        """Load the trained model and preprocessing components"""
        try:
            model_path = os.path.join(model_dir, "house_price_model.joblib")
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            feature_path = os.path.join(model_dir, "feature_names.joblib")
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_path]):
                raise FileNotFoundError("Model files not found. Please train the model first.")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feature_path)
            
            logger.info(f"Model loaded successfully. Version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def engineer_features(self, features: HouseFeatures) -> Dict[str, float]:
        """Engineer features same as training"""
        base_features = features.dict()
        
        # Add engineered features
        base_features['total_rooms'] = features.bedrooms + features.bathrooms
        base_features['luxury_score'] = (features.location_score + features.school_rating) / 2
        
        return base_features
    
    def predict(self, features: HouseFeatures) -> Dict[str, Any]:
        """Make a single prediction"""
        try:
            # Engineer features
            engineered_features = self.engineer_features(features)
            
            # Create feature vector in correct order
            feature_vector = [engineered_features[name] for name in self.feature_names]
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            # Calculate confidence interval (using prediction intervals from trees)
            tree_predictions = [tree.predict(scaled_features)[0] for tree in self.model.estimators_]
            std_dev = np.std(tree_predictions)
            confidence_interval = {
                "lower": float(prediction - 1.96 * std_dev),
                "upper": float(prediction + 1.96 * std_dev)
            }
            
            self.prediction_count += 1
            
            return {
                "prediction": float(prediction),
                "confidence_interval": confidence_interval,
                "feature_importance": self.get_feature_importance()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return {k: float(v) for k, v in importance_dict.items()}
        return {}

# Initialize the model predictor
try:
    predictor = MLModelPredictor()
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None
    model_loaded = False

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Production-grade ML API for house price prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("House Price Prediction API starting up...")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    uptime = time.time() - (predictor.load_time if predictor else time.time())
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
        total_predictions=predictor.prediction_count if predictor else 0
    )

# Readiness probe
@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_price(
    features: HouseFeatures,
    background_tasks: BackgroundTasks
):
    """Predict house price for a single house"""
    if not model_loaded:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Make prediction
        result = predictor.predict(features)
        
        # Generate prediction ID
        prediction_id = f"pred_{int(time.time())}_{predictor.prediction_count}"
        
        # Record metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        # Log prediction (in background)
        background_tasks.add_task(
            log_prediction,
            prediction_id,
            features.dict(),
            result["prediction"]
        )
        
        return PredictionResponse(
            predicted_price=result["prediction"],
            confidence_interval=result["confidence_interval"],
            model_version=predictor.model_version,
            prediction_id=prediction_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict prices for multiple houses"""
    if not model_loaded:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=503, detail="Model not available")
    
    if len(request.houses) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 houses per batch")
    
    try:
        predictions = []
        for i, house in enumerate(request.houses):
            result = predictor.predict(house)
            predictions.append({
                "house_index": i,
                "predicted_price": result["prediction"],
                "confidence_interval": result["confidence_interval"]
            })
            PREDICTION_COUNTER.inc()
        
        return {
            "predictions": predictions,
            "model_version": predictor.model_version,
            "batch_size": len(request.houses),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")
    
    return {
        "model_version": predictor.model_version,
        "model_type": "RandomForestRegressor",
        "feature_names": predictor.feature_names,
        "feature_importance": predictor.get_feature_importance(),
        "total_predictions": predictor.prediction_count,
        "uptime_seconds": time.time() - predictor.load_time
    }

# Background task for logging
async def log_prediction(prediction_id: str, features: dict, prediction: float):
    """Log prediction for monitoring and retraining"""
    log_entry = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "model_version": predictor.model_version
    }
    
    # In production, this would go to a database or data lake
    logger.info(f"Prediction logged: {json.dumps(log_entry)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "status": "healthy" if model_loaded else "model not loaded",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)