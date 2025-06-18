import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, data_path="data/housing.csv"):
        """Load and prepare housing data"""
        try:
            # Create synthetic housing data if file doesn't exist
            if not os.path.exists(data_path):
                logger.info("Creating synthetic housing data...")
                self.create_synthetic_data(data_path)
            
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_synthetic_data(self, data_path, n_samples=10000):
        """Create synthetic housing data for demo"""
        np.random.seed(42)
        
        # Generate realistic housing features
        data = {
            'size_sqft': np.random.normal(2000, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'age_years': np.random.randint(0, 50, n_samples),
            'garage': np.random.randint(0, 3, n_samples),
            'location_score': np.random.uniform(1, 10, n_samples),
            'school_rating': np.random.uniform(1, 10, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic price based on features
        df['price'] = (
            df['size_sqft'] * 150 +
            df['bedrooms'] * 10000 +
            df['bathrooms'] * 15000 +
            (50 - df['age_years']) * 1000 +
            df['garage'] * 8000 +
            df['location_score'] * 20000 +
            df['school_rating'] * 15000 +
            np.random.normal(0, 20000, n_samples)
        )
        
        # Ensure positive prices
        df['price'] = np.abs(df['price'])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Created synthetic data: {data_path}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        # Handle missing values
        df = df.fillna(df.median())
        
        # Feature engineering
        df['price_per_sqft'] = df['price'] / df['size_sqft']
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['luxury_score'] = (df['location_score'] + df['school_rating']) / 2
        
        # Separate features and target
        feature_cols = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 
                       'garage', 'location_score', 'school_rating', 
                       'total_rooms', 'luxury_score']
        
        X = df[feature_cols]
        y = df['price']
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Start MLflow run
        with mlflow.start_run():
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics to MLflow
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info(f"Model trained - RMSE: {rmse:.2f}, R2: {r2:.4f}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'model': self.model,
                'scaler': self.scaler
            }
    
    def save_model(self, model_dir="models"):
        """Save model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "house_price_model.joblib")
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        feature_path = os.path.join(model_dir, "feature_names.joblib")
        joblib.dump(self.feature_names, feature_path)
        
        logger.info(f"Model saved to {model_dir}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'feature_path': feature_path
        }

def main():
    """Main training function"""
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("house-price-prediction")
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load and preprocess data
    df = predictor.load_data()
    X, y = predictor.preprocess_data(df)
    
    # Train model
    results = predictor.train(X, y)
    
    # Save model
    paths = predictor.save_model()
    
    # Print results
    print(f"Training completed!")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"R2 Score: {results['r2']:.4f}")
    print(f"Model saved to: {paths['model_path']}")

if __name__ == "__main__":
    main()