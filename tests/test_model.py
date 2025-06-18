import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import HousePricePredictor

class TestHousePricePredictor:
    
    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for testing"""
        return HousePricePredictor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'size_sqft': [1500, 2000, 2500, 1800, 2200],
            'bedrooms': [2, 3, 4, 3, 3],
            'bathrooms': [1, 2, 3, 2, 2],
            'age_years': [5, 10, 15, 8, 12],
            'garage': [1, 2, 2, 1, 2],
            'location_score': [7.5, 8.0, 9.0, 7.0, 8.5],
            'school_rating': [8.0, 8.5, 9.5, 7.5, 8.0],
            'price': [300000, 400000, 550000, 350000, 450000]
        })
    
    def test_create_synthetic_data(self, predictor, tmp_path):
        """Test synthetic data creation"""
        data_path = tmp_path / "test_housing.csv"
        df = predictor.create_synthetic_data(str(data_path), n_samples=100)
        
        # Check that file was created
        assert data_path.exists()
        
        # Check data properties
        assert len(df) == 100
        assert 'price' in df.columns
        assert 'size_sqft' in df.columns
        assert df['price'].min() > 0  # All prices should be positive
        
        # Check that data is realistic
        assert df['bedrooms'].min() >= 1
        assert df['bedrooms'].max() <= 6
        assert df['bathrooms'].min() >= 1
        assert df['bathrooms'].max() <= 4
    
    def test_preprocess_data(self, predictor, sample_data):
        """Test data preprocessing"""
        X, y = predictor.preprocess_data(sample_data)
        
        # Check shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Check feature engineering
        assert 'total_rooms' in X.columns
        assert 'luxury_score' in X.columns
        
        # Check feature values
        assert all(X['total_rooms'] == X['bedrooms'] + X['bathrooms'])
        expected_luxury = (sample_data['location_score'] + sample_data['school_rating']) / 2
        assert all(abs(X['luxury_score'] - expected_luxury) < 0.001)
        
        # Check target variable
        assert all(y == sample_data['price'])
    
    def test_preprocess_data_with_missing_values(self, predictor):
        """Test preprocessing with missing values"""
        data_with_na = pd.DataFrame({
            'size_sqft': [1500, np.nan, 2500],
            'bedrooms': [2, 3, np.nan],
            'bathrooms': [1, 2, 3],
            'age_years': [5, 10, 15],
            'garage': [1, 2, 2],
            'location_score': [7.5, 8.0, 9.0],
            'school_rating': [8.0, 8.5, 9.5],
            'price': [300000, 400000, 550000]
        })
        
        X, y = predictor.preprocess_data(data_with_na)
        
        # Should not have any NaN values after preprocessing
        assert not X.isnull().any().any()
        assert not y.isnull().any()
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('mlflow.sklearn.log_model')
    def test_train_model(self, mock_log_model, mock_log_param, mock_log_metric, 
                        mock_start_run, predictor, sample_data):
        """Test model training"""
        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        X, y = predictor.preprocess_data(sample_data)
        results = predictor.train(X, y)
        
        # Check that model was trained
        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict')
        
        # Check that scaler was fitted
        assert predictor.scaler is not None
        
        # Check results
        assert 'mse' in results
        assert 'rmse' in results
        assert 'r2' in results
        assert results['mse'] >= 0
        assert results['rmse'] >= 0
        assert -1 <= results['r2'] <= 1
        
        # Check MLflow logging was called
        mock_log_metric.assert_called()
        mock_log_param.assert_called()
        mock_log_model.assert_called_once()
    
    def test_save_and_load_model(self, predictor, sample_data, tmp_path):
        """Test saving and loading model"""
        # Train model first
        X, y = predictor.preprocess_data(sample_data)
        with patch('mlflow.start_run'), patch('mlflow.log_metric'), \
             patch('mlflow.log_param'), patch('mlflow.sklearn.log_model'):
            predictor.train(X, y)
        
        # Save model
        model_dir = tmp_path / "models"
        paths = predictor.save_model(str(model_dir))
        
        # Check files were created
        assert os.path.exists(paths['model_path'])
        assert os.path.exists(paths['scaler_path'])
        assert os.path.exists(paths['feature_path'])
        
        # Test loading
        loaded_model = joblib.load(paths['model_path'])
        loaded_scaler = joblib.load(paths['scaler_path'])
        loaded_features = joblib.load(paths['feature_path'])
        
        # Check loaded objects
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_scaler, 'transform')
        assert isinstance(loaded_features, list)
        assert len(loaded_features) > 0
    
    def test_model_prediction_consistency(self, predictor, sample_data):
        """Test that model predictions are consistent"""
        X, y = predictor.preprocess_data(sample_data)
        
        with patch('mlflow.start_run'), patch('mlflow.log_metric'), \
             patch('mlflow.log_param'), patch('mlflow.sklearn.log_model'):
            predictor.train(X, y)
        
        # Make predictions multiple times
        X_scaled = predictor.scaler.transform(X)
        pred1 = predictor.model.predict(X_scaled)
        pred2 = predictor.model.predict(X_scaled)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_feature_names_consistency(self, predictor, sample_data):
        """Test that feature names are consistent"""
        X, y = predictor.preprocess_data(sample_data)
        
        expected_features = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 
                           'garage', 'location_score', 'school_rating', 
                           'total_rooms', 'luxury_score']
        
        assert predictor.feature_names == expected_features
        assert len(predictor.feature_names) == X.shape[1]
    
    def test_model_performance_threshold(self, predictor, sample_data):
        """Test that model meets minimum performance threshold"""
        X, y = predictor.preprocess_data(sample_data)
        
        with patch('mlflow.start_run'), patch('mlflow.log_metric'), \
             patch('mlflow.log_param'), patch('mlflow.sklearn.log_model'):
            results = predictor.train(X, y)
        
        # Model should have reasonable R2 score (at least better than random)
        assert results['r2'] > 0.5, f"R2 score {results['r2']} is too low"
        
        # RMSE should be reasonable relative to target values
        mean_price = y.mean()
        assert results['rmse'] < mean_price * 0.5, f"RMSE {results['rmse']} is too high relative to mean price {mean_price}"

# Integration test
class TestEndToEndWorkflow:
    
    def test_complete_workflow(self, tmp_path):
        """Test complete training workflow"""
        # Initialize predictor
        predictor = HousePricePredictor()
        
        # Create synthetic data
        data_path = tmp_path / "housing.csv"
        df = predictor.create_synthetic_data(str(data_path), n_samples=1000)
        
        # Load and preprocess data
        df_loaded = predictor.load_data(str(data_path))
        X, y = predictor.preprocess_data(df_loaded)
        
        # Train model
        with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'), \
             patch('mlflow.start_run'), patch('mlflow.log_metric'), \
             patch('mlflow.log_param'), patch('mlflow.sklearn.log_model'):
            results = predictor.train(X, y)
        
        # Save model
        model_dir = tmp_path / "models"
        paths = predictor.save_model(str(model_dir))
        
        # Verify everything worked
        assert os.path.exists(paths['model_path'])
        assert results['r2'] > 0.7  # Should have good performance on synthetic data
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.feature_names is not None