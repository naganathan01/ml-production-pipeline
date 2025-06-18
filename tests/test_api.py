import pytest
import json
import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock the model loading before importing the app
with patch('src.serve.MLModelPredictor') as mock_predictor:
    mock_instance = MagicMock()
    mock_instance.model_version = "1.0.0"
    mock_instance.prediction_count = 0
    mock_instance.load_time = 1234567890
    mock_instance.feature_names = ["size_sqft", "bedrooms", "bathrooms", "age_years", 
                                  "garage", "location_score", "school_rating", 
                                  "total_rooms", "luxury_score"]
    mock_predictor.return_value = mock_instance
    
    from serve import app

# Create test client
client = TestClient(app)

class TestHealthEndpoints:
    
    def test_health_check_healthy(self):
        """Test health check when model is loaded"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
        assert "uptime_seconds" in data
        assert "total_predictions" in data
    
    def test_readiness_check(self):
        """Test readiness probe"""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "House Price Prediction API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "healthy"

class TestModelInfoEndpoint:
    
    def test_model_info(self):
        """Test model information endpoint"""
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.model_version = "1.0.0"
            mock_predictor.feature_names = ["size_sqft", "bedrooms"]
            mock_predictor.prediction_count = 5
            mock_predictor.load_time = 1234567890
            mock_predictor.get_feature_importance.return_value = {"size_sqft": 0.7, "bedrooms": 0.3}
            
            response = client.get("/model/info")
            assert response.status_code == 200
            
            data = response.json()
            assert data["model_version"] == "1.0.0"
            assert "prediction_id" in data
            assert "timestamp" in data
    
    def test_single_prediction_validation_errors(self):
        """Test prediction with validation errors"""
        invalid_data = {
            "size_sqft": -1000,  # Invalid: negative
            "bedrooms": 0,       # Invalid: too low
            "bathrooms": 15,     # Invalid: too high
            "age_years": -5,     # Invalid: negative
            "garage": 10,        # Invalid: too high
            "location_score": 15, # Invalid: too high
            "school_rating": 0    # Invalid: too low
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_single_prediction_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_data = {
            "size_sqft": 2000.0,
            "bedrooms": 3
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    def test_batch_prediction_success(self, valid_house_data):
        """Test successful batch prediction"""
        batch_data = {
            "houses": [valid_house_data, valid_house_data]
        }
        
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 450000.0,
                "confidence_interval": {"lower": 400000.0, "upper": 500000.0}
            }
            mock_predictor.model_version = "1.0.0"
            
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == 200
            
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["batch_size"] == 2
            assert data["model_version"] == "1.0.0"
            
            # Check individual predictions
            for i, pred in enumerate(data["predictions"]):
                assert pred["house_index"] == i
                assert pred["predicted_price"] == 450000.0
                assert "confidence_interval" in pred
    
    def test_batch_prediction_too_large(self, valid_house_data):
        """Test batch prediction with too many houses"""
        large_batch = {
            "houses": [valid_house_data] * 101  # Exceeds limit of 100
        }
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 400
        assert "Maximum 100 houses per batch" in response.json()["detail"]
    
    def test_prediction_with_model_error(self, valid_house_data):
        """Test prediction when model throws an error"""
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.side_effect = Exception("Model prediction failed")
            
            response = client.post("/predict", json=valid_house_data)
            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]

class TestModelNotLoaded:
    """Test behavior when model is not loaded"""
    
    def test_prediction_when_model_not_loaded(self, valid_house_data):
        """Test prediction when model is not available"""
        with patch('src.serve.model_loaded', False):
            response = client.post("/predict", json=valid_house_data)
            assert response.status_code == 503
            assert "Model not available" in response.json()["detail"]
    
    def test_batch_prediction_when_model_not_loaded(self, valid_house_data):
        """Test batch prediction when model is not available"""
        batch_data = {"houses": [valid_house_data]}
        
        with patch('src.serve.model_loaded', False):
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == 503
            assert "Model not available" in response.json()["detail"]
    
    def test_model_info_when_model_not_loaded(self):
        """Test model info when model is not available"""
        with patch('src.serve.model_loaded', False):
            response = client.get("/model/info")
            assert response.status_code == 503
            assert "Model not available" in response.json()["detail"]
    
    def test_health_check_when_model_not_loaded(self):
        """Test health check when model is not loaded"""
        with patch('src.serve.model_loaded', False), \
             patch('src.serve.predictor', None):
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] == False
    
    def test_readiness_when_model_not_loaded(self):
        """Test readiness when model is not loaded"""
        with patch('src.serve.model_loaded', False):
            response = client.get("/ready")
            assert response.status_code == 503

class TestInputValidation:
    """Test various input validation scenarios"""
    
    def test_edge_case_values(self):
        """Test edge case but valid values"""
        edge_case_data = {
            "size_sqft": 1.0,      # Minimum valid
            "bedrooms": 1,         # Minimum valid
            "bathrooms": 1,        # Minimum valid
            "age_years": 0,        # Minimum valid
            "garage": 0,           # Minimum valid
            "location_score": 1.0, # Minimum valid
            "school_rating": 1.0   # Minimum valid
        }
        
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 100000.0,
                "confidence_interval": {"lower": 90000.0, "upper": 110000.0}
            }
            
            response = client.post("/predict", json=edge_case_data)
            assert response.status_code == 200
    
    def test_maximum_values(self):
        """Test maximum valid values"""
        max_values_data = {
            "size_sqft": 10000.0,
            "bedrooms": 10,
            "bathrooms": 10,
            "age_years": 100,
            "garage": 5,
            "location_score": 10.0,
            "school_rating": 10.0
        }
        
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 2000000.0,
                "confidence_interval": {"lower": 1800000.0, "upper": 2200000.0}
            }
            
            response = client.post("/predict", json=max_values_data)
            assert response.status_code == 200
    
    def test_float_precision(self):
        """Test handling of float precision"""
        precise_data = {
            "size_sqft": 2000.123456789,
            "bedrooms": 3,
            "bathrooms": 2,
            "age_years": 5,
            "garage": 2,
            "location_score": 8.567890123,
            "school_rating": 9.123456789
        }
        
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 450000.123,
                "confidence_interval": {"lower": 400000.0, "upper": 500000.0}
            }
            
            response = client.post("/predict", json=precise_data)
            assert response.status_code == 200

class TestConcurrency:
    """Test concurrent requests"""
    
    def test_multiple_concurrent_predictions(self, valid_house_data):
        """Test multiple simultaneous predictions"""
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 450000.0,
                "confidence_interval": {"lower": 400000.0, "upper": 500000.0}
            }
            mock_predictor.model_version = "1.0.0"
            mock_predictor.prediction_count = 0
            
            # Simulate multiple concurrent requests
            responses = []
            for i in range(5):
                response = client.post("/predict", json=valid_house_data)
                responses.append(response)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["predicted_price"] == 450000.0

class TestResponseFormat:
    """Test response format compliance"""
    
    def test_prediction_response_schema(self, valid_house_data):
        """Test that prediction response matches expected schema"""
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 450000.0,
                "confidence_interval": {"lower": 400000.0, "upper": 500000.0}
            }
            mock_predictor.model_version = "1.0.0"
            mock_predictor.prediction_count = 1
            
            response = client.post("/predict", json=valid_house_data)
            assert response.status_code == 200
            
            data = response.json()
            
            # Check required fields
            required_fields = [
                "predicted_price", "confidence_interval", 
                "model_version", "prediction_id", "timestamp"
            ]
            for field in required_fields:
                assert field in data
            
            # Check data types
            assert isinstance(data["predicted_price"], (int, float))
            assert isinstance(data["confidence_interval"], dict)
            assert "lower" in data["confidence_interval"]
            assert "upper" in data["confidence_interval"]
            assert isinstance(data["model_version"], str)
            assert isinstance(data["prediction_id"], str)
            assert isinstance(data["timestamp"], str)
    
    def test_error_response_format(self):
        """Test error response format"""
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data

# Performance and load testing
class TestPerformance:
    """Basic performance tests"""
    
    def test_prediction_latency(self, valid_house_data):
        """Test that predictions complete within reasonable time"""
        import time
        
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 450000.0,
                "confidence_interval": {"lower": 400000.0, "upper": 500000.0}
            }
            
            start_time = time.time()
            response = client.post("/predict", json=valid_house_data)
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    def test_batch_prediction_efficiency(self, valid_house_data):
        """Test that batch predictions are reasonably efficient"""
        import time
        
        batch_data = {"houses": [valid_house_data] * 10}
        
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 450000.0,
                "confidence_interval": {"lower": 400000.0, "upper": 500000.0}
            }
            
            start_time = time.time()
            response = client.post("/predict/batch", json=batch_data)
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 2.0  # Batch of 10 should complete within 2 seconds"
            assert data["model_type"] == "RandomForestRegressor"
            assert "feature_names" in data
            assert "feature_importance" in data

class TestPredictionEndpoints:
    
    @pytest.fixture
    def valid_house_data(self):
        """Valid house data for testing"""
        return {
            "size_sqft": 2000.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "age_years": 5,
            "garage": 2,
            "location_score": 8.5,
            "school_rating": 9.0
        }
    
    def test_single_prediction_success(self, valid_house_data):
        """Test successful single prediction"""
        with patch('src.serve.predictor') as mock_predictor:
            mock_predictor.predict.return_value = {
                "prediction": 450000.0,
                "confidence_interval": {"lower": 400000.0, "upper": 500000.0},
                "feature_importance": {"size_sqft": 0.5}
            }
            mock_predictor.model_version = "1.0.0"
            mock_predictor.prediction_count = 1
            
            response = client.post("/predict", json=valid_house_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["predicted_price"] == 450000.0
            assert "confidence_interval" in data
            assert data["model_version"] == "1.0.0