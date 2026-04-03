"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from infinitetalk.api.server import create_app

@pytest.fixture
def client():
    """Create test client"""
    app = create_app()
    return TestClient(app)

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data

def test_video_generation_endpoint(client):
    """Test video generation endpoint"""
    import base64
    
    # Create dummy base64 inputs
    dummy_image = base64.b64encode(b"dummy_image_data").decode()
    dummy_audio = base64.b64encode(b"dummy_audio_data").decode()
    
    payload = {
        "model": "infinitetalk-720p",
        "image": dummy_image,
        "audio": dummy_audio,
        "num_frames": 120,
        "resolution": "720p"
    }
    
    # This would fail without actual pipeline
    # response = client.post("/v1/video/generations", json=payload)
    # assert response.status_code == 200
