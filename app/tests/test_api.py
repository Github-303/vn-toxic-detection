"""Tests for API endpoints."""
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from app.config.settings import settings

def test_root(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Vietnamese Toxic Comment Detection API"}

def test_docs_available(client: TestClient) -> None:
    """Test OpenAPI documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200
    response = client.get("/openapi.json")
    assert response.status_code == 200

def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_auth_flow(client: TestClient) -> None:
    """Test complete authentication flow."""
    # Test registration
    register_data = {
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "Test User"
    }
    response = client.post(f"{settings.API_V1_STR}/auth/register", json=register_data)
    assert response.status_code == 201
    assert "id" in response.json()
    
    # Test login
    login_data = {
        "username": register_data["email"],
        "password": register_data["password"]
    }
    response = client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()
    token = response.json()["access_token"]
    
    # Test protected endpoint
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get(f"{settings.API_V1_STR}/users/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["email"] == register_data["email"]

def test_unauthorized_access(client: TestClient) -> None:
    """Test unauthorized access to protected endpoints."""
    response = client.get(f"{settings.API_V1_STR}/users/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

def test_model_prediction(client: TestClient, auth_token: Dict[str, str]) -> None:
    """Test model prediction endpoint."""
    test_comment = {
        "text": "Đây là một bình luận tích cực",
        "model_type": "transformer"
    }
    response = client.post(
        f"{settings.API_V1_STR}/detect",
        json=test_comment,
        headers=auth_token
    )
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert "processing_time" in result
