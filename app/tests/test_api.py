"""Test API endpoints."""
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio

from app.main import app, MOCK_USER_ID
from app.utils.auth import create_access_token

client = TestClient(app)

@pytest.mark.asyncio
async def test_root():
    """Test root endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to Toxic Comment Detection API"}

@pytest.mark.asyncio
async def test_docs_available():
    """Test OpenAPI docs are available."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/docs")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_auth_flow():
    """Test authentication flow (register -> login -> access protected endpoint)."""
    # Registration data
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepassword123"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Register user
        register_response = await ac.post("/auth/register", json=user_data)
        assert register_response.status_code == 201
        
        # Login
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        login_response = await ac.post("/auth/login", data=login_data)
        assert login_response.status_code == 200
        assert "access_token" in login_response.json()
        
        # Access protected endpoint with token
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        protected_response = await ac.get("/users/me", headers=headers)
        assert protected_response.status_code == 200
        assert protected_response.json()["username"] == user_data["username"]

@pytest.mark.asyncio
async def test_unauthorized_access():
    """Test unauthorized access to protected endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/users/me")
        assert response.status_code == 401
        assert "detail" in response.json()

@pytest.mark.asyncio
async def test_model_prediction():
    """Test model prediction endpoint."""
    # Create a test token
    token = create_access_token(data={"sub": MOCK_USER_ID})
    headers = {"Authorization": f"Bearer {token}"}
    
    # Comment to analyze
    test_comment = {
        "content": "This is a test comment",
        "platform": "web"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/comments/analyze", json=test_comment, headers=headers)
        assert response.status_code == 200
        
        # Verify response structure
        data = response.json()
        assert "id" in data
        assert "content" in data
        assert "toxicity_level" in data
        assert "toxicity_score" in data
        assert "created_at" in data
