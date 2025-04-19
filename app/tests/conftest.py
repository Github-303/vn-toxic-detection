"""Test fixtures and configuration."""
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Generator
from unittest.mock import MagicMock
import uuid

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from app.main import app
from app.models.user import User
from app.models.comment import Comment
from app.models.comment_vector import CommentVector
from app.models.log import Log
from app.schemas.user import UserCreate
from app.schemas.comment import CommentCreate, ToxicityLevel
from app.utils.auth import create_access_token
from app.config.settings import settings

@pytest.fixture
def test_db():
    """Create test database engine and session."""
    test_db_url = "postgresql+asyncpg://postgres:secret@localhost:5432/test_db"
    engine = create_async_engine(test_db_url, echo=True)
    TestingSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    return TestingSessionLocal()

@pytest.fixture
def client() -> Generator:
    """Create test client."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_user() -> Dict:
    """Create mock user data."""
    return {
        "id": uuid.uuid4(),
        "username": "testuser",
        "email": "test@example.com",
        "password_hash": "hashed_password",
        "is_active": True,
        "role": "user",
        "created_at": datetime.utcnow()
    }

@pytest.fixture
def mock_admin_user() -> Dict:
    """Create mock admin user data."""
    return {
        "id": uuid.uuid4(),
        "username": "admin",
        "email": "admin@example.com",
        "password_hash": "hashed_password",
        "is_active": True,
        "role": "admin",
        "created_at": datetime.utcnow()
    }

@pytest.fixture
def user_token(mock_user: Dict) -> str:
    """Create JWT token for test user."""
    return create_access_token(
        data={"sub": str(mock_user["id"])},
        expires_delta=timedelta(minutes=30)
    )

@pytest.fixture
def admin_token(mock_admin_user: Dict) -> str:
    """Create JWT token for admin user."""
    return create_access_token(
        data={"sub": str(mock_admin_user["id"])},
        expires_delta=timedelta(minutes=30)
    )

@pytest.fixture
def mock_comment() -> Dict:
    """Create mock comment data."""
    return {
        "id": uuid.uuid4(),
        "content": "This is a test comment",
        "user_id": uuid.uuid4(),
        "platform": "web",
        "label": ToxicityLevel.SAFE.value,
        "prediction_code": 0,
        "preprocessed_text": "test comment",
        "detected_at": datetime.utcnow()
    }

@pytest.fixture
def mock_toxic_comment() -> Dict:
    """Create mock toxic comment data."""
    return {
        "id": uuid.uuid4(),
        "content": "This is a toxic comment",
        "user_id": uuid.uuid4(),
        "platform": "web",
        "label": ToxicityLevel.TOXIC.value,
        "prediction_code": 1,
        "preprocessed_text": "toxic comment",
        "detected_at": datetime.utcnow()
    }

@pytest.fixture
def mock_comment_vector() -> Dict:
    """Create mock comment vector data."""
    return {
        "comment_id": uuid.uuid4(),
        "embedding": [0.1] * 768,
        "created_at": datetime.utcnow()
    }

@pytest.fixture
def mock_ml_prediction() -> Dict:
    """Create mock ML prediction result."""
    return {
        "text": "Test comment",
        "label": ToxicityLevel.SAFE.value,
        "preprocessed_text": "test comment",
        "prediction": 0,
        "confidence": 0.95,
        "processing_time": 0.123
    }

@pytest.fixture
def mock_ml_service():
    """Create mock ML service."""
    service = MagicMock()
    service.predict.return_value = {
        "success": True,
        "text": "Test comment",
        "label": ToxicityLevel.SAFE.value,
        "preprocessed_text": "test comment",
        "prediction": 0
    }
    return service

@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    session = MagicMock(spec=AsyncSession)
    return session

@pytest.fixture
def user_registration_data():
    """Create user registration data."""
    return UserCreate(
        username="newuser",
        email="newuser@example.com",
        password="StrongPass123!",
        password_confirm="StrongPass123!"
    )

@pytest.fixture
def comment_creation_data():
    """Create comment creation data."""
    return CommentCreate(
        content="This is a new comment for testing",
        platform="web"
    ) 