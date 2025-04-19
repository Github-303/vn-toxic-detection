"""
Tests for authentication controller.
"""
import uuid
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime, timedelta

from app.controllers.auth import AuthController
from app.models.user import User
from app.schemas.user import UserCreate
from app.utils.auth import verify_password, create_access_token
from app.middleware.auth import AuthMiddleware

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db():
    """Fixture for database session mock."""
    db = MagicMock(spec=Session)
    return db


@pytest.fixture
def user_data():
    """Fixture for user data."""
    return UserCreate(
        username="testuser",
        email="test@example.com",
        password="password123",
        password_confirm="password123"
    )


async def test_register_user_success(mock_db_session, user_registration_data):
    """Test successful user registration."""
    # Configure mock
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
    
    # Call register_user
    user = await AuthController.register_user(
        db=mock_db_session,
        username=user_registration_data.username,
        email=user_registration_data.email,
        password=user_registration_data.password
    )
    
    # Verify user properties
    assert user.username == user_registration_data.username
    assert user.email == user_registration_data.email
    assert verify_password(user_registration_data.password, user.password_hash)
    assert user.role == "user"
    assert user.is_active is True
    
    # Verify database operations
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once_with(user)


async def test_register_user_email_exists(mock_db_session, user_registration_data):
    """Test registration with existing email."""
    # Configure mock to simulate existing email
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = MagicMock()
    
    # Check if ValueError is raised
    with pytest.raises(ValueError, match="Email already registered"):
        await AuthController.register_user(
            db=mock_db_session,
            username=user_registration_data.username,
            email=user_registration_data.email,
            password=user_registration_data.password
        )


async def test_register_user_username_exists(mock_db_session, user_registration_data):
    """Test registration with existing username."""
    # Configure mocks
    mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [None, MagicMock()]
    
    # Check if ValueError is raised
    with pytest.raises(ValueError, match="Username already taken"):
        await AuthController.register_user(
            db=mock_db_session,
            username=user_registration_data.username,
            email=user_registration_data.email,
            password=user_registration_data.password
        )


async def test_authenticate_user_success(mock_db_session, user_registration_data):
    """Test successful user authentication."""
    # Create mock user
    mock_user = MagicMock(spec=User)
    mock_user.password_hash = AuthController.get_password_hash(user_registration_data.password)
    mock_user.is_active = True
    
    # Configure mock
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user
    
    # Call authenticate_user
    user = await AuthController.authenticate_user(
        db=mock_db_session,
        email=user_registration_data.email,
        password=user_registration_data.password
    )
    
    # Verify result
    assert user is not None
    assert user == mock_user
    assert user.last_login is not None


async def test_authenticate_user_invalid_password(mock_db_session, user_registration_data):
    """Test authentication with invalid password."""
    # Create mock user
    mock_user = MagicMock(spec=User)
    mock_user.password_hash = AuthController.get_password_hash(user_registration_data.password)
    
    # Configure mock
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user
    
    # Call authenticate_user with wrong password
    user = await AuthController.authenticate_user(
        db=mock_db_session,
        email=user_registration_data.email,
        password="wrongpassword"
    )
    
    # Verify result
    assert user is None


async def test_authenticate_user_inactive(mock_db_session, user_registration_data):
    """Test authentication with inactive user."""
    # Create mock user
    mock_user = MagicMock(spec=User)
    mock_user.password_hash = AuthController.get_password_hash(user_registration_data.password)
    mock_user.is_active = False
    
    # Configure mock
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user
    
    # Call authenticate_user
    user = await AuthController.authenticate_user(
        db=mock_db_session,
        email=user_registration_data.email,
        password=user_registration_data.password
    )
    
    # Verify result
    assert user is None


async def test_authenticate_user_not_found(mock_db_session):
    """Test authentication with non-existent user."""
    # Configure mock
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
    
    # Call authenticate_user
    user = await AuthController.authenticate_user(
        db=mock_db_session,
        email="nonexistent@example.com",
        password="password123"
    )
    
    # Verify result
    assert user is None


def test_create_access_token():
    """Test JWT token creation."""
    # Create test data
    user_id = "test-user-id"
    expires_delta = timedelta(minutes=15)
    
    # Create token
    token = create_access_token(
        data={"sub": user_id},
        expires_delta=expires_delta
    )
    
    # Verify token
    assert isinstance(token, str)
    assert len(token.split(".")) == 3  # JWT format: header.payload.signature


async def test_auth_middleware_valid_token(mock_db_session, mock_user):
    """Test auth middleware with valid token."""
    # Create middleware instance
    middleware = AuthMiddleware()
    
    # Create token and mock request
    token = create_access_token(data={"sub": str(mock_user["id"])})
    mock_request = MagicMock()
    mock_request.headers = {"Authorization": f"Bearer {token}"}
    
    # Configure mock
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user
    
    # Call middleware
    user = await middleware(mock_request)
    
    # Verify result
    assert user is not None
    assert user == mock_user


async def test_auth_middleware_invalid_token():
    """Test auth middleware with invalid token."""
    # Create middleware instance
    middleware = AuthMiddleware()
    
    # Create mock request with invalid token
    mock_request = MagicMock()
    mock_request.headers = {"Authorization": "Bearer invalid_token"}
    
    # Check if HTTPException is raised
    with pytest.raises(HTTPException) as exc_info:
        await middleware(mock_request)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Invalid token"


async def test_auth_middleware_missing_token():
    """Test auth middleware with missing token."""
    # Create middleware instance
    middleware = AuthMiddleware()
    
    # Create mock request without token
    mock_request = MagicMock()
    mock_request.headers = {}
    
    # Check if HTTPException is raised
    with pytest.raises(HTTPException) as exc_info:
        await middleware(mock_request)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Missing token"


async def test_auth_middleware_expired_token(mock_db_session, mock_user):
    """Test auth middleware with expired token."""
    # Create middleware instance
    middleware = AuthMiddleware()
    
    # Create expired token
    expired_delta = timedelta(minutes=-1)  # Token expired 1 minute ago
    token = create_access_token(
        data={"sub": str(mock_user["id"])},
        expires_delta=expired_delta
    )
    
    # Create mock request
    mock_request = MagicMock()
    mock_request.headers = {"Authorization": f"Bearer {token}"}
    
    # Check if HTTPException is raised
    with pytest.raises(HTTPException) as exc_info:
        await middleware(mock_request)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Token has expired"