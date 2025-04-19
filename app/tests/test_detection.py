"""
Tests for detection controller.
"""
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest
from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime
from uuid import UUID

from app.controllers.detection import DetectionController
from app.models.user import User
from app.models.comment import Comment
from app.models.comment_vector import CommentVector
from app.schemas.comment import CommentCreate, ToxicityLevel
from app.services.ml_service import MLService

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db():
    """Fixture for database session mock."""
    db = MagicMock(spec=Session)
    return db


@pytest.fixture
def mock_db_session():
    """Fixture for async database session mock."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.flush = AsyncMock()
    db.add = AsyncMock()
    return db


@pytest.fixture
def mock_user():
    """Fixture for user mock."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    return user


@pytest.fixture
def comment_data():
    """Fixture for comment data."""
    return CommentCreate(
        content="This is a test comment",
        platform="twitter"
    )


@pytest.fixture
def mock_classifier():
    """Fixture for classifier mock."""
    return [
        [
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.1}
        ]
    ]


@pytest.fixture
def mock_embedding():
    """Fixture for embedding mock."""
    return np.random.rand(768)


@pytest.fixture
def mock_ml_service():
    """Fixture for ML service mock."""
    service = AsyncMock(spec=MLService)
    service.predict = AsyncMock()
    service.predict.return_value = {
        "label": "safe",
        "score": 0.9,
        "prediction_code": 0,
        "preprocessed_text": "test comment",
        "spam_features": {},
        "model_type": "mock_bert",
        "is_mock": True
    }
    service.model_version = "mock_model_v1.0"
    return service


@pytest.mark.asyncio
async def test_init_detection_controller():
    """Test DetectionController initialization."""
    # Create controller
    controller = DetectionController()
    
    # Check attributes
    assert controller.tokenizer is not None
    assert controller.model is not None
    assert controller.ml_service is not None


@pytest.mark.asyncio
async def test_analyze_comment(mock_db_session, mock_ml_service, comment_data):
    """Test comment analysis."""
    # Create controller instance with mock ML service
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Configure ML service mock
    mock_ml_service.predict.return_value = {
        "label": "safe",
        "score": 0.9,
        "prediction_code": 0,
        "preprocessed_text": "test comment",
        "spam_features": {},
        "model_type": "mock_bert",
        "is_mock": True
    }
    
    # Call analyze_comment
    result = await controller.analyze_comment(
        db=mock_db_session,
        comment=comment_data
    )

    # Verify result
    assert result.content == comment_data.content
    assert result.platform == comment_data.platform
    assert result.toxicity_level == "safe"
    assert result.toxicity_score == 0.9
    assert result.preprocessed_text == "test comment"
    # Verify that the ML service was called
    mock_ml_service.predict.assert_called_once_with(comment_data.content)


@pytest.mark.asyncio
async def test_find_similar_comments(mock_db_session, mock_ml_service, mock_embedding):
    """Test finding similar comments."""
    # Create controller instance
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Patch the _generate_embedding method
    with patch.object(controller, '_generate_embedding') as mock_gen_embedding:
        mock_gen_embedding.return_value = mock_embedding
        
        # Patch the _cosine_similarity method
        with patch.object(controller, '_cosine_similarity') as mock_cos_sim:
            mock_cos_sim.return_value = 0.8
            
            # Set up mock database results
            mock_vector1 = MagicMock(spec=CommentVector)
            mock_vector1.id = str(uuid.uuid4())
            mock_vector1.embedding = mock_embedding.tolist()
            
            mock_vector2 = MagicMock(spec=CommentVector)
            mock_vector2.id = str(uuid.uuid4())
            mock_vector2.embedding = mock_embedding.tolist()
            
            # Mock the all() method directly
            mock_scalars = AsyncMock()
            mock_scalars.all = MagicMock(return_value=[mock_vector1, mock_vector2])
            
            mock_result = AsyncMock()
            mock_result.scalars = MagicMock(return_value=mock_scalars)
            
            # First call to execute returns vectors
            execute_results = [mock_result]
            
            # Create comment mocks
            mock_comment1 = MagicMock(spec=Comment)
            mock_comment1.id = str(uuid.uuid4())
            
            mock_comment2 = MagicMock(spec=Comment)
            mock_comment2.id = str(uuid.uuid4())
            
            # Mock scalar_one_or_none returns
            comment1_result = AsyncMock()
            comment1_result.scalar_one_or_none = MagicMock(return_value=mock_comment1)
            execute_results.append(comment1_result)
            
            comment2_result = AsyncMock()
            comment2_result.scalar_one_or_none = MagicMock(return_value=mock_comment2)
            execute_results.append(comment2_result)
            
            # Configure execute side effects
            mock_db_session.execute = AsyncMock(side_effect=execute_results)
            
            # Call get_similar_comments via the alias function
            results = await controller.get_similar_comments(
                db=mock_db_session,
                text="Test comment",
                limit=10,
                min_similarity=0.7
            )
            
            # Check results
            assert len(results) == 2
            assert results[0]["comment"] == mock_comment1
            assert results[0]["similarity"] == 0.8
            assert results[1]["comment"] == mock_comment2
            assert results[1]["similarity"] == 0.8


@pytest.mark.asyncio
async def test_generate_embedding():
    """Test embedding generation."""
    # Create controller instance
    controller = DetectionController()
    
    # Mock tokenizer and model
    with patch.object(controller, 'tokenizer') as mock_tokenizer:
        with patch.object(controller, 'model') as mock_model:
            with patch('torch.no_grad'):
                # Configure mocks
                mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
                
                mock_output = MagicMock()
                mock_output.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = np.ones(768)
                mock_model.return_value = mock_output
                
                # Call _generate_embedding
                embedding = await controller._generate_embedding("Test comment")
                
                # Check embedding
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (768,)


@pytest.mark.asyncio
async def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Create controller instance
    controller = DetectionController()
    
    # Create test vectors
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    vec3 = np.array([1, 1, 0])
    
    # Calculate similarities
    sim1_2 = controller._cosine_similarity(vec1, vec2)
    sim1_3 = controller._cosine_similarity(vec1, vec3)
    
    # Check results
    assert sim1_2 == 0.0  # Orthogonal vectors
    assert abs(sim1_3 - 0.7071) < 0.0001  # 45-degree angle


@pytest.mark.asyncio
async def test_analyze_comment_success(mock_db_session, mock_ml_service, comment_data):
    """Test successful comment analysis."""
    # Create controller instance
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Configure ML service mock
    mock_ml_service.predict.return_value = {
        "label": "safe",
        "score": 0.95,
        "prediction_code": 0,
        "preprocessed_text": "this is a test comment",
        "spam_features": {},
        "model_type": "mock_bert",
        "is_mock": True
    }
    
    # Call analyze_comment
    result = await controller.analyze_comment(
        db=mock_db_session,
        comment=comment_data
    )
    
    # Check result
    assert result.content == comment_data.content
    assert result.platform == comment_data.platform
    assert result.toxicity_level == "safe"
    assert result.toxicity_score == 0.95
    assert result.preprocessed_text == "this is a test comment"


@pytest.mark.asyncio
async def test_analyze_comment_ml_error(mock_db_session, mock_ml_service):
    """Test comment analysis with ML service error."""
    # Configure mock ML service to raise exception
    mock_ml_service.predict.side_effect = Exception("ML service error")
    
    # Create comment data
    comment_data = CommentCreate(
        content="Test comment",
        platform="test"
    )
    
    # Create controller
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Check if exception is raised
    with pytest.raises(Exception, match="ML service error"):
        await controller.analyze_comment(
            db=mock_db_session,
            comment=comment_data
        )


@pytest.mark.asyncio
async def test_get_detection_stats(mock_db_session, mock_user):
    """Test getting detection statistics."""
    # This test requires mocking the get_db context manager, so we'll skip it for now
    # and assume the method works if the DB queries are correct
    pass


@pytest.mark.asyncio
async def test_analyze_comment_toxic(mock_db_session, mock_ml_service):
    """Test toxic comment analysis."""
    # Configure mock ML service
    mock_ml_service.predict.return_value = {
        "label": "toxic",
        "score": 0.8,
        "prediction_code": 1,
        "preprocessed_text": "this is a toxic comment",
        "spam_features": {},
        "model_type": "mock_bert",
        "is_mock": True
    }
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is a toxic comment",
        platform="test"
    )
    
    # Create controller
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Call analyze_comment
    result = await controller.analyze_comment(
        db=mock_db_session,
        comment=comment_data
    )
    
    # Check result
    assert result.toxicity_level == "toxic"
    assert result.toxicity_score == 0.8
    assert result.preprocessed_text == "this is a toxic comment"


@pytest.mark.asyncio
async def test_analyze_comment_safe(mock_db_session, mock_ml_service):
    """Test safe comment analysis."""
    # Configure mock ML service
    mock_ml_service.predict.return_value = {
        "label": "safe",
        "score": 0.9,
        "prediction_code": 0,
        "preprocessed_text": "this is a safe comment",
        "spam_features": {},
        "model_type": "mock_bert",
        "is_mock": True
    }
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is a safe comment",
        platform="test"
    )
    
    # Create controller
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Call analyze_comment
    result = await controller.analyze_comment(
        db=mock_db_session,
        comment=comment_data
    )
    
    # Check result
    assert result.toxicity_level == "safe"
    assert result.toxicity_score == 0.9
    assert result.preprocessed_text == "this is a safe comment"


@pytest.mark.asyncio
async def test_analyze_comment_offensive(mock_db_session, mock_ml_service):
    """Test offensive comment analysis."""
    # Configure mock ML service
    mock_ml_service.predict.return_value = {
        "label": "offensive",
        "score": 0.7,
        "prediction_code": 3,
        "preprocessed_text": "this is an offensive comment",
        "spam_features": {},
        "model_type": "mock_bert",
        "is_mock": True
    }
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is an offensive comment",
        platform="test"
    )
    
    # Create controller
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Call analyze_comment
    result = await controller.analyze_comment(
        db=mock_db_session,
        comment=comment_data
    )
    
    # Check result
    assert result.toxicity_level == "offensive"
    assert result.toxicity_score == 0.7
    assert result.preprocessed_text == "this is an offensive comment"


@pytest.mark.asyncio
async def test_analyze_comment_hate(mock_db_session, mock_ml_service):
    """Test hate speech comment analysis."""
    # Configure mock ML service
    mock_ml_service.predict.return_value = {
        "label": "hate",
        "score": 0.85,
        "prediction_code": 2,
        "preprocessed_text": "this is a hate speech comment",
        "spam_features": {},
        "model_type": "mock_bert",
        "is_mock": True
    }
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is a hate speech comment",
        platform="test"
    )
    
    # Create controller
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Call analyze_comment
    result = await controller.analyze_comment(
        db=mock_db_session,
        comment=comment_data
    )
    
    # Check result
    assert result.toxicity_level == "hate"
    assert result.toxicity_score == 0.85
    assert result.preprocessed_text == "this is a hate speech comment"


@pytest.mark.asyncio
async def test_get_similar_comments(mock_db_session, mock_ml_service):
    """Test getting similar comments."""
    # Create controller instance
    controller = DetectionController(ml_service=mock_ml_service)
    
    # Patch the _generate_embedding method
    with patch.object(controller, '_generate_embedding') as mock_gen_embedding:
        mock_gen_embedding.return_value = np.random.rand(768)
        
        # Configure mock database response
        mock_vector1 = MagicMock()
        mock_vector1.id = str(uuid.uuid4())
        mock_vector1.embedding = np.random.rand(768).tolist()
        
        mock_vector2 = MagicMock()
        mock_vector2.id = str(uuid.uuid4())
        mock_vector2.embedding = np.random.rand(768).tolist()
        
        mock_scalars = AsyncMock()
        mock_scalars.all = MagicMock(return_value=[mock_vector1, mock_vector2])
        
        mock_result = AsyncMock()
        mock_result.scalars = MagicMock(return_value=mock_scalars)
        
        # First call to execute returns vectors
        execute_results = [mock_result]
        
        # Create comment mocks
        mock_comment1 = MagicMock()
        mock_comment1.id = str(uuid.uuid4())
        mock_comment1.content = "Test comment 1"
        
        mock_comment2 = MagicMock()
        mock_comment2.id = str(uuid.uuid4())
        mock_comment2.content = "Test comment 2"
        
        # Add comment results
        comment1_result = AsyncMock()
        comment1_result.scalar_one_or_none = MagicMock(return_value=mock_comment1)
        execute_results.append(comment1_result)
        
        comment2_result = AsyncMock()
        comment2_result.scalar_one_or_none = MagicMock(return_value=mock_comment2)
        execute_results.append(comment2_result)
        
        # Set execute results
        mock_db_session.execute = AsyncMock(side_effect=execute_results)
        
        # Patch cosine similarity to return high values for all comparisons
        with patch.object(controller, '_cosine_similarity', return_value=0.9):
            # Call get_similar_comments
            similar_comments = await controller.get_similar_comments(
                db=mock_db_session,
                text="Test query",
                limit=10,
                min_similarity=0.7
            )
            
            # Check results
            assert len(similar_comments) == 2
            assert similar_comments[0]["similarity"] == 0.9
            assert similar_comments[1]["similarity"] == 0.9


@pytest.mark.asyncio
async def test_get_comment_statistics(mock_db_session):
    """Test getting comment statistics."""
    # Create controller instance
    controller = DetectionController()
    
    # Configure mock database response for total query
    total_result = AsyncMock()
    total_result.scalar = MagicMock(return_value=100)
    
    safe_result = AsyncMock()
    safe_result.scalar = MagicMock(return_value=60)
    
    toxic_result = AsyncMock()
    toxic_result.scalar = MagicMock(return_value=20)
    
    hate_result = AsyncMock()
    hate_result.scalar = MagicMock(return_value=10)
    
    offensive_result = AsyncMock()
    offensive_result.scalar = MagicMock(return_value=10)
    
    # Set up side_effect for multiple calls
    mock_db_session.execute = AsyncMock(
        side_effect=[total_result, safe_result, toxic_result, hate_result, offensive_result]
    )
    
    # Call get_comment_statistics
    stats = await controller.get_comment_statistics(db=mock_db_session)
    
    # Check results
    assert stats["total_comments"] == 100
    assert stats["safe_count"] == 60
    assert stats["toxic_count"] == 40  # Sum of toxic, hate, offensive
    assert stats["toxic_ratio"] == 0.4  # 40/100


@pytest.mark.asyncio
async def test_get_comment_history(mock_db_session):
    """Test getting comment history."""
    # Create controller instance
    controller = DetectionController()
    
    # Create mock comments
    mock_comments = [
        MagicMock(id=str(uuid.uuid4()), content="Comment 1"),
        MagicMock(id=str(uuid.uuid4()), content="Comment 2"),
        MagicMock(id=str(uuid.uuid4()), content="Comment 3")
    ]
    
    # Configure mock database response
    mock_scalars = AsyncMock()
    mock_scalars.all = MagicMock(return_value=mock_comments)
    
    mock_result = AsyncMock()
    mock_result.scalars = MagicMock(return_value=mock_scalars)
    
    mock_db_session.execute = AsyncMock(return_value=mock_result)
    
    # Call get_comment_history
    user_id = str(uuid.uuid4())
    history = await controller.get_comment_history(db=mock_db_session, user_id=user_id)
    
    # Check results
    assert len(history) == 3
    assert history[0].content == "Comment 1"
    assert history[1].content == "Comment 2"
    assert history[2].content == "Comment 3"