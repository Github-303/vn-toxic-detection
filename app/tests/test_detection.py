"""
Tests for detection controller.
"""
import uuid
from unittest.mock import patch, MagicMock, ANY

import numpy as np
import pytest
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.controllers.detection import DetectionController
from app.controllers.ml_controller import MLController
from app.models.user import User
from app.models.comment import Comment, CommentVector
from app.schemas.comment import CommentCreate, ToxicityLevel

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db():
    """Fixture for database session mock."""
    db = MagicMock(spec=Session)
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


@patch('transformers.pipeline')
def test_init_detection_controller(mock_pipeline):
    """Test DetectionController initialization."""
    # Create controller
    controller = DetectionController()
    
    # Check classifier
    assert controller.classifier is not None
    mock_pipeline.assert_called_once()


@patch.object(DetectionController, '_generate_embedding')
@patch.object(DetectionController, '__init__', return_value=None)
def test_analyze_comment(mock_init, mock_generate_embedding, mock_db, mock_user, comment_data, mock_classifier, mock_embedding):
    """Test comment analysis."""
    # Create controller instance
    controller = DetectionController()
    controller.classifier = MagicMock(return_value=mock_classifier)
    
    # Configure mocks
    mock_generate_embedding.return_value = mock_embedding
    
    # Call analyze_comment
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
        result = controller.analyze_comment(
            db=mock_db,
            comment_data=comment_data,
            user=mock_user
        )
    
    # Check result
    assert result["content"] == comment_data.content
    assert result["label"] == ToxicityLevel.SAFE.value
    assert result["confidence"] == 0.9
    assert "breakdown" in result
    assert "comment_id" in result
    
    # Verify database operations
    mock_db.add.assert_called()
    mock_db.commit.assert_called_once()


@patch.object(DetectionController, '_cosine_similarity')
@patch.object(DetectionController, '_generate_embedding')
@patch.object(DetectionController, '__init__', return_value=None)
def test_find_similar_comments(mock_init, mock_generate_embedding, mock_cosine_similarity, mock_db, mock_embedding):
    """Test finding similar comments."""
    # Create controller instance
    controller = DetectionController()
    
    # Configure mocks
    mock_generate_embedding.return_value = mock_embedding
    mock_cosine_similarity.return_value = 0.8
    
    # Create mock vectors and comments
    mock_vector1 = MagicMock(spec=CommentVector)
    mock_vector1.embedding = mock_embedding.tolist()
    mock_vector1.comment_id = uuid.uuid4()
    
    mock_vector2 = MagicMock(spec=CommentVector)
    mock_vector2.embedding = mock_embedding.tolist()
    mock_vector2.comment_id = uuid.uuid4()
    
    mock_comment1 = MagicMock(spec=Comment)
    mock_comment1.id = mock_vector1.comment_id
    
    mock_comment2 = MagicMock(spec=Comment)
    mock_comment2.id = mock_vector2.comment_id
    
    # Configure query mocks
    mock_db.query.return_value.all.return_value = [mock_vector1, mock_vector2]
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_comment1, mock_comment2]
    
    # Call find_similar_comments
    results = controller.find_similar_comments(
        db=mock_db,
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


@patch('torch.no_grad')
@patch('transformers.AutoModel.from_pretrained')
@patch('transformers.AutoTokenizer.from_pretrained')
@patch.object(DetectionController, '__init__', return_value=None)
def test_generate_embedding(mock_init, mock_tokenizer_from_pretrained, mock_model_from_pretrained, mock_no_grad, mock_embedding):
    """Test embedding generation."""
    # Create controller instance
    controller = DetectionController()
    
    # Configure mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    
    mock_model = MagicMock()
    mock_model_from_pretrained.return_value = mock_model
    
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = mock_embedding
    mock_model.return_value = mock_outputs
    
    controller.tokenizer = mock_tokenizer
    controller.model = mock_model
    
    # Call _generate_embedding
    embedding = controller._generate_embedding("Test comment")
    
    # Check embedding
    assert embedding is mock_embedding


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Create controller instance
    controller = DetectionController()
    
    # Create vectors
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])  # Same direction, similarity = 1
    c = np.array([0, 1, 0])  # Perpendicular, similarity = 0
    d = np.array([-1, 0, 0])  # Opposite direction, similarity = -1
    
    # Calculate similarities
    sim_same = controller._cosine_similarity(a, b)
    sim_perp = controller._cosine_similarity(a, c)
    sim_opp = controller._cosine_similarity(a, d)
    
    # Check results
    assert sim_same == pytest.approx(1.0)
    assert sim_perp == pytest.approx(0.0)
    assert sim_opp == pytest.approx(-1.0)


def test_analyze_comment_success(mock_db, mock_user, comment_data):
    """Test successful comment analysis."""
    # Create controller instance
    controller = DetectionController()
    
    # Configure ML controller mock
    mock_prediction = {
        "label": "safe",
        "prediction_code": 0,
        "preprocessed_text": "test comment"
    }
    controller.ml_controller = MagicMock()
    controller.ml_controller.analyze_text.return_value = mock_prediction
    
    # Call analyze_comment
    result = controller.analyze_comment(
        text=comment_data.content,
        user=mock_user,
        platform=comment_data.platform
    )
    
    # Check result
    assert result["content"] == comment_data.content
    assert result["label"] == mock_prediction["label"]
    assert result["prediction_code"] == mock_prediction["prediction_code"]
    assert result["preprocessed_text"] == mock_prediction["preprocessed_text"]
    assert "comment_id" in result


def test_analyze_comment_ml_error(mock_db, mock_user, comment_data):
    """Test comment analysis with ML error."""
    # Create controller instance
    controller = DetectionController()
    
    # Configure ML controller mock to raise error
    controller.ml_controller = MagicMock()
    controller.ml_controller.analyze_text.side_effect = Exception("ML error")
    
    # Check if error is raised
    with pytest.raises(Exception) as exc_info:
        controller.analyze_comment(
            text=comment_data.content,
            user=mock_user,
            platform=comment_data.platform
        )
    assert str(exc_info.value) == "ML error"


def test_get_detection_stats(mock_db, mock_user):
    """Test getting detection statistics."""
    # Create controller instance
    controller = DetectionController()
    
    # Create mock comments
    mock_comments = [
        MagicMock(label="safe"),
        MagicMock(label="toxic"),
        MagicMock(label="hate"),
        MagicMock(label="offensive")
    ]
    
    # Configure mock
    mock_db.execute.return_value.scalars.return_value.all.return_value = mock_comments
    
    # Call get_detection_stats
    stats = controller.get_detection_stats(user=mock_user)
    
    # Check result
    assert stats["total_comments"] == 4
    assert stats["stats"]["safe"] == 1
    assert stats["stats"]["toxic"] == 1
    assert stats["stats"]["hate"] == 1
    assert stats["stats"]["offensive"] == 1


def test_ml_controller_initialization():
    """Test ML controller initialization."""
    # Create controller instance
    controller = MLController()
    
    # Check if service is initialized
    assert controller._service is not None
    assert controller._lock is not None


def test_ml_controller_predict(mock_db, comment_data):
    """Test ML controller prediction."""
    # Create controller instance
    controller = MLController()
    
    # Configure service mock
    mock_prediction = {
        "text": comment_data.content,
        "label": "safe",
        "preprocessed_text": "test comment",
        "prediction": 0
    }
    controller._service = MagicMock()
    controller._service.predict.return_value = mock_prediction
    
    # Call predict
    result = controller.predict(comment_data.content)
    
    # Check result
    assert result == mock_prediction


def test_ml_controller_batch_predict(mock_db):
    """Test ML controller batch prediction."""
    # Create controller instance
    controller = MLController()
    
    # Configure service mock
    mock_predictions = [
        {"text": "comment 1", "label": "safe"},
        {"text": "comment 2", "label": "toxic"}
    ]
    controller._service = MagicMock()
    controller._service.batch_predict.return_value = mock_predictions
    
    # Call batch_predict
    texts = ["comment 1", "comment 2"]
    results = controller.batch_predict(texts)
    
    # Check result
    assert len(results) == 2
    assert results[0]["label"] == "safe"
    assert results[1]["label"] == "toxic"


async def test_analyze_comment_toxic(mock_db_session, mock_ml_service, mock_comment):
    """Test toxic comment analysis."""
    # Configure mock ML service
    mock_prediction = {
        "toxicity": 0.95,
        "severe_toxicity": 0.85,
        "obscene": 0.75,
        "threat": 0.65,
        "insult": 0.55,
        "identity_hate": 0.45
    }
    mock_ml_service.predict.return_value = mock_prediction
    mock_ml_service.get_embedding.return_value = np.random.rand(768)
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is a toxic comment",
        platform="test"
    )
    
    # Call analyze_comment
    result = await DetectionController.analyze_comment(
        db=mock_db_session,
        comment_data=comment_data,
        ml_service=mock_ml_service
    )
    
    # Verify result
    assert result.toxicity_level == ToxicityLevel.TOXIC
    assert result.toxicity_score == mock_prediction["toxicity"]
    assert result.severe_toxicity_score == mock_prediction["severe_toxicity"]
    assert result.obscene_score == mock_prediction["obscene"]
    assert result.threat_score == mock_prediction["threat"]
    assert result.insult_score == mock_prediction["insult"]
    assert result.identity_hate_score == mock_prediction["identity_hate"]
    
    # Verify database operations
    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called()
    mock_db_session.refresh.assert_called()


async def test_analyze_comment_safe(mock_db_session, mock_ml_service):
    """Test safe comment analysis."""
    # Configure mock ML service
    mock_prediction = {
        "toxicity": 0.1,
        "severe_toxicity": 0.05,
        "obscene": 0.03,
        "threat": 0.02,
        "insult": 0.01,
        "identity_hate": 0.01
    }
    mock_ml_service.predict.return_value = mock_prediction
    mock_ml_service.get_embedding.return_value = np.random.rand(768)
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is a safe comment",
        platform="test"
    )
    
    # Call analyze_comment
    result = await DetectionController.analyze_comment(
        db=mock_db_session,
        comment_data=comment_data,
        ml_service=mock_ml_service
    )
    
    # Verify result
    assert result.toxicity_level == ToxicityLevel.SAFE
    assert result.toxicity_score == mock_prediction["toxicity"]
    assert result.severe_toxicity_score == mock_prediction["severe_toxicity"]
    
    # Verify database operations
    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called()
    mock_db_session.refresh.assert_called()


async def test_analyze_comment_offensive(mock_db_session, mock_ml_service):
    """Test offensive comment analysis."""
    # Configure mock ML service
    mock_prediction = {
        "toxicity": 0.7,
        "severe_toxicity": 0.3,
        "obscene": 0.6,
        "threat": 0.2,
        "insult": 0.65,
        "identity_hate": 0.15
    }
    mock_ml_service.predict.return_value = mock_prediction
    mock_ml_service.get_embedding.return_value = np.random.rand(768)
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is an offensive comment",
        platform="test"
    )
    
    # Call analyze_comment
    result = await DetectionController.analyze_comment(
        db=mock_db_session,
        comment_data=comment_data,
        ml_service=mock_ml_service
    )
    
    # Verify result
    assert result.toxicity_level == ToxicityLevel.OFFENSIVE
    assert 0.6 <= result.toxicity_score < 0.8
    assert result.severe_toxicity_score == mock_prediction["severe_toxicity"]
    
    # Verify database operations
    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called()
    mock_db_session.refresh.assert_called()


async def test_analyze_comment_hate(mock_db_session, mock_ml_service):
    """Test hate speech comment analysis."""
    # Configure mock ML service
    mock_prediction = {
        "toxicity": 0.9,
        "severe_toxicity": 0.8,
        "obscene": 0.7,
        "threat": 0.6,
        "insult": 0.85,
        "identity_hate": 0.9
    }
    mock_ml_service.predict.return_value = mock_prediction
    mock_ml_service.get_embedding.return_value = np.random.rand(768)
    
    # Create comment data
    comment_data = CommentCreate(
        content="This is a hate speech comment",
        platform="test"
    )
    
    # Call analyze_comment
    result = await DetectionController.analyze_comment(
        db=mock_db_session,
        comment_data=comment_data,
        ml_service=mock_ml_service
    )
    
    # Verify result
    assert result.toxicity_level == ToxicityLevel.HATE
    assert result.toxicity_score >= 0.8
    assert result.identity_hate_score >= 0.8
    
    # Verify database operations
    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called()
    mock_db_session.refresh.assert_called()


async def test_analyze_comment_ml_error(mock_db_session, mock_ml_service):
    """Test comment analysis with ML service error."""
    # Configure mock ML service to raise exception
    mock_ml_service.predict.side_effect = Exception("ML service error")
    
    # Create comment data
    comment_data = CommentCreate(
        content="Test comment",
        platform="test"
    )
    
    # Check if exception is raised
    with pytest.raises(Exception, match="ML service error"):
        await DetectionController.analyze_comment(
            db=mock_db_session,
            comment_data=comment_data,
            ml_service=mock_ml_service
        )


async def test_get_similar_comments(mock_db_session, mock_comment_vector):
    """Test retrieving similar comments."""
    # Configure mock
    mock_db_session.execute.return_value.fetchall.return_value = [
        (mock_comment_vector, 0.95),
        (mock_comment_vector, 0.85),
        (mock_comment_vector, 0.75)
    ]
    
    # Create test embedding
    test_embedding = np.random.rand(768)
    
    # Call get_similar_comments
    similar_comments = await DetectionController.get_similar_comments(
        db=mock_db_session,
        embedding=test_embedding,
        limit=3
    )
    
    # Verify result
    assert len(similar_comments) == 3
    for comment, similarity in similar_comments:
        assert isinstance(comment, CommentVector)
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1


async def test_get_similar_comments_empty(mock_db_session):
    """Test retrieving similar comments with no results."""
    # Configure mock
    mock_db_session.execute.return_value.fetchall.return_value = []
    
    # Create test embedding
    test_embedding = np.random.rand(768)
    
    # Call get_similar_comments
    similar_comments = await DetectionController.get_similar_comments(
        db=mock_db_session,
        embedding=test_embedding,
        limit=3
    )
    
    # Verify result
    assert len(similar_comments) == 0


async def test_get_comment_statistics(mock_db_session):
    """Test retrieving comment statistics."""
    # Configure mock
    mock_stats = {
        "total_comments": 100,
        "toxic_comments": 20,
        "hate_comments": 10,
        "offensive_comments": 30,
        "safe_comments": 40
    }
    mock_db_session.execute.return_value.first.return_value = mock_stats
    
    # Call get_comment_statistics
    stats = await DetectionController.get_comment_statistics(db=mock_db_session)
    
    # Verify result
    assert stats["total_comments"] == 100
    assert stats["toxic_comments"] == 20
    assert stats["hate_comments"] == 10
    assert stats["offensive_comments"] == 30
    assert stats["safe_comments"] == 40


async def test_get_comment_history(mock_db_session, mock_comment):
    """Test retrieving comment history."""
    # Configure mock
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = [
        mock_comment for _ in range(5)
    ]
    
    # Call get_comment_history
    history = await DetectionController.get_comment_history(
        db=mock_db_session,
        limit=5,
        offset=0
    )
    
    # Verify result
    assert len(history) == 5
    for comment in history:
        assert isinstance(comment, Comment)


async def test_get_comment_history_empty(mock_db_session):
    """Test retrieving empty comment history."""
    # Configure mock
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
    
    # Call get_comment_history
    history = await DetectionController.get_comment_history(
        db=mock_db_session,
        limit=5,
        offset=0
    )
    
    # Verify result
    assert len(history) == 0