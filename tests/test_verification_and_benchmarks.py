import time
import pytest
import numpy as np
from app import create_app
from database.models import db, User, UserBiometrics, AuthenticationLog
from models.face_embedding import FaceEmbeddingExtractor
from models.template_builder import BiometricTemplateBuilder
from vector_search import FAISSVectorEngine
from services.session_manager import BiometricSessionManager
from services.authentication_engine import AuthenticationEngine

@pytest.fixture
def app_client():
    """Configures a test app instance with an in-memory SQLite database."""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SESSION_TTL_SECONDS'] = 1.0  # Accelerated TTL for test suite

    with app.app_context():
        db.create_all()
        # Seed test user
        user = User(username="test_user", is_active=True)
        db.session.add(user)
        db.session.commit()
        yield app.test_client(), user.id
        db.drop_all()

class TestVerificationAndBenchmarks:

    def test_01_bad_frame_rejection(self, app_client):
        """Verifies that empty or garbage base64 image strings are gracefully rejected."""
        client, user_id = app_client

        # 1. Test empty payload
        res = AuthenticationEngine.process_verification_request(image_b64="")
        assert res['success'] is False
        assert res['reason'] == "INVALID_IMAGE"

        # 2. Test invalid non-image payload
        res = AuthenticationEngine.process_verification_request(image_b64="invalid_base64_data")
        assert res['success'] is False
        assert res['reason'] == "INVALID_IMAGE"

    def test_02_faiss_search_speed_benchmark(self, app_client):
        """Benchmarks FAISS IndexFlatIP lookup speed against a generated vector set."""
        client, user_id = app_client

        # Populate FAISS index with 1,000 synthetic 512-D normalized vectors
        num_vectors = 1000
        dim = 512
        synthetic_vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        
        # Normalize vectors
        norms = np.linalg.norm(synthetic_vectors, axis=1, keepdims=True)
        synthetic_vectors = synthetic_vectors / norms

        # Initialize FAISS engine index manually
        engine = FAISSVectorEngine
        engine._index = None
        engine._user_map = [i for i in range(num_vectors)]
        
        index = engine.get_index()
        index.add(synthetic_vectors)

        # Benchmark query latency over 100 random lookups
        query_vector = synthetic_vectors[42]
        start_time = time.time()
        iterations = 100

        for _ in range(iterations):
            results = engine.search(query_vector, top_k=1)

        total_time = time.time() - start_time
        avg_latency_ms = (total_time / iterations) * 1000.0

        print(f"\n[BENCHMARK] FAISS Search Latency over {num_vectors} vectors: {avg_latency_ms:.4f} ms/query")
        assert len(results) > 0
        assert results[0]['user_id'] == 42
        assert avg_latency_ms < 5.0  # Must complete under 5ms per query

    def test_03_expired_session_rejection(self, app_client):
        """Validates that session tokens expire strictly after the TTL threshold."""
        client, user_id = app_client

        # Create session with 1.0 second TTL
        token = BiometricSessionManager.create_session(user_id=user_id, username="test_user", ttl_seconds=1.0)
        
        # Immediate validation -> should pass
        valid, data, reason = BiometricSessionManager.validate_session(token)
        assert valid is True
        assert reason == "VALID"

        # Wait for TTL expiration (1.2s)
        time.sleep(1.2)

        # Post-expiration validation -> should fail
        valid, data, reason = BiometricSessionManager.validate_session(token)
        assert valid is False
        assert reason == "SESSION_EXPIRED"

    def test_04_template_builder_outlier_filtering(self):
        """Validates template synthesis and outlier vector rejection."""
        dim = 512
        base_vector = np.random.randn(dim).astype(np.float32)
        base_vector /= np.linalg.norm(base_vector)

        # Create 3 highly similar vectors and 1 noisy outlier
        vec1 = base_vector + np.random.normal(0, 0.01, dim).astype(np.float32)
        vec2 = base_vector + np.random.normal(0, 0.01, dim).astype(np.float32)
        vec3 = base_vector + np.random.normal(0, 0.01, dim).astype(np.float32)
        outlier = -base_vector  # Opposite direction outlier

        embeddings = [vec1, vec2, vec3, outlier]
        master_template = BiometricTemplateBuilder.synthesize_template(embeddings)

        assert master_template.shape == (512,)
        # Ensure master template maintains high similarity to original base vector
        sim = BiometricTemplateBuilder.compute_similarity(master_template, base_vector)
        assert sim > 0.90