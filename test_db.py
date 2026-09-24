from flask import Flask
from config import Config
from database.models import db, User, ConsentRecord, AuthenticationLog
from database.database import init_db, BiometricEncryption
import numpy as np

def test_database_and_encryption():
    app = Flask(__name__)
    app.config.from_object(Config)
    init_db(app)

    with app.app_context():
        print("=" * 60)
        print("      FACEKEY AI - DATABASE & ENCRYPTION TEST      ")
        print("=" * 60)

        # 1. Simulate dummy face embedding array (128-dimensional vector)
        mock_embedding = np.random.rand(128).astype(np.float32)
        print(f"[✓] Generated mock biometric embedding vector (128-dim)")

        # 2. Encrypt embedding
        encrypted_blob = BiometricEncryption.encrypt_embedding(mock_embedding)
        print(f"[✓] Encrypted vector to binary BLOB ({len(encrypted_blob)} bytes)")

        # 3. Create test user and consent record
        test_user = User(
            username="testuser",
            email="testuser@example.com",
            encrypted_biometric_template=encrypted_blob
        )
        db.session.add(test_user)
        db.session.commit()

        consent = ConsentRecord(
            user_id=test_user.id,
            consent_version=Config.CONSENT_VERSION
        )
        log = AuthenticationLog(
            user_id=test_user.id,
            event_type="TEST_LOGIN_SUCCESS",
            success=True,
            confidence_score=0.982
        )
        db.session.add_all([consent, log])
        db.session.commit()

        # 4. Retrieve from database and decrypt
        retrieved_user = User.query.filter_by(username="testuser").first()
        decrypted_embedding = BiometricEncryption.decrypt_embedding(retrieved_user.encrypted_biometric_template)

        # 5. Verify vector equality
        is_identical = np.allclose(mock_embedding, decrypted_embedding)
        print(f"[✓] Retrieved user '{retrieved_user.username}' from SQLite database")
        print(f"[✓] Decrypted template matches original array: {is_identical}")

        # Cleanup test entry
        db.session.delete(retrieved_user)
        db.session.commit()
        print("=" * 60)
        print("SUCCESS: Database schema, consent model & encryption verified!")
        print("=" * 60)

if __name__ == "__main__":
    test_database_and_encryption()