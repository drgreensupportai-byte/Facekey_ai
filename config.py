import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # -------------------------------------------------------------------------
    # Core Base Paths
    # -------------------------------------------------------------------------
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # -------------------------------------------------------------------------
    # Core Flask & Security Settings
    # -------------------------------------------------------------------------
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'default-dev-key')
    
    # Path explicitly points to FaceKey_AI/database/facekey.db
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'database', 'facekey.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    BIOMETRIC_ENCRYPTION_KEY = os.getenv('BIOMETRIC_ENCRYPTION_KEY')
    BIOMETRIC_RETENTION_DAYS = int(os.getenv('BIOMETRIC_RETENTION_DAYS', 365))
    ALLOW_RAW_IMAGE_STORAGE = os.getenv('ALLOW_RAW_IMAGE_STORAGE', 'False').lower() == 'true'
    CONSENT_VERSION = os.getenv('CONSENT_VERSION', '1.0')

    # Configurable Session Timeout (in seconds) for automatic logout/scanning behavior
    AUTO_LOGOUT_SECONDS = int(os.getenv('AUTO_LOGOUT_SECONDS', 5))

    MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', 5))
    LOGIN_LOCKOUT_MINUTES = int(os.getenv('LOGIN_LOCKOUT_MINUTES', 15))

    # -------------------------------------------------------------------------
    # File & Directory Paths
    # -------------------------------------------------------------------------
    DATASET_DIR = os.getenv('DATASET_DIR', os.path.join(BASE_DIR, 'dataset'))
    INDEX_DIR = os.getenv('INDEX_DIR', os.path.join(BASE_DIR, 'indexes'))
    FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'facekey_faiss.index')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # -------------------------------------------------------------------------
    # Biometric Sample & Multi-Angle Registration Constraints
    # -------------------------------------------------------------------------
    MIN_SAMPLES = int(os.getenv('MIN_SAMPLES', 10))
    MAX_SAMPLES = int(os.getenv('MAX_SAMPLES', 20))

    # Required angle coverage during enrollment (Yaw, Pitch limits in degrees)
    POSE_ANGLES = {
        'FRONT': {'yaw': (-8, 8), 'pitch': (-8, 8)},
        'LEFT': {'yaw': (-30, -15), 'pitch': (-10, 10)},
        'RIGHT': {'yaw': (15, 30), 'pitch': (-10, 10)},
        'UP': {'yaw': (-10, 10), 'pitch': (12, 25)},
        'DOWN': {'yaw': (-10, 10), 'pitch': (-25, -12)},
        'LEFT_UP': {'yaw': (-30, -12), 'pitch': (10, 22)},
        'LEFT_DOWN': {'yaw': (-30, -12), 'pitch': (-22, -10)},
        'RIGHT_UP': {'yaw': (12, 30), 'pitch': (10, 22)},
        'RIGHT_DOWN': {'yaw': (12, 30), 'pitch': (-22, -10)}
    }

    # -------------------------------------------------------------------------
    # Face Quality Engine Thresholds (Optimized for faster capture)
    # -------------------------------------------------------------------------
    MIN_FACE_WIDTH_PX = int(os.getenv('MIN_FACE_WIDTH_PX', 80))        # Lowered from 120px to prevent rejections on webcam stream
    MIN_LAPLACIAN_VAR = float(os.getenv('MIN_LAPLACIAN_VAR', 40.0))     # Lowered from 70.0 to significantly speed up frame validation
    MIN_BRIGHTNESS = int(os.getenv('MIN_BRIGHTNESS', 40))               # Lowered from 50
    MAX_BRIGHTNESS = int(os.getenv('MAX_BRIGHTNESS', 220))             # Raised from 210

    # -------------------------------------------------------------------------
    # Verification & Vector Search Parameters
    # -------------------------------------------------------------------------
    FAISS_TOP_K = int(os.getenv('FAISS_TOP_K', 5))
    EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 512))
    
    # Cosine Similarity Matching Threshold (validated via ROC/EER curves)
    VERIFICATION_THRESHOLD = float(os.getenv('VERIFICATION_THRESHOLD', 0.65))

    @classmethod
    def init_app(cls, app):
        """Creates necessary system directories at startup."""
        directories = [
            cls.DATASET_DIR,
            cls.INDEX_DIR,
            cls.MODELS_DIR,
            os.path.join(cls.BASE_DIR, "database"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)