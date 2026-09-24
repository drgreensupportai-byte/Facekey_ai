import os
import sys
import logging
import numpy as np
import cv2
import torch
import torchaudio

# Configure Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("FaceKey-Trainer")

# Import system encryption and feature extraction engines
from database.database import BiometricEncryption
from ml.face_engine import FaceQualityEngine


# ==========================================
# HELPER: DYNAMIC FILE RESOLUTION (MULTI-DIRECTORY & MULTI-EXTENSION)
# ==========================================
def find_user_image(app, username: str) -> str:
    """
    Ishakisha ifoto y'umuser muri folders zose zishoboka n'ubwoko bwose bw'amashusho.
    """
    possible_folders = [
        os.path.join(app.root_path, 'dataset'),
        os.path.join(app.root_path, 'static', 'uploads', 'dataset'),
        os.path.join(app.root_path, 'static', 'uploads', 'profile_pictures'),
        os.path.join(app.root_path, 'static', 'uploads', 'profiles'),
        os.path.join(app.root_path, 'static', 'uploads')
    ]
    
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.svg', '.avif']

    for folder in possible_folders:
        if os.path.exists(folder):
            for ext in extensions:
                image_path = os.path.join(folder, f"{username}{ext}")
                if os.path.exists(image_path):
                    return image_path
                # Shakisha n'izina rihamagarwa mu nyuguti nto (lowercase)
                image_path_lower = os.path.join(folder, f"{username.lower()}{ext}")
                if os.path.exists(image_path_lower):
                    return image_path_lower

    return None


def find_user_audio(app, username: str) -> str:
    """
    Ishakisha ijwi (audio) ry'umuser muri folders zose zishoboka.
    """
    possible_folders = [
        os.path.join(app.root_path, 'dataset'),
        os.path.join(app.root_path, 'static', 'uploads', 'audio'),
        os.path.join(app.root_path, 'static', 'uploads')
    ]
    extensions = ['.wav', '.mp3', '.ogg', '.flac']

    for folder in possible_folders:
        if os.path.exists(folder):
            for ext in extensions:
                audio_path = os.path.join(folder, f"{username}{ext}")
                if os.path.exists(audio_path):
                    return audio_path
    return None


# ==========================================
# 1. FACIAL EMBEDDING EXTRACTION
# ==========================================
def extract_face_embedding(image_path: str) -> np.ndarray:
    """
    Reads image from disk, validates quality via FaceQualityEngine,
    and returns normalized face embedding vector.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image dataset file not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not decode image at {image_path}")

    # Assess quality (e.g., face presence, lighting, blur)
    is_valid, msg = FaceQualityEngine.assess_quality(img_bgr)
    if not is_valid:
        logger.warning(f"Quality assessment warning for '{image_path}': {msg}")

    # Extract feature embedding vector
    embedding = FaceQualityEngine.extract_embedding(img_bgr)
    return embedding


# ==========================================
# 2. VOICE EMBEDDING EXTRACTION
# ==========================================
def extract_voice_embedding(audio_path: str) -> np.ndarray:
    """
    Extracts voice print embedding using MFCCs via torchaudio.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio sample file not found: {audio_path}")

    waveform, sample_rate = torchaudio.load(audio_path)

    # Extract Mel-Frequency Cepstral Coefficients (MFCC)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=40,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}
    )
    mfcc = mfcc_transform(waveform)

    # Take mean & std along time axis to construct fixed-size embedding
    mean_mfcc = torch.mean(mfcc, dim=2).squeeze().numpy()
    std_mfcc = torch.std(mfcc, dim=2).squeeze().numpy()

    voice_vector = np.concatenate([mean_mfcc, std_mfcc])
    voice_vector = voice_vector / (np.linalg.norm(voice_vector) + 1e-10)  # L2 normalize
    return voice_vector


# ==========================================
# 3. DATABASE RETRAIN / UPDATE WORKFLOW
# ==========================================
def train_single_user(user_id: int, app, db, User) -> bool:
    """
    Retrains a single user's face/voice embeddings from their dataset/profile image or audio.
    """
    with app.app_context():
        user = db.session.get(User, user_id) if hasattr(db.session, 'get') else User.query.get(user_id)
        if not user:
            logger.error(f"User with ID {user_id} not found.")
            return False

        # Find image and audio across all directories & formats
        image_path = find_user_image(app, user.username)
        audio_path = find_user_audio(app, user.username)

        if not image_path:
            logger.error(f"❌ Nta foto y'umuser '{user.username}' yabonywe muri folder n'imwe (dataset/ cyangwa static/uploads/).")
            return False

        try:
            logger.info(f"📸 Extracting facial embeddings for '{user.username}' from: {image_path}")
            face_embedding = extract_face_embedding(image_path)
            
            # Encrypt facial biometric template
            user.encrypted_biometric_template = BiometricEncryption.encrypt_embedding(face_embedding)

            # Optional Voice embedding update if audio file exists
            if audio_path:
                logger.info(f"🎙️ Extracting voice embeddings for '{user.username}' from: {audio_path}")
                voice_embedding = extract_voice_embedding(audio_path)
                user.encrypted_voice_template = BiometricEncryption.encrypt_embedding(voice_embedding)

            db.session.commit()
            logger.info(f"✅ Successfully retrained and updated biometrics for '{user.username}'.")
            return True

        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Failed to retrain user '{user.username}': {str(e)}")
            return False


def retrain_all_users(app, db, User):
    """
    Iterates through all registered users and regenerates biometrics.
    """
    with app.app_context():
        users = User.query.all()
        logger.info(f"🚀 Starting bulk retraining task for {len(users)} registered users...")

        if not users:
            logger.warning("⚠️ Nta ma-users agaragara muri Database!")
            return

        success_count = 0
        for user in users:
            res = train_single_user(user.id, app, db, User)
            if res:
                success_count += 1

        logger.info(f"🎉 Bulk retraining complete. {success_count}/{len(users)} users successfully updated.")


# ==========================================
# 4. CLI EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    logger.info("Initializing FaceKey AI Standalone Biometric Trainer...")

    # Dynamic Flask app contextual import
    try:
        from app import app
        from database.models import db, User
    except ImportError:
        logger.error("Could not import Flask 'app', 'db', or 'User' model. Run this script from project root.")
        sys.exit(1)

    if len(sys.argv) > 1:
        # CLI Mode: Retrain specific user ID (e.g., `python ml/train.py 5`)
        try:
            target_id = int(sys.argv[1])
            success = train_single_user(target_id, app, db, User)
            sys.exit(0 if success else 1)
        except ValueError:
            logger.error("User ID must be an integer.")
            sys.exit(1)
    else:
        # CLI Mode: Retrain all registered users
        retrain_all_users(app, db, User)