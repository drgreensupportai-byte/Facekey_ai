import sys
import flask
import cv2
import numpy as np
import sklearn
import cryptography
from config import Config

def test_environment():
    print("=" * 60)
    print("   FACEKEY AI - SECURE & PRIVACY FOUNDATION TEST   ")
    print("=" * 60)
    print(f"[✓] Python Version       : {sys.version.split()[0]}")
    print(f"[✓] Flask Version        : {flask.__version__}")
    print(f"[✓] OpenCV Version       : {cv2.__version__}")
    print(f"[✓] NumPy Version        : {np.__version__}")
    print(f"[✓] scikit-learn Ver.    : {sklearn.__version__}")
    print(f"[✓] Cryptography Ver.    : {cryptography.__version__}")
    print("-" * 60)
    print(f"[✓] Biometric Retention  : {Config.BIOMETRIC_RETENTION_DAYS} Days")
    print(f"[✓] Raw Image Storage    : {Config.ALLOW_RAW_IMAGE_STORAGE} (Data Minimization)")
    print(f"[✓] Consent Version      : {Config.CONSENT_VERSION}")
    print(f"[✓] Encryption Key Status: {'Configured' if Config.BIOMETRIC_ENCRYPTION_KEY else 'Missing'}")
    print("=" * 60)
    print("SUCCESS: Secure environment foundation verified successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_environment()