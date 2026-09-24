from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    
    # Biometric Encryption Templates (Binary BLOBs)
    encrypted_biometric_template = db.Column(db.LargeBinary, nullable=False)  # Face
    encrypted_voice_template = db.Column(db.LargeBinary, nullable=True)       # Voice MFA (New)
    
    # Role & status flags
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime, 
        default=lambda: datetime.now(timezone.utc), 
        onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    consents = db.relationship('ConsentRecord', backref='user', cascade="all, delete-orphan", lazy=True)
    auth_logs = db.relationship('AuthenticationLog', backref='user', cascade="all, delete-orphan", lazy=True)


class ConsentRecord(db.Model):
    __tablename__ = 'consent_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    consent_type = db.Column(db.String(50), nullable=False, default='biometric_authentication')
    consent_version = db.Column(db.String(10), nullable=False)
    consented_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    withdrawn_at = db.Column(db.DateTime, nullable=True)


class AuthenticationLog(db.Model):
    __tablename__ = 'authentication_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)  # e.g., 'BYAKUNZE.', 'BYANZE', 'Kugenzura ko ikintu gikora byanze'
    success = db.Column(db.Boolean, nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)  # Similarity score (no biometric data logged)