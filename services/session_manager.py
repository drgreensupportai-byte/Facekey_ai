import time
import uuid
import logging
from typing import Dict, Any, Optional
from flask import session, request, redirect, url_for, jsonify
from functools import wraps
from config import Config

logger = logging.getLogger("FaceKeyAI.SessionManager")

class BiometricSessionManager:
    """
    Server-side session store providing short-lived (5-second TTL) session state management.
    """
    
    # In-memory session store: { session_token: { 'user_id': int, 'username': str, 'created_at': float, 'expires_at': float } }
    _active_sessions: Dict[str, Dict[str, Any]] = {}
    DEFAULT_TTL_SECONDS = 5.0

    @classmethod
    def create_session(cls, user_id: int, username: str, ttl_seconds: float = None) -> str:
        """
        Creates a new short-lived session token and sets it in Flask session cookies.
        """
        ttl = ttl_seconds if ttl_seconds is not None else getattr(Config, 'SESSION_TTL_SECONDS', cls.DEFAULT_TTL_SECONDS)
        now = time.time()
        token = str(uuid.uuid4())

        session_data = {
            "user_id": user_id,
            "username": username,
            "created_at": now,
            "expires_at": now + ttl
        }

        cls._active_sessions[token] = session_data
        
        # Store token identifier in Flask session
        session['session_token'] = token
        session['user_id'] = user_id
        session['username'] = username
        
        logger.info(f"Session created for user '{username}' (ID: {user_id}). Token: {token[:8]}... TTL: {ttl}s")
        return token

    @classmethod
    def validate_session(cls, token: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Validates whether a session token exists and is within its valid TTL window.
        
        Returns:
            Tuple[bool, dict, str]: (is_valid, session_data, reason)
        """
        if not token:
            token = session.get('session_token')

        if not token or token not in cls._active_sessions:
            return False, None, "NO_SESSION"

        sess_data = cls._active_sessions[token]
        now = time.time()

        if now > sess_data['expires_at']:
            # Expired session cleanup
            cls.revoke_session(token)
            return False, None, "SESSION_EXPIRED"

        return True, sess_data, "VALID"

    @classmethod
    def revoke_session(cls, token: Optional[str] = None) -> None:
        """Destroys an active session token and clears the Flask session."""
        if not token:
            token = session.get('session_token')

        if token and token in cls._active_sessions:
            del cls._active_sessions[token]
            logger.info(f"Session token {token[:8]}... revoked.")

        session.clear()

    @classmethod
    def cleanup_expired_sessions(cls) -> int:
        """Background routine to prune all expired tokens from memory."""
        now = time.time()
        expired_keys = [t for t, data in cls._active_sessions.items() if now > data['expires_at']]
        for t in expired_keys:
            del cls._active_sessions[t]
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired sessions.")
        return len(expired_keys)


def require_short_session(f):
    """
    Decorator for Flask route endpoints enforcing 5-second TTL validity.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        is_valid, sess_data, reason = BiometricSessionManager.validate_session()
        
        if not is_valid:
            if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({
                    "success": False,
                    "reason": reason,
                    "message": "Session expired or invalid. Please re-authenticate."
                }), 401
            return redirect(url_for('login_page', expired=1))
            
        return f(*args, **kwargs)
    return decorated_function