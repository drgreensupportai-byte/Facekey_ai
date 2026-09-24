from datetime import datetime, timezone
from database.models import db, ConsentRecord

def record_user_consent(user_id: int, consent_version: str) -> ConsentRecord:
    """Creates and persists an explicit biometric consent record in the database."""
    consent = ConsentRecord(
        user_id=user_id,
        consent_type='biometric_authentication',
        consent_version=consent_version,
        consented_at=datetime.now(timezone.utc)
    )
    db.session.add(consent)
    db.session.commit()
    return consent

def withdraw_user_consent(user_id: int) -> bool:
    """Marks existing active biometric consent records as withdrawn."""
    consents = ConsentRecord.query.filter_by(user_id=user_id, withdrawn_at=None).all()
    if not consents:
        return False
    
    now = datetime.now(timezone.utc)
    for c in consents:
        c.withdrawn_at = now
    db.session.commit()
    return True