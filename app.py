import base64
import json
import os
import queue
import subprocess
import base64
import time
from datetime import datetime
from functools import wraps
import numpy as np
from flask import (
    Flask, render_template, request, jsonify, redirect, 
    url_for, session, flash, send_from_directory, current_app , Response
)
from config import Config
from database.database import init_db, BiometricEncryption
from database.models import db, User, AuthenticationLog
from privacy.consent import record_user_consent
from ml.face_engine import FaceQualityEngine
from ml.verification import VerificationEngine

app = Flask(__name__)
app.config.from_object(Config)

# Configurable session timeout duration
app.config['AUTO_LOGOUT_SECONDS'] = getattr(Config, 'AUTO_LOGOUT_SECONDS', 5)
AUTH_LOCK_DEBOUNCE_SEC = 10  # Debounce cooldown for continuous scanning

# Allowed dataset image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'svg', 'avif', 'bmp'}

init_db(app)

# Real-time SSE Broadcast Queue list
sse_subscribers = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_user_dataset_image(username, dataset_dir):
    """Searches for a user's image in the dataset folder across all allowed extensions."""
    if not os.path.exists(dataset_dir):
        return None
    for ext in ALLOWED_EXTENSIONS:
        candidate = os.path.join(dataset_dir, f"{username}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def broadcast_admin_event(event_data):
    """Pushes live authentication events to connected Admin SSE streams."""
    global sse_subscribers
    message = f"data: {json.dumps(event_data)}\n\n"
    for q in sse_subscribers[:]:
        try:
            q.put_nowait(message)
        except Exception:
            sse_subscribers.remove(q)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('admin_secret_login'))
        user = User.query.get(session['user_id'])
        if not user or not getattr(user, 'is_admin', False):
            flash("Ntabwo wemerewe kwinjira muri Admin Panel.", "error")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_config():
    """Injects AUTO_LOGOUT_SECONDS into template context."""
    return dict(AUTO_LOGOUT_SECONDS=app.config['AUTO_LOGOUT_SECONDS'])

@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    """Serves raw image files from the dataset directory."""
    dataset_dir = os.path.join(app.root_path, 'dataset')
    return send_from_directory(dataset_dir, filename)

@app.route('/')
def index():
    return redirect(url_for('login_page'))

@app.route('/privacy')
def privacy_notice():
    return render_template(
        'privacy.html',
        consent_version=app.config.get('CONSENT_VERSION', '1.0'),
        retention_days=app.config.get('BIOMETRIC_RETENTION_DAYS', 30)
    )

@app.route('/consent', methods=['GET'])
def consent_page():
    return render_template('consent.html')

@app.route('/consent', methods=['POST'])
def process_consent():
    consent_given = request.form.get('consent_given')
    if consent_given:
        session['biometric_consent_granted'] = True
        session['consent_version'] = app.config.get('CONSENT_VERSION', '1.0')
        return redirect(url_for('register_page'))
    flash("Consent is required to register.", "error")
    return redirect(url_for('consent_page'))

@app.route('/register')
def register_page():
    if not session.get('biometric_consent_granted'):
        return redirect(url_for('consent_page'))
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def register_api():
    consent_granted = session.get('biometric_consent_granted', True)
    if not consent_granted:
        return jsonify({'success': False, 'message': 'Explicit consent missing.'}), 403

    data = request.get_json() or {}
    username = data.get('username') or data.get('name')
    email = data.get('email') or (f"{username}@facekey.local" if username else None)
    images_list = data.get('images', [])  # Urutonde rw'amafoto ku mpande zose
    audio_data = data.get('audio_data')

    # Backup: Niba haratanzwe image_data imwe gusa
    if not images_list and data.get('image_data'):
        images_list = [data.get('image_data')]

    if not username or not images_list:
        return jsonify({
            'success': False, 
            'message': 'Missing registration details. Username and face captures are required.'
        }), 400

    # Rema Folder y'Uyu Mukoresha (dataset/username/)
    base_dataset_dir = getattr(Config, 'DATASET_DIR', os.path.join(current_app.root_path, 'dataset'))
    user_dataset_dir = os.path.join(base_dataset_dir, username)
    os.makedirs(user_dataset_dir, exist_ok=True)

    saved_count = 0
    first_valid_embedding = None

    try:
        # Bika amashusho yose yafashwe ku mpande zose
        for idx, img_b64 in enumerate(images_list):
            if "," in img_b64:
                header, encoded = img_b64.split(",", 1)
                ext = 'jpg'
                if 'image/png' in header: ext = 'png'
                elif 'image/webp' in header: ext = 'webp'
            else:
                encoded = img_b64
                ext = 'jpg'

            image_bytes = base64.b64decode(encoded)
            image = FaceQualityEngine.decode_image_bytes(image_bytes)

            # Extract embedding kuri foto ya mberta
            if first_valid_embedding is None:
                first_valid_embedding = FaceQualityEngine.extract_embedding(image)

            # Izina ry'ifoto: sample_1_1690000.jpg, sample_2_1690000.jpg ...
            image_filename = f"sample_{idx+1}_{int(time.time())}.{ext}"
            image_path = os.path.join(user_dataset_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)
            saved_count += 1

        # Encrypt Embedding
        encrypted_template = BiometricEncryption.encrypt_embedding(first_valid_embedding)

        # Process Audio (niba ihari)
        encrypted_voice_template = None
        if audio_data and isinstance(audio_data, str) and len(audio_data) > 0:
            if "," in audio_data:
                _, aud_encoded = audio_data.split(",", 1)
            else:
                aud_encoded = audio_data
            audio_bytes = base64.b64decode(aud_encoded)
            encrypted_voice_template = BiometricEncryption.encrypt_embedding(audio_bytes)

        # Bika Umukoresha muri Database
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if not existing_user:
            new_user = User(
                username=username,
                email=email,
                encrypted_biometric_template=encrypted_template,
                encrypted_voice_template=encrypted_voice_template
            )
            db.session.add(new_user)
            db.session.commit()

            consent_ver = session.get('consent_version', current_app.config.get('CONSENT_VERSION', '1.0'))
            record_user_consent(new_user.id, consent_ver)
            db.session.commit()

        return jsonify({
            'success': True, 
            'message': f'Successfully registered {username}! Saved {saved_count} face samples into dataset/{username}/.', 
            'redirect': '/login'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Processing error: {str(e)}'}), 500
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def login_api():
    start_time = time.time()
    data = request.get_json() or {}
    image_data = data.get('image_data')

    if not image_data:
        return jsonify({'success': False, 'message': 'No face frame received.'}), 400

    # Authentication Lock Check
    last_auth = session.get('last_auth_timestamp', 0)
    if time.time() - last_auth < AUTH_LOCK_DEBOUNCE_SEC:
        return jsonify({'success': False, 'message': 'Cooldown active. Please wait...'}), 429

    try:
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data

        image_bytes = base64.b64decode(encoded)
        image = FaceQualityEngine.decode_image_bytes(image_bytes)

        # 1. Check Liveness & Quality
        is_live, liveness_msg = VerificationEngine.verify_liveness(image)
        if not is_live:
            proc_time = int((time.time() - start_time) * 1000)
            log = AuthenticationLog(
                user_id=None,
                event_type='Liveness Failed',
                success=False,
                confidence_score=0.0
            )
            if hasattr(log, 'processing_time_ms'):
                log.processing_time_ms = proc_time
            if hasattr(log, 'liveness_status'):
                log.liveness_status = 'FAILED'
            db.session.add(log)
            db.session.commit()

            return jsonify({'success': False, 'message': f'Liveness Check Failed: {liveness_msg}'}), 400

        # 2. Extract Embedding & Calculate Similarity
        incoming_embedding = FaceQualityEngine.extract_embedding(image)
        all_users = User.query.all()
        best_match_user = None
        highest_score = 0.0
        SIMILARITY_THRESHOLD = 0.94

        for user in all_users:
            if not user.encrypted_biometric_template:
                continue
            
            stored_embedding = BiometricEncryption.decrypt_embedding(user.encrypted_biometric_template)
            score = VerificationEngine.calculate_cosine_similarity(incoming_embedding, stored_embedding)
            
            if score > highest_score:
                highest_score = score
                best_match_user = user

        proc_time = int((time.time() - start_time) * 1000)
        curr_time_str = datetime.now().strftime('%H:%M:%S')

        # Successful Match Flow
        if best_match_user and highest_score >= SIMILARITY_THRESHOLD:
            session['user_id'] = best_match_user.id
            session['username'] = best_match_user.username
            session['last_auth_timestamp'] = time.time()

            log = AuthenticationLog(
                user_id=best_match_user.id, 
                event_type='Face Recognition Success', 
                success=True, 
                confidence_score=highest_score
            )
            if hasattr(log, 'processing_time_ms'):
                log.processing_time_ms = proc_time
            if hasattr(log, 'liveness_status'):
                log.liveness_status = 'PASSED'
            
            db.session.add(log)
            db.session.commit()

            # Real-time Admin Notification Broadcast
            broadcast_admin_event({
                'user': best_match_user.username,
                'status': 'SUCCESS',
                'method': 'Face Recognition',
                'liveness': 'PASSED',
                'time': curr_time_str,
                'duration': f"{app.config['AUTO_LOGOUT_SECONDS']} seconds",
                'auth_id': f"#{log.id}",
                'proc_time': f"{proc_time}ms"
            })

            return jsonify({
                'success': True, 
                'username': best_match_user.username,
                'confidence': round(highest_score * 100, 1),
                'redirect': url_for('dashboard')
            })
        else:
            # Unknown Face Handling
            log = AuthenticationLog(
                user_id=None,
                event_type='Unrecognized Face',
                success=False,
                confidence_score=highest_score
            )
            if hasattr(log, 'processing_time_ms'):
                log.processing_time_ms = proc_time
            if hasattr(log, 'liveness_status'):
                log.liveness_status = 'PASSED'
            
            db.session.add(log)
            db.session.commit()

            broadcast_admin_event({
                'user': 'Unknown',
                'status': 'DENIED',
                'method': 'Face Recognition',
                'liveness': 'PASSED',
                'time': curr_time_str,
                'duration': 'N/A',
                'auth_id': f"#{log.id}",
                'proc_time': f"{proc_time}ms"
            })

            return jsonify({
                'success': False, 
                'message': 'Identity not recognized.'
            }), 401

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Authentication error: {str(e)}'}), 500

# ==================== REAL-TIME SSE ADMIN STREAM ====================
@app.route('/admin/stream')
@admin_required
def admin_stream():
    """Server-Sent Events Endpoint for real-time admin panel updates."""
    def event_stream():
        q = queue.Queue()
        sse_subscribers.append(q)
        try:
            while True:
                msg = q.get()
                yield msg
        except GeneratorExit:
            sse_subscribers.remove(q)

    return Response(event_stream(), content_type='text/event-stream')

# ==================== SECRET ADMIN ROUTES ====================
@app.route('/secret/sifa/togy/admin', methods=['GET', 'POST'])
def admin_secret_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'AsTogy' and password == '12345':
            user = User.query.filter_by(username='AsTogy').first()

            if not user:
                dummy_template = BiometricEncryption.encrypt_embedding(np.zeros(128, dtype=np.float32))
                user = User(
                    username='AsTogy',
                    email='drgreen.support.ai@gmail.com',
                    encrypted_biometric_template=dummy_template
                )
                user.is_admin = True
                db.session.add(user)
                db.session.commit()
            elif not getattr(user, 'is_admin', False):
                user.is_admin = True
                db.session.commit()

            session['user_id'] = user.id
            session['username'] = user.username

            return redirect(url_for('admin_dashboard'))
        else:
            flash("Izina cyangwa ijambo ry'ibanga ntibyo.", "error")
            return redirect(url_for('admin_secret_login'))

    return render_template('admin-login.html')

@app.route('/admin')
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    all_users = User.query.all()
    recent_logs = AuthenticationLog.query.order_by(AuthenticationLog.timestamp.desc()).limit(15).all()

    # Analytics Calculations
    total_logins = AuthenticationLog.query.count()
    successful_logins = AuthenticationLog.query.filter_by(success=True).count()
    failed_attempts = AuthenticationLog.query.filter_by(success=False).count()

    return render_template(
        'admin.html', 
        total_users=total_users, 
        users=all_users, 
        logs=recent_logs,
        total_logins=total_logins,
        successful_logins=successful_logins,
        failed_attempts=failed_attempts
    )

# ==================== ADMIN AI TRAINING DASHBOARD ====================
@app.route('/admin/train', methods=['GET'])
@app.route('/admin/train-ai', methods=['GET'])
@admin_required
def admin_train_ai():
    """Renders Admin AI Training Page."""
    all_users = User.query.all()
    dataset_dir = os.path.join(app.root_path, 'dataset')
    users_in_dataset = []
    
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            if allowed_file(f):
                base_name = os.path.splitext(f)[0]
                if base_name not in users_in_dataset:
                    users_in_dataset.append(base_name)
        
    return render_template('admin_train.html', users=all_users, dataset_usernames=users_in_dataset)

@app.route('/api/train-model', methods=['POST'])
@admin_required
def trigger_training():
    """Triggers dataset re-indexing or batch feature extraction."""
    try:
        users = User.query.all()
        retrained_count = 0
        dataset_dir = os.path.join(app.root_path, 'dataset')
        
        for user in users:
            dataset_path = find_user_dataset_image(user.username, dataset_dir)
            if dataset_path and os.path.exists(dataset_path):
                with open(dataset_path, "rb") as f:
                    image_bytes = f.read()

                image = FaceQualityEngine.decode_image_bytes(image_bytes)
                new_embedding = FaceQualityEngine.extract_embedding(image)
                user.encrypted_biometric_template = BiometricEncryption.encrypt_embedding(new_embedding)
                retrained_count += 1

        db.session.commit()
        return jsonify({
            'status': 'success', 
            'message': f'AI Training Yarangije Neza! Model yongeye kwigishwa abakozi/users {retrained_count}.'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'AI Training failed: {str(e)}'}), 500

@app.route('/admin/logs')
@admin_required
def admin_logs_api():
    logs = AuthenticationLog.query.order_by(AuthenticationLog.timestamp.desc()).limit(30).all()
    logs_data = []
    for l in logs:
        user = User.query.get(l.user_id) if l.user_id else None
        logs_data.append({
            'id': l.id,
            'user': user.username if user else 'Unknown',
            'event': l.event_type,
            'status': 'SUCCESS' if l.success else 'DENIED',
            'time': l.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'score': round((l.confidence_score or 0) * 100, 1)
        })
    return jsonify(logs_data)

@app.route('/admin/retrain/<int:user_id>', methods=['POST'])
@admin_required
def admin_retrain_user(user_id):
    user = User.query.get_or_404(user_id)
    dataset_dir = os.path.join(app.root_path, 'dataset')
    dataset_path = find_user_dataset_image(user.username, dataset_dir)

    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'success': False, 'message': f'Ifoto ya {user.username} ntiyabonetse muri dataset.'}), 404

    try:
        with open(dataset_path, "rb") as f:
            image_bytes = f.read()

        image = FaceQualityEngine.decode_image_bytes(image_bytes)

        new_embedding = FaceQualityEngine.extract_embedding(image)
        new_encrypted_template = BiometricEncryption.encrypt_embedding(new_embedding)

        user.encrypted_biometric_template = new_encrypted_template
        db.session.commit()

        return jsonify({'success': True, 'message': f'AI model successfully retrained for {user.username}!'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Retraining error: {str(e)}'}), 500

@app.route('/admin/delete/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    try:
        dataset_dir = os.path.join(app.root_path, 'dataset')
        dataset_path = find_user_dataset_image(user.username, dataset_dir)
        if dataset_path and os.path.exists(dataset_path):
            os.remove(dataset_path)

        db.session.delete(user)
        db.session.commit()

        return jsonify({'success': True, 'message': f'User {user.username} successfully deleted.'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Delete error: {str(e)}'}), 500

# ==================== USER DASHBOARD & LOGOUT ====================
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login_page'))

    logs = AuthenticationLog.query.filter_by(user_id=user.id).order_by(AuthenticationLog.timestamp.desc()).limit(5).all()
    return render_template('dashboard.html', user=user, logs=logs)

@app.route('/logout')
def logout():
    # Clear user session and keep the logout seamless for continuous scanner loop
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)