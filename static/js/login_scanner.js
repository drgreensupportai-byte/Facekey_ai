/**
 * Hands-Free Biometric Authentication Scanner
 * Continuous WebRTC processing with asynchronous frame selection and debounce support.
 */

class BiometricLoginScanner {
    constructor(config = {}) {
        this.video = document.getElementById(config.videoId || 'webcam');
        this.canvas = document.getElementById(config.canvasId || 'mesh-canvas');
        this.statusEl = document.getElementById(config.statusId || 'status-msg');
        this.loaderEl = document.getElementById(config.loaderId || 'outer-loader');

        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d');

        this.apiUrl = config.apiUrl || '/api/verify';
        this.frameIntervalMs = config.frameIntervalMs || 400; // Frame sampling rate
        this.audioRecordMs = config.audioRecordMs || 1200;

        this.mediaStream = null;
        this.faceMesh = null;
        this.camera = null;

        this.isProcessingFrame = false;
        this.isSessionLocked = false;
        this.lastVerificationTime = 0;
        this.lockoutDurationMs = config.lockoutDurationMs || 5000; // Session debounce duration

        this.init();
    }

    async init() {
        this.updateStatus("Kugana kuri kamera n'amajwi...", "#38bdf8");
        
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 320 }, height: { ideal: 240 }, frameRate: { ideal: 30 } },
                audio: true
            });

            this.video.srcObject = this.mediaStream;
            await this.video.play();

            this.initFaceMesh();
        } catch (err) {
            this.updateStatus("Ikosa rya kamera/mic: " + err.message, "#f87171");
            console.error("Camera access failed:", err);
        }
    }

    initFaceMesh() {
        this.faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        this.faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: false,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6
        });

        this.faceMesh.onResults((results) => this.handleLandmarks(results));

        this.camera = new Camera(this.video, {
            onFrame: async () => {
                if (this.video && this.video.readyState >= 2) {
                    await this.faceMesh.send({ image: this.video });
                }
            },
            width: 320,
            height: 240
        });

        this.camera.start();
        this.updateStatus("🔍 Reba mu kamera ucecetse...", "#38bdf8");
    }

    handleLandmarks(results) {
        if (!this.canvas || !this.ctx) return;

        // Synchronize dimensions
        if (this.canvas.width !== this.video.videoWidth || this.canvas.height !== this.video.videoHeight) {
            this.canvas.width = this.video.videoWidth || 320;
            this.canvas.height = this.video.videoHeight || 240;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];

            // Render landmark dots
            this.ctx.fillStyle = this.isSessionLocked ? '#a855f7' : '#38bdf8';
            for (const landmark of landmarks) {
                const x = landmark.x * this.canvas.width;
                const y = landmark.y * this.canvas.height;
                this.ctx.beginPath();
                this.ctx.arc(x, y, 1.0, 0, 2 * Math.PI);
                this.ctx.fill();
            }

            // Trigger continuous asynchronous evaluation loop if not locked/busy
            this.evaluateFrameForVerification();
        }
    }

    evaluateFrameForVerification() {
        const now = Date.now();

        // Enforce session lock / debounce limit
        if (this.isProcessingFrame || this.isSessionLocked) {
            if (this.isSessionLocked && (now - this.lastVerificationTime > this.lockoutDurationMs)) {
                this.isSessionLocked = false;
                this.updateStatus("🔍 Reba mu kamera...", "#38bdf8");
            }
            return;
        }

        if (now - this.lastVerificationTime < this.frameIntervalMs) {
            return;
        }

        this.processVerificationScan();
    }

    async processVerificationScan() {
        this.isProcessingFrame = true;
        this.lastVerificationTime = Date.now();

        // Extract video frame
        this.captureCanvas.width = 320;
        this.captureCanvas.height = 240;
        this.captureCtx.drawImage(this.video, 0, 0, 320, 240);
        const imageData = this.captureCanvas.toDataURL('image/jpeg', 0.6);

        // Record short audio snippet in parallel
        const audioData = await this.recordAudioSnippet();

        // Dispatch verification payload
        await this.sendVerificationPayload(imageData, audioData);
    }

    recordAudioSnippet() {
        return new Promise((resolve) => {
            if (!this.mediaStream) return resolve(null);

            const audioTracks = this.mediaStream.getAudioTracks();
            if (audioTracks.length === 0 || !audioTracks[0].enabled) {
                return resolve(null);
            }

            let recorder = null;
            let chunks = [];

            try {
                const mimeType = this.getSupportedMimeType();
                recorder = mimeType ? new MediaRecorder(this.mediaStream, { mimeType }) : new MediaRecorder(this.mediaStream);
            } catch (e) {
                return resolve(null);
            }

            recorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) chunks.push(e.data);
            };

            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/wav' });
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result);
                reader.onerror = () => resolve(null);
                reader.readAsDataURL(blob);
            };

            recorder.start();
            setTimeout(() => {
                if (recorder.state === 'recording') {
                    recorder.stop();
                }
            }, this.audioRecordMs);
        });
    }

    async sendVerificationPayload(imageData, audioData) {
        if (this.loaderEl) this.loaderEl.style.display = 'block';

        const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';

        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    image_data: imageData,
                    audio_data: audioData
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                this.handleSuccess(result);
            } else {
                this.handleFailure(result);
            }
        } catch (err) {
            console.error("Verification network error:", err);
            this.isProcessingFrame = false;
        } finally {
            if (this.loaderEl) this.loaderEl.style.display = 'none';
        }
    }

    handleSuccess(result) {
        this.isSessionLocked = true;
        this.updateStatus(`✓ Murakaza neza, ${result.username}!`, "#2ef8a0");

        // Clean up sensors
        if (this.camera && typeof this.camera.stop === 'function') this.camera.stop();
        if (this.mediaStream) this.mediaStream.getTracks().forEach(t => t.stop());

        setTimeout(() => {
            window.location.href = result.redirect || "/dashboard";
        }, 1000);
    }

    handleFailure(result) {
        this.isProcessingFrame = false;

        if (result.reason === 'LIVENESS_FAILED') {
            this.updateStatus("⚠️ Ntabwo tuzi niba uri umuntu muzima. Ongera ugerageze.", "#fbbf24");
        } else if (result.reason === 'NO_MATCH') {
            this.updateStatus("❌ Ntabwo twaguhawe uburenganzira (No Match).", "#f87171");
        } else {
            this.updateStatus("🔍 Reba neza muri kamera...", "#38bdf8");
        }
    }

    updateStatus(message, color) {
        if (this.statusEl) {
            this.statusEl.innerText = message;
            if (color) this.statusEl.style.color = color;
        }
    }

    getSupportedMimeType() {
        const types = ['audio/webm', 'audio/mp4', 'audio/ogg', 'audio/wav'];
        for (let t of types) {
            if (MediaRecorder.isTypeSupported(t)) return t;
        }
        return '';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('webcam')) {
        window.biometricScanner = new BiometricLoginScanner();
    }
});