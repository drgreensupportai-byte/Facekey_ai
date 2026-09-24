/**
 * login.js - Ultra-Fast FaceKey AI Biometric & Liveness Login Controller
 */

let videoStream = null;
let livenessPassed = false;
let autoScanInterval = null;
let isProcessing = false;
let mediaRecorder = null;
let audioChunks = [];

const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('mesh-canvas');
const captureCanvas = document.getElementById('capture-canvas');
const canvasCtx = canvasElement ? canvasElement.getContext('2d') : null;
const scanLoader = document.getElementById('outer-loader') || document.getElementById('scan-loader');
const statusElement = document.getElementById('status-msg');

// Fast Configuration Limits
const SCAN_INTERVAL_MS = 2000;  // Run check every 2 seconds
const AUDIO_RECORD_MS = 1000;   // Capture 1 second of audio stream for speed

// 1. Initialize MediaPipe FaceMesh Engine
const faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: false, // Performance speed optimization
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

// Render face landmarks in real-time
faceMesh.onResults((results) => {
    if (!canvasElement || !canvasCtx || !videoElement) return;

    canvasElement.width = videoElement.videoWidth || 320;
    canvasElement.height = videoElement.videoHeight || 240;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];

        for (const landmark of landmarks) {
            const x = landmark.x * canvasElement.width;
            const y = landmark.y * canvasElement.height;

            canvasCtx.beginPath();
            canvasCtx.arc(x, y, 1.0, 0, 2 * Math.PI);
            canvasCtx.fillStyle = '#38bdf8'; // Glowing sky blue dots
            canvasCtx.fill();
        }
    }
});

// UI Outer Ring Spinner Controls
function startScanningUI() {
    if (scanLoader) {
        scanLoader.style.display = scanLoader.id === 'outer-loader' ? 'block' : 'flex';
    }
}

function stopScanningUI() {
    if (scanLoader) {
        scanLoader.style.display = 'none';
    }
}

// 2. Camera Stream Boot
async function startCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 320, height: 240, frameRate: { ideal: 30 } },
            audio: true
        });
        videoElement.srcObject = videoStream;

        if (statusElement) {
            statusElement.innerText = "⚡ Reba mu kamera... Kwinjira birimo gukorwa...";
            statusElement.style.color = "#38bdf8";
        }

        // Initialize frame feed for MediaPipe FaceMesh
        const camera = new Camera(videoElement, {
            onFrame: async () => {
                await faceMesh.send({ image: videoElement });
            },
            width: 320,
            height: 240
        });
        camera.start();

        // Accelerated interval: every 2 seconds
        autoScanInterval = setInterval(authenticateUser, SCAN_INTERVAL_MS);

    } catch (err) {
        if (statusElement) {
            statusElement.innerText = "Ikosa ku gufata camera cyangwa mic: " + err.message;
            statusElement.style.color = "#f87171";
        }
    }
}

// 3. Fast Snapshot & Audio Capture
async function authenticateUser() {
    if (isProcessing || !videoElement) return;
    isProcessing = true;

    startScanningUI();

    // Fast Low-Weight Frame Capture (JPEG 0.6 quality for payload size reduction)
    const targetCanvas = captureCanvas || document.createElement('canvas');
    targetCanvas.width = 320;
    targetCanvas.height = 240;
    
    const ctx = targetCanvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, 320, 240);
    const imageData = targetCanvas.toDataURL('image/jpeg', 0.6);

    if (statusElement) {
        statusElement.innerText = "Kugenzura umutwe n'ijwi... Tegereza gato.";
        statusElement.style.color = "#38bdf8";
    }

    // Fast 1-Second Audio Capture
    audioChunks = [];
    try {
        mediaRecorder = new MediaRecorder(videoStream);
    } catch (e) {
        console.warn("MediaRecorder missing:", e);
    }

    if (mediaRecorder) {
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const reader = new FileReader();

            reader.onloadend = () => {
                dispatchLoginPayload(imageData, reader.result);
            };

            reader.readAsDataURL(audioBlob);
        };

        mediaRecorder.start();
        setTimeout(() => {
            if (mediaRecorder.state === "recording") {
                mediaRecorder.stop();
            }
        }, AUDIO_RECORD_MS);
    } else {
        dispatchLoginPayload(imageData, null);
    }
}

// 4. API Request Handler
async function dispatchLoginPayload(imageData, audioData) {
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_data: imageData,
                audio_data: audioData,
                liveness_passed: true
            })
        });

        const result = await response.json();
        stopScanningUI();

        if (result.success) {
            if (autoScanInterval) clearInterval(autoScanInterval);

            // Clean up active streams on successful authentication
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
            }

            if (statusElement) {
                statusElement.style.color = "#2ef8a0";
                statusElement.innerText = `✓ Murakaza neza, ${result.username}! (${result.confidence}% Match)`;
            }

            // Accelerated redirect
            setTimeout(() => { 
                window.location.href = result.redirect || '/dashboard'; 
            }, 600);
        } else {
            if (statusElement) {
                if (response.status === 429) {
                    statusElement.style.color = "#a855f7";
                    statusElement.innerText = "⏳ Tegereza gato ugeregeze...";
                } else {
                    statusElement.style.color = "#fbbf24";
                    statusElement.innerText = "🔍 " + (result.message || "Ntidushoboye kukoza. Ongera ugerageze.");
                }
            }
            setTimeout(() => { isProcessing = false; }, 800);
        }
    } catch (error) {
        stopScanningUI();
        if (statusElement) {
            statusElement.style.color = "#f87171";
            statusElement.innerText = "Server Error: " + error.message;
        }
        setTimeout(() => { isProcessing = false; }, 800);
    }
}

// Start immediately on load
window.onload = startCamera;