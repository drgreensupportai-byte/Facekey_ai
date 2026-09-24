// static/js/register.js

let currentStream = null;
let mediaRecorder = null;
let recordedChunks = [];

/**
 * 1. Gushaka MIME type yemewe n'integanya-muryango (Browser)
 */
function getSupportedMimeType() {
    const types = [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm',
        'video/mp4'
    ];
    for (let type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }
    return ''; // Ukoreshe default niba ntayibonetse
}

/**
 * 2. Tangiza Kamera (Webcam) ku gipimo cyoroshye gukora (640x480)
 */
async function startCamera() {
    const videoElement = document.getElementById('video'); // Reba niba id='video'
    if (!videoElement) return;

    try {
        if (currentStream && currentStream.active) {
            return;
        }

        // Set optimal webcam constraints (640x480 resolution for fast processing)
        currentStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 15 }
            },
            audio: true // Shyiraho 'false' niba udakeneye gufata amajwi
        });

        videoElement.srcObject = currentStream;
        await videoElement.play();
        console.log("Kamera yafungutse neza!");
    } catch (err) {
        console.warn("Kamera ntiyafungutse n'audio, tutangije video gusa:", err);
        try {
            // Fallback: Tangiza amashusho gusa niba microfon ifite ikosa
            currentStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 640 }, height: { ideal: 480 } }, 
                audio: false 
            });
            videoElement.srcObject = currentStream;
            await videoElement.play();
        } catch (fallbackErr) {
            console.error("Kamera yanze gufunguka burundu:", fallbackErr);
            alert("Ntiwabashije gufungura kamera! Reba permission muri browser.");
        }
    }
}

/**
 * 3. MediaRecorder Capture logic (Ikemura NotSupportedError)
 */
async function captureAndRegister() {
    try {
        if (!currentStream || !currentStream.active) {
            await startCamera();
        }

        recordedChunks = [];
        const mimeType = getSupportedMimeType();
        const options = mimeType ? { mimeType: mimeType } : {};

        // Tangiza MediaRecorder hatabayeho crash
        mediaRecorder = new MediaRecorder(currentStream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const blob = new Blob(recordedChunks, { type: mimeType || 'video/webm' });
            await sendDataToBackend(blob);
        };

        // Tangira gufata chunk y'amashusho buri sec 1
        mediaRecorder.start(1000);
        console.log("MediaRecorder yatangiye gukora...");

        // Urugero: Fata amashusho y'amasegonda 3 hanyuma uhagarike
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                mediaRecorder.stop();
            }
        }, 3000);

    } catch (err) {
        console.error("MediaRecorder error:", err);
        alert("Ikosa mu gukoresha MediaRecorder: " + err.message);
    }
}

/**
 * 4. Yoherereza Data kuri Flask API (/register)
 */
async function sendDataToBackend(videoBlob) {
    const usernameInput = document.getElementById('username'); // Reba input ID ya username
    const username = usernameInput ? usernameInput.value.trim() : '';

    if (!username) {
        alert("Nyamuneka andika Izina (Username) mbere yo kwiyandikisha!");
        return;
    }

    const formData = new FormData();
    formData.append('username', username);
    formData.append('video', videoBlob, 'registration_capture.webm');

    try {
        const response = await fetch('/register', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            alert("Kwiyandikisha byagenze neza!");
            window.location.href = "/login";
        } else {
            alert("Kwiyandikisha byanze: " + (result.message || "Server Error"));
        }
    } catch (err) {
        console.error("Backend request error:", err);
        alert("Ntiwabashije koherereza data kuri server. Reba Terminal ya Python.");
    }
}

// Tangiza Kamera ku paji ikimara gufunguka (DOMContentLoaded)
document.addEventListener("DOMContentLoaded", () => {
    startCamera();

    const registerBtn = document.getElementById('registerBtn'); // Reba niba button yawe ifite id='registerBtn'
    if (registerBtn) {
        registerBtn.addEventListener('click', (e) => {
            e.preventDefault();
            captureAndRegister();
        });
    }
});