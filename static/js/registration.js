document.addEventListener("DOMContentLoaded", async () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const progressBar = document.getElementById('progressBar');
    const sampleCounter = document.getElementById('sampleCounter');
    const progressPercentage = document.getElementById('progressPercentage');
    const statusFeedback = document.getElementById('statusFeedback');
    const stepInstruction = document.getElementById('stepInstruction');
    const stepTitle = document.getElementById('stepTitle');

    // Registration Target Positions State Machine
    const TARGET_STEPS = [
        { angle: 'FRONT', label: 'Tumbira kamera hagati (Look straight at camera)' },
        { angle: 'LEFT', label: 'Pimisha umutwe kidogo ku MUKASO (Turn head slightly LEFT)' },
        { angle: 'RIGHT', label: 'Pimisha umutwe kidogo iburyo (Turn head slightly RIGHT)' },
        { angle: 'UP', label: 'Umutwe wamurike HEURU (Look slightly UP)' },
        { angle: 'DOWN', label: 'Umutwe uwunike HASI (Look slightly DOWN)' },
        { angle: 'LEFT_UP', label: 'Pimisha ibumoso unarunguruka HEURU (Turn LEFT & UP)' },
        { angle: 'RIGHT_UP', label: 'Pimisha iburyo unarunguruka HEURU (Turn RIGHT & UP)' }
    ];

    let currentStepIndex = 0;
    let capturedSamples = [];
    let isProcessingFrame = false;

    // Start WebRTC Camera
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
    } catch (err) {
        statusFeedback.innerText = "❌ Camera access denied or not available.";
        return;
    }

    // MediaPipe FaceMesh Initialization
    const faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
    });

    faceMesh.onResults(onResults);

    // Continuous Frame Processing Loop
    async function renderLoop() {
        if (!video.paused && !video.ended) {
            await faceMesh.send({ image: video });
        }
        requestAnimationFrame(renderLoop);
    }

    video.addEventListener('loadeddata', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        renderLoop();
    });

    async function onResults(results) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            statusFeedback.innerText = "⚠️ Nta sura ibonetse (No face detected)";
            return;
        }

        if (isProcessingFrame || currentStepIndex >= TARGET_STEPS.length) return;

        const currentStep = TARGET_STEPS[currentStepIndex];
        stepTitle.innerText = `STEP ${currentStepIndex + 1} / ${TARGET_STEPS.length}`;
        stepInstruction.innerText = currentStep.label;

        // Capture canvas frame as Blob
        isProcessingFrame = true;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        tempCanvas.getContext('2d').drawImage(video, 0, 0);

        tempCanvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob);
            formData.append('target_angle', currentStep.angle);
            formData.append('landmarks', JSON.stringify(results.multiFaceLandmarks[0]));

            try {
                const response = await fetch('/api/process-registration-frame', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.success) {
                    // Update UI checklist badge
                    const badge = document.querySelector(`[data-angle="${currentStep.angle}"]`);
                    if (badge) {
                        badge.className = "angle-item p-2 rounded bg-green-900/80 text-green-300 font-bold";
                        badge.innerText = `✅ ${currentStep.angle} PASSED`;
                    }

                    capturedSamples.push(data.sample);
                    currentStepIndex++;

                    // Update Progress
                    const pct = Math.round((currentStepIndex / TARGET_STEPS.length) * 100);
                    progressBar.style.width = `${pct}%`;
                    progressPercentage.innerText = `${pct}%`;
                    sampleCounter.innerText = `${currentStepIndex} / ${TARGET_STEPS.length} Samples`;

                    if (currentStepIndex >= TARGET_STEPS.length) {
                        statusFeedback.innerText = "🎉 All angles captured successfully!";
                        document.getElementById('completeRegistrationForm').classList.remove('hidden');
                    }
                } else {
                    statusFeedback.innerText = `⚠️ ${data.message}`;
                }
            } catch (err) {
                statusFeedback.innerText = "❌ Frame processing error.";
            } finally {
                setTimeout(() => { isProcessingFrame = false; }, 400); // Debounce
            }
        }, 'image/jpeg', 0.9);
    }
});