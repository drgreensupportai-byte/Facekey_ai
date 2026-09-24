<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>let blinkPassed = false;
let headTurnPassed = false;

// Calculate Eye Aspect Ratio (EAR)
function calculateEAR(landmarks, eyeIndices) {
    const p1 = landmarks[eyeIndices[0]];
    const p2 = landmarks[eyeIndices[1]];
    const p3 = landmarks[eyeIndices[2]];
    const p4 = landmarks[eyeIndices[3]];
    const p5 = landmarks[eyeIndices[4]];
    const p6 = landmarks[eyeIndices[5]];

    const distV1 = Math.hypot(p2.x - p6.x, p2.y - p6.y);
    const distV2 = Math.hypot(p3.x - p5.x, p3.y - p5.y);
    const distH = Math.hypot(p1.x - p4.x, p1.y - p4.y);

    return (distV1 + distV2) / (2.0 * distH);
}

const faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

faceMesh.onResults((results) => {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;

    const landmarks = results.multiFaceLandmarks[0];

    // 1. Blink Detection (Left Eye Indices: [33, 160, 158, 133, 153, 144])
    const leftEAR = calculateEAR(landmarks, [33, 160, 158, 133, 153, 144]);
    if (leftEAR < 0.18) {
        blinkPassed = true;
        document.getElementById('status').innerText = "Blink detected! Now turn your head to the right.";
    }

    // 2. Head Motion Detection (Nose Tip x-position relative to cheeks)
    const noseX = landmarks[1].x;
    const leftCheekX = landmarks[234].x;
    const rightCheekX = landmarks[454].x;
    const yawRatio = (noseX - leftCheekX) / (rightCheekX - leftCheekX);

    if (blinkPassed && (yawRatio < 0.35 || yawRatio > 0.65)) {
        headTurnPassed = true;
        document.getElementById('status').innerText = "Liveness verified! Capturing face...";
        captureAndAuthenticate();
    }
});