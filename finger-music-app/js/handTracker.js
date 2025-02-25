const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleSelect = document.getElementById('scaleSelect');
let model, hands;

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadModel() {
    model = await handPoseDetection.createDetector(handPoseDetection.SupportedModels.MediaPipeHands);
}

async function detectHands() {
    const predictions = await model.estimateHands(video);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawHands(predictions);
    requestAnimationFrame(detectHands);
}

function drawHands(predictions) {
    predictions.forEach(prediction => {
        const keypoints = prediction.keypoints;
        keypoints.forEach((keypoint, index) => {
            if (keypoint.score > 0.5) {
                ctx.fillStyle = 'red';
                ctx.beginPath();
                ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = 'blue';
                ctx.lineWidth = 2;
                ctx.stroke();
                if (index > 0) {
                    ctx.beginPath();
                    ctx.moveTo(keypoints[index - 1].x, keypoints[index - 1].y);
                    ctx.lineTo(keypoint.x, keypoint.y);
                    ctx.stroke();
                }
            }
        });
    });
}

async function init() {
    await setupCamera();
    await loadModel();
    video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detectHands();
}

document.getElementById('startBtn').addEventListener('click', init);