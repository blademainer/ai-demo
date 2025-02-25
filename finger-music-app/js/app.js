// This file contains the main JavaScript logic for the Finger Music application.

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const scaleSelect = document.getElementById('scaleSelect');
const volumeSlider = document.getElementById('volumeSlider');

let model, handTracker, audioPlayer;
let isTracking = false;

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
    model = await handPoseDetection.SupportedModels.MediaPipeHands;
    handTracker = await handPoseDetection.createDetector(model);
}

async function detectHands() {
    const hands = await handTracker.estimateHands(video);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (hands.length > 0) {
        drawHands(hands);
        audioPlayer.playNotes(hands);
    }
    if (isTracking) {
        requestAnimationFrame(detectHands);
    }
}

function drawHands(hands) {
    hands.forEach(hand => {
        hand.keypoints.forEach(keypoint => {
            if (keypoint.score > 0.5) {
                ctx.beginPath();
                ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'red';
                ctx.fill();
            }
        });
        drawLines(hand.keypoints);
    });
}

function drawLines(keypoints) {
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
    ];
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    connections.forEach(connection => {
        const [startIdx, endIdx] = connection;
        const start = keypoints[startIdx];
        const end = keypoints[endIdx];
        if (start.score > 0.5 && end.score > 0.5) {
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();
        }
    });
}

startBtn.addEventListener('click', async () => {
    await setupCamera();
    await loadModel();
    video.play();
    isTracking = true;
    detectHands();
    startBtn.disabled = true;
    stopBtn.disabled = false;
});

stopBtn.addEventListener('click', () => {
    isTracking = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

scaleSelect.addEventListener('change', () => {
    audioPlayer.setScale(scaleSelect.value);
});

volumeSlider.addEventListener('input', () => {
    audioPlayer.setVolume(volumeSlider.value);
});

// Initialize audio player
audioPlayer = new AudioPlayer(); // Assuming AudioPlayer is defined in audioPlayer.js
audioPlayer.loadSounds(); // Load sounds based on the selected scale

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;