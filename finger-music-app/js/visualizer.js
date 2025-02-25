const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function drawFingerLines(fingerPositions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas

    ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)'; // Set line color
    ctx.lineWidth = 5; // Set line width

    fingerPositions.forEach((position, index) => {
        ctx.beginPath();
        ctx.moveTo(position.x, position.y);
        ctx.lineTo(position.x, position.y - 50); // Draw line upwards
        ctx.stroke();
        ctx.closePath();
    });
}

function updateCanvasSize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

window.addEventListener('resize', updateCanvasSize);
updateCanvasSize(); // Initial canvas size setup

export { drawFingerLines };