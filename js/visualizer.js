class Visualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.fingerColors = [
            '#FF0000', // Thumb - Red
            '#00FF00', // Index - Green
            '#0000FF', // Middle - Blue
            '#FFFF00', // Ring - Yellow
            '#FF00FF'  // Pinky - Magenta
        ];
        
        // Store previous finger positions for drawing trails
        this.prevPositions = {};
        
        // Set the maximum length of the trails
        this.maxTrailLength = 20;
    }
    
    updateCanvasSize() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.offsetWidth;
        this.canvas.height = container.offsetHeight;
    }
    
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    // Draw hand landmarks and connections
    drawHand(hand) {
        if (!hand || !hand.keypoints || hand.keypoints.length === 0) return;
        
        const keypoints = hand.keypoints;
        
        // Draw connections between landmarks
        this.drawConnections(keypoints);
        
        // Draw landmarks
        keypoints.forEach((keypoint) => {
            this.drawLandmark(keypoint);
        });
        
        // Draw fingertip trails
        this.updateFingerTrails(hand);
        this.drawFingerTrails();
    }
    
    drawLandmark(keypoint) {
        if (!keypoint.x || !keypoint.y) return;
        
        const x = keypoint.x * this.canvas.width;
        const y = keypoint.y * this.canvas.height;
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
        this.ctx.fillStyle = 'white';
        this.ctx.fill();
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#3498db';
        this.ctx.fill();
    }
    
    drawConnections(keypoints) {
        // Define the connections between keypoints
        const connections = [
            // Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            // Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            // Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            // Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            // Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
            // Palm
            [0, 5], [5, 9], [9, 13], [13, 17]
        ];
        
        connections.forEach(([i, j]) => {
            const start = keypoints[i];
            const end = keypoints[j];
            
            if (!start || !end || !start.x || !start.y || !end.x || !end.y) return;
            
            this.ctx.beginPath();
            this.ctx.moveTo(start.x * this.canvas.width, start.y * this.canvas.height);
            this.ctx.lineTo(end.x * this.canvas.width, end.y * this.canvas.height);
            this.ctx.strokeStyle = '#ffffff80';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
    }
    
    updateFingerTrails(hand) {
        if (!hand || !hand.keypoints) return;
        
        // Map of fingertip indices
        const fingertips = [4, 8, 12, 16, 20];
        
        fingertips.forEach((tipIndex, fingerIndex) => {
            const keypoint = hand.keypoints[tipIndex];
            
            if (!keypoint || !keypoint.x || !keypoint.y) return;
            
            if (!this.prevPositions[fingerIndex]) {
                this.prevPositions[fingerIndex] = [];
            }
            
            this.prevPositions[fingerIndex].push({
                x: keypoint.x * this.canvas.width,
                y: keypoint.y * this.canvas.height
            });
            
            // Limit the trail length
            if (this.prevPositions[fingerIndex].length > this.maxTrailLength) {
                this.prevPositions[fingerIndex].shift();
            }
        });
    }
    
    drawFingerTrails() {
        Object.keys(this.prevPositions).forEach(fingerIndex => {
            const positions = this.prevPositions[fingerIndex];
            const color = this.fingerColors[fingerIndex];
            
            if (positions.length < 2) return;
            
            this.ctx.beginPath();
            this.ctx.moveTo(positions[0].x, positions[0].y);
            
            for (let i = 1; i < positions.length; i++) {
                this.ctx.lineTo(positions[i].x, positions[i].y);
            }
            
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 4;
            this.ctx.lineCap = 'round';
            this.ctx.stroke();
        });
    }
    
    resetTrails() {
        this.prevPositions = {};
    }
}
