class HandTracker {
    constructor() {
        this.detector = null;
        this.hands = null;
        this.isRunning = false;
        this.lastFingerPositions = {};
    }
    
    async initialize() {
        const model = handPoseDetection.SupportedModels.MediaPipeHands;
        const detectorConfig = {
            runtime: 'mediapipe',
            solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
            modelType: 'full'
        };
        
        this.detector = await handPoseDetection.createDetector(model, detectorConfig);
        console.log('Hand tracking model loaded');
    }
    
    async detectHands(video) {
        if (!this.detector) return [];
        
        try {
            this.hands = await this.detector.estimateHands(video, {
                flipHorizontal: true
            });
            
            return this.hands;
        } catch (error) {
            console.error('Error detecting hands:', error);
            return [];
        }
    }
    
    getFingerPositionY(hand, fingerIndex) {
        if (!hand || !hand.keypoints) return null;
        
        // Get the fingertip keypoint
        // Fingertips are at indices 4, 8, 12, 16, 20 for thumb, index, middle, ring, pinky
        const fingertipIndices = [4, 8, 12, 16, 20];
        const tipIndex = fingertipIndices[fingerIndex];
        
        if (!tipIndex) return null;
        
        const keypoint = hand.keypoints[tipIndex];
        if (!keypoint) return null;
        
        // Return normalized y-position (0 at top, 1 at bottom)
        return keypoint.y;
    }
    
    getFingertipPositions(hand) {
        if (!hand || !hand.keypoints) return null;
        
        const fingertipIndices = [4, 8, 12, 16, 20];
        const positions = {};
        
        fingertipIndices.forEach((tipIndex, fingerIndex) => {
            const keypoint = hand.keypoints[tipIndex];
            if (keypoint) {
                positions[fingerIndex] = {
                    x: keypoint.x,
                    y: keypoint.y
                };
            }
        });
        
        return positions;
    }
    
    checkFingerMovement(hand) {
        const currentPositions = this.getFingertipPositions(hand);
        if (!currentPositions) return null;
        
        const movingFingers = {};
        
        // Compare current positions with last positions
        Object.keys(currentPositions).forEach(fingerIndex => {
            const current = currentPositions[fingerIndex];
            const previous = this.lastFingerPositions[fingerIndex];
            
            if (previous) {
                // Check if the finger has moved significantly
                const distance = Math.sqrt(
                    Math.pow(current.x - previous.x, 2) + 
                    Math.pow(current.y - previous.y, 2)
                );
                
                // If finger moved more than threshold, consider it moving
                if (distance > 0.01) {
                    movingFingers[fingerIndex] = {
                        position: current.y, // Normalized y position (0-1)
                        velocity: distance
                    };
                }
            }
        });
        
        // Update last positions
        this.lastFingerPositions = currentPositions;
        
        return movingFingers;
    }
}
