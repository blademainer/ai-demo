document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const scaleSelect = document.getElementById('scaleSelect');
    const volumeSlider = document.getElementById('volumeSlider');
    
    // Initialize classes
    const visualizer = new Visualizer(canvas);
    const audioPlayer = new AudioPlayer();
    const handTracker = new HandTracker();
    
    let isRunning = false;
    let animationFrame;
    
    // Set up event listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    scaleSelect.addEventListener('change', e => audioPlayer.setScale(e.target.value));
    volumeSlider.addEventListener('input', e => audioPlayer.setVolume(e.target.value));
    
    // Handle window resize
    window.addEventListener('resize', () => {
        visualizer.updateCanvasSize();
    });
    
    // Initialize the application
    async function init() {
        try {
            // Initialize hand tracking model
            await handTracker.initialize();
            console.log('Application initialized successfully');
            startBtn.disabled = false;
        } catch (error) {
            console.error('Failed to initialize application:', error);
            alert('Failed to initialize the application. Please try again or check console for details.');
        }
    }
    
    // Start the camera and tracking
    async function startCamera() {
        if (isRunning) return;
        
        try {
            // Access camera
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480,
                    facingMode: 'user'
                },
                audio: false
            });
            
            video.srcObject = stream;
            await video.play();
            
            // Update canvas size to match video dimensions
            visualizer.updateCanvasSize();
            
            // Start processing frames
            isRunning = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            // Resume audio context if suspended
            if (audioPlayer.audioContext.state === 'suspended') {
                await audioPlayer.audioContext.resume();
            }
            
            // Start the processing loop
            processFrame();
        } catch (error) {
            console.error('Error starting camera:', error);
            alert('Failed to access the camera. Please make sure you have given permission and your camera is working.');
        }
    }
    
    // Stop the camera and tracking
    function stopCamera() {
        if (!isRunning) return;
        
        // Stop the animation frame
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
            animationFrame = null;
        }
        
        // Stop all audio
        audioPlayer.stopAll();
        
        // Stop the camera stream
        if (video.srcObject) {
            const tracks = video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }
        
        // Reset the visualizer
        visualizer.resetTrails();
        visualizer.clear();
        
        // Update UI
        isRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
    
    // Process each video frame
    async function processFrame() {
        if (!isRunning) return;
        
        try {
            // Clear the canvas
            visualizer.clear();
            
            // Detect hands in the current frame
            const hands = await handTracker.detectHands(video);
            
            if (hands && hands.length > 0) {
                const hand = hands[0]; // Process only the first hand for simplicity
                
                // Draw the hand
                visualizer.drawHand(hand);
                
                // Process landmarks and play sounds
                if (hand.landmarks) {
                    processHandLandmarks(hand.landmarks);
                }
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
        
        // Continue the loop
        animationFrame = requestAnimationFrame(processFrame);
    }
    
    // Function to process hand landmarks and play sounds
    function processHandLandmarks(landmarks) {
        // Get video dimensions for normalization
        const videoWidth = video.videoWidth || 640;
        const videoHeight = video.videoHeight || 480;
        
        landmarks.forEach((landmark, i) => {
            if (i > 0 && i <= 5) { // Only process finger landmarks (exclude palm)
                const fingerPosition = {
                    x: landmark[0] / videoWidth,
                    y: landmark[1] / videoHeight
                };
                
                // Ensure values are within valid ranges
                const normalizedY = Math.min(Math.max(fingerPosition.y, 0), 1);
                
                // Play the note based on finger position
                audioPlayer.playNote(
                    i.toString(), // Use finger index as the identifier
                    fingerPosition
                );
            }
        });
    }
    
    // Initialize the application
    init();
});
