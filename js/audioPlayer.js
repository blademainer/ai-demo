class AudioPlayer {
    constructor() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.activeOscillators = {};
        this.volume = 0.5;
        this.setScale('pentatonic'); // Default scale
    }
    
    // Define musical scales (frequencies in Hz)
    setScale(scaleType) {
        const scales = {
            'major': [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
            'minor': [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],
            'pentatonic': [261.63, 293.66, 329.63, 392.00, 440.00, 523.25],
            'blues': [261.63, 311.13, 349.23, 370.00, 392.00, 466.16, 523.25],
            'chromatic': [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25]
        };
        
        this.scale = scales[scaleType] || scales['pentatonic'];
        this.scaleType = scaleType;
    }
    
    setVolume(value) {
        this.volume = parseFloat(value);
    }
    
    playNote(fingerId, position) {
        if (!position || typeof position.y !== 'number') {
            console.warn('Invalid position for note playing');
            return;
        }
        
        // Calculate frequency based on y-position
        const normalizedY = 1 - Math.min(Math.max(position.y, 0), 1); // Invert Y so higher position = higher pitch
        const noteIndex = Math.floor(normalizedY * this.scale.length);
        const frequency = this.scale[Math.min(Math.max(noteIndex, 0), this.scale.length - 1)];
        
        if (!Number.isFinite(frequency)) {
            console.warn('Skipping note with invalid frequency:', frequency);
            return;
        }
        
        // Stop previous oscillator for this finger if exists
        this.stopNote(fingerId);
        
        // Create new oscillator
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        // Configure oscillator
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(frequency, this.audioContext.currentTime);
        
        // Configure gain (volume)
        gainNode.gain.setValueAtTime(this.volume, this.audioContext.currentTime);
        
        // Connect nodes
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        // Start oscillator
        oscillator.start();
        
        // Store reference to stop later
        this.activeOscillators[fingerId] = {
            oscillator,
            gainNode
        };
    }
    
    stopNote(fingerId) {
        if (this.activeOscillators[fingerId]) {
            const { oscillator, gainNode } = this.activeOscillators[fingerId];
            
            // Fade out to avoid clicks
            gainNode.gain.setValueAtTime(gainNode.gain.value, this.audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.0001, this.audioContext.currentTime + 0.03);
            
            // Schedule stop
            oscillator.stop(this.audioContext.currentTime + 0.03);
            
            // Remove reference
            delete this.activeOscillators[fingerId];
        }
    }
    
    stopAll() {
        Object.keys(this.activeOscillators).forEach(fingerId => {
            this.stopNote(fingerId);
        });
    }
}
