// This file handles audio playback functionality. It loads sound files based on the selected scale and plays the corresponding notes when fingers are detected.

const audioPlayer = {
    audioContext: new (window.AudioContext || window.webkitAudioContext)(),
    sounds: {
        pentatonic: [
            'assets/sounds/pentatonic/note1.mp3',
            'assets/sounds/pentatonic/note2.mp3',
            'assets/sounds/pentatonic/note3.mp3'
        ],
        major: [
            'assets/sounds/major/note1.mp3',
            'assets/sounds/major/note2.mp3',
            'assets/sounds/major/note3.mp3'
        ],
        minor: [
            'assets/sounds/minor/note1.mp3',
            'assets/sounds/minor/note2.mp3',
            'assets/sounds/minor/note3.mp3'
        ]
    },
    currentScale: 'pentatonic',
    buffers: [],

    loadSounds: function() {
        const scaleSounds = this.sounds[this.currentScale];
        const promises = scaleSounds.map((soundUrl) => this.loadSound(soundUrl));
        return Promise.all(promises);
    },

    loadSound: function(url) {
        return fetch(url)
            .then(response => response.arrayBuffer())
            .then(arrayBuffer => this.audioContext.decodeAudioData(arrayBuffer))
            .then(buffer => {
                this.buffers.push(buffer);
            });
    },

    playNote: function(noteIndex) {
        if (this.buffers[noteIndex]) {
            const source = this.audioContext.createBufferSource();
            source.buffer = this.buffers[noteIndex];
            source.connect(this.audioContext.destination);
            source.start(0);
        }
    },

    setScale: function(scale) {
        this.currentScale = scale;
        this.buffers = [];
        return this.loadSounds();
    }
};

export default audioPlayer;