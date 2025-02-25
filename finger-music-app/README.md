# Finger Music App

## Overview
The Finger Music App is a web application that allows users to play musical notes by moving their fingers in front of a camera. The application utilizes TensorFlow.js and MediaPipe Handpose model to detect finger movements and visually represents them on the screen. Users can select different musical scales and adjust the volume of the notes being played.

## Features
- Real-time finger tracking using the camera.
- Play notes by moving fingers, with each finger triggering a different note.
- Select between different musical scales: Pentatonic, Major, and Minor.
- Adjustable volume control for audio playback.
- Visual representation of finger positions on a canvas.

## Project Structure
```
finger-music-app
├── index.html          # Main entry point for the application
├── styles.css         # CSS styles for the application
├── js
│   ├── app.js         # Main application logic
│   ├── audioPlayer.js  # Audio playback functionality
│   ├── handTracker.js   # Hand tracking logic
│   └── visualizer.js    # Visual representation of fingers
├── assets
│   ├── sounds
│   │   ├── pentatonic  # Audio files for the pentatonic scale
│   │   ├── major       # Audio files for the major scale
│   │   └── minor       # Audio files for the minor scale
│   └── fonts
│       └── main-font.woff2  # Web font for styling
└── README.md          # Documentation for the project
```

## Setup Instructions
1. Clone the repository to your local machine.
2. Open `index.html` in a web browser to run the application.
3. Ensure your camera is enabled for the application to detect finger movements.

## Usage
- Click the "Start Camera" button to begin tracking your fingers.
- Move your fingers in front of the camera to play notes.
- Use the scale selector to change the musical scale.
- Adjust the volume slider to control the audio output.

## License
This project is licensed under the MIT License.