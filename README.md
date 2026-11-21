# AI-Based Hospital Assistance System Using Indian Sign Language Translation

## Overview

The **AI-Based Hospital Assistance System Using Indian Sign Language Translation** is a computer vision-based application designed to help patients with speech or hearing difficulties communicate more easily in healthcare environments.

The system uses real-time hand gesture recognition to identify Indian Sign Language gestures through a webcam. The recognized gestures are converted into text and spoken aloud using Text-to-Speech (TTS), enabling communication between patients and hospital staff.

The system also incorporates an emergency alert mechanism that sends an email notification when a predefined emergency gesture is detected.

It aims to improve accessibility, reduce communication barriers, and provide an additional means of conveying basic needs and emergency situations.

## Features

- **Real-Time Sign Language Recognition** – Detects hand gestures through a webcam.
- **Hand Landmark Detection** – Extracts hand coordinates using MediaPipe.
- **Machine Learning Classification** – Classifies hand landmarks using a trained model.
- **Gesture Stabilization** – Confirms gestures by maintaining the predicted sign for a specified duration to reduce false detections.
- **Text-to-Speech Conversion** – Converts recognized gestures into audible speech.
- **Emergency Email Alerts** – Sends an email notification when a predefined emergency gesture, such as "Call my family," is detected.
- **Visual Feedback** – Displays the recognized sign on the webcam feed.

## System Workflow

1. The webcam captures live video frames.
2. MediaPipe detects the hand and extracts its landmarks.
3. The landmark coordinates are processed and passed to the trained machine learning model.
4. The model predicts the corresponding sign-language gesture.
5. The predicted gesture is validated using a time-based stabilization mechanism.
6. Once confirmed:
   - The recognized gesture is displayed on the screen.
   - The gesture is converted into speech.
   - An email alert is triggered if the gesture represents an emergency.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand detection and landmark extraction |
| NumPy | Numerical operations and landmark processing |
| Scikit-learn | Machine learning classification |
| Joblib | Saving and loading trained models |
| gTTS | Text-to-speech conversion |
| Pygame | Audio playback |
| SMTP | Email notification system |

## Project Structure

```text
AI_Hospital_Sign_Assistant/
│
├── detection.py
├── train.py
├── landmark.py
├── asl_landmarks.csv
├── asl_model.pkl
├── label_encoder.pkl
├── demo.mp4
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tis123-boop/AI_Hospital_Sign_Assistant.git
```

### 2. Navigate to the project directory

```bash
cd AI_Hospital_Sign_Assistant
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Ensure that your webcam is connected and the trained model files are available in the project directory.

Run the real-time detection application:

```bash
python detection.py
```

The webcam window will display the detected hand landmarks and the recognized gesture.

Press **Q** to exit the application.

## Machine Learning Pipeline

The project follows a landmark-based gesture recognition approach.

### Data Processing

Hand landmarks are extracted from sign-language gestures using MediaPipe. Each landmark contains spatial coordinates that represent the position of key points on the hand.

### Model Training

The extracted landmark features are used to train a machine learning model to recognize different sign-language gestures.

### Prediction

During real-time operation, the webcam captures hand movements, extracts landmarks, and passes them to the trained model for gesture classification.

### Gesture Confirmation

A predicted gesture must remain stable for a specified duration before it is confirmed. This helps prevent accidental recognition caused by momentary changes in hand position.

## Emergency Alert Mechanism

The system includes an emergency communication feature.

When the predefined emergency gesture is recognized and confirmed:

- The detected emergency sign is identified.
- The current timestamp is recorded.
- An email notification is generated.
- The notification is sent to the configured recipient.

This feature is intended to help patients communicate urgent needs when verbal communication is difficult.

## Future Enhancements

- Expand the sign-language gesture vocabulary.
- Improve recognition under varying lighting and camera conditions.
- Support continuous gesture and sentence-level recognition.
- Add multilingual speech output.
- Develop a mobile-friendly interface.

## Disclaimer

This project is an assistive prototype intended for educational and demonstration purposes. It is not a replacement for professional interpreters, clinical assessment, or established hospital emergency communication systems.
