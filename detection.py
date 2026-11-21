import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
from gtts import gTTS
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import time

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("asl_model.pkl")
le = joblib.load("label_encoder.pkl")

# ===============================
# INIT AUDIO
# ===============================
pygame.mixer.init()

# ===============================
# EMAIL CONFIG
# ===============================
SENDER_EMAIL = "bloghosting34@gmail.com"
APP_PASSWORD = "fcmipzqwssucvdtv"
RECEIVER_EMAIL = "rtsri1416@gmail.com"

EMERGENCY_SIGNS = ["Call my family"]
last_email_sent = ""

# ===============================
# HOLD SETTINGS
# ===============================
HOLD_TIME = 1  # seconds

current_sign = ""
sign_start_time = 0

# ===============================
# SPEAK FUNCTION (BLOCKING)
# ===============================
def speak(text):
    filename = "voice.mp3"

    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

    if os.path.exists(filename):
        os.remove(filename)

# ===============================
# EMAIL FUNCTION
# ===============================
def send_email(sign_name):
    global last_email_sent

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = MIMEText(f"""
Emergency Alert!

Detected Sign: {sign_name}
Time: {current_time}
""")

    msg['Subject'] = "Patient Emergency Alert"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()

        print("🚨 Email Sent")

    except Exception as e:
        print("❌ Email Error:", e)

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)

is_speaking = False
last_confirmed_sign = ""

# ===============================
# MAIN LOOP
# ===============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_sign = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([landmark_list])
            predicted_sign = le.inverse_transform(prediction)[0]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # ===============================
    # HOLD LOGIC (2 SECONDS)
    # ===============================
    if not is_speaking:

        if predicted_sign != "":

            if predicted_sign != current_sign:
                current_sign = predicted_sign
                sign_start_time = time.time()

            else:
                elapsed = time.time() - sign_start_time

                if elapsed >= HOLD_TIME:

                    if current_sign != last_confirmed_sign:

                        print("Detected Sign:", current_sign)

                        is_speaking = True  # 🔒 lock

                        speak(current_sign)

                        # Email alert
                        if current_sign in EMERGENCY_SIGNS:
                            if current_sign != last_email_sent:
                                send_email(current_sign)
                                last_email_sent = current_sign

                        last_confirmed_sign = current_sign
                        is_speaking = False  # 🔓 unlock

                        current_sign = ""
                        sign_start_time = 0

    # ===============================
    # DISPLAY
    # ===============================
    display_text = current_sign if current_sign != "" else last_confirmed_sign

    cv2.putText(frame,
                display_text,
                (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.imshow("Patient Sign Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
pygame.quit()