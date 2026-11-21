import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATASET_PATH = "asl_alphabet_train"
data = []
labels = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                data.append(landmark_list)
                labels.append(label)

print("Data Extracted Successfully")

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("asl_landmarks.csv", index=False)