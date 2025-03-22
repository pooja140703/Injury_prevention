# app.py
from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load ideal pose dataset
ideal_pose_df = pd.read_csv('body_landmarks_dataset.csv')  # Ensure this file exists

mp_pose = mp.solutions.pose
THRESHOLD = 0.1  # Adjust for tolerance

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def check_pose(landmarks, ideal_pose_df):
    total_difference = 0
    for i, landmark in enumerate(landmarks):
        real_x, real_y = landmark.x, landmark.y
        ideal_x = ideal_pose_df.iloc[0][f"x_{i}"]
        ideal_y = ideal_pose_df.iloc[0][f"y_{i}"]
        total_difference += euclidean_distance((real_x, real_y), (ideal_x, ideal_y))
    avg_difference = total_difference / len(landmarks)
    return avg_difference > 0.2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame'].read()
    nparr = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    with mp_pose.Pose(static_image_mode=True) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            pose_correct = check_pose(results.pose_landmarks.landmark, ideal_pose_df)
            confidence_score = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
            return jsonify({'pose_correct': pose_correct, 'confidence_score': confidence_score})

    return jsonify({'pose_correct': False, 'confidence_score': 0})

if __name__ == '__main__':
    app.run(debug=True)
