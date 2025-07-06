import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
from playsound import playsound
import threading
import time

import os
import csv
from datetime import datetime

# Function to compute eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds and consecutive frame count for drowsiness
eye_ar_thresh = 0.25
eye_ar_consec_frames = 20
counter = 0
alarm_on = False

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start video stream
vs = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = vs.read()
    if not ret or frame is None:
        print("[WARNING] Empty frame received.")
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.dtype != 'uint8':
        gray = gray.astype('uint8')

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < eye_ar_thresh:
            counter += 1

            if counter >= eye_ar_consec_frames:
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=playsound, args=('alarm.wav',), daemon=True).start()

                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            counter = 0
            alarm_on = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()