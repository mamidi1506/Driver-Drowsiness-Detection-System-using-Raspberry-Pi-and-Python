import cv2

import numpy as np

import dlib

from imutils import face_utils

from pygame import mixer 

from tkinter import *

from PIL import Image, ImageTk

import tkinter.messagebox as msg


def a():
    msg.showinfo("Note","SYSTEM ACTIVATED...!!!")

    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    sleep = 0
    drowsy = 0
    active = 0
    status = ""
    color = (0, 0, 0)

    mixer.init()
    mixer.music.load("alarm.wav")

    def compute(ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(a, b, c, d, e, f):
        up = compute(b, d) + compute(c, e)
        down = compute(a, f)
        ratio = up / (2.0 * down)

        if (ratio > 0.25):
            return 2
        elif (ratio > 0.21 and ratio <= 0.25):
            return 1
        else:
            return 0

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        face_frame = frame.copy()

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if (left_blink == 0 or right_blink == 0):
                sleep += 1
                drowsy = 0
                active = 0
                if (sleep > 15):
                    status = "Sleeping...!!!"
                    mixer.music.play()
                    color = (128, 0, 0)

            elif (left_blink == 1 or right_blink == 1):
                sleep = 0
                active = 0
                drowsy += 1
                mixer.music.stop()
                if (drowsy > 6):
                    status = "Drawsy..."
                    color = (225, 0, 255)

            else:
                drowsy = 0
                sleep = 0
                active += 1
                mixer.music.stop()
                if (active > 6):
                    status = "Active"
                    color = (0, 255, 0)

            cv2.putText(frame, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)
        cv2.waitKey(1)



root = Tk()
root.geometry("1200x900")
root.configure(bg="black")

label1 = Label(root, text="Driver Drowsiness Detection", bg="cyan", font="sail 30 bold", relief=SUNKEN)
label1.pack(fill=X, pady=15)

image = Image.open("DDD_image.jpg")
photo = ImageTk.PhotoImage(image)
label = Label(image=photo)
label.pack(pady = 20)

f1 = Frame(root, bg="black",relief=SUNKEN)
f1.pack(fill=X, pady=20,padx=10)

b1 = Button(f1, text="ACTIVATE", font=" sail 25 bold", command=a, bg="red", borderwidth=10, fg="white", relief=RAISED)
b1.pack(side=TOP, padx=20,pady = 40)


root.mainloop()