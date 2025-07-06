import cv2
import numpy as np
import dlib
from imutils import face_utils
from pygame import mixer
from tkinter import *
from PIL import Image, ImageTk
import tkinter.messagebox as msg
import os
import datetime
import csv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Global control flag and driver info
running = True
driver_id = ""

def generate_pdf(driver_id, drowsy_count, sleep_count, driver_class):
    filename = f"report_{driver_id}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, f"Driver Drowsiness Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Driver ID: {driver_id}")
    c.drawString(100, 700, f"Drowsy Detections: {drowsy_count}")
    c.drawString(100, 680, f"Sleep Detections: {sleep_count}")
    c.drawString(100, 660, f"Total Alerts: {drowsy_count + sleep_count}")
    c.drawString(100, 640, f"Driver Classification: {driver_class}")
    c.drawString(100, 600, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()

def a():
    global running, driver_id
    running = True
    driver_id = entry_driver.get().strip()

    if not driver_id:
        msg.showerror("Missing Info", "Please enter Driver ID")
        return

    msg.showinfo("Note", "SYSTEM ACTIVATED...!!!")

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

    save_dir = f"drowsiness_records/{driver_id}"
    os.makedirs(save_dir, exist_ok=True)
    drowsy_count = 0
    sleep_count = 0

    def compute(ptA, ptB):
        return np.linalg.norm(ptA - ptB)

    def blinked(a, b, c, d, e, f):
        up = compute(b, d) + compute(c, e)
        down = compute(a, f)
        ratio = up / (2.0 * down)
        if ratio > 0.25:
            return 2
        elif 0.21 < ratio <= 0.25:
            return 1
        else:
            return 0

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            face_frame = frame.copy()

            for face in faces:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()

                cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

                if left_blink == 0 or right_blink == 0:
                    sleep += 1
                    drowsy = 0
                    active = 0
                    if sleep > 15:
                        status = "Sleeping...!!!"
                        mixer.music.play()
                        color = (128, 0, 0)
                        sleep_count += 1
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(os.path.join(save_dir, f"sleep_{timestamp}.jpg"), frame)

                elif left_blink == 1 or right_blink == 1:
                    sleep = 0
                    active = 0
                    drowsy += 1
                    mixer.music.stop()
                    if drowsy > 6:
                        status = "Drawsy..."
                        color = (225, 0, 255)
                        drowsy_count += 1
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(os.path.join(save_dir, f"drowsy_{timestamp}.jpg"), frame)

                else:
                    drowsy = 0
                    sleep = 0
                    active += 1
                    mixer.music.stop()
                    if active > 6:
                        status = "Active"
                        color = (0, 255, 0)

                cv2.putText(frame, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                for n in range(0, 68):
                    (x, y) = landmarks[n]
                    cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

            cv2.imshow("Frame", frame)
            cv2.imshow("Result of detector", face_frame)
            cv2.waitKey(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()

        total_alerts = drowsy_count + sleep_count
        driver_class = "Good" if total_alerts <= 5 else "Bad"

        # Save to CSV
        with open("driver_stats.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([driver_id, drowsy_count, sleep_count, total_alerts, driver_class])

        # Generate PDF
        generate_pdf(driver_id, drowsy_count, sleep_count, driver_class)

        # Show result
        msg.showinfo("Session Ended",
                     f"Driver ID: {driver_id}\nDrowsy: {drowsy_count}\nSleep: {sleep_count}\nTotal: {total_alerts}\nClass: {driver_class}\nPDF saved as report_{driver_id}.pdf")

def stop():
    global running
    running = False

# ========== GUI ==========

root = Tk()
root.geometry("700x500")
root.configure(bg="black")
root.title("Drowsiness Detection")

label1 = Label(root, text="Driver Drowsiness Detection", bg="cyan", font="sail 20 bold", relief=SUNKEN)
label1.pack(fill=X, pady=15)

try:
    image = Image.open("DDD_image.jpg")
    image = image.resize((200, 150))
    photo = ImageTk.PhotoImage(image)
    label_img = Label(image=photo)
    label_img.pack(pady=10)
except:
    pass

f1 = Frame(root, bg="black", relief=SUNKEN)
f1.pack(pady=10)

label_driver = Label(f1, text="Driver ID:", bg="black", fg="white", font="Arial 14")
label_driver.pack(side=LEFT, padx=5)
entry_driver = Entry(f1, font="Arial 14", width=20)
entry_driver.pack(side=LEFT, padx=5)

f2 = Frame(root, bg="black")
f2.pack(pady=20)

b1 = Button(f2, text="ACTIVATE", font="Arial 16 bold", command=a, bg="red", fg="white", width=10)
b1.pack(side=LEFT, padx=10)
b2 = Button(f2, text="STOP", font="Arial 16 bold", command=stop, bg="gray", fg="white", width=10)
b2.pack(side=LEFT, padx=10)

root.mainloop()
