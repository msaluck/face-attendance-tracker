import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime

# Constants
ENCODINGS_PATH = "encodings.pkl"
ATTENDANCE_PATH = "attendance.csv"

# Initialize face data
known_face_encodings = []
known_face_names = []


# Load saved encodings
def load_encodings():
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    print("[DEBUG] Loaded names:", known_face_names)
    print("[DEBUG] Loaded encodings:", len(known_face_encodings), "encodings loaded.")


# Save updated encodings
def save_encodings():
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)


# Mark attendance
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_PATH, "a") as f:
        f.write(f"{name},{dt_string}\n")


# Recognize face from webcam
def recognize_face():
    cap = cv2.VideoCapture(0)
    recognized_names = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(
                known_face_encodings, encoding
            )
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name not in recognized_names:
                        mark_attendance(name)
                        recognized_names.add(name)
            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# Register face from webcam
def register_face():
    name = simpledialog.askstring("Register", "Enter your name:")
    if not name:
        return
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if boxes:
            encodings = face_recognition.face_encodings(rgb, boxes)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                save_encodings()
                messagebox.showinfo("Success", f"{name} registered successfully.")
                break
        cv2.imshow("Register Face - Press 'q' to Cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# Register face from image
def register_face_from_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    if not file_path:
        return
    name = simpledialog.askstring("Register from Image", "Enter your name:")
    if not name:
        return
    image = cv2.imread(file_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if not boxes:
        messagebox.showerror("Error", "No face detected in the image.")
        return
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        messagebox.showerror("Error", "Could not extract encoding from face.")
        return
    known_face_encodings.append(encodings[0])
    known_face_names.append(name)
    save_encodings()
    messagebox.showinfo("Success", f"{name} registered successfully from image.")


# Recognize face from image
def recognize_from_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    if not file_path:
        return
    image = cv2.imread(file_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)
        top, right, bottom, left = box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            image,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
    cv2.imshow("Recognize from Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# UI Setup
load_encodings()
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("400x350")

btn_recognize = tk.Button(
    root, text="Recognize Face (Webcam)", command=recognize_face, width=30, pady=10
)
btn_register = tk.Button(
    root, text="Register Face (Webcam)", command=register_face, width=30, pady=10
)
btn_register_image = tk.Button(
    root,
    text="Register from Image",
    command=register_face_from_image,
    width=30,
    pady=10,
)
btn_recognize_image = tk.Button(
    root, text="Recognize from Image", command=recognize_from_image, width=30, pady=10
)
btn_exit = tk.Button(root, text="Exit", command=root.destroy, width=30, pady=10)

btn_recognize.pack(pady=5)
btn_register.pack(pady=5)
btn_register_image.pack(pady=5)
btn_recognize_image.pack(pady=5)
btn_exit.pack(pady=5)

root.mainloop()
