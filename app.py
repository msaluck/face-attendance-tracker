import cv2
import face_recognition
import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import pickle
import numpy as np
import datetime

# Load known faces if exists
known_encodings = []
known_names = []
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]

# Save attendance
attendance_file = "attendance.csv"


def mark_attendance(name):
    now = datetime.datetime.now()
    with open(attendance_file, "a") as f:
        f.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")


# Register new face
def register_face():
    name = simpledialog.askstring("Register Face", "Enter name:")
    if not name:
        return

    encodings = []
    cap = cv2.VideoCapture(0)
    registered = 0

    messagebox.showinfo(
        "Registering", f"Please look at the camera. Registering {name}..."
    )

    while registered < 5:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        if boxes:
            encoding = face_recognition.face_encodings(rgb, boxes)[0]
            encodings.append(encoding)
            registered += 1
            cv2.putText(
                frame,
                f"Sample {registered}/5",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Register Face", frame)
            cv2.waitKey(1000)

    cap.release()
    cv2.destroyAllWindows()

    known_names.extend([name] * len(encodings))
    known_encodings.extend(encodings)

    with open("known_faces.pkl", "wb") as f:
        pickle.dump({"names": known_names, "encodings": known_encodings}, f)

    messagebox.showinfo("Success", f"{name} has been registered.")


# Face recognition
def recognize_faces():
    cap = cv2.VideoCapture(0)
    recognized = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for i, b in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    name = known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

                if name not in recognized:
                    mark_attendance(name)
                    recognized.add(name)

            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# GUI setup
root = tk.Tk()
root.title("Face Attendance App")

btn_register = tk.Button(root, text="Register Face", command=register_face)
btn_register.pack(pady=10)

btn_recognize = tk.Button(root, text="Start Recognition", command=recognize_faces)
btn_recognize.pack(pady=10)

checkbox_vars = [tk.BooleanVar() for _ in range(4)]
checkbox_labels = ["Option A", "Option B", "Option C", "Option D"]

for i, label in enumerate(checkbox_labels):
    cb = tk.Checkbutton(root, text=label, variable=checkbox_vars[i])
    cb.pack()

root.mainloop()
