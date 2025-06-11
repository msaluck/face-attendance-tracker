import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import cv2
import face_recognition
import os
import numpy as np
import pickle
import csv
from datetime import datetime

# File paths
ENCODINGS_PATH = "encodings.pkl"
ATTENDANCE_PATH = "attendance.csv"

# Load encodings
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

# Create attendance CSV if not exists
if not os.path.exists(ATTENDANCE_PATH):
    with open(ATTENDANCE_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time"])


def mark_attendance(name):
    with open(ATTENDANCE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, time_str])


def recognize_from_source(source):
    if isinstance(source, str):  # Image file
        image = cv2.imread(source)
    else:  # Webcam
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cap.release()
    if image is None:
        messagebox.showerror("Error", "Could not load image or webcam.")
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names_found = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names_found.append(name)
        if name != "Unknown":
            mark_attendance(name)
    if names_found:
        messagebox.showinfo("Recognized", f"Recognized: {', '.join(names_found)}")
    else:
        messagebox.showinfo("No Faces", "No known faces recognized.")


def register_from_source(source):
    name = simpledialog.askstring("Register Face", "Enter your name:")
    if not name:
        return
    if isinstance(source, str):
        image = cv2.imread(source)
    else:
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cap.release()
    if image is None:
        messagebox.showerror("Error", "Could not load image or webcam.")
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        messagebox.showwarning("No Face", "No face found in the image.")
        return
    data["encodings"].extend(encodings)
    data["names"].extend([name] * len(encodings))
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)
    messagebox.showinfo("Success", f"Registered {name} with {len(encodings)} face(s).")


def recognize_face():
    source = tk.messagebox.askquestion(
        "Input Method", "Use webcam? Click 'Yes' for webcam or 'No' to upload an image."
    )
    if source == "yes":
        recognize_from_source(None)
    else:
        path = filedialog.askopenfilename()
        if path:
            recognize_from_source(path)


def register_face():
    source = tk.messagebox.askquestion(
        "Input Method", "Use webcam? Click 'Yes' for webcam or 'No' to upload an image."
    )
    if source == "yes":
        register_from_source(None)
    else:
        path = filedialog.askopenfilename()
        if path:
            register_from_source(path)


def view_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        messagebox.showinfo("No Data", "No attendance data available.")
        return
    window = tk.Toplevel(root)
    window.title("Attendance Logs")
    tree = ttk.Treeview(window, columns=("Name", "Time"), show="headings")
    tree.heading("Name", text="Name")
    tree.heading("Time", text="Time")
    tree.pack(fill="both", expand=True)
    with open(ATTENDANCE_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            tree.insert("", "end", values=row)


def export_attendance():
    path = filedialog.asksaveasfilename(
        defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
    )
    if path:
        with open(ATTENDANCE_PATH, "r") as src:
            with open(path, "w", newline="") as dst:
                dst.write(src.read())
        messagebox.showinfo("Exported", f"Attendance exported to {path}")


# UI Setup
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("400x300")

btn_register = tk.Button(root, text="Register Face", command=register_face, width=25)
btn_register.pack(pady=10)
btn_recognize = tk.Button(root, text="Recognize Face", command=recognize_face, width=25)
btn_recognize.pack(pady=10)
btn_view = tk.Button(
    root, text="View Attendance Logs", command=view_attendance, width=25
)
btn_view.pack(pady=10)
btn_export = tk.Button(
    root, text="Export Attendance Report", command=export_attendance, width=25
)
btn_export.pack(pady=10)

root.mainloop()
