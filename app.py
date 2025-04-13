import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import os
import sqlite3
import csv
from datetime import datetime

# Database setup
conn = sqlite3.connect('attendance.db')
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        time TEXT
    )
""")
conn.commit()

# Load known faces
known_encodings = []
known_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(f"known_faces/{filename}")
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])

# Mark attendance
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    c.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if c.fetchone() is None:
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
        conn.commit()
        print(f"[INFO] Marked attendance for {name}")

# Recognition and attendance
def recognize_face():
    cap = cv2.VideoCapture(0)
    recognized = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                if name not in recognized:
                    mark_attendance(name)
                    recognized.add(name)

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Face Recognition - Press 'q' to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Export attendance to CSV
def export_csv():
    try:
        with open("attendance_export.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Name", "Date", "Time"])
            for row in c.execute("SELECT * FROM attendance"):
                writer.writerow(row)
        messagebox.showinfo("Export Success", "Attendance exported to attendance_export.csv")
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export: {e}")

# Tkinter UI
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("300x250")

label = tk.Label(root, text="Attendance Tracker", font=("Helvetica", 16))
label.pack(pady=20)

start_button = tk.Button(root, text="Start Recognition", command=recognize_face)
start_button.pack(pady=10)

export_button = tk.Button(root, text="Export to CSV", command=export_csv)
export_button.pack(pady=5)

quit_button = tk.Button(root, text="Quit", command=root.quit)
quit_button.pack(pady=5)

root.mainloop()
