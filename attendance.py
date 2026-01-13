import cv2
import os
import csv
import time
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from tkcalendar import DateEntry
from PIL import Image
import numpy as np
from datetime import datetime
from ttkthemes import ThemedStyle   # 🔹 NEW for modern ttk styles

# ------------------- Paths -------------------
DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
ATTENDANCE_DIR = "attendance"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Haarcascade path
haarcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(haarcascadePath)

# ------------------- Timetable -------------------
timetable = {
    "Monday": [
        ("09:00", "10:00", "OS"),
        ("10:00", "11:00", "DWDM"),
        ("11:00", "12:00", "ADMIN LAB"),
        ("13:00", "14:00", "IS"),
        ("14:00", "15:00", "DMGT"),
        ("15:00", "16:00", "IOT"),
    ],
    "Tuesday": [
        ("09:00", "10:00", "EE"),
        ("10:00", "11:00", "SE"),
        ("11:00", "12:00", "TDPC"),
        ("13:00", "14:00", "DWDM"),
        ("14:00", "15:00", "IS"),
        ("15:00", "16:00", "COA"),
    ],
    "Wednesday": [
        ("09:00", "10:00", "OS"),
        ("10:00", "11:00", "TST"),
        ("11:00", "12:00", "ADMIN LAB"),
        ("13:00", "14:00", "MENTORING"),
        ("14:00", "15:00", "DMGT"),
        ("15:00", "16:00", "IS"),
    ],
    "Thursday": [
        ("09:00", "10:00", "OS"),
        ("10:00", "11:00", "DWDM"),
        ("11:00", "12:00", "IOT"),
        ("13:00", "14:00", "SE"),
        ("14:00", "15:00", "DMGT"),
        ("15:00", "16:00", "COA"),
    ],
    "Friday": [
        ("09:00", "10:00", "OS"),
        ("10:00", "11:00", "DWDM"),
        ("11:00", "12:50", "ADMIN LAB"),
        ("13:00", "14:00", "LIB"),
        ("14:00", "15:00", "DMGT"),
        ("15:00", "16:00", "IOT"),
    ],
    "Saturday": [
        ("09:00", "10:00", "EE"),
        ("10:00", "11:00", "DWDM"),
        ("11:00", "12:00", "IS"),
        ("13:00", "14:00", "SE"),
        ("14:00", "15:00", "COA"),
        ("15:00", "16:00", "DMGT"),
    ]
}

all_subjects = sorted({subj for day in timetable.values() for _, _, subj in day})

def get_current_subject():
    now = datetime.now()
    weekday = now.strftime("%A")
    current_time = now.strftime("%H:%M")

    if weekday in timetable:
        for start, end, subject in timetable[weekday]:
            if start <= current_time < end:
                return subject
    return None

# ------------------- Functions -------------------
def take_images(user_id, name):
    cam = cv2.VideoCapture(0)
    count = 0
    directions = ["Front", "Left", "Right"]

    for direction in directions:
        start_time = time.time()
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            elapsed = int(time.time() - start_time)
            remaining = 10 - elapsed
            cv2.putText(img, f"{direction} - Capture in {remaining}s",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Capture", img)

            if remaining <= 0:
                for (x, y, w, h) in faces:
                    count += 1
                    face_img = gray[y:y+h, x:x+w]
                    file_path = os.path.join(DATASET_DIR, f"{name}.{user_id}.{count}.jpg")
                    cv2.imwrite(file_path, face_img)
                break

            if cv2.waitKey(100) & 0xFF == 27:
                break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Images saved for {name}")

    with open("students.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([user_id, name])

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".jpg"):
            img_path = os.path.join(DATASET_DIR, file)
            img = Image.open(img_path).convert("L")
            np_img = np.array(img, "uint8")
            user_id = int(file.split(".")[1])
            faces.append(np_img)
            ids.append(user_id)

    recognizer.train(faces, np.array(ids))
    recognizer.save(os.path.join(TRAINER_DIR, "trainer.yml"))
    messagebox.showinfo("Training", "Training completed successfully!")

def mark_attendance():
    subject = get_current_subject()
    if not subject:
        messagebox.showerror("Error", "No subject scheduled at this time.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(TRAINER_DIR, "trainer.yml"))

    cam = cv2.VideoCapture(0)
    students = {}
    if os.path.exists("students.csv"):
        with open("students.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].isdigit():
                    students[int(row[0])] = row[1]

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(ATTENDANCE_DIR, f"Attendance_{subject}_{date_str}.csv")

    already_marked = set()
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            already_marked = {row[0] for row in reader}

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 60:
                name = students.get(id_, "Unknown")
                if str(id_) not in already_marked:
                    with open(filename, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([id_, name, datetime.now().strftime("%H:%M:%S")])
                    already_marked.add(str(id_))

                cv2.putText(img, f"{name} ({id_}) - Marked Present",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Attendance", img)
        if cv2.waitKey(100) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

def delete_user(user_id):
    new_rows = []
    with open("students.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] != str(user_id):
                new_rows.append(row)

    with open("students.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    for file in os.listdir(DATASET_DIR):
        if file.split(".")[1] == str(user_id):
            os.remove(os.path.join(DATASET_DIR, file))

    messagebox.showinfo("Delete", f"User {user_id} deleted successfully!")

def view_attendance(date_str, subject):
    for row in attendance_tree.get_children():
        attendance_tree.delete(row)

    filename = os.path.join(ATTENDANCE_DIR, f"Attendance_{subject}_{date_str}.csv")

    # Load all students
    all_students = {}
    if os.path.exists("students.csv"):
        with open("students.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    all_students[row[0]] = row[1]

    # Load present students
    present_students = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    present_students[row[0]] = (row[1], row[2])

    # Insert Present
    for sid, (name, time) in present_students.items():
        attendance_tree.insert("", "end", values=(sid, name, time, "Present"))

    # Insert Absent (students not in present list)
    for sid, name in all_students.items():
        if sid not in present_students:
            attendance_tree.insert("", "end", values=(sid, name, "-", "Absent"))

def view_users():
    for row in user_tree.get_children():
        user_tree.delete(row)

    if not os.path.exists("students.csv"):
        return

    with open("students.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                user_tree.insert("", "end", values=(row[0], row[1]))

# ------------------- UI -------------------

root = ctk.CTk()
root.title("Smart Attendance System (Auto-Subject)")
root.geometry("1000x700")

# Apply ttkthemes to ttk widgets
style = ThemedStyle(root)
style.set_theme("equilux")   # 🔹 Options: "arc", "plastik", "clearlooks", "radiance", "breeze"

tabview = ctk.CTkTabview(root, width=950, height=650)
tabview.pack(pady=20, padx=20, fill="both", expand=True)

# Tab: Add Student
tab_add = tabview.add("Add Student")
ctk.CTkLabel(tab_add, text="Student ID").pack(pady=5)
entry_id = ctk.CTkEntry(tab_add)
entry_id.pack(pady=5)
ctk.CTkLabel(tab_add, text="Student Name").pack(pady=5)
entry_name = ctk.CTkEntry(tab_add)
entry_name.pack(pady=5)

def handle_add():
    if entry_id.get() and entry_name.get():
        take_images(entry_id.get(), entry_name.get())
    else:
        messagebox.showerror("Error", "Enter ID and Name")

ctk.CTkButton(tab_add, text="Capture Images", command=handle_add).pack(pady=10)
ctk.CTkButton(tab_add, text="Train Model", command=train_model).pack(pady=10)

# Tab: Attendance
tab_att = tabview.add("Mark Attendance")
ctk.CTkLabel(tab_att, text="Attendance will be auto-marked based on current subject & time").pack(pady=10)
ctk.CTkButton(tab_att, text="Start Attendance", command=mark_attendance).pack(pady=20)

# Tab: Delete User
tab_delete = tabview.add("Delete User")
ctk.CTkLabel(tab_delete, text="Enter Student ID to Delete").pack(pady=5)
delete_id = ctk.CTkEntry(tab_delete)
delete_id.pack(pady=5)
ctk.CTkButton(tab_delete, text="Delete User", command=lambda: delete_user(delete_id.get())).pack(pady=10)

# Tab: View Attendance
tab_view = tabview.add("View Attendance")
ctk.CTkLabel(tab_view, text="Select Date").pack(pady=5)
entry_date = DateEntry(tab_view, width=12, background="darkblue", foreground="white", borderwidth=2, date_pattern="yyyy-mm-dd")
entry_date.pack(pady=5)

ctk.CTkLabel(tab_view, text="Select Subject").pack(pady=5)
subject_var = tk.StringVar()
subject_dropdown = ttk.Combobox(tab_view, textvariable=subject_var, values=all_subjects, state="readonly")
subject_dropdown.pack(pady=5)

attendance_tree = ttk.Treeview(tab_view, columns=("ID", "Name", "Time", "Status"), show="headings")
attendance_tree.heading("ID", text="ID")
attendance_tree.heading("Name", text="Name")
attendance_tree.heading("Time", text="Time")
attendance_tree.heading("Status", text="Status")
attendance_tree.pack(fill="both", expand=True, padx=20, pady=20)

def handle_view():
    date_str = entry_date.get()
    subject = subject_var.get()
    if date_str and subject:
        view_attendance(date_str, subject)
    else:
        messagebox.showerror("Error", "Select Date and Subject")

ctk.CTkButton(tab_view, text="Load Attendance", command=handle_view).pack(pady=10)

# Tab: View Users
tab_users = tabview.add("View Students")
user_tree = ttk.Treeview(tab_users, columns=("ID", "Name"), show="headings")
user_tree.heading("ID", text="ID")
user_tree.heading("Name", text="Name")
user_tree.pack(fill="both", expand=True, padx=20, pady=20)
ctk.CTkButton(tab_users, text="Refresh", command=view_users).pack(pady=10)

# ------------------- Run -------------------
root.mainloop()