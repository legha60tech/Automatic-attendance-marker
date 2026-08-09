import streamlit as st
import cv2
import os
import pandas as pd
import face_recognition
import pickle
from datetime import datetime
import pyttsx3

# --------------------------
# FILE PATHS
# --------------------------
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings/encodings.pkl"
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs("dataset", exist_ok=True)
os.makedirs("encodings", exist_ok=True)

# --------------------------
# TEXT TO SPEECH
# --------------------------
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)  # Zira female voice


def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass


# --------------------------
# LOAD/SAVE STUDENTS
# --------------------------
def load_students():
    if os.path.exists(STUDENTS_FILE):
        return pd.read_csv(STUDENTS_FILE)
    return pd.DataFrame(columns=["Name", "RegisterNo", "College", "Department", "Year"])


def save_students(df):
    df.to_csv(STUDENTS_FILE, index=False)


# --------------------------
# LOAD/SAVE ATTENDANCE
# --------------------------
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)
    return pd.DataFrame(columns=["Name", "RegisterNo", "Date", "Day", "Time"])


def save_attendance(df):
    df.to_csv(ATTENDANCE_FILE, index=False)


# --------------------------
# REGISTER STUDENT
# --------------------------
def register_student(name, reg, college, dept, year):
    df = load_students()

    if reg in df["RegisterNo"].values:
        return "⚠ Register Number already exists."

    new = {
        "Name": name,
        "RegisterNo": reg,
        "College": college,
        "Department": dept,
        "Year": year
    }

    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    save_students(df)

    folder = os.path.join(DATASET_DIR, reg)
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    st.info("📸 Capturing 20 images. Please look at the camera.")

    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(os.path.join(folder, f"{count}.jpg"), frame)
        count += 1

    cap.release()

    speak("Registration completed successfully.")

    return f"✔ {name} Registered Successfully!"


# --------------------------
# ENCODE ALL FACES
# --------------------------
def encode_all_faces():
    students = load_students()
    encodings = []
    ids = []

    for idx, row in students.iterrows():
        reg = row["RegisterNo"]
        folder = os.path.join(DATASET_DIR, reg)

        if not os.path.exists(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = face_recognition.load_image_file(img_path)
            loc = face_recognition.face_locations(img, model="hog")

            if len(loc) == 0:
                continue

            enc = face_recognition.face_encodings(
                img,
                known_face_locations=loc,
                model="small"
            )[0]

            encodings.append(enc)
            ids.append(reg)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": ids}, f)

    return "✔ Encoding Completed Successfully."


# --------------------------
# TAKE ATTENDANCE
# --------------------------
def take_attendance():
    if not os.path.exists(ENCODINGS_FILE):
        return "⚠ No encodings found. Please encode faces first."

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_ids = data["names"]

    students = load_students()
    attendance_df = load_attendance()

    cap = cv2.VideoCapture(0)
    st.info("Camera started. Please look at the screen.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "⚠ Camera Error."

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb, model="hog")

    if len(faces) == 0:
        return "⚠ No face detected."

    encs = face_recognition.face_encodings(
        rgb,
        known_face_locations=faces,
        model="small"
    )

    if len(encs) == 0:
        return "⚠ Unable to generate encoding."

    face_enc = encs[0]

    matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.45)

    if True in matches:
        idx = matches.index(True)
        reg = known_ids[idx]

        student = students[students["RegisterNo"] == reg].iloc[0]
        name = student["Name"]

        today = datetime.now().strftime("%d-%m-%Y")
        day = datetime.now().strftime("%A")
        time_now = datetime.now().strftime("%I:%M:%S %p")

        if ((attendance_df["RegisterNo"] == reg) & (attendance_df["Date"] == today)).any():
            speak(f"{name}, your attendance is already marked.")
            return f"✔ Attendance already marked for {name}."

        new = {
            "Name": name,
            "RegisterNo": reg,
            "Date": today,
            "Day": day,
            "Time": time_now
        }

        attendance_df = pd.concat([attendance_df, pd.DataFrame([new])], ignore_index=True)
        save_attendance(attendance_df)

        speak(f"{name}, your attendance has been marked successfully.")

        return f"🎉 Attendance marked for {name}."

    else:
        return "❌ Face not recognized."


# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("📸 AI Face Attendance System")

menu = st.sidebar.radio("Menu", ["Register Student", "Encode Faces", "Take Attendance", "View Attendance"])

# Register Page
if menu == "Register Student":
    st.subheader("📝 Register New Student")

    name = st.text_input("Student Name")
    reg = st.text_input("Register Number")
    college = st.text_input("College Name")
    dept = st.text_input("Department")
    year = st.text_input("Year of Study")

    if st.button("Register"):
        msg = register_student(name, reg, college, dept, year)
        st.success(msg)

# Encode
if menu == "Encode Faces":
    if st.button("Start Encoding"):
        msg = encode_all_faces()
        st.success(msg)

# Attendance
if menu == "Take Attendance":
    if st.button("Start Camera"):
        msg = take_attendance()
        st.success(msg)

# View Attendance
if menu == "View Attendance":
    st.subheader("📄 Attendance Records")
    df = load_attendance()
    st.dataframe(df)

    today = datetime.now().strftime("%d-%m-%Y")
    count_today = len(df[df["Date"] == today])
    st.info(f"Total Present Today: {count_today}")
