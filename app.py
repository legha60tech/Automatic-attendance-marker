
import streamlit as st
import os
import pandas as pd
import face_recognition
import pickle
from datetime import datetime
from PIL import Image
import numpy as np

# --------------------------
# FILE PATHS
# --------------------------
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings/encodings.pkl"
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("encodings", exist_ok=True)


# --------------------------
# LOAD / SAVE STUDENTS
# --------------------------
def load_students():
    if os.path.exists(STUDENTS_FILE):
        return pd.read_csv(STUDENTS_FILE, dtype={"RegisterNo": str})

    return pd.DataFrame(columns=[
        "Name", "RegisterNo", "College", "Department", "Year"
    ])


def save_students(df):
    df.to_csv(STUDENTS_FILE, index=False)


# --------------------------
# LOAD / SAVE ATTENDANCE
# --------------------------
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE, dtype={"RegisterNo": str})

    return pd.DataFrame(columns=[
        "Name", "RegisterNo", "Date", "Day", "Time"
    ])


def save_attendance(df):
    df.to_csv(ATTENDANCE_FILE, index=False)


# --------------------------
# REGISTER STUDENT
# --------------------------
def register_student(name, reg, college, dept, year, photo):

    df = load_students()

    if reg in df["RegisterNo"].values:
        return "Register Number already exists.", False

    # Read browser camera image
    image = face_recognition.load_image_file(photo)

    locations = face_recognition.face_locations(
        image, model="hog"
    )

    if len(locations) != 1:
        return "Please capture a photo containing exactly one face.", False

    folder = os.path.join(DATASET_DIR, reg)
    os.makedirs(folder, exist_ok=True)

    image_path = os.path.join(folder, "student.jpg")
    Image.fromarray(image).save(image_path)

    new_student = {
        "Name": name,
        "RegisterNo": reg,
        "College": college,
        "Department": dept,
        "Year": year
    }

    df = pd.concat(
        [df, pd.DataFrame([new_student])],
        ignore_index=True
    )

    save_students(df)

    return f"{name} registered successfully!", True


# --------------------------
# ENCODE ALL FACES
# --------------------------
def encode_all_faces():

    students = load_students()

    encodings = []
    ids = []

    for _, row in students.iterrows():

        reg = str(row["RegisterNo"])
        folder = os.path.join(DATASET_DIR, reg)

        image_path = os.path.join(folder, "student.jpg")

        if not os.path.exists(image_path):
            continue

        image = face_recognition.load_image_file(image_path)

        locations = face_recognition.face_locations(
            image, model="hog"
        )

        if len(locations) != 1:
            continue

        face_enc = face_recognition.face_encodings(
            image,
            known_face_locations=locations
        )[0]

        encodings.append(face_enc)
        ids.append(reg)

    if not encodings:
        return "No valid face images found.", False

    with open(ENCODINGS_FILE, "wb") as file:
        pickle.dump({
            "encodings": encodings,
            "names": ids
        }, file)

    return f"Successfully encoded {len(encodings)} students.", True


# --------------------------
# TAKE ATTENDANCE
# --------------------------
def take_attendance(photo):

    if not os.path.exists(ENCODINGS_FILE):
        return "Please encode faces first.", False

    with open(ENCODINGS_FILE, "rb") as file:
        data = pickle.load(file)

    known_encodings = data["encodings"]
    known_ids = data["names"]

    # Read captured image
    image = face_recognition.load_image_file(photo)

    locations = face_recognition.face_locations(
        image, model="hog"
    )

    if len(locations) != 1:
        return "Please capture exactly one face.", False

    face_encodings = face_recognition.face_encodings(
        image,
        known_face_locations=locations
    )

    if not face_encodings:
        return "Unable to recognize the face.", False

    face_encoding = face_encodings[0]

    distances = face_recognition.face_distance(
        known_encodings, face_encoding
    )

    best_match = np.argmin(distances)

    if distances[best_match] > 0.45:
        return "Face not recognized.", False

    reg = str(known_ids[best_match])

    students = load_students()
    attendance_df = load_attendance()

    student = students[
        students["RegisterNo"] == reg
    ]

    if student.empty:
        return "Student record not found.", False

    name = student.iloc[0]["Name"]

    now = datetime.now()
    today = now.strftime("%d-%m-%Y")

    already_marked = (
        (attendance_df["RegisterNo"] == reg) &
        (attendance_df["Date"] == today)
    ).any()

    if already_marked:
        return f"Attendance already marked for {name}.", False

    new_record = {
        "Name": name,
        "RegisterNo": reg,
        "Date": today,
        "Day": now.strftime("%A"),
        "Time": now.strftime("%I:%M:%S %p")
    }

    attendance_df = pd.concat(
        [attendance_df, pd.DataFrame([new_record])],
        ignore_index=True
    )

    save_attendance(attendance_df)

    return f"Attendance successfully marked for {name}!", True


# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(
    page_title="AI Face Attendance System",
    page_icon="📸",
    layout="centered"
)

st.title("📸 AI Face Attendance System")

st.write("Automatic Student Attendance Using Face Recognition")

menu = st.sidebar.radio(
    "Menu",
    [
        "Register Student",
        "Encode Faces",
        "Take Attendance",
        "View Attendance"
    ]
)


# --------------------------
# REGISTER PAGE
# --------------------------
if menu == "Register Student":

    st.subheader("📝 Register New Student")

    name = st.text_input("Student Name")
    reg = st.text_input("Register Number")
    college = st.text_input("College Name")
    dept = st.text_input("Department")
    year = st.text_input("Year of Study")

    photo = st.camera_input("Capture Student Photo")

    if st.button("Register Student"):

        if not all([name.strip(), reg.strip(),
                    college.strip(), dept.strip(),
                    year.strip()]):
            st.warning("Please fill in all student details.")

        elif photo is None:
            st.warning("Please capture a student photo.")

        else:
            msg, success = register_student(
                name.strip(),
                reg.strip(),
                college.strip(),
                dept.strip(),
                year.strip(),
                photo
            )

            if success:
                st.success(msg)
            else:
                st.error(msg)


# --------------------------
# ENCODE PAGE
# --------------------------
elif menu == "Encode Faces":

    st.subheader("🔍 Encode Student Faces")

    if st.button("Start Encoding"):

        msg, success = encode_all_faces()

        if success:
            st.success(msg)
        else:
            st.error(msg)


# --------------------------
# ATTENDANCE PAGE
# --------------------------
elif menu == "Take Attendance":

    st.subheader("📷 Mark Attendance")

    photo = st.camera_input("Capture Face for Attendance")

    if st.button("Mark Attendance"):

        if photo is None:
            st.warning("Please capture a photo first.")

        else:
            msg, success = take_attendance(photo)

            if success:
                st.success(msg)
            else:
                st.warning(msg)


# --------------------------
# VIEW ATTENDANCE
# --------------------------
elif menu == "View Attendance":

    st.subheader("📄 Attendance Records")

    df = load_attendance()

    st.dataframe(df, use_container_width=True)

    today = datetime.now().strftime("%d-%m-%Y")

    count_today = len(df[df["Date"] == today])

    st.info(f"Total Present Today: {count_today}")
