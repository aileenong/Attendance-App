# - Authentication (signup/login), profiles with roles
# - Admin dashboard for role management
# - Employee registration, sample capture, retraining to Supabase Storage
# - Face attendance with clock‑in/out confirmation
# - Timesheet calculation and reporting dashboard

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
from datetime import datetime, date, timedelta
from supabase import create_client, Client
import time

# -------------------------------
# Config & setup
# -------------------------------
st.set_page_config(page_title="Face Attendance (Supabase)", layout="wide")

# Secrets (Service Role key required for storage + privileged ops)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Paths and constants
DATA_DIR = "data/faces"
LOCAL_MODELS_DIR = "models"
LBPH_MODEL_FILENAME = "lbph_model.xml"
LABEL_MAP_FILENAME = "label_to_empid.npy"
STORAGE_MODELS_BUCKET = "models"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# LBPH confidence: lower is better match; adjust as needed
CONFIDENCE_THRESHOLD = 70.0

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

ensure_dirs()

# -------------------------------
# Auth: signup, login, role helpers
# -------------------------------
def signup_ui():
    st.subheader("Sign up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    empid = st.text_input("Employee ID")
    name = st.text_input("Full name")

    if st.button("Create account"):
        if not email or not password or not empid or not name:
            st.error("Fill in all fields.")
            return
        try:
            res = supabase.auth.sign_up({"email": email, "password": password})
            user = res.user
            if user:
                # Create profile with default role employee
                supabase.table("profiles").insert({
                    "id": user.id,
                    "role": "employee",
                    "employee_id": empid.upper().strip(),
                    "name": name.strip()
                }).execute()
                # Ensure users table mirrors for attendance linkage
                try:
                    supabase.table("users").insert({
                        "employee_id": empid.upper().strip(),
                        "name": name.strip()
                    }).execute()
                except Exception:
                    pass
                st.success("Account created. You can now log in.")
        except Exception as e:
            st.error(f"Sign up failed: {e}")

def login_ui():
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            user = res.user
            if user:
                st.session_state["user"] = user
                st.success("Logged in successfully")
                # Small delay so Streamlit rerun catches the new session state
                time.sleep(1)
                st.rerun()

        except Exception as e:
            st.error(f"Login failed: {e}")

def logout_ui():
    if st.sidebar.button("Logout"):
        st.session_state.pop("user", None)
        st.success("Logged out.")
        # Small delay so Streamlit rerun catches the new session state
        time.sleep(1)
        st.rerun()


def get_role(user_id):
    res = supabase.table("profiles").select("role").eq("id", user_id).execute()
    if res.data:
        return res.data[0]["role"]
    return "employee"

def get_profile(user_id):
    res = supabase.table("profiles").select("*").eq("id", user_id).execute()
    return res.data[0] if res.data else None

# -------------------------------
# DB helpers
# -------------------------------
def register_employee(empid: str, name: str):
    empid = empid.upper().strip()
    name = name.strip()
    supabase.table("users").insert({"employee_id": empid, "name": name}).execute()

def get_user(empid: str):
    empid = empid.upper().strip()
    res = supabase.table("users").select("*").eq("employee_id", empid).execute()
    data = res.data or []
    return data[0] if data else None

def list_users():
    res = supabase.table("users").select("*").order("employee_id").execute()
    return res.data or []

def log_attendance(user_id: int, method: str):
    supabase.table("attendance").insert({
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "method": method
    }).execute()

def list_attendance(limit=1000):
    res = supabase.table("attendance").select("id,user_id,timestamp,method").order("timestamp", desc=True).limit(limit).execute()
    return res.data or []

def has_attendance_today(user_id: int):
    today_str = date.today().isoformat()
    res = supabase.table("attendance")\
        .select("id,timestamp,method")\
        .eq("user_id", user_id)\
        .gte("timestamp", today_str)\
        .execute()
    return len(res.data or []) > 0

# -------------------------------
# Storage helpers (models)
# -------------------------------
def upload_model_to_storage():
    model_path = os.path.join(LOCAL_MODELS_DIR, LBPH_MODEL_FILENAME)
    label_path = os.path.join(LOCAL_MODELS_DIR, LABEL_MAP_FILENAME)

    if not os.path.exists(model_path) or not os.path.exists(label_path):
        st.error("Local model files not found. Capture samples and retrain first.")
        return False

    with open(model_path, "rb") as f:
        supabase.storage.from_(STORAGE_MODELS_BUCKET).upload(LBPH_MODEL_FILENAME, f.read())
    with open(label_path, "rb") as f:
        supabase.storage.from_(STORAGE_MODELS_BUCKET).upload(LABEL_MAP_FILENAME, f.read())
    return True

def load_model_from_supabase():
    try:
        model_bytes = supabase.storage.from_(STORAGE_MODELS_BUCKET).download(LBPH_MODEL_FILENAME)
        label_bytes = supabase.storage.from_(STORAGE_MODELS_BUCKET).download(LABEL_MAP_FILENAME)
    except Exception:
        return None, None

    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(LOCAL_MODELS_DIR, LBPH_MODEL_FILENAME)
    label_path = os.path.join(LOCAL_MODELS_DIR, LABEL_MAP_FILENAME)
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    with open(label_path, "wb") as f:
        f.write(label_bytes)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    label_to_empid = np.load(label_path, allow_pickle=True).item()
    return recognizer, label_to_empid

# -------------------------------
# Face helpers
# -------------------------------
def detect_faces(gray_img):
    return face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

def preprocess_face(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        return None, None, None
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    face_gray = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_gray, (200, 200))
    return face_resized, (x, y, w, h), gray

# -------------------------------
# Training (LBPH) from app samples
# -------------------------------
def retrain_lbph_from_local_samples():
    ensure_dirs()
    faces, labels = [], []
    label_to_empid, empid_to_label = {}, {}
    current_label = 0

    for empid in os.listdir(DATA_DIR):
        emp_dir = os.path.join(DATA_DIR, empid)
        if not os.path.isdir(emp_dir):
            continue
        empid_to_label[empid] = current_label
        label_to_empid[current_label] = empid
        current_label += 1

        for fname in os.listdir(emp_dir):
            fpath = os.path.join(emp_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(empid_to_label[empid])

    if not faces:
        st.error("No face samples found. Capture samples first.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    model_path = os.path.join(LOCAL_MODELS_DIR, LBPH_MODEL_FILENAME)
    label_path = os.path.join(LOCAL_MODELS_DIR, LABEL_MAP_FILENAME)
    recognizer.write(model_path)
    np.save(label_path, label_to_empid)

    ok = upload_model_to_storage()
    return ok

# -------------------------------
# UI components
# -------------------------------
def ui_header():
    st.title("Face Recognition Attendance with Supabase")
    st.caption("Auth → Roles → Register → Capture → Retrain → Recognize → Timesheets → Reports")

def sidebar_menu_admin():
    return st.sidebar.radio("Menu", [
        "Admin dashboard",
        "Register employee",
        "Capture samples",
        "Retrain model",
        "Mark attendance",
        "Attendance logs",
        "Timesheet",
        "Monthly Timesheet",
        "Reporting dashboard"
    ])

# Admin dashboard for role management
def admin_dashboard_ui():
    st.subheader("Admin dashboard")
    st.markdown("Manage roles and profiles")

    # List profiles
    res = supabase.table("profiles").select("*").execute()
    profiles = res.data or []
    if not profiles:
        st.info("No profiles yet.")
        return

    df = pd.DataFrame(profiles)
    st.dataframe(df, width='stretch')

    # Promote/demote
    st.markdown("Update role")
    target_empid = st.text_input("Employee ID to update")
    new_role = st.selectbox("New role", ["employee", "admin"])
    if st.button("Apply role update"):
        if not target_empid:
            st.error("Enter an Employee ID.")
        else:
            try:
                # Find profile by employee_id
                res2 = supabase.table("profiles").select("id").eq("employee_id", target_empid.upper().strip()).execute()
                if res2.data:
                    pid = res2.data[0]["id"]
                    supabase.table("profiles").update({"role": new_role}).eq("id", pid).execute()
                    st.success(f"Updated role for {target_empid} to {new_role}.")
                else:
                    st.error("Profile not found.")
            except Exception as e:
                st.error(f"Role update failed: {e}")

# Register employee UI (creates in users table)
def register_employee_ui():
    st.subheader("Register employee")
    empid = st.text_input("Employee ID")
    name = st.text_input("Full name")
    if st.button("Register"):
        if not empid or not name:
            st.error("Enter both Employee ID and Name.")
            return
        try:
            register_employee(empid, name)
            st.success(f"Registered {empid.upper()} - {name}")
        except Exception as e:
            st.error(f"Registration failed: {e}")

# Capture samples UI
def capture_samples_ui():
    st.subheader("Capture face samples")
    empid = st.text_input("Employee ID")
    name = st.text_input("Full name")
    st.info("Tip: capture 10–20 samples in different lighting and angles for better accuracy.")

    img = st.camera_input("Take a photo")
    if img and empid and name:
        empid_norm = empid.upper().strip()
        emp_dir = os.path.join(DATA_DIR, empid_norm)
        os.makedirs(emp_dir, exist_ok=True)

        pil_img = Image.open(img)
        face_resized, rect, gray = preprocess_face(pil_img)
        if face_resized is None:
            st.error("No face detected. Try again.")
            return

        filename = f"{empid_norm}_{len(os.listdir(emp_dir)) + 1}.png"
        filepath = os.path.join(emp_dir, filename)
        cv2.imwrite(filepath, face_resized)
        st.success(f"Saved sample {filename} for {name} ({empid_norm})")

        # Auto-register user if missing
        user = get_user(empid_norm)
        if not user:
            try:
                register_employee(empid_norm, name)
                st.info("Employee auto-registered in database.")
            except Exception as e:
                st.warning(f"Auto-registration failed: {e}")

# Retrain UI
def retrain_ui():
    st.subheader("Retrain LBPH model from captured samples")
    if st.button("Retrain model"):
        with st.spinner("Training LBPH…"):
            ok = retrain_lbph_from_local_samples()
        if ok:
            st.success("Model retrained and uploaded to Supabase Storage.")
        else:
            st.error("Retraining failed.")

# Attendance marking UI
def mark_attendance_ui():
    st.subheader("Mark attendance (face recognition)")
    img = st.camera_input("Take a photo to mark attendance")
    if img is None:
        st.info("Awaiting photo…")
        return

    pil_img = Image.open(img)
    face_resized, rect, gray = preprocess_face(pil_img)
    if face_resized is None:
        st.error("No face detected.")
        return

    recognizer, label_to_empid = load_model_from_supabase()
    if recognizer is None or label_to_empid is None:
        st.error("Model not available. Retrain and upload first.")
        return

    label, confidence = recognizer.predict(face_resized)
    st.write(f"Match score (LBPH): {confidence:.2f}")
    if label not in label_to_empid or confidence > CONFIDENCE_THRESHOLD:
        st.error("Face not recognized. Consider capturing more samples or retraining.")
        return

    empid = label_to_empid[label]
    user = get_user(empid)
    if not user:
        st.error(f"Employee {empid} not found in database.")
        return

    # Clock-in/out prompt
    already = has_attendance_today(user["id"])
    if already:
        confirm = st.radio("Already clocked in today. Is this a clock-out?", ["No", "Yes"])
        if confirm == "Yes":
            log_attendance(user["id"], method="CLOCK_OUT")
            st.success(f"Clock-out recorded for {user['name']} ({empid}) at {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.info("No action taken.")
    else:
        log_attendance(user["id"], method="CLOCK_IN")
        st.success(f"Clock-in recorded for {user['name']} ({empid}) at {datetime.now().strftime('%H:%M:%S')}")

# Attendance logs UI
def attendance_logs_ui():
    st.subheader("Attendance logs")
    rows = list_attendance(limit=2000)
    if not rows:
        st.info("No attendance records yet.")
        return

    df = pd.DataFrame(rows)
    users_df = pd.DataFrame(list_users())
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if not users_df.empty:
        df = df.merge(users_df[["id", "employee_id", "name"]], left_on="user_id", right_on="id", how="left")
        df.drop(columns=["id_y"], inplace=True)
        df.rename(columns={"id_x": "id"}, inplace=True)

    st.dataframe(df.sort_values("timestamp", ascending=False), width='stretch')

# Timesheet helpers and UI
def calculate_daily_hours_for_user(user_id: int, day: date):
    start = pd.Timestamp(day.isoformat())
    end = start + pd.Timedelta(days=1)
    res = supabase.table("attendance")\
        .select("timestamp,method")\
        .eq("user_id", user_id)\
        .gte("timestamp", start.isoformat())\
        .lt("timestamp", end.isoformat())\
        .order("timestamp")\
        .execute()
    recs = res.data or []
    if not recs:
        return 0.0

    total = pd.Timedelta(0)
    pending_in = None
    for r in recs:
        ts = pd.Timestamp(r["timestamp"])
        if r["method"] == "CLOCK_IN":
            pending_in = ts
        elif r["method"] == "CLOCK_OUT" and pending_in is not None:
            total += (ts - pending_in)
            pending_in = None
    return round(total.total_seconds() / 3600.0, 2)

def timesheet_ui(mode='Daily'):
    if mode == 'Daily':
        st.subheader("Daily timesheet")
    else:
        st.subheader("Monthly timesheet")

    users = list_users()
    if not users:
        st.info("No users registered.")
        return

    label = st.selectbox("Select employee", [f"{u['employee_id']} - {u['name']}" for u in users])
    user = next(u for u in users if f"{u['employee_id']} - {u['name']}" == label)
    
    if mode == 'Daily': 
        day = st.date_input("Select date", date.today())
        hours = calculate_daily_hours_for_user(user["id"], day)
        if hours == 0.0:
            st.warning(f"No complete clock-in/out records for {user['name']} on {day}.")
        else:
            st.success(f"{user['name']} worked {hours} hours on {day}.")
    else:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
        month = st.number_input("Month", min_value=1, max_value=12, value=date.today().month)
        day = date(year, month, 1)
        days_in_month = (date(year + (month // 12), (month % 12) + 1, 1) - timedelta(days=1)).day
        records = []
        for d in range(1, days_in_month + 1):
            current_day = date(year, month, d)
            hours = calculate_daily_hours_for_user(user["id"], current_day)
            if hours > 0.0:
                records.append({"date": current_day, "hours": hours})
        if not records:
            st.warning(f"No attendance records for {user['name']} in {month}/{year}.")  
        else:
            df = pd.DataFrame(records)
            st.dataframe(df, width='stretch')
            total_hours = df["hours"].sum()
            st.success(f"Total hours worked in {month}/{year}: {total_hours} hours.")

# Reporting dashboard UI
def reporting_dashboard_ui():
    st.subheader("Reporting dashboard")
    rows = list_attendance(limit=5000)
    if not rows:
        st.info("No attendance records yet.")
        return

    df = pd.DataFrame(rows)
    users_df = pd.DataFrame(list_users())
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    if not users_df.empty:
        df = df.merge(users_df[["id", "employee_id", "name"]], left_on="user_id", right_on="id", how="left")
        df.drop(columns=["id_y"], inplace=True)
        df.rename(columns={"id_x": "id"}, inplace=True)

    # Compute daily hours per user
    daily_hours = []
    for (uid, d), group in df.groupby(["user_id", "date"]):
        timeline = group.sort_values("timestamp")[["timestamp", "method"]].values.tolist()
        total = pd.Timedelta(0)
        pending_in = None
        for ts, method in timeline:
            ts = pd.Timestamp(ts)
            if method == "CLOCK_IN":
                pending_in = ts
            elif method == "CLOCK_OUT" and pending_in is not None:
                total += (ts - pending_in)
                pending_in = None
        hrs = round(total.total_seconds() / 3600.0, 2)
        name = group["name"].iloc[0] if "name" in group.columns and not group["name"].isna().all() else str(uid)
        empid = group["employee_id"].iloc[0] if "employee_id" in group.columns and not group["employee_id"].isna().all() else ""
        daily_hours.append({"user_id": uid, "employee_id": empid, "name": name, "date": d, "hours": hrs})

    dh = pd.DataFrame(daily_hours)
    if dh.empty:
        st.info("No complete in/out pairs to report.")
        return

    employees = sorted(dh["name"].unique().tolist())
    selected_names = st.multiselect("Filter employees", employees, default=employees)
    start_date = st.date_input("Start date", date.today() - timedelta(days=7))
    end_date = st.date_input("End date", date.today())

    mask = (dh["name"].isin(selected_names)) & (dh["date"] >= start_date) & (dh["date"] <= end_date)
    filtered = dh[mask]

    chart = alt.Chart(filtered).mark_bar().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("hours:Q", title="Hours"),
        color=alt.Color("name:N", title="Employee"),
        tooltip=["name", "employee_id", "date", "hours"]
    ).properties(height=300)

    st.altair_chart(chart, width='stretch')
    st.dataframe(filtered.sort_values(["name", "date"]), width='stretch')

# -------------------------------
# Main routing with roles
# -------------------------------
def main():
    ui_header()

    # Auth gate
    if "user" not in st.session_state:
        auth_choice = st.sidebar.radio("Auth", ["Login", "Sign up"])
        if auth_choice == "Login":
            login_ui()
        else:
            signup_ui()
        return

    user = st.session_state["user"]
    role = get_role(user.id)
    profile = get_profile(user.id)
    st.sidebar.markdown(f"Signed in as: {profile['name'] if profile else user.email}")
    st.sidebar.markdown(f"Role: {role}")
    logout_ui()

    if role == "admin":
        choice = sidebar_menu_admin()
        if choice == "Admin dashboard":
            admin_dashboard_ui()
        elif choice == "Register employee":
            register_employee_ui()
        elif choice == "Capture samples":
            capture_samples_ui()
        elif choice == "Retrain model":
            retrain_ui()
        elif choice == "Mark attendance":
            mark_attendance_ui()
        elif choice == "Attendance logs":
            attendance_logs_ui()
        elif choice == "Timesheet":
            timesheet_ui('Daily')
        elif choice == 'Monthly Timesheet':
            timesheet_ui('Monthly')
        elif choice == "Reporting dashboard":
            reporting_dashboard_ui()
    else:
        choice = st.sidebar.radio("Menu", ["Mark attendance", "Timesheet"])
        if choice == "Mark attendance":
            mark_attendance_ui()
        elif choice == "Timesheet":
            timesheet_ui()

if __name__ == "__main__":
    main()