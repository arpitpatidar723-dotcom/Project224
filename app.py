import os
import cv2
import sqlite3
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "faces"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Haar cascade for face detection
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            face_encoding BLOB
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT,
            name TEXT,
            date TEXT,
            time TEXT,
            status TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_student(name, roll_no, face_encoding):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO students (name, roll_no, face_encoding)
        VALUES (?, ?, ?)
        """,
        (name, roll_no, pickle.dumps(face_encoding)),
    )
    conn.commit()
    conn.close()


def get_all_students():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    data = cursor.fetchall()
    conn.close()
    return data


def attendance_already_marked(roll_no, date_str):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM attendance WHERE roll_no=? AND date=? LIMIT 1",
        (roll_no, date_str),
    )
    row = cursor.fetchone()
    conn.close()
    return row is not None


def mark_attendance(roll_no, name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if attendance_already_marked(roll_no, date_str):
        return False

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO attendance (roll_no, name, date, time, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (roll_no, name, date_str, time_str, "Present"),
    )
    conn.commit()
    conn.close()
    return True


def get_attendance_today():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        "SELECT * FROM attendance WHERE date=? ORDER BY time DESC",
        (today,),
    )
    data = cursor.fetchall()
    conn.close()
    return data


# ---------------- FACE PROCESSING ----------------
def extract_face_features(image_path):
    """
    Read image, detect largest face, convert to grayscale,
    resize to fixed size, normalize and flatten.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No face detected in image.")

    # largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (100, 100))
    face = cv2.equalizeHist(face)
    face = face.astype("float32") / 255.0

    return face.flatten()


class FaceAttendance:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_roll_numbers = []
        self.load_known_faces()

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_roll_numbers = []

        students = get_all_students()
        for student in students:
            # student = (id, name, roll_no, face_encoding)
            if student[3]:
                try:
                    encoding = pickle.loads(student[3])
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(student[1])
                    self.known_roll_numbers.append(student[2])
                except Exception:
                    pass

    def register_face(self, name, roll_no, image_path):
        encoding = extract_face_features(image_path)
        add_student(name, roll_no, encoding)
        self.load_known_faces()
        return True

    def recognize_face(self, face_vector, threshold=0.38):
        """
        Compare current face vector with stored vectors.
        Lower distance = better match.
        """
        if not self.known_face_encodings:
            return None, None, None

        distances = [
            np.linalg.norm(face_vector - known_vec)
            for known_vec in self.known_face_encodings
        ]

        best_index = int(np.argmin(distances))
        best_distance = distances[best_index]

        if best_distance < threshold:
            return (
                self.known_face_names[best_index],
                self.known_roll_numbers[best_index],
                float(best_distance),
            )

        return None, None, float(best_distance)

    def start_recognition(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return "Camera not opened."

        attended_now = set()

        print("Press 'q' to quit recognition...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (100, 100))
                face_roi = cv2.equalizeHist(face_roi)
                face_roi = face_roi.astype("float32") / 255.0
                face_vector = face_roi.flatten()

                name, roll_no, distance = self.recognize_face(face_vector)

                if name is not None and roll_no is not None:
                    label = f"{name} ({roll_no})"

                    if roll_no not in attended_now:
                        marked = mark_attendance(roll_no, name)
                        if marked:
                            print(f"Attendance marked: {name} - {roll_no}")
                        attended_now.add(roll_no)

                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    label,
                    (x + 6, y + h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("AI Face Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return "Recognition completed."


# ---------------- INIT ----------------
init_db()
face_attendance = FaceAttendance()


# ---------------- HTML ----------------
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Face Attendance</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;color:white;}
        .container{max-width:800px;margin:50px auto;padding:20px;text-align:center;}
        h1{font-size:3em;margin-bottom:50px;text-shadow:2px 2px 4px rgba(0,0,0,0.3);}
        .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:30px;margin:50px 0;}
        .card{background:rgba(255,255,255,0.1);padding:40px;border-radius:20px;cursor:pointer;transition:all 0.3s ease;backdrop-filter:blur(10px);}
        .card:hover{transform:translateY(-10px);background:rgba(255,255,255,0.2);}
        .status{color:#b8ffb8;font-size:1.2em;margin:20px 0;}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Face Attendance System</h1>
        <div class="status">Ready to use</div>
        <div class="cards">
            <div class="card" onclick="window.location.href='/start_attendance'">
                <h3>Start Attendance</h3>
                <p>Live camera attendance marking</p>
            </div>
            <div class="card" onclick="window.location.href='/dashboard'">
                <h3>Dashboard</h3>
                <p>View today's attendance</p>
            </div>
            <div class="card" onclick="window.location.href='/register'">
                <h3>Register Student</h3>
                <p>Add new student with photo</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

REGISTER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Register Student</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;color:white;}
        .container{max-width:500px;margin:50px auto;padding:30px;background:rgba(255,255,255,0.1);border-radius:20px;backdrop-filter:blur(10px);}
        h1{text-align:center;margin-bottom:30px;}
        input,button{width:100%;padding:15px;margin:10px 0;border:none;border-radius:10px;font-size:16px;}
        input{background:rgba(255,255,255,0.95);color:#333;}
        button{background:#4CAF50;color:white;cursor:pointer;font-weight:bold;}
        button:hover{background:#45a049;}
        .btn-secondary{background:rgba(255,255,255,0.2);color:white;}
        #preview{width:200px;height:200px;object-fit:cover;border-radius:10px;margin:20px auto;display:none;}
    </style>
</head>
<body>
    <div class="container">
        <h1>Register New Student</h1>
        <form id="registerForm">
            <input type="text" id="name" placeholder="Full Name" required>
            <input type="text" id="rollno" placeholder="Roll Number" required>
            <input type="file" id="photo" accept="image/*" required>
            <img id="preview">
            <button type="submit">Register Student</button>
        </form>
        <button class="btn-secondary" onclick="location.href='/'">Back to Home</button>
    </div>

    <script>
        document.getElementById('photo').addEventListener('change', function(e){
            const file = e.target.files[0];
            if(file){
                const preview = document.getElementById('preview');
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
            }
        });

        document.getElementById('registerForm').addEventListener('submit', async function(e){
            e.preventDefault();

            const formData = new FormData();
            formData.append('name', document.getElementById('name').value);
            formData.append('roll_no', document.getElementById('rollno').value);
            formData.append('file', document.getElementById('photo').files[0]);

            const response = await fetch('/register_student', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            alert(result.success || result.error);

            if(result.success){
                this.reset();
                document.getElementById('preview').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Dashboard</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;color:white;}
        .container{max-width:1000px;margin:20px auto;padding:20px;}
        h1{text-align:center;margin-bottom:30px;}
        .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-bottom:30px;}
        .stat{background:rgba(255,255,255,0.1);padding:20px;border-radius:15px;text-align:center;}
        table{width:100%;background:rgba(255,255,255,0.1);border-radius:15px;overflow:hidden;border-collapse:collapse;}
        th,td{padding:15px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.2);}
        th{background:rgba(255,255,255,0.2);}
        .btn{display:inline-block;padding:10px 20px;background:#4CAF50;color:white;border-radius:25px;text-decoration:none;margin:10px;}
    </style>
</head>
<body>
    <div class="container">
        <h1>Today's Attendance Dashboard</h1>

        <div class="stats">
            <div class="stat">
                <h2>{{ attendance|length }}</h2>
                <p>Total Present</p>
            </div>
            <div class="stat">
                <h2>{{ date }}</h2>
                <p>Date</p>
            </div>
        </div>

        <table>
            <tr>
                <th>Name</th>
                <th>Roll No</th>
                <th>Time</th>
                <th>Status</th>
            </tr>
            {% for record in attendance %}
            <tr>
                <td>{{ record[2] }}</td>
                <td>{{ record[1] }}</td>
                <td>{{ record[4] }}</td>
                <td>{{ record[5] }}</td>
            </tr>
            {% endfor %}
        </table>

        <div style="text-align:center; margin-top:20px;">
            <a href="/" class="btn">Home</a>
            <a href="/start_attendance" class="btn">Start Attendance</a>
        </div>
    </div>
</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template_string(HOME_HTML)


@app.route("/register")
def register_page():
    return render_template_string(REGISTER_HTML)


@app.route("/register_student", methods=["POST"])
def register_student():
    name = request.form.get("name", "").strip()
    roll_no = request.form.get("roll_no", "").strip()

    if not name or not roll_no:
        return jsonify({"error": "Name and roll number are required."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    filename = secure_filename(f"{roll_no}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        face_attendance.register_face(name, roll_no, filepath)
        return jsonify({"success": f"Student {name} registered successfully."})
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 400


@app.route("/start_attendance")
def start_attendance():
    return """
    <html>
    <head><title>Start Attendance</title></head>
    <body style="background:#000;color:#fff;font-family:Arial;text-align:center;padding-top:80px;">
        <h1>Starting Face Recognition...</h1>
        <p>Camera will open automatically. Press <strong>Q</strong> to quit.</p>
        <script>
            setTimeout(() => {
                window.location.href = '/recognize';
            }, 1500);
        </script>
    </body>
    </html>
    """


@app.route("/recognize")
def recognize():
    result = face_attendance.start_recognition()
    return f"<h2 style='font-family:Arial'>{result}</h2><a href='/dashboard'>Go to Dashboard</a>"


@app.route("/dashboard")
def dashboard():
    today_attendance = get_attendance_today()
    date = datetime.now().strftime("%Y-%m-%d")
    return render_template_string(
        DASHBOARD_HTML,
        attendance=today_attendance,
        date=date
    )


if __name__ == "__main__":
    print("AI Face Attendance System Starting...")
    print("Open: http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)