from flask import Flask, render_template, send_from_directory, request, redirect
import os
from datetime import datetime
from collections import Counter
from werkzeug.utils import secure_filename
import subprocess
import cv2

app = Flask(__name__)

EVENT_FOLDER = "../events"
UPLOAD_FOLDER = "../uploads"
THUMB_FOLDER = os.path.join(EVENT_FOLDER, "thumbs")

os.makedirs(EVENT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMB_FOLDER, exist_ok=True)


def format_filename(filename):
    try:
        parts = filename.split("_")

        cam = parts[0]
        date = parts[1]
        time = parts[2].split(".")[0]

        dt = datetime.strptime(date + time, "%Y%m%d%H%M%S")

        return cam.upper(), dt.strftime("%d-%m-%Y %H:%M:%S")

    except:
        return "UPLOAD", filename


# 🔥 GENERATE THUMBNAIL
def generate_thumbnail(video_path, filename):
    thumb_name = filename.replace(".mp4", ".jpg")
    thumb_path = os.path.join(THUMB_FOLDER, thumb_name)

    if os.path.exists(thumb_path):
        return thumb_name

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(thumb_path, frame)

    cap.release()
    return thumb_name


@app.route("/")
def index():

    files = [f for f in os.listdir(EVENT_FOLDER) if f.endswith(".mp4")]
    files.sort(reverse=True)

    videos = []
    cam_counter = Counter()

    for i, f in enumerate(files):
        cam, time = format_filename(f)
        cam_counter[cam] += 1

        video_path = os.path.join(EVENT_FOLDER, f)
        thumb = generate_thumbnail(video_path, f)

        videos.append({
            "id": i + 1,
            "filename": f,
            "camera": cam,
            "time": time,
            "thumb": thumb
        })

    return render_template(
        "index.html",
        videos=videos,
        total=len(videos),
        cam_stats=dict(cam_counter)
    )


@app.route("/video/<path:filename>")
def video(filename):
    return send_from_directory(EVENT_FOLDER, filename)


@app.route("/thumb/<path:filename>")
def thumb(filename):
    return send_from_directory(THUMB_FOLDER, filename)


# 🔥 UPLOAD ROUTE
@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return redirect("/")

    file = request.files["file"]

    if file.filename == "":
        return redirect("/")

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(path)

    print(f"📁 Uploaded: {path}")

    subprocess.Popen([
        "python",
        "../process_video.py",
        path
    ])

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)