from flask import Flask, render_template, request, send_file
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -----------------------------
# Convert image to circle
# -----------------------------
def make_circle_image(path):
    img = Image.open(path).convert("RGBA")

    size = min(img.size)
    img = img.resize((size, size))

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    output = Image.new("RGBA", (size, size))
    output.paste(img, (0, 0), mask)

    base, _ = os.path.splitext(path)
    circle_path = base + "_circle.png"

    output.save(circle_path)
    return circle_path


# -----------------------------
# SIMPLE + STABLE DETECTION
# (fast + no timing inflation)
# -----------------------------
def detect_active_range(video_path, threshold=0.015):
    cap = cv2.VideoCapture(video_path)

    prev = None
    start = None
    end = None

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is not None:
            diff = np.mean(cv2.absdiff(gray, prev)) / 255.0

            if diff > threshold:
                if start is None:
                    start = frame_id
                end = frame_id

        prev = gray
        frame_id += 1

    cap.release()

    if start is None:
        return 0, frame_id

    return start, end


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():

    video_file = request.files["video"]
    image_file = request.files["image"]

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)

    video_file.save(video_path)
    image_file.save(image_path)

    video = VideoFileClip(video_path)

    image_path = make_circle_image(image_path)

    # detect timing
    start_f, end_f = detect_active_range(video_path)

    fps = video.fps if video.fps else 24

    start_t = start_f / fps
    end_t = end_f / fps

    duration = max(0, end_t - start_t)

    # FIX: prevent too long overlays
    if duration > video.duration:
        duration = video.duration * 0.6

    overlay = (
        ImageClip(image_path)
        .with_start(start_t)
        .with_duration(duration)
        .resized(width=200)
        .with_position(("center", "center"))
    )

    final = CompositeVideoClip([video, overlay])

    output_path = os.path.join(OUTPUT_FOLDER, "result.mp4")

    final.write_videofile(
        output_path,
        codec="libx264",
        fps=24,
        audio=False,
        logger=None
    )

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=8000)