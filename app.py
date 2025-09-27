import os
from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# YOLO model
MODEL_PATH = "C:\\Users\\hadap\\runs\\detect\\train8\\weights\\best.pt"
model = YOLO(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    detections = []
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "Please upload an image."
            return render_template("index.html", error=error)

        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # Run YOLO prediction
        results = model.predict(source=upload_path, conf=0.25, save=False)
        img = cv2.imread(upload_path)

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = results[0].names[cls_id]
            detections.append(f"{label} ({conf*100:.1f}%)")

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf*100:.1f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save result
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, img)

        # Relative path for HTML
        result_img = "/static/results/" + filename

    return render_template("index.html", result_img=result_img, detections=detections, error=error)

if __name__ == "__main__":
    app.run(debug=True)
