import os
import cv2
import pandas as pd
import numpy as np
import joblib
import requests       # For downloading the model file
import matplotlib.pyplot as plt
import logging
import threading
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Disable logging so nothing prints to the console.
# -----------------------------------------------------------------------------
logging.disable(logging.CRITICAL)

# -----------------------------------------------------------------------------
# Define expected feature order (must match training order)
# -----------------------------------------------------------------------------
FEATURES_ORDER = [
    'pupil_size',
    'sclera_redness',
    'vein_prominence',
    'pupil_response_time',
    'ir_intensity',
    'scleral_vein_density',
    'ir_temperature',
    'tear_film_reflectivity',
    'pupil_dilation_rate',
    'sclera_color_balance',
    'vein_pulsation_intensity',
    'birefringence_index'
]

# -----------------------------------------------------------------------------
# Feature Extraction Functions
# -----------------------------------------------------------------------------
def get_pupil_size(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=80
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return round(np.mean([r for (_, _, r) in circles]), 5)
        else:
            return 0.0
    except Exception as e:
        return 0.0

def get_sclera_redness(image):
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        redness = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1]) * 100
        return round(redness, 5)
    except Exception as e:
        return 0.0

def get_vein_prominence(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        prominence = np.sum(edges) / (255.0 * image.shape[0] * image.shape[1])
        return round(prominence * 10, 5)
    except Exception as e:
        return 0.0

def get_ir_intensity(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.mean(gray), 5)
    except Exception as e:
        return 0.0

def get_scleral_vein_density(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.sum(edges) / (255.0 * image.shape[0] * image.shape[1])
        return round(density, 5)
    except Exception as e:
        return 0.0

def get_ir_temperature(image):
    try:
        return round(np.mean(image[:, :, 2]), 5)
    except Exception as e:
        return 0.0

def get_tear_film_reflectivity(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.std(gray), 5)
    except Exception as e:
        return 0.0

def get_sclera_color_balance(image):
    try:
        r_mean = np.mean(image[:, :, 2])
        g_mean = np.mean(image[:, :, 1])
        return round(r_mean / g_mean, 5) if g_mean > 0 else 1.0
    except Exception as e:
        return 1.0

def get_vein_pulsation_intensity(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pulsation = np.mean(cv2.Laplacian(gray, cv2.CV_64F))
        return round(pulsation, 5)
    except Exception as e:
        return 0.0

def get_birefringence_index(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return round(np.var(gray) / 255.0, 5)
    except Exception as e:
        return 0.0

# -----------------------------------------------------------------------------
# DataClass for Detection Results
# -----------------------------------------------------------------------------
@dataclass
class EyeDetection:
    left_eye: any          # Detected left eye bounding boxes.
    right_eye: any         # Detected right eye bounding boxes.
    ir_intensity: float    # Mean grayscale intensity.
    timestamp: datetime
    is_valid: bool         # True if a valid face was detected.
    eyes_open: bool        # True if eyes appear open.
    face_rect: tuple = None  # Face bounding box (x, y, w, h)

# -----------------------------------------------------------------------------
# Main Prediction & Monitoring Class
# -----------------------------------------------------------------------------
class EyeGlucoseMonitor:
    def __init__(self, model_path: str = "eye_glucose_model.pkl"):
        self.model_path = model_path
        self.model_url = "https://github.com/jtb21091/Cherry/releases/download/1.0/eye_glucose_model.pkl"
        self.model = self._load_model()
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascades = {
            "left": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'),
            "right": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        }
        self.last_valid_detection_time = time.time()
        self.invalid_detection_threshold = 3.0  # seconds
        self.prediction_lock = threading.Lock()
        
        self.latest_smoothed_prediction = None
        self.latest_instantaneous_prediction = None
        self.last_instantaneous_prediction = None
        self.alpha = 0.009  # EMA factor
        
        self.pupil_history = deque(maxlen=30)
        
        self.MIN_EYE_ASPECT_RATIO = 0.2
        self.MIN_IR_INTENSITY = 30
        
        self.time_history = deque(maxlen=200)
        self.instantaneous_history = deque(maxlen=200)
        self.smoothed_history = deque(maxlen=200)

    def _download_model(self) -> None:
        try:
            response = requests.get(self.model_url, stream=True)
            response.raise_for_status()
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            pass  # Suppress any errors from being printed.

    def _load_model(self) -> Any:
        if not os.path.exists(self.model_path):
            self._download_model()
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                return model
            except Exception as e:
                return None
        else:
            return None

    def detect_face_and_eyes(self, frame: np.ndarray) -> EyeDetection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_intensity = np.mean(gray)
        enhanced = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
        
        if len(faces) == 0 and ir_intensity < self.MIN_IR_INTENSITY:
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            faces = self.face_cascade.detectMultiScale(blurred, scaleFactor=1.1, minNeighbors=2, minSize=(50, 50))
        
        detection = EyeDetection([], [], ir_intensity, datetime.now(), False, False)
        if len(faces) > 0:
            face = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = face
            detection.face_rect = face
            face_roi = enhanced[y:y+h, x:x+w]
            left_eyes = self.eye_cascades["left"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15))
            right_eyes = self.eye_cascades["right"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15))
            if ir_intensity < self.MIN_IR_INTENSITY:
                detection = EyeDetection(left_eyes, right_eyes, ir_intensity, datetime.now(), True, True, face)
            else:
                if len(left_eyes) > 0 or len(right_eyes) > 0:
                    ear_list = []
                    for (ex, ey, ew, eh) in (list(left_eyes) + list(right_eyes)):
                        ratio = eh / ew if ew > 0 else 0
                        ear_list.append(ratio)
                    avg_ear = np.mean(ear_list) if ear_list else 0
                    eyes_open = avg_ear > self.MIN_EYE_ASPECT_RATIO
                    detection = EyeDetection(left_eyes, right_eyes, ir_intensity, datetime.now(), True, eyes_open, face)
        return detection

    def update_pupil_history(self, pupil_size: float):
        current_time = time.time()
        self.pupil_history.append((current_time, pupil_size))

    def compute_pupil_dilation_rate(self) -> float:
        if len(self.pupil_history) < 2:
            return 0.0
        t0, p0 = self.pupil_history[-2]
        t1, p1 = self.pupil_history[-1]
        dt = t1 - t0
        if dt == 0:
            return 0.0
        rate = (p1 - p0) / dt
        return round(rate, 5)

    def compute_pupil_response_time(self) -> float:
        rate = self.compute_pupil_dilation_rate()
        if rate == 0:
            return 0.0
        response_time = 1.0 / abs(rate)
        return round(response_time, 5)

    def extract_features(self, frame: np.ndarray) -> Dict:
        detection = self.detect_face_and_eyes(frame)
        if not detection.is_valid or (len(detection.left_eye) == 0 and len(detection.right_eye) == 0):
            return {}
        if detection.face_rect is None:
            return {}

        fx, fy, fw, fh = detection.face_rect
        face_roi = frame[fy:fy+fh, fx:fx+fw]

        eye_boxes = []
        for (ex, ey, ew, eh) in detection.left_eye:
            eye_boxes.append((ex, ey, ex+ew, ey+eh))
        for (ex, ey, ew, eh) in detection.right_eye:
            eye_boxes.append((ex, ey, ex+ew, ey+eh))
        if not eye_boxes:
            return {}
        ex_min = min(box[0] for box in eye_boxes)
        ey_min = min(box[1] for box in eye_boxes)
        ex_max = max(box[2] for box in eye_boxes)
        ey_max = max(box[3] for box in eye_boxes)
        eye_roi_x = fx + ex_min
        eye_roi_y = fy + ey_min
        eye_roi_w = ex_max - ex_min
        eye_roi_h = ey_max - ey_min

        eye_roi = frame[eye_roi_y:eye_roi_y+eye_roi_h, eye_roi_x:eye_roi_x+eye_roi_w]

        pupil_size = get_pupil_size(eye_roi)
        self.update_pupil_history(pupil_size)
        pupil_dilation_rate = self.compute_pupil_dilation_rate()
        pupil_response_time = self.compute_pupil_response_time()

        features = {
            "pupil_size": pupil_size,
            "sclera_redness": get_sclera_redness(eye_roi),
            "vein_prominence": get_vein_prominence(eye_roi),
            "pupil_response_time": pupil_response_time,
            "ir_intensity": get_ir_intensity(eye_roi),
            "scleral_vein_density": get_scleral_vein_density(eye_roi),
            "ir_temperature": get_ir_temperature(eye_roi),
            "tear_film_reflectivity": get_tear_film_reflectivity(eye_roi),
            "pupil_dilation_rate": pupil_dilation_rate,
            "sclera_color_balance": get_sclera_color_balance(eye_roi),
            "vein_pulsation_intensity": get_vein_pulsation_intensity(eye_roi),
            "birefringence_index": get_birefringence_index(eye_roi)
        }
        ordered_features = {key: features.get(key, 0) for key in FEATURES_ORDER}
        return ordered_features

    def predict_glucose(self, features: Dict):
        result = None
        try:
            if self.model is not None and features:
                df = pd.DataFrame([features], columns=FEATURES_ORDER)
                prediction = self.model.predict(df)
                if len(prediction) > 0:
                    result = prediction[0]
                    if result is None or (isinstance(result, float) and np.isnan(result)):
                        result = None
        except Exception as e:
            result = None

        with self.prediction_lock:
            if result is not None:
                self.latest_instantaneous_prediction = result
            else:
                self.latest_instantaneous_prediction = None

            if result is not None:
                if self.latest_smoothed_prediction is None:
                    self.latest_smoothed_prediction = result
                else:
                    self.latest_smoothed_prediction = self.alpha * result + (1 - self.alpha) * self.latest_smoothed_prediction
            else:
                self.latest_smoothed_prediction = None

            if self.last_instantaneous_prediction is not None and result is not None:
                # Rate-of-change check (logic kept but not printed)
                _ = result - self.last_instantaneous_prediction
            self.last_instantaneous_prediction = result

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        line_inst, = ax.plot([], [], label="Instantaneous", color="green")
        line_avg, = ax.plot([], [], label="Smoothed", color="orange")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Blood Glucose (mg/dL)")
        ax.legend()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.detect_face_and_eyes(frame)
            if detection.is_valid and (len(detection.left_eye) > 0 or len(detection.right_eye) > 0):
                features = self.extract_features(frame)
                self.predict_glucose(features)
                current_time = time.time() - start_time
                with self.prediction_lock:
                    inst_pred = self.latest_instantaneous_prediction if self.latest_instantaneous_prediction is not None else np.nan
                    smooth_pred = self.latest_smoothed_prediction if self.latest_smoothed_prediction is not None else np.nan
                self.time_history.append(current_time)
                self.instantaneous_history.append(inst_pred)
                self.smoothed_history.append(smooth_pred)

                if detection.face_rect is not None:
                    (x, y, w, h) = detection.face_rect
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    for (ex, ey, ew, eh) in detection.left_eye:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                    for (ex, ey, ew, eh) in detection.right_eye:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
            else:
                with self.prediction_lock:
                    self.latest_smoothed_prediction = None
                    self.latest_instantaneous_prediction = None

            line_inst.set_data(self.time_history, self.instantaneous_history)
            line_avg.set_data(self.time_history, self.smoothed_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Overlay prediction text on the frame without printing to console.
            with self.prediction_lock:
                inst_text = f"Inst: {self.latest_instantaneous_prediction:.1f} mg/dL" if self.latest_instantaneous_prediction is not None else "Inst: No Reading"
                smooth_text = f"Avg: {self.latest_smoothed_prediction:.1f} mg/dL" if self.latest_smoothed_prediction is not None else "Avg: No Reading"
            cv2.putText(frame, inst_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, smooth_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.imshow("Blood Glucose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            plt.pause(0.001)

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    monitor = EyeGlucoseMonitor()
    monitor.run()
