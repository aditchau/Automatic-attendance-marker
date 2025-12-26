import sys
import cv2
import time
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QCheckBox, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

# =========================
# Kalman Filter 2D
# =========================
class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    def predict(self):
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

# =========================
# Main PyQt5 Application
# =========================
class SmartTrackerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Tracker Control Panel")
        self.setGeometry(50, 50, 1250, 750)
        self.setStyleSheet("background-color: #1c1c1c; color: white;")

        # YOLO Models
        self.yolo_general = YOLO("yolo11n.pt")
        self.yolo_person = YOLO("yolov8n-person.pt")
        self.yolo_face = YOLO("yolov8n-face.pt")

        # State variables
        self.cap = None
        self.running = False
        self.kalman_enabled = True
        self.mode = "person"
        self.selection_mode = False
        self.selected_target = None
        self.kalman_filters = {}
        self.prediction_trails = {}
        self.current_boxes = {}
        self.prev_time = time.time()
        self.fps = 0

        self.init_ui()

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # =========================
    # GUI Layout
    # =========================
    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left: Video display
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(900, 700)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.mousePressEvent = self.on_video_click
        main_layout.addWidget(self.video_label)

        # Right: Control Panel
        ctrl_panel = QFrame()
        ctrl_panel.setStyleSheet("background-color: #2e2e2e;")
        ctrl_panel.setFixedWidth(300)
        ctrl_layout = QVBoxLayout()
        ctrl_panel.setLayout(ctrl_layout)
        main_layout.addWidget(ctrl_panel)

        # Headline
        headline = QLabel("Smart Tracker Control Panel")
        headline.setFont(QFont("Arial", 16, QFont.Bold))
        headline.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(headline)
        ctrl_layout.addSpacing(20)

        # Start/Stop Buttons
        self.btn_start = self.create_button("Start Camera Feed", self.start_video)
        self.btn_stop = self.create_button("Stop Camera Feed", self.stop_video)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addSpacing(15)

        # Detection Mode
        mode_label = QLabel("Detection Type")
        mode_label.setFont(QFont("Arial", 14, QFont.Bold))
        ctrl_layout.addWidget(mode_label)
        ctrl_layout.addSpacing(5)

        self.mode_group = QButtonGroup()
        self.rb_object = QRadioButton("Object")
        self.rb_person = QRadioButton("Person")
        self.rb_face = QRadioButton("Face")
        self.rb_person.setChecked(True)
        self.mode_group.addButton(self.rb_object)
        self.mode_group.addButton(self.rb_person)
        self.mode_group.addButton(self.rb_face)
        self.rb_object.toggled.connect(lambda: self.set_mode("object"))
        self.rb_person.toggled.connect(lambda: self.set_mode("person"))
        self.rb_face.toggled.connect(lambda: self.set_mode("face"))
        ctrl_layout.addWidget(self.rb_object)
        ctrl_layout.addWidget(self.rb_person)
        ctrl_layout.addWidget(self.rb_face)
        ctrl_layout.addSpacing(15)

        # Kalman Toggle
        self.cb_kalman = QCheckBox("Kalman Filter ON")
        self.cb_kalman.setChecked(True)
        self.cb_kalman.stateChanged.connect(self.toggle_kalman)
        ctrl_layout.addWidget(self.cb_kalman)
        ctrl_layout.addSpacing(10)

        # Select/Delete Target
        self.btn_select = self.create_button("Select Target for Tracking", self.enable_selection)
        self.btn_delete = self.create_button("Delete Selected Target", self.delete_target)
        ctrl_layout.addWidget(self.btn_select)
        ctrl_layout.addWidget(self.btn_delete)
        ctrl_layout.addSpacing(15)

        # Selected Target Label
        self.lbl_target = QLabel("Selected Target: None")
        self.lbl_target.setFont(QFont("Arial", 12, QFont.StyleItalic))
        self.lbl_target.setStyleSheet("color: cyan;")
        ctrl_layout.addWidget(self.lbl_target)
        ctrl_layout.addStretch()

    def create_button(self, text, func):
        btn = QPushButton(text)
        btn.setFont(QFont("Arial", 12, QFont.Bold))
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; border-radius: 6px; height: 40px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3e8e41; }
        """)
        btn.clicked.connect(func)
        return btn

    # =========================
    # Mode/Kalman/Target Functions
    # =========================
    def set_mode(self, mode):
        self.mode = mode

    def toggle_kalman(self, state):
        self.kalman_enabled = state == Qt.Checked

    def enable_selection(self):
        self.selection_mode = True
        self.selected_target = None
        self.lbl_target.setText("Click on target in video feed")

    def delete_target(self):
        if self.selected_target:
            if self.selected_target in self.kalman_filters:
                del self.kalman_filters[self.selected_target]
            if self.selected_target in self.prediction_trails:
                del self.prediction_trails[self.selected_target]
            self.selected_target = None
            self.lbl_target.setText("Selected Target: None")

    def on_video_click(self, event):
        if not self.selection_mode:
            return
        x = event.pos().x()
        y = event.pos().y()
        # Map click to frame coordinates
        frame_w, frame_h = 900, 700
        vid_w, vid_h = self.video_label.width(), self.video_label.height()
        fx = int(x * frame_w / vid_w)
        fy = int(y * frame_h / vid_h)
        # Find clicked box
        for key, (x1, y1, x2, y2) in self.current_boxes.items():
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                self.selected_target = key
                self.selection_mode = False
                self.prediction_trails[key] = deque(maxlen=20)
                self.lbl_target.setText(f"Selected Target: {key}")
                break

    # =========================
    # Video Functions
    # =========================
    def start_video(self):
        if not self.running:
            self.cap = cv2.VideoCapture(1)
            self.running = True
            self.timer.start(30)

    def stop_video(self):
        self.running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()

    # =========================
    # Frame Processing
    # =========================
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (900, 700))
        self.current_boxes = {}

        # FPS calculation
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        # Detection
        boxes = []
        if self.mode == "person":
            results = self.yolo_person(frame, conf=0.4, verbose=False)[0]
            boxes = results.boxes
        elif self.mode == "face":
            results = self.yolo_face(frame, conf=0.4, verbose=False)[0]
            boxes = results.boxes
        elif self.mode == "object":
            results = self.yolo_general(frame, conf=0.4, verbose=False)[0]
            # Exclude person from object
            boxes = [b for b in results.boxes if self.yolo_general.names[int(b.cls[0])] != "person"]

        # Draw boxes
        for idx, box in enumerate(boxes):
            label = self.mode
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            key = f"{label}_{idx}"
            self.current_boxes[key] = (x1, y1, x2, y2)

            # Hide other objects if target selected
            if self.selected_target and self.selected_target != key:
                continue

            # Selected target highlight
            color = (0, 255, 255) if self.selected_target == key else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID
            cv2.putText(frame, key, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Kalman prediction & trail
            if self.selected_target == key and self.kalman_enabled:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if key not in self.kalman_filters:
                    self.kalman_filters[key] = KalmanFilter2D()
                kf = self.kalman_filters[key]
                kf.correct(cx, cy)
                px, py = kf.predict()
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(frame, (px - w//2, py - h//2), (px + w//2, py + h//2), (0,0,255), 2)
                cv2.circle(frame, (px, py), 5, (0,0,255), -1)

                # Trail
                self.prediction_trails[key].append((px, py))
                for i in range(1, len(self.prediction_trails[key])):
                    cv2.line(frame, self.prediction_trails[key][i-1], self.prediction_trails[key][i], (0,0,255), 2)

        # FPS display
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # Convert to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

# =========================
# Run Application
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartTrackerApp()
    window.show()
    sys.exit(app.exec_())
