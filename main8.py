# ================= IMPORTS =================
import os
os.environ["ULTRALYTICS_SETTINGS"] = "false"
os.environ["YOLO_OFFLINE"] = "true"

import cv2
import numpy as np
import threading
import time
from collections import deque
from ultralytics import YOLO
import supervision as sv

from logic.object_memory import ObjectMemory
from logic.state_machine import update_state
from utils.geometry import centroid

# ================= CONFIG =================

CAMERA_1 = "rtsp://admin:123456@192.168.1.109:554/stream1"
CAMERA_2 = "rtsp://admin:123456@192.168.1.188:554/stream1"

CONF_THRESHOLD = 0.15  # 🔥 slightly lower for better recall
FRAME_SKIP = 2         # 🔥 more frequent detection

PERSON_BAG_DISTANCE = 220
PERSON_MEMORY_TIME = 4.0
STATIONARY_DISTANCE = 15
STATIONARY_TIME = 5.0
MIN_STATE_TIME = 3.0

VALID_CLASSES = [0, 24, 26, 28]

EVENT_DIR = "events"
os.makedirs(EVENT_DIR, exist_ok=True)

WINDOW = "AI-Based Lost & Found Monitoring System"

# ================= MODEL =================

model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()

object_memory_cam1 = ObjectMemory(distance_threshold=60)
object_memory_cam2 = ObjectMemory(distance_threshold=60)

# ================= STREAMS =================

latest_frames = {"cam1": None, "cam2": None}
raw_frames = {"cam1": None, "cam2": None}

frame_buffer = {
    "cam1": deque(maxlen=300),
    "cam2": deque(maxlen=300)
}

recording = {
    "cam1": {"active": False, "frames": [], "start_time": 0},
    "cam2": {"active": False, "frames": [], "start_time": 0}
}

# 🔥 detection history (NEW)
bag_history = {"cam1": deque(maxlen=5), "cam2": deque(maxlen=5)}

# ================= IMAGE ENHANCEMENT =================

def enhance_frame(frame):
    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    frame = cv2.filter2D(frame, -1, kernel)

    # contrast boost
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

    return frame

def normalize_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0,(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ================= CAMERA =================

def camera_thread(source, cam_key):

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ {cam_key} failed")
        return

    print(f"✅ {cam_key} started")

    while True:
        cap.grab()
        ret, frame = cap.retrieve()

        if ret:
            raw_frames[cam_key] = frame.copy()

            # 🔥 enhance before detection
            processed = enhance_frame(frame)
            processed = normalize_frame(processed)

            latest_frames[cam_key] = processed
            frame_buffer[cam_key].append(frame.copy())

        else:
            time.sleep(0.005)

threading.Thread(target=camera_thread, args=(CAMERA_1,"cam1"), daemon=True).start()
threading.Thread(target=camera_thread, args=(CAMERA_2,"cam2"), daemon=True).start()

print("📡 Cameras running...")

# ================= CACHE =================

last_detections_cam1 = None
last_detections_cam2 = None
last_persons_cam1 = []
last_persons_cam2 = []
last_bags_cam1 = []
last_bags_cam2 = []

frame_count = 0

# ================= VIDEO =================

def get_video_writer(path, w, h):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(path, fourcc, 25, (w,h))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, 25, (w,h))

    return out

def save_event_video(cam_key):
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"{EVENT_DIR}/{cam_key}_{ts}.mp4"

    frames = [f for f in recording[cam_key]["frames"] if f is not None]
    if not frames:
        return

    h, w = frames[0].shape[:2]
    out = get_video_writer(path, w, h)

    for f in frames:
        out.write(f)

    out.release()
    print(f"🎬 Saved: {path}")

# ================= PROCESS =================

def process_frame(frame, last_detections, last_persons, last_bags, memory, cam_key):

    global frame_count
    frame_count += 1
    now = time.time()

    persons = last_persons
    bags = last_bags
    detections = last_detections

    if frame_count % FRAME_SKIP == 0:

        results = model(frame, conf=CONF_THRESHOLD, imgsz=960, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results)

        if detections.class_id is not None:
            detections = detections[np.isin(detections.class_id, VALID_CLASSES)]

        detections = tracker.update_with_detections(detections)

        persons, bags = [], []

        for box, cls in zip(detections.xyxy, detections.class_id):
            if cls == 0:
                persons.append(centroid(box))
            else:
                bags.append(box)

        # 🔥 store history
        bag_history[cam_key].append(bags)

    # 🔥 merge last few frames (STABILITY)
    merged_bags = []
    for history in bag_history[cam_key]:
        merged_bags.extend(history)

    bags = merged_bags if merged_bags else bags

    # ================= LOGIC =================

    for bag_box in bags:

        bag_center = centroid(bag_box)
        obj = memory.match_or_create(bag_center)

        movement = np.linalg.norm(np.array(bag_center) - np.array(obj.last_position))

        person_near = any(
            np.linalg.norm(np.array(p) - np.array(bag_center)) < PERSON_BAG_DISTANCE
            for p in persons
        )

        if movement > 50 or person_near:
            obj.state = "ATTENDED"
            obj.event_triggered = False
            obj.last_person_near_time = now
            obj.stationary_since = None
            obj.update_position(bag_center)
            continue

        if movement < STATIONARY_DISTANCE:
            if obj.stationary_since is None:
                obj.stationary_since = now
        else:
            obj.stationary_since = None

        is_stationary = obj.stationary_since and (now - obj.stationary_since) >= STATIONARY_TIME

        obj.update_position(bag_center)

        person_absent = (now - obj.last_person_near_time) >= PERSON_MEMORY_TIME

        if obj.state != "UNATTENDED":
            update_state(obj, person_absent, is_stationary, now, MIN_STATE_TIME)

        if obj.state == "UNATTENDED" and not obj.event_triggered:

            obj.event_triggered = True
            print(f"\n🚨 EVENT on {cam_key}")

            recording[cam_key]["active"] = True
            recording[cam_key]["frames"] = list(frame_buffer[cam_key])
            recording[cam_key]["start_time"] = now

    # RECORD
    if recording[cam_key]["active"]:
        if raw_frames[cam_key] is not None:
            recording[cam_key]["frames"].append(raw_frames[cam_key].copy())

        if now - recording[cam_key]["start_time"] > 10:
            save_event_video(cam_key)
            recording[cam_key]["active"] = False

    return frame, detections, persons, bags

# ================= MAIN =================

while True:

    f1 = latest_frames["cam1"]
    f2 = latest_frames["cam2"]

    if f1 is None and f2 is None:
        continue

    if f1 is not None:
        f1, last_detections_cam1, last_persons_cam1, last_bags_cam1 = process_frame(
            f1, last_detections_cam1, last_persons_cam1, last_bags_cam1, object_memory_cam1, "cam1"
        )

    if f2 is not None:
        f2, last_detections_cam2, last_persons_cam2, last_bags_cam2 = process_frame(
            f2, last_detections_cam2, last_persons_cam2, last_bags_cam2, object_memory_cam2, "cam2"
        )

    frames = [f for f in [f1,f2] if f is not None]

    if len(frames) == 2:
        h = min(frames[0].shape[0], frames[1].shape[0])
        frames = [cv2.resize(f,(int(f.shape[1]*h/f.shape[0]),h)) for f in frames]
        combined = cv2.hconcat(frames)
    else:
        combined = frames[0]

    cv2.imshow(WINDOW, combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()