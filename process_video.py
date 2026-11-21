import cv2
import numpy as np
import time
import os
from collections import deque
from ultralytics import YOLO
import supervision as sv

# ================= CONFIG =================

CONF_THRESHOLD = 0.18
FRAME_SKIP = 3

PERSON_BAG_DISTANCE = 220
PERSON_MEMORY_TIME = 4.0
STATIONARY_DISTANCE = 12
STATIONARY_TIME = 5.0
MIN_STATE_TIME = 3.0

VALID_CLASSES = [0, 24, 26, 28]

EVENT_DIR = "events"
os.makedirs(EVENT_DIR, exist_ok=True)

# ================= MODEL =================

model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()

# ================= OBJECT MEMORY =================

class Obj:
    def __init__(self, pos, id):
        self.logical_id = id
        self.last_position = pos
        self.last_seen = time.time()
        self.state = "ATTENDED"
        self.state_enter_time = time.time()
        self.last_person_near_time = time.time()
        self.stationary_since = None
        self.event_triggered = False

    def update(self, pos):
        self.last_position = pos
        self.last_seen = time.time()


class ObjectMemory:
    def __init__(self):
        self.objects = []
        self.next_id = 1

    def match_or_create(self, pos):
        for obj in self.objects:
            if np.linalg.norm(np.array(pos) - np.array(obj.last_position)) < 60:
                return obj

        obj = Obj(pos, self.next_id)
        self.next_id += 1
        self.objects.append(obj)
        return obj


memory = ObjectMemory()

# ================= VIDEO INPUT =================

video_path = __import__("sys").argv[1]
cap = cv2.VideoCapture(video_path)

frame_buffer = deque(maxlen=300)
recording = {"active": False, "frames": [], "start_time": 0}

frame_count = 0

# ================= HELPERS =================

def centroid(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def save_event_video(frames):

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"{EVENT_DIR}/upload_{ts}.mp4"

    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 20, (w,h))

    for f in frames:
        out.write(f)

    out.release()

    print(f"🎬 Saved Event: {path}")

# ================= MAIN LOOP =================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame.copy())

    frame_count += 1
    now = time.time()

    if frame_count % FRAME_SKIP != 0:
        continue

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    if detections.class_id is not None:
        detections = detections[np.isin(detections.class_id, VALID_CLASSES)]

    detections = tracker.update_with_detections(detections)

    persons = []
    bags = []

    for box, cls in zip(detections.xyxy, detections.class_id):
        if cls == 0:
            persons.append(centroid(box))
        else:
            bags.append(box)

    # ================= LOGIC =================

    for bag_box in bags:

        c = centroid(bag_box)
        obj = memory.match_or_create(c)

        movement = np.linalg.norm(
            np.array(c) - np.array(obj.last_position)
        )

        person_near = any(
            np.linalg.norm(np.array(p) - np.array(c)) < PERSON_BAG_DISTANCE
            for p in persons
        )

        # RESET
        if movement > 40 or person_near:
            obj.state = "ATTENDED"
            obj.event_triggered = False
            obj.last_person_near_time = now
            obj.stationary_since = None
            obj.update(c)
            continue

        # STATIONARY
        if movement < STATIONARY_DISTANCE:
            if obj.stationary_since is None:
                obj.stationary_since = now
        else:
            obj.stationary_since = None

        is_stationary = (
            obj.stationary_since and
            (now - obj.stationary_since) >= STATIONARY_TIME
        )

        obj.update(c)

        person_absent = (
            now - obj.last_person_near_time
        ) >= PERSON_MEMORY_TIME

        # STATE CHANGE
        if obj.state != "UNATTENDED":

            if person_absent and is_stationary:
                if now - obj.state_enter_time > MIN_STATE_TIME:
                    obj.state = "UNATTENDED"
                    obj.state_enter_time = now

        # EVENT TRIGGER
        if obj.state == "UNATTENDED" and not obj.event_triggered:

            obj.event_triggered = True

            print("🚨 UNATTENDED DETECTED")

            recording["active"] = True
            recording["frames"] = list(frame_buffer)
            recording["start_time"] = now

    # RECORD AFTER EVENT
    if recording["active"]:

        recording["frames"].append(frame.copy())

        if now - recording["start_time"] > 10:
            save_event_video(recording["frames"])
            recording["active"] = False

cap.release()
print("✅ Processing Complete")