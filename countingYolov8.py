import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import datetime
import threading
from queue import Queue
from pymongo import MongoClient

# Load YOLO model and initialize tracker
model = YOLO('yolov8s.pt')
tracker = Tracker()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Portrait window size
window_width, window_height = 720, 1280
output = cv2.VideoWriter('output_final.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, (window_width, window_height))

# Counters and storage
offset = 6
count = 0
persondown = {}
personup = {}
counter_down = []
counter_up = []

# MongoDB setup
client = MongoClient("mongodb+srv://seanbartolome7slm:cap2419it@busmateph.vfi4r.mongodb.net/?tlsAllowInvalidCertificates=true")
db = client["BusMatePH"]
collection = db["capacity"]
data_queue = Queue()

# Logging thread function
def log_data_worker():
    while True:
        try:
            capacity = data_queue.get()
            query = {"busID": 4}
            update = {
                "$set": {
                    "capacity": capacity,
                    "date": datetime.datetime.now()
                }
            }
            collection.update_one(query, update, upsert=True)
            print(f"[LOGGED] Bus Capacity = {capacity}")
        except Exception as e:
            print("[MongoDB ERROR]:", e)
        finally:
            data_queue.task_done()

# Start logging thread
threading.Thread(target=log_data_worker, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    count += 1
    if count % 3 != 0:
        continue

    cam_height, cam_width = frame.shape[:2]
    scale = window_width / cam_width
    new_w = window_width
    new_h = int(cam_height * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    y_offset = (window_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, 0:new_w] = resized_frame

    cy1 = int(new_h * 0.48) + y_offset
    cy2 = int(new_h * 0.52) + y_offset

    results = model.predict(resized_frame)
    boxes = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(boxes).astype(float)

    detected_boxes = []
    for idx, row in px.iterrows():
        x1, y1, x2, y2, score, cls = row[:6]
        class_id = int(cls)
        class_name = model.names[class_id] if hasattr(model, 'names') else 'person'
        if 'person' in class_name:
            detected_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    bbox_id = tracker.update(detected_boxes)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        cx_c = cx
        cy_c = cy + y_offset
        x3_c = x3
        y3_c = y3 + y_offset
        x4_c = x4
        y4_c = y4 + y_offset

        cv2.circle(canvas, (cx_c, cy_c), 4, (255, 0, 255), -1)

        if cy1 - offset < cy_c < cy1 + offset:
            cv2.rectangle(canvas, (x3_c, y3_c), (x4_c, y4_c), (0, 0, 255), 2)
            cvzone.putTextRect(canvas, f'{id}', (x3_c, y3_c), 1, 2)
            persondown[id] = (cx_c, cy_c)

        if id in persondown:
            if cy2 - offset < cy_c < cy2 + offset:
                cv2.rectangle(canvas, (x3_c, y3_c), (x4_c, y4_c), (0, 255, 255), 2)
                cvzone.putTextRect(canvas, f'{id}', (x3_c, y3_c), 1, 2)
                if id not in counter_down:
                    counter_down.append(id)

        if cy2 - offset < cy_c < cy2 + offset:
            cv2.rectangle(canvas, (x3_c, y3_c), (x4_c, y4_c), (0, 255, 0), 2)
            cvzone.putTextRect(canvas, f'{id}', (x3_c, y3_c), 1, 2)
            personup[id] = (cx_c, cy_c)

        if id in personup:
            if cy1 - offset < cy_c < cy1 + offset:
                cv2.rectangle(canvas, (x3_c, y3_c), (x4_c, y4_c), (0, 255, 255), 2)
                cvzone.putTextRect(canvas, f'{id}', (x3_c, y3_c), 1, 2)
                if id not in counter_up:
                    counter_up.append(id)

    # Draw lines
    cv2.line(canvas, (0, cy1), (window_width, cy1), (0, 255, 0), 2)
    cv2.line(canvas, (0, cy2), (window_width, cy2), (0, 255, 255), 2)

    downcount = len(counter_down)
    upcount = len(counter_up)
    current_capacity = upcount - downcount

    # Log capacity (only when changed)
    if not hasattr(log_data_worker, "last_logged") or current_capacity != log_data_worker.last_logged:
        data_queue.put(current_capacity)
        log_data_worker.last_logged = current_capacity

    cvzone.putTextRect(canvas, f'Up: {downcount}', (50, 60), 2, 2)
    cvzone.putTextRect(canvas, f'Down: {upcount}', (50, 160), 2, 2)

    output.write(canvas)
    cv2.imshow('RGB', canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
