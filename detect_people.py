import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# 1. VIDEO AND MODEL
# -----------------------------
VIDEO_PATH = r"F:\Research\WinterInternship\crowd-risk-project\videos\crowd2.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video")
    exit()

# -----------------------------
# 2. HOMOGRAPHY USING YOUR POINTS
# -----------------------------
# Your 4 clicked ground corners (image pixels)
img_pts = np.array([
    [418, 299],  # Point 1
    [978, 362],  # Point 2
    [454, 616],  # Point 3
    [134, 489],  # Point 4
], dtype=np.float32)

# Map them to a simple 20m x 20m square in real world
ground_pts = np.array([
    [0.0, 0.0],     # corresponds to Point 1
    [20.0, 0.0],    # Point 2
    [20.0, 20.0],   # Point 3
    [0.0, 20.0],    # Point 4
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, ground_pts)
area_m2 = 20.0 * 20.0  # 400 m^2

# -----------------------------
# 3. MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    r = results[0]

    people_ground = []

    # ---- detect people and project to ground plane ----
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # bottom-center of bounding box as foot point
            px = (x1 + x2) / 2.0
            py = y2

            pt_img = np.array([[[px, py]]], dtype=np.float32)
            gx, gy = cv2.perspectiveTransform(pt_img, H)[0][0]
            people_ground.append((gx, gy))

    person_count = len(people_ground)

    # ---- density (people per m^2) and risk ----
    density = person_count / area_m2 if area_m2 > 0 else 0.0

    if density < 0.3:
        risk = "LOW"
        risk_color = (0, 255, 0)
    elif density < 1.5:
        risk = "MEDIUM"
        risk_color = (0, 255, 255)
    else:
        risk = "HIGH"
        risk_color = (0, 0, 255)

    # ---- draw person boxes with risk color ----
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), risk_color, 2)

    # ---- draw the 4 ground-corner points (blue) ----
    for pt in img_pts:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (255, 0, 0), -1)

    # ---- overlay text ----
    cv2.putText(frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Density: {density:.2f} ppl/m^2", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Risk: {risk}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_color, 2)

    cv2.imshow("Crowd density and risk", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
