import cv2
import numpy as np
import csv
from ultralytics import YOLO

# -------- 1. VIDEO AND MODEL --------
VIDEO_PATH = r"F:\Research\WinterInternship\crowd-risk-project\videos\crowd3.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video")
    exit()

# -------- 2. HOMOGRAPHY (your 4 clicked points) --------
img_pts = np.array([
    [1523, 4],  # Point 1
    [2, 5],  # Point 2
    [6, 823],  # Point 3
    [1521, 823],  # Point 4
], dtype=np.float32)

ground_pts = np.array([
    [0.0, 0.0],     # -> Point 1
    [20.0, 0.0],    # -> Point 2
    [20.0, 20.0],   # -> Point 3
    [0.0, 20.0],    # -> Point 4
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, ground_pts)
area_m2 = 20.0 * 20.0  # 400 m^2

# ---- risk thresholds (YOU TUNE THESE) ----
LOW_MAX = 0.025   # density < 0.05 ppl/m^2  -> LOW
MED_MAX = 0.030   # 0.05 <= density < 0.15  -> MEDIUM
# density >= MED_MAX                       -> HIGH

# -------- 3. CSV FILES --------
ground_csv = open("ground_points.csv", "w", newline="")
ground_writer = csv.writer(ground_csv)
ground_writer.writerow(["frame_id", "person_id", "gx", "gy"])

risk_csv = open("frame_risk.csv", "w", newline="")
risk_writer = csv.writer(risk_csv)
risk_writer.writerow(["frame_id", "density", "risk"])

frame_id = 0

# (optional) video writer for demo output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "crowd_risk_output.mp4",
    fourcc,
    25,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

# -------- 4. MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model(frame)
    r = results[0]

    people_ground = []

    # project each detected person to ground plane
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            px = (x1 + x2) / 2.0   # bottom-centre x
            py = y2                # bottom y

            pt_img = np.array([[[px, py]]], dtype=np.float32)
            gx, gy = cv2.perspectiveTransform(pt_img, H)[0][0]
            people_ground.append((gx, gy))

    person_count = len(people_ground)
    density = person_count / area_m2 if area_m2 > 0 else 0.0

    # ---- risk from density (using thresholds above) ----
    if density < LOW_MAX:
        risk = "LOW"
        risk_color = (0, 255, 0)
    elif density < MED_MAX:
        risk = "MEDIUM"
        risk_color = (0, 255, 255)
    else:
        risk = "HIGH"
        risk_color = (0, 0, 255)

    # draw boxes
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), risk_color, 2)

    # draw homography points
    for pt in img_pts:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (255, 0, 0), -1)

    # overlay text
    cv2.putText(frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Density: {density:.3f} ppl/m^2", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Risk: {risk}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_color, 2)

    cv2.imshow("Crowd density and risk", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # save ground positions of this frame
    for i, (gx, gy) in enumerate(people_ground):
        ground_writer.writerow([frame_id, i, gx, gy])

    # save frame-level risk
    risk_writer.writerow([frame_id, density, risk])

cap.release()
ground_csv.close()
risk_csv.close()
out.release()
cv2.destroyAllWindows()
