import cv2

cap = cv2.VideoCapture(r"F:\Research\WinterInternship\crowd-risk-project\videos\crowd3.mp4")
ret, frame = cap.read()
cv2.imwrite("ref_frame.png", frame)
cap.release()
