import cv2

img = cv2.imread("ref_frame.png")
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("Image", img)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

cv2.imshow("Image", img)
print("Click 4 ground corners, press ESC when done.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
print("Final points:", points)
