import cv2
import numpy as np
import time

# Configuration Constants
WIDTH = 640
HEIGHT = 480

# HSV Range for skin (adjust per lighting)
HSV_MIN = np.array([0, 48, 80], np.uint8)
HSV_MAX = np.array([20, 255, 255], np.uint8)
MIN_CONTOUR_AREA = 3000

# Virtual Box
BOX_TOP_LEFT = (int(0.75 * WIDTH), int(0.3 * HEIGHT))
BOX_BOTTOM_RIGHT = (WIDTH - 10, int(0.7 * HEIGHT))

# Distance thresholds
DANGER_THRESHOLD = 0.05 * WIDTH
WARNING_THRESHOLD = 0.15 * WIDTH

# Load face detector once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     "haarcascade_frontalface_default.xml")


def get_fingertip_points(frame_bgr):
    """Return list of fingertip points using contour, hull & convexity defects."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)

    # Remove face so it isn't detected as hand
    faces = face_cascade.detectMultiScale(frame_bgr, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 0), -1)

    # Clean-up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Largest contour → hand candidate
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return []

    # Reject non-hand shapes
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    if len(approx) < 4 or len(approx) > 25:
        return []

    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return []

    defects = cv2.convexityDefects(contour, hull)
    if defects is None or len(defects) < 2:
        return []

    # Extract fingertip points
    fingertip_points = []

    for defect in defects:
        start_idx, end_idx, far_idx, depth = defect.squeeze()
        start_pt = tuple(contour[start_idx][0])
        end_pt = tuple(contour[end_idx][0])

        # Add these as fingertip candidates
        fingertip_points.append(start_pt)
        fingertip_points.append(end_pt)

    # Remove duplicates & cluster nearby points
    unique = []
    for pt in fingertip_points:
        if all(np.linalg.norm(np.array(pt) - np.array(u)) > 20 for u in unique):
            unique.append(pt)

    return unique


def distance_to_box(pt):
    px, py = pt
    x = max(BOX_TOP_LEFT[0], min(px, BOX_BOTTOM_RIGHT[0]))
    y = max(BOX_TOP_LEFT[1], min(py, BOX_BOTTOM_RIGHT[1]))
    return np.sqrt((px - x) ** 2 + (py - y) ** 2)


# Main Application 
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

frame_count = 0
start_time = time.time()

print("Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Detect fingertips
    fingertips = get_fingertip_points(frame)

    # Default state
    current_state = "SAFE"
    color = (0, 255, 0)

    min_dist = float("inf")

    if fingertips:
        for pt in fingertips:
            cv2.circle(frame, pt, 8, (255, 0, 0), -1)
            d = distance_to_box(pt)
            min_dist = min(min_dist, d)

        # Apply SAFE / WARNING / DANGER logic
        if min_dist <= DANGER_THRESHOLD:
            current_state = "DANGER"
            color = (0, 0, 255)
        elif min_dist <= WARNING_THRESHOLD:
            current_state = "WARNING"
            color = (0, 165, 255)

    # Draw the virtual box
    cv2.rectangle(frame, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, color, 3)

    # State text
    cv2.putText(frame, f"State: {current_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if current_state == "DANGER":
        cv2.putText(frame, "DANGER DANGER", (WIDTH // 4, HEIGHT // 2),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)

    # FPS counter
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = 30 / elapsed
        start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Tracking POC", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
