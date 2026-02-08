import cv2
import numpy as np
import os

# ---------------- CONFIG ----------------
SAVE_ROOT = "data/air_digits"
CANVAS_SIZE = 400
DIGIT_SIZE = 28
STROKE_THICKNESS = 12

current_label = 0
drawing = False
prev_point = None

canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)


# ---------------- PREPROCESS (SAME AS CNN) ----------------
def preprocess(img):
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]

    kernel = np.ones((2, 2), np.uint8)
    cropped = cv2.dilate(cropped, kernel, iterations=1)

    scale = min((DIGIT_SIZE - 4) / h, (DIGIT_SIZE - 4) / w)
    resized = cv2.resize(cropped, None, fx=scale, fy=scale)

    final = np.zeros((DIGIT_SIZE, DIGIT_SIZE), dtype=np.uint8)
    h, w = resized.shape
    y_off = (DIGIT_SIZE - h) // 2
    x_off = (DIGIT_SIZE - w) // 2

    final[y_off:y_off+h, x_off:x_off+w] = resized
    return final


# ---------------- MOUSE CALLBACK ----------------
def draw(event, x, y, flags, param):
    global drawing, prev_point, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, prev_point, (x, y), 255, STROKE_THICKNESS)
        prev_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_point = None


# ---------------- MAIN LOOP ----------------
cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)

print("Controls:")
print(" Draw with mouse")
print(" 0–9 : set label")
print(" s   : save image")
print(" c   : clear canvas")
print(" ESC : quit")

while True:
    display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(display, (50, 50), (350, 350), (100, 100, 100), 1)
    cv2.putText(display, f"LABEL: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Draw Digit", display)

    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == 27:
        break

    # Clear
    if key == ord('c'):
        canvas[:] = 0

    # Set label
    if ord('0') <= key <= ord('9'):
        current_label = key - ord('0')
        print("Label set to:", current_label)

    # Save image
    if key == ord('s'):
        processed = preprocess(canvas)
        if processed is None:
            print("Nothing to save")
            continue

        save_dir = os.path.join(SAVE_ROOT, str(current_label))
        os.makedirs(save_dir, exist_ok=True)

        count = len(os.listdir(save_dir))
        filename = f"{count}.png"
        path = os.path.join(save_dir, filename)

        cv2.imwrite(path, processed)
        print("Saved:", path)

        cv2.imshow("Saved 28x28", processed)
        canvas[:] = 0

cv2.destroyAllWindows()