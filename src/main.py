import cv2
from hand_tracking import HandTracker
from drawing_canvas import DrawingCanvas
from ui_buttons import Button
from inference import InferenceModel

cap = cv2.VideoCapture(0)
tracker = HandTracker()

ret, frame = cap.read()
canvas = DrawingCanvas(frame.shape[1], frame.shape[0])
model = InferenceModel("../models/cnn_model.pth")

ok_button = Button(500, 20, 100, 50, "OK", (0,255,0))

while True:
    ret, frame = cap.read()
    landmarks = tracker.get_landmarks(frame)

    if landmarks:
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        if abs(index_tip[1] - middle_tip[1]) > 40:
            canvas.draw(index_tip)
        else:
            canvas.reset_point()

        if ok_button.check_hover(index_tip):
            pred = model.predict(canvas.canvas)
            print("Prediction:", pred)

    ok_button.draw(frame)

    combined = cv2.addWeighted(frame, 0.7, canvas.canvas, 0.3, 0)
    cv2.imshow("Air Draw", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()