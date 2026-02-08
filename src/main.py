import cv2

from src.hand_tracking import HandTracker
from src.drawing_canvas import DrawingCanvas
from src.ui_buttons import Button
from src.inference import InferenceModel


def draw_tips(frame):
    tips = [
        "Tips for better recognition:",
        "- Draw BIG and centered",
        "- Use ONE continuous stroke",
        "- Avoid tiny digits",
        "- Hold finger steady on OK",
        "- Digits work best (0-9)"
    ]

    y = 100
    for tip in tips:
        cv2.putText(
            frame,
            tip,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )
        y += 18


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        return

    h, w, _ = frame.shape
    canvas = DrawingCanvas(w, h)

    model = InferenceModel("models/cnn_model.pth")

    # UI buttons
    red_btn    = Button(20, 20, 80, 45, "RED", (0, 0, 255))
    blue_btn   = Button(110, 20, 80, 45, "BLUE", (255, 0, 0))
    green_btn  = Button(200, 20, 80, 45, "GREEN", (0, 255, 0))
    eraser_btn = Button(290, 20, 100, 45, "ERASER", (180, 180, 180))
    clear_btn  = Button(400, 20, 110, 45, "CLEAR ALL", (0, 0, 0))
    ok_btn     = Button(530, 20, 80, 45, "OK", (0, 255, 0))

    last_prediction = None
    last_confidence = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks = tracker.get_landmarks(frame)

        if landmarks:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]

            # ----- BUTTON LOGIC -----
            if red_btn.check_hover(index_tip):
                canvas.set_color((0, 0, 255))
                canvas.thickness = 5

            if blue_btn.check_hover(index_tip):
                canvas.set_color((255, 0, 0))
                canvas.thickness = 5

            if green_btn.check_hover(index_tip):
                canvas.set_color((0, 255, 0))
                canvas.thickness = 5

            if eraser_btn.check_hover(index_tip):
                canvas.set_color((0, 0, 0))
                canvas.thickness = 30

            if clear_btn.check_hover(index_tip):
                canvas.clear()
                last_prediction = None
                last_confidence = None

            if ok_btn.check_hover(index_tip):
                result = model.predict(canvas.canvas)
                if result is not None:
                    last_prediction, last_confidence = result

            # ----- DRAWING LOGIC -----
            if abs(index_tip[1] - middle_tip[1]) > 40:
                canvas.draw(index_tip)
            else:
                canvas.reset_point()

        # Draw buttons
        red_btn.draw(frame)
        blue_btn.draw(frame)
        green_btn.draw(frame)
        eraser_btn.draw(frame)
        clear_btn.draw(frame)
        ok_btn.draw(frame)

        # Draw tips
        draw_tips(frame)

        # Show prediction
        if last_prediction is not None:
            text = f"Prediction: {last_prediction} ({last_confidence*100:.1f}%)"
            cv2.putText(
                frame,
                text,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        combined = cv2.addWeighted(frame, 0.7, canvas.canvas, 0.3, 0)
        cv2.imshow("Air Drawing Recognition", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()