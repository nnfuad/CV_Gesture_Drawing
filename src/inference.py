import os
import cv2
import torch
import numpy as np

from train.model import SketchCNN


class InferenceModel:
    def __init__(self, model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, model_path)

        self.model = SketchCNN()
        self.model.load_state_dict(torch.load(full_path))
        self.model.eval()

    def preprocess(self, canvas):
        # 1. Convert to grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # 2. Threshold (remove background noise)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # 3. Find bounding box of drawing
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        cropped = thresh[y:y+h, x:x+w]

        # 4. Thicken strokes (MNIST-like)
        kernel = np.ones((2, 2), np.uint8)
        cropped = cv2.dilate(cropped, kernel, iterations=1)

        # 5. Resize while preserving aspect ratio
        size = 28
        h, w = cropped.shape
        scale = min((size - 4) / h, (size - 4) / w)
        resized = cv2.resize(cropped, None, fx=scale, fy=scale)

        # 6. Center the digit
        canvas_28 = np.zeros((size, size), dtype=np.uint8)
        h, w = resized.shape
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        canvas_28[y_offset:y_offset+h, x_offset:x_offset+w] = resized

        return canvas_28

    def predict(self, canvas):
        processed = self.preprocess(canvas)
        if processed is None:
            return None

        cv2.imshow("CNN Input (28x28)", processed)

        tensor = torch.tensor(processed, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        return prediction.item(), confidence.item()