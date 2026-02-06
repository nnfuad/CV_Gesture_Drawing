import cv2
import torch
import numpy as np
from train.model import SketchCNN

class InferenceModel:
    def __init__(self, model_path):
        self.model = SketchCNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, canvas):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            output = self.model(tensor)
            return torch.argmax(output).item()