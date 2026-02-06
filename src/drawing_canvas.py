import cv2
import numpy as np

class DrawingCanvas:
    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_point = None
        self.color = (255, 0, 0)
        self.thickness = 5

    def draw(self, point):
        if self.prev_point is None:
            self.prev_point = point
            return

        cv2.line(self.canvas, self.prev_point, point, self.color, self.thickness)
        self.prev_point = point

    def reset_point(self):
        self.prev_point = None

    def clear(self):
        self.canvas[:] = 0

    def set_color(self, color):
        self.color = color