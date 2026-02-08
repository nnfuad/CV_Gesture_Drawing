import cv2

class Button:
    def __init__(self, x, y, w, h, label, color):
        self.rect = (x, y, w, h)
        self.label = label
        self.color = color
        self.counter = 0

    def draw(self, frame):
        x, y, w, h = self.rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, 2)
        cv2.putText(frame, self.label, (x+10, y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)

    def check_hover(self, point, threshold=15):
        x, y, w, h = self.rect
        px, py = point

        if x < px < x+w and y < py < y+h:
            self.counter += 1
            if self.counter > threshold:
                self.counter = 0
                return True
        else:
            self.counter = 0
        return False