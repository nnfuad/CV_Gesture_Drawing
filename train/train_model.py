import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
from train.model import SketchCNN

def pil_to_tensor(img):
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

dataset = datasets.MNIST(
    root="../data/raw",
    train=True,
    download=True,
    transform=pil_to_tensor
)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

model = SketchCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    for images, labels in loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# torch.save(model.state_dict(), "../models/cnn_model.pth")

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(
    model.state_dict(),
    os.path.join(MODEL_DIR, "cnn_model.pth")
)