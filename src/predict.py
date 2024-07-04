import os
import sys
import torch
import requests
from PIL import Image
from io import BytesIO
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python3 predict.py <url>")
    sys.exit(1)

url = sys.argv[1]

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train/')
classes = sorted(os.listdir(train_dir))
n_classes = len(classes)

model = models.vgg16(weights='IMAGENET1K_V1')
n_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(n_features, n_classes)

model.load_state_dict(torch.load('src/model.pt'))
model.eval()

response = requests.get(url)
img = Image.open(BytesIO(response.content))

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0) # adds new dimension at index 0, i.e. (1, C, H, W)

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)
model = model.to(device)
img_tensor = img_tensor.to(device)

output = model(img_tensor)
_, pred_tensor = torch.max(output, 1)
pred = pred_tensor.item()

plt.imshow(img)
plt.title(f'Predicted Pokemon: {classes[pred]}', fontweight='bold', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
