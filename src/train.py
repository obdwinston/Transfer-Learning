import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

batch_size = 20
learning_rate = 1e-3
epochs = 3

# +===========+
# | Load Data |
# +===========+

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')
classes = sorted(os.listdir(train_dir))

train_transforms = transforms.Compose([
    # transforms.RandomRotation(30), # rotated [-30., 30.] degrees
    # transforms.RandomHorizontalFlip(), # flipped horizontally
    # transforms.RandomResizedCrop(224), # downsampled and cropped
    transforms.Resize(255), # downsampled
    transforms.CenterCrop(224), # cropped
    transforms.ToTensor() # converted to tensor and scaled [0., 1.]
])
test_transforms = transforms.Compose([
    transforms.Resize(255), # downsampled
    transforms.CenterCrop(224), # cropped
    transforms.ToTensor() # converted to tensor and scaled [0., 1.]
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

print(f'Train Examples: {len(train_data)}')
print(f'Test Examples: {len(test_data)}')

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

images, labels = next(iter(train_dataloader)) # obtain first (next) batch
print(f'Images Shape (N, C, H, W): {images.size()}')
images = images.numpy() # convert image pytorch tensors to numpy arrays

fig = plt.figure(figsize=(10, 5))
fig.suptitle('Training Samples', fontweight='bold', fontsize=16)
for idx in np.arange(10):
    ax = fig.add_subplot(2, 10//2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
plt.tight_layout()
plt.show()

# +==============+
# | Define Model |
# +==============+

model = models.vgg16(weights='IMAGENET1K_V1') # freeze parameters (fixed feature extractor)
# print(model)

for param in model.features.parameters(): # pooling and activation layers do not contain learnable parameters
    param.requires_grad = False

n_features = model.classifier[6].in_features
n_classes = len(classes)
model.classifier[6] = nn.Linear(n_features, n_classes)
print(model)

# for name, param in model.named_parameters():
#     print(f'Parameter: {name}, Requires gradient: {param.requires_grad}')

# +====================+
# | Train & Save Model |
# +====================+

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Device: {device}')

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.classifier.parameters(), lr=learning_rate)

train_size = len(train_dataloader.dataset)
train_batches = len(train_dataloader)
test_size = len(test_dataloader.dataset)
test_batches = len(test_dataloader)

for epoch in range(epochs):

    print(f'=== Epoch {epoch + 1}/{epochs} ===')

    # train model

    train_loss = 0.
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        train_loss += loss.item()

        if (batch % 20 == 19) or (batch + 1 == train_batches):
            print(f'Batch: {batch + 1}/{train_batches}, Loss: {(train_loss / 20): .10f}')
            train_loss = 0.

    # test model

    print(f'>>> Testing <<<')

    test_loss = 0.
    correct = {classname: 0 for classname in classes}
    total = {classname: 0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)
            test_loss += loss.item()

            _, predicted = torch.max(pred, 1) # returns (maximum_value, maximum_value_index)
            for i in range(len(predicted)):
                label = y[i]
                if predicted[i] == label:
                    correct[classes[label]] += 1
                total[classes[label]] += 1
    
    test_loss /= test_batches
    print(f'Test Loss: {test_loss:.10f}')

    for classname, correct_count in correct.items():
        accuracy = 100 * correct_count / total[classname]
        print(f'Test Accuracy of {classname.capitalize()}: {accuracy:.0f}% ({correct_count}/{total[classname]})')

    overall_accuracy = 100 * sum(correct.values()) / sum(total.values())
    print(f'Overall Test Accuracy: {overall_accuracy:.0f}% ({sum(correct.values())}/{sum(total.values())})')

    # save model

    torch.save(model.state_dict(), 'src/model.pt')
    print('Saved model state dictionary to model.pt.')
