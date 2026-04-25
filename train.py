import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import glob

class TabletDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.images = glob.glob(os.path.join(folder, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = img_path.replace(".jpg", ".png")
        image = Image.open(img_path).convert("RGB")
        label = 0
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            if mask.max() > 0:
                label = 1
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = TabletDataset("dataset/train", transform)
valid_ds = TabletDataset("dataset/valid", transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False)

print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_acc = 0
for epoch in range(5):
    model.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, pred = model(imgs).max(1)
            vc += pred.eq(labels).sum().item()
            vt += labels.size(0)

    val_acc = 100. * vc / vt
    print(f"Epoch [{epoch+1}/5] Train: {100.*correct/total:.1f}% | Val: {val_acc:.1f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model": model.state_dict(), "classes": ["good", "defective"]}, "tablet_model.pth")
        print(f"  Best model saved! ({val_acc:.1f}%)")

print(f"Done! Best accuracy: {best_acc:.1f}%")