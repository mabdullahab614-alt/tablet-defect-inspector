import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pygame
import time

# Load model
checkpoint = torch.load("tablet_model.pth", map_location="cpu")
CLASSES = checkpoint["classes"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize sound
pygame.mixer.init()

def beep():
    pygame.mixer.Sound.play(pygame.mixer.Sound(
        pygame.sndarray.make_sound(
            np.array([np.sin(2 * np.pi * 440 * t / 44100) * 32767
                      for t in range(44100)], dtype=np.int16)
        )
    ))

def predict(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor)[0], dim=0)
    idx = probs.argmax().item()
    return CLASSES[idx], float(probs[idx]) * 100

# ─── CHANGE THIS TO YOUR PHONE IP ───
PHONE_IP = "192.168.1.8"  # Change this!
STREAM_URL = f"http://{PHONE_IP}:8080/video"

print("Connecting to phone camera...")
print(f"URL: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Could not connect! Using PC webcam instead...")
    cap = cv2.VideoCapture(0)

last_alert = 0
print("Starting inspection... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received!")
        break

    label, conf = predict(frame)

    if label == "defective":
        color = (0, 0, 255)  # Red
        text = f"DEFECTIVE! {conf:.1f}%"
        # Play alarm every 2 seconds
        if time.time() - last_alert > 2:
            try:
                beep()
            except:
                pass
            last_alert = time.time()
        # Flash red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    else:
        color = (0, 255, 0)  # Green
        text = f"GOOD {conf:.1f}%"

    # Draw result on frame
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("💊 Tablet Defect Inspector - LIVE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Inspection stopped.")