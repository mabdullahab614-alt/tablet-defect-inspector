import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

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

def inspect(image, history):
    if image is None:
        return "Please upload an image!", None, history

    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor)[0], dim=0)

    scores = {c: round(float(probs[i]) * 100, 2) for i, c in enumerate(CLASSES)}
    pred = max(scores, key=scores.get)
    conf = scores[pred]

    if pred == "defective":
        result = f"❌ DEFECTIVE TABLET — {conf:.1f}% confidence"
    else:
        result = f"✅ GOOD TABLET — {conf:.1f}% confidence"

    history = (history or "") + f"\n{result}\n" + "─"*30
    return result, scores, history

def clear():
    return None, "_Upload a tablet image to inspect!_", None, ""

with gr.Blocks(title="💊 Tablet Defect Inspector") as app:
    gr.Markdown("""
    # 💊 Tablet Defect Inspector AI
    ### Upload a tablet photo — AI instantly detects if it's good or defective!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="📷 Upload Tablet Photo", type="pil", height=280)
            inspect_btn = gr.Button("🔍 Inspect Tablet!", variant="primary")
            clear_btn = gr.Button("🗑️ Clear")
            gr.Markdown("""
---
### 🔍 Detects:
- ✅ Good tablets
- ❌ Cracked tablets
- ❌ Contaminated tablets
            """)

        with gr.Column(scale=2):
            result_text = gr.Markdown("_Upload a tablet image to inspect!_")
            conf_out = gr.Label(label="Confidence Scores", num_top_classes=2)
            history_box = gr.Textbox(label="Inspection History", lines=6, interactive=False)

    history_state = gr.State("")

    inspect_btn.click(fn=inspect,
                      inputs=[img_input, history_state],
                      outputs=[result_text, conf_out, history_state])
    history_state.change(fn=lambda h: h, inputs=history_state, outputs=history_box)
    clear_btn.click(fn=clear, outputs=[img_input, result_text, conf_out, history_box])

app.launch()