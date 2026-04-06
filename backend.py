from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
from captum.attr import IntegratedGradients
import sqlite3
import matplotlib.pyplot as plt

app = FastAPI()

# Allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
           'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
           'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# 1. Load the Model
model = mobilenet_v2(num_classes=14)
# Make sure nih_model.pth is in the same folder!
model.load_state_dict(torch.load('nih_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# 2. Setup SQLite Database
conn = sqlite3.connect('local_xai.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS diagnostics 
             (id INTEGER PRIMARY KEY, image_name TEXT, predicted_disease TEXT, heatmap BLOB)''')
conn.commit()

# 3. API Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs[0])
        predicted_idx = torch.argmax(probabilities).item()
        disease_name = CLASSES[predicted_idx]
        
        # --- NEW CODE: Calculate "Accuracy" / Confidence ---
        confidence_score = round(probabilities[predicted_idx].item() * 100, 2)
        
        # --- NEW CODE: Print Convolution Math for the Professor ---
        print("\n=== EXECUTING CONVOLUTIONAL NEURAL NETWORK ===")
        print(f"1. Input Image Tensor Shape: {input_tensor.shape} (1 image, 3 RGB channels, 224x224 pixels)")
        
        # Pass the image through just the very first Convolution layer
        first_conv_layer_output = model.features[0](input_tensor)
        
        print(f"2. After 1st Convolution Layer: {first_conv_layer_output.shape}")
        print(f"-> The 2D Convolution applied 32 mathematical filters, turning 3 colors into 32 feature maps!")
        print("==============================================\n")
        
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(input_tensor, target=predicted_idx, return_convergence_delta=True)
    heatmap = np.sum(np.abs(attributions.squeeze().detach().numpy()), axis=0)
    
    # Save raw data to SQLite
    heatmap_blob = heatmap.astype(np.float32).tobytes()
    c.execute("INSERT INTO diagnostics (image_name, predicted_disease, heatmap) VALUES (?, ?, ?)", 
              (file.filename, disease_name, heatmap_blob))
    conn.commit()

    # --- UPDATED HEATMAP COLORING ---
    # Normalize between 0 and 1 safely
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.ptp(heatmap) + 1e-8)
    
    # Apply the medical 'jet' colormap (Blue -> Green -> Yellow -> Red)
    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap_norm)
    
    # Convert RGBA to RGB format for the web
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored).resize((image.width, image.height))
    # --------------------------------
    
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Added confidence to the return dictionary
    return {
        "disease": disease_name,
        "confidence": confidence_score,
        "heatmap": f"data:image/png;base64,{heatmap_b64}"
    }