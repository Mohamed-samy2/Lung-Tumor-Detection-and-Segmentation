#from segment_anything import SamPredictor, sam_model_registry
from sam import SamPredictor, sam_model_registry
import os
import torch
import argparse
import json
import cv2
import matplotlib.pyplot as plt
import  numpy as np
# Step 1: Read JSON file
json_file_path = r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation\Src\Models\Fine_Tuned_SAM\args.json"  # Path to your JSON file
with open(json_file_path, "r") as file:
    json_data = json.load(file)  # Load JSON data into a dictionary


args = argparse.Namespace(**json_data)

arch = 'vit_b'
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[arch](args,os.path.join(os.path.dirname( os.path.abspath(__file__)),'checkpoint_best.pth'),num_classes=2)
sam.to(device)
sam.eval()

img = cv2.imread(r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation\Data\val\images\Subject_60\47.png")

predictor = SamPredictor(sam)
predictor.set_image(img)

mask , x ,y = predictor.predict(multimask_output=True)
print(mask.shape)

# Extract background and prediction
background = mask[0, :, :]  # First channel (background)
prediction = mask[1, :, :]  # Second channel (prediction)

# Plot the background and prediction
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot Background
axes[0].imshow(background, cmap='gray')
axes[0].set_title("Background")
axes[0].axis('off')  # Turn off axis

# Plot Prediction
axes[1].imshow(prediction, cmap='jet')
axes[1].set_title("Prediction")
axes[1].axis('off')  # Turn off axis

plt.tight_layout()
plt.show()




