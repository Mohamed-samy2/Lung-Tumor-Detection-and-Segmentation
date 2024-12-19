from .UNet_Model import UNet
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from ..BaseModel import BaseModel
import cv2

class Segmentation(BaseModel):
    def __init__(self):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        torch.cuda.empty_cache()
        
        self.Weight_Path = os.path.join(os.path.dirname( os.path.abspath(__file__)),'UNet_weights.pth')
        self.model = UNet(1)
        self.model.load_state_dict(torch.load(self.Weight_Path,weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
    def run(self,img):
        
        tumor = False
        input = self._image_preprocessing(img)
        with torch.no_grad():
            pred = self.model(input)
        
        pred = self._image_postprocessing(pred)
        
        height , width = self._get_hw(pred)
        
        if height != 0 and width != 0:
            tumor = True
        
        return pred.numpy(), height , width , tumor
    
    def _image_preprocessing(self,img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        img = transform(img)
        img = img.unsqueeze(0) # add batch dimension
        
        return img.to(self.device)
    
    def _image_postprocessing(self,pred):
        
        pred = pred.squeeze(0) # remove batch dimension
        
        pred = torch.sigmoid(pred)
        
        pred = (pred > 0.90).float()
        
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu()
        
        if pred.ndim == 3 and pred.shape[0] == 1:  
            pred = pred.squeeze(0) # remove channel dimension
        
        return pred
    
    def _get_hw(self,pred):
        
        non_zero_indices = torch.nonzero(pred)
        if len(non_zero_indices) == 0:
            return 0, 0  

        
        min_y, min_x = non_zero_indices.min(dim=0)[0]  
        max_y, max_x = non_zero_indices.max(dim=0)[0]  
        # Calculate the height and width of the bounding box
        height = max_y.item() - min_y.item() + 1
        width = max_x.item() - min_x.item() + 1

        return height, width

if __name__ == '__main__':
    
    # img = Image.open(r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation-\Data\val\images\Subject_60\49.png")
    
    img = cv2.imread(r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation-\Data\val\images\Subject_60\49.png", cv2.IMREAD_GRAYSCALE)
    
    Segmentation_Model = Segmentation()
    pred, h, w, tumor = Segmentation_Model.run(img)
    
    plt.figure(figsize=(10, 5))
    
    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
    plt.title("Input Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="gray" if pred.ndim == 2 else None)
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
    plt.imshow(pred, cmap="gray", alpha=0.7)  # 'jet' for color mask and transparency with alpha
    plt.title("Image with Predicted Mask")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
