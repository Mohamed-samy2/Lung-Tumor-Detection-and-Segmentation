# environment and functions
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from ..BaseModel import BaseModel
import cv2

class MedSam(BaseModel):
    def __init__(self):
        #  load model and image
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        torch.cuda.empty_cache()
        
        MedSAM_CKPT_PATH = os.path.join(os.path.dirname( os.path.abspath(__file__)),'medsam_vit_b.pth')
        self.medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        self.medsam_model.to(self.device)
        self.medsam_model.eval()
        
    
    
    def run(self,img,boxes):
        
        masks = []
        for box in boxes:
            img1024 , H, W = self._image_preprocessing(img)
            box_np = np.array([box])
            # transfer box_np t0 1024x1024 scale
            box_1024 = box_np / np.array([W, H, W, H]) * 1024
            
            with torch.no_grad():
                image_embedding = self.medsam_model.image_encoder(img1024) # (1, 256, 64, 64)
            medsam_seg = self.medsam_inference(image_embedding, box_1024, H, W)
            masks.append(medsam_seg)
        
        return masks ##(img_size, img_size)
        
    def showmask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, 0.6])
        
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_box(self,box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
    
    
    @torch.no_grad()
    def medsam_inference(self, img_embed, box_1024, H, W):
        
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings =  self.medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ =  self.medsam_model.mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe= self.medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg
    
    
    def _image_preprocessing(self,img):
        
        if len(img.shape) == 2:
            img_3c = np.repeat(img[:, :, None], 3, axis=-1)
        else:
            img_3c = img
        
        H, W, _ = img_3c.shape
        
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)        
        return img_1024_tensor, H, W
    

if __name__ == '__main__':
    img = cv2.imread(r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation-\Data\val\images\Subject_59\86.png", cv2.IMREAD_GRAYSCALE)
    box = [64,158,77,177]
    medsam = MedSam()
    medsam.run(img,box)