from ..BaseModel import BaseModel
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision.transforms.functional as F

class FasterRCNN(BaseModel):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        torch.cuda.empty_cache()
        
        self.weights_path = os.path.join(os.path.dirname( os.path.abspath(__file__)),'Faster_RCNN_last.pth')
        self.model = self.__get_model(num_classes=2)
        self.model.load_state_dict(torch.load(self.weights_path,weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.COCO_CLASSES = {0:"Background" , 1:"tumor"}
    
    def run(self,img):
        img = self._image_preprocessing(img)
        
        with torch.no_grad():
            prediction = self.model(img)
        
        boxes, labels, scores,cropped_images,w_h = self._postprocessing(img,prediction)
        return boxes, labels, scores,cropped_images,w_h
    
    def __get_model(self,num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    
    def _postprocessing(self,img,prediction):
        
        boxes=prediction[0]['boxes'].cpu().numpy()
        labels=prediction[0]['labels'].cpu().numpy()
        scores=prediction[0]['scores'].cpu().numpy()
        cropped_images = []
        w_h = []
        # Apply the threshold
        threshold = 0.20
        mask = scores > threshold

        # Filter out predictions based on threshold
        filtered_boxes = boxes[mask]
        filtered_labels = labels[mask]
        filtered_scores = scores[mask]
        
        for box in filtered_boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_image = img[ymin : ymax , xmin : xmax]
            w_h.append((xmax-xmin,ymax-ymin))
            cropped_images.append(cropped_image)
        
        filtered_labels = [self.COCO_CLASSES[label] for label in filtered_labels]
        
        return filtered_boxes, filtered_labels, filtered_scores,cropped_images,w_h
    
    def _image_preprocessing(self,img):
        image_tensor = F.to_tensor(img).unsqueeze(0)
        return image_tensor.to(self.device)
