from .BaseModel import BaseModel
from .Segmentation import Segmentation
from .Yolo import YoloDetection
from .Faster_RCNN import FasterRCNN
from .MedSam import MedSam
from .Cropped_Segmentation import Cropped_Segmentation
import cv2
import numpy as np  

class MainModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.Segmentation_Model = Segmentation()
        self.Yolo_Model = YoloDetection()
        self.Faster_RCNN_Model = FasterRCNN()
        self.Medsam_Model = MedSam()
        self.Cropped_Segmentation_Model = Cropped_Segmentation()
        
    def run(self,img):
        
        detect = False
        boxes, labels, scores, cropped_fast = self.Faster_RCNN_Model.run(img.copy())
        annotated_image, bounding_boxes, resized_cropped_tumors, tumor_count, tumor_scores = self.Yolo_Model.run(img.copy())
        annotate_image_fast = self.__draw_bounding_boxes(img.copy(),boxes,labels,scores)
        
        if len(boxes) > len(bounding_boxes):
            medsam_out = self._medsam_mask(self.Medsam_Model.run(img.copy(),boxes)) ## take img and list of boxes and return list of masks (256,256)
            cropped ,h_w = self.Cropped_Segmentation_Model.run(cropped_fast,boxes) ## take img and list of masks and return list of cropped images (256,256)
            segmented_image= self._segment_cropped(img.copy(),cropped,boxes)
            detect = True
            return segmented_image , medsam_out , annotate_image_fast , h_w , len(labels) , boxes , detect
    
        elif len(boxes) < len(bounding_boxes):
            medsam_out = self._medsam_mask(self.Medsam_Model.run(img.copy(),bounding_boxes)) ## take img and list of boxes and return list of masks (256,256)
            cropped , h_w = self.Cropped_Segmentation_Model.run(resized_cropped_tumors,bounding_boxes) ## take img and list of masks and return list of cropped images (256,256)
            segmented_image = self._segment_cropped(img.copy(),cropped,bounding_boxes) ## take img and list of masks and return list of cropped images (256,256)
            detect = True
            return segmented_image , medsam_out , annotated_image , h_w , tumor_count , bounding_boxes,detect
        
        elif len(boxes) == 0 and len(bounding_boxes) == 0:
            pred, h, w, tumor = self.Segmentation_Model.run(img.copy()) ## take img and return mask , height , width , tumor existence
            if tumor == True:
                tumor = 1
            else: 
                tumor = 0
            return annotated_image , pred , h , w , bounding_boxes , tumor , detect
        
        yolo , fast = 0 , 0
        for fast_score , yolo_score in zip(scores, tumor_scores):
            if fast_score > yolo_score:
                fast += 1
            else:
                yolo += 1
        
        if fast > yolo:
            medsam_out = self._medsam_mask(self.Medsam_Model.run(img.copy(),boxes)) ## take img and list of boxes and return list of masks (256,256)
            cropped , h_w = self.Cropped_Segmentation_Model.run(cropped_fast,boxes) ## take img and list of masks and return list of cropped images (256,256)
            segmented_image = self._segment_cropped(img.copy(),cropped,boxes)
            detect = True
            return segmented_image , medsam_out , annotate_image_fast , h_w , len(labels) , boxes , detect

        else:
            medsam_out = self._medsam_mask(self.Medsam_Model.run(img.copy(),bounding_boxes)) ## take img and list of boxes and return list of masks (256,256)
            cropped , h_w = self.Cropped_Segmentation_Model.run(resized_cropped_tumors,bounding_boxes)
            segmented_image = self._segment_cropped(img.copy(),cropped,bounding_boxes) ## take img and list of masks and return list of cropped images (256,256)
            detect = True
            return segmented_image , medsam_out , annotated_image , h_w , tumor_count , bounding_boxes,detect
        
    def _segment_cropped(self,img,cropped,boxes):
        
        for crop,box in zip(cropped,boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            img[ymin:ymax, xmin:xmax] = crop*255
            
        return img
    
    def _medsam_mask(self,masks):
        final_mask = np.zeros((256, 256))
        for mask in masks:
            final_mask += mask
        
        return final_mask
    
    def __draw_bounding_boxes(self,img,boxes,labels,scores):
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int,box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return img


if __name__ == "__main__":
    
    img = cv2.imread(r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation-\Data\val\images\Subject_60\49.png", cv2.IMREAD_GRAYSCALE)
    MainModel().run(img)