from .BaseModel import BaseModel
from .Segmentation import Segmentation
from .Yolo import YoloDetection
from .Faster_RCNN import FasterRCNN
from .MedSam import MedSam
from .Cropped_Segmentation import Cropped_Segmentation
import cv2


class MainModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.Segmentation_Model = Segmentation()
        self.Yolo_Model = YoloDetection()
        self.Faster_RCNN_Model = FasterRCNN()
        self.Medsam_Model = MedSam()
        self.Cropped_Segmentation_Model = Cropped_Segmentation()
        
    def run(self,img):

        boxes, labels, scores,cropped_fast,w_h_fast = self.Faster_RCNN_Model.run(img)
        
        annotated_image, bounding_boxes, resized_cropped_tumors, tumor_count, tumor_scores,w_h = self.Yolo_Model.run(img)
        
        annotate_image = self.__draw_bounding_boxes(img.copy(),boxes,labels,scores)
        
        pred, h, w, tumor = self.Segmentation_Model.run(img)
        
        if tumor == True or tumor_count >= 1: 
            return annotate_image ,pred , h , w , bounding_boxes ,max(tumor_count,1)
        
        return annotate_image , pred , h , w , bounding_boxes , 0

    def __draw_bounding_boxes(self,img,boxes,labels,scores):
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int,box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return img


if __name__ == "__main__":
    
    img = cv2.imread(r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation-\Data\val\images\Subject_60\49.png", cv2.IMREAD_GRAYSCALE)
    MainModel().run(img)