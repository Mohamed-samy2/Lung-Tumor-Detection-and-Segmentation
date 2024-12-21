import os
import cv2
import numpy as np
import tensorflow as tf
from ..BaseModel import BaseModel
import matplotlib.pyplot as plt
# Define the custom loss function
class BCEDiceLoss(tf.keras.losses.Loss):
    def _init_(self, epsilon=1e-6, prob_of_bce=0.8, **kwargs):
        super(BCEDiceLoss, self)._init_(**kwargs)
        self.epsilon = epsilon
        self.prob_of_bce = prob_of_bce
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        # Binary cross-entropy loss from logits
        bce_loss = self.bce(y_true, y_pred)

        # Flatten the tensors
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        # Compute Dice coefficient
        intersection = tf.reduce_sum(y_pred * y_true)
        union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)

        # Dice loss is 1 - Dice coefficient
        dice_loss = 1 - dice


class Cropped_Segmentation(BaseModel):
    def __init__(self):
        # Load your saved model
        self.model = tf.keras.models.load_model(
            os.path.join(os.path.dirname( os.path.abspath(__file__)),'resnetUnetPP256.h5'),
            custom_objects={'BCEDiceLoss': BCEDiceLoss},
            compile=False  # Disable compilation if you're not retraining the model
            )
    
    def run(self,images , boxes):
        results = []
        h_w = []
        for img, box in zip(images,boxes):
            preprocessed_image = self._image_preprocessing(img)
            # Make predictions
            predicted_mask = self.model.predict(preprocessed_image)
            mask = self._image_postprocessing(predicted_mask,box)
            h_w.append(self._get_hw(mask))
            results.append(mask)
        
        return results , h_w
    
    def _image_postprocessing(self,mask,box):
        
        mask = np.squeeze(mask, axis=(0, -1)) # (256,256)
        
        xmin, ymin, xmax, ymax = map(int, box)
        mask = cv2.resize(
        mask,
        (xmax-xmin,ymax-ymin),  # W , H
        interpolation=cv2.INTER_LANCZOS4 
        )
        mask = (mask > 0.7).astype(np.uint8)
        return mask
        
    def _image_preprocessing(self,img):
        img  = cv2.resize(img,(256,256))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1) # (1,256,256,1)
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _get_hw(self,mask):
        rows, cols = np.where(mask == 1)
        if rows.size > 0 and cols.size > 0:
            height = rows.max() - rows.min() + 1
            width = cols.max() - cols.min() + 1
        else:
            # If no non-zero pixels equal to 1 are found, return 0 for height and width
            height = 0
            width = 0
        return (height, width)

if __name__ == "__main__":
    img = cv2.imread(r"E:\Computer Vision Project\Lung-Tumor-Detection-and-Segmentation-\Data\val\images\Subject_60\49.png", cv2.IMREAD_GRAYSCALE)
    print(Cropped_Segmentation().run([img],None))
