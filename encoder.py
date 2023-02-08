import cv2
import numpy as np



def encode(origimage, file_name=True):
    if file_name:
        img = cv2.imread(origimage, cv2.IMREAD_GRAYSCALE)
    else:
        img = origimage
 

    dx = np.gradient(img, axis=1)
    dy = np.gradient(img, axis=0)
    #dx = np.gradient(result, axis=1)
    #dy = np.gradient(result, axis=0)

    saliency = np.abs(dx) + np.abs(dy)

    orientation = np.pi / 2 + np.arctan2(dy, dx)

    return saliency, orientation
    
