import numpy as np
import cv2

def getLabelFromScore (score):
    """
    Returns the label based on the probability
    if score >= 0.5, return 'mask'
    else return 'non_mask'
    """
    if (score >= 0.5): return 'Mask'
    else: return 'Non mask'

def predict_classification (img, model):
	img = np.array(img)
	img = cv2.resize(img,(150,150))
	img = img / 255.0
	x = np.expand_dims(img, axis=0)
	score = model.predict(x)[0][0]
	return getLabelFromScore (score)