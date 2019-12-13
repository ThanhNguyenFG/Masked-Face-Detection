# -*- coding: utf-8 -*-
# @Author: User
# @Date:   2019-10-18 15:54:30
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-22 17:27:24
import cv2
import os
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import box_utils
from tensorflow.keras.models import load_model
from inceptionv3_binary_classification import *

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


filename = 'test.jpg'
model = 'ultra-light'
scale = 1

classification_path = 'classification/models/model-10ep.h5'
model = load_model(classification_path)

onnx_path = 'UltraLight/models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)

input_name = ort_session.get_inputs()[0].name

raw_img = cv2.imread(os.path.join('TestImg',filename))
h, w, _ = raw_img.shape
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 480))
img_mean = np.array([127, 127, 127])
img = (img - img_mean) / 128
img = np.transpose(img, [2, 0, 1])
img = np.expand_dims(img, axis=0)
img = img.astype(np.float32)

t0 = time.time()
confidences, boxes = ort_session.run(None, {input_name: img})
boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
t1 = time.time()
print(f"took {round(t1-t0, 3)} to get {boxes.shape[0]} faces")

for i in range(boxes.shape[0]):
    box = boxes[i, :]
    x1, y1, x2, y2 = box
    cv2.rectangle(raw_img, (x1, y1), (x2, y2), (80,18,236), 2)
    img_clss = raw_img[y1:y2,x1:x2]
    label = predict_classification(img_clss, model)

font = cv2.FONT_HERSHEY_DUPLEX
text = f'took {round(t1-t0, 3)} to get {boxes.shape[0]} faces, Label: {label}'
cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)
# raw_img = draw_labels_and_boxes(raw_img, boxes, confidences, classids, idxs, colors, labels)

while True:
    cv2.imshow('IMG', raw_img)
    cv2.imshow('IMGxx', img_clss)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
