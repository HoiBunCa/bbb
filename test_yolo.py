import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


import numpy as np
import cv2

conf_thres = 0.5
iou_thres = 0.4
device = torch.device("cuda:0")

model = torch.load("yolov5s.pt")['model']
model.half().to(device)
# print(model)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
cap = cv2.VideoCapture(0)

class TrackingClass(object):
    def __int__(self):

        self.pred = None
        self.pred_new = None

    def initial_frame(self,pred):
        self.pred = pred

    def get_input(self, pred):
        self.pred_new = pred

    def calculate_input(self):

        li_center_old = []
        for i in self.pred:
            center = self.get_center(i)
            li_center_old.append(center)
        print("****************")

        li_out = []
        for i in range(len(self.pred_new)):
            center_i = self.get_center(self.pred_new[i])

            li_dis = []
            for j in li_center_old:
                dis_eu = self.distance_euclid(center_i, j)
                li_dis.append(dis_eu)
            try:
                min_dis_indices = li_dis.index(min(li_dis))

                self.pred_new[i][-1] = self.pred[min_dis_indices][-1]

                li_out.append(self.pred_new[i])
                print("li_out", li_out)
                self.pred = li_out
            except:
                pass

    def get_output(self):
        return self.pred

    def get_center(self, li):
        x_center = int((li[2]-li[0])/2) + li[0]
        y_center = int((li[3]-li[1])/2) + li[1]
        return (x_center, y_center)

    def distance_euclid(self, A, B):
        dis = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
        return dis

count1 = 0
Tracking = TrackingClass()


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (416, 416))
    # Our operations on the frame come here
    img = [letterbox(x, new_shape=640)[0] for x in [frame]]
    # Stack
    img = np.stack(img, 0)
    # Convert

    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[67,0], agnostic=False)
    # Inference
    for i, det in enumerate(pred):
        im0 = frame
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label="", color=colors[int(cls)], line_thickness=3)
    if len(pred) != 0:
        pred[0] = pred[0].tolist()
        pred = pred[0]
        print("pred", pred)

        for i in range(len(pred)):
            pred[i].append(i)

        if count1 == 0:
            Tracking.initial_frame(pred)

        Tracking.get_input(pred)
        Tracking.calculate_input()
        out_pred = Tracking.get_output()
        # print("out_pred", out_pred)
        for i in range(len(out_pred)):
            x1 = out_pred[i][0]
            y1 = out_pred[i][1]
            x2 = out_pred[i][2]
            y2 = out_pred[i][3]
            c1, c2 = (int(out_pred[i][0]), int(out_pred[i][1])), (int(out_pred[i][2]), int(out_pred[i][3]))
            cc1 = (int(out_pred[i][0]), int(out_pred[i][1])-10)
            text = out_pred[i][-1]
            cv2.putText(frame, str(text), cc1, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 3, cv2.LINE_AA)
            cv2.rectangle(frame, c1, c2, (0,0,255), 3)

        count1 +=1






    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()