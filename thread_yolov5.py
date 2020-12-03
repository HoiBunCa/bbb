from threading import Thread
import cv2
import torch
import shutil
import time
import torch.backends.cudnn as cudnn
import numpy as np

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class YoloStreamClass(object):

    def __init__(self, src=0):
        self.source = src
        self.weights = "yolov5/weights/yolov5s.pt"
        self.config_deepsort = "deep_sort/configs/deep_sort.yaml"
        self.imgsz = 640
        self.pred = None
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        self.cfg = get_config()
        self.cfg.merge_from_file(self.config_deepsort)
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                            max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.device = torch.device("cuda")
        self.half = self.device.type != 'cpu'
        self.model = torch.load(self.weights, map_location=self.device)['model'].float()  # load to FP32
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()  # to FP16
        self.view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.threadx = Thread(target=self.stream)
        self.threadx.daemon = True
        self.threadx.start()

    def stream(self):
        self.dataset = LoadStreams(self.source, img_size=self.imgsz)
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(self.dataset):
            print("imgimgimg", img.shape)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            self.pred = non_max_suppression(pred, 0.5, 0.5, classes=None,
                                       agnostic=False)
            t2 = time_synchronized()
            print("AAAAAAAAAA", self.pred)
            for i, det in enumerate(self.pred):  # detections per image

                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        img_h, img_w, _ = im0.shape
                        x_c, y_c, bbox_w, bbox_h = self.bbox_rel(img_w, img_h, *xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = self.deepsort.update(xywhs, confss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        self.draw_boxes(im0, bbox_xyxy, identities)
                    cv2.imshow("image", im0)

    def bbox_rel(self, image_width, image_height, *xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    def draw_boxes(self, img, bbox, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def get_pred(self):
        p = self.pred
        return p


Yolo = YoloStreamClass(src=0)
while True:
    print("11111111111", Yolo.get_pred())