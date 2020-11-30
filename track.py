from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
import numpy as np
import threading
# https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './yolov5')


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def get_center(li):
    x_c = int(((li[2] - li[0]) / 2) + li[0])
    y_c = int(((li[3] - li[1]) / 2) + li[1])
    return [x_c, y_c]


def cal_distance(A, B):
    di = math.sqrt(math.pow((A[0] - B[0]), 2) + math.pow((A[1] - B[1]), 2))
    return di


def append_class(output, det):
    li_center_output = []
    li_center_det = []
    for i in output:
        li_center_output.append(get_center(i))
    for i in det:
        li_center_det.append(get_center(i))
    #     print(li_center_output)
    #     print(li_center_det)
    li_dist = []
    for i in li_center_output:
        list_dis_output_det = []
        for j in li_center_det:
            list_dis_output_det.append(cal_distance(i, j))
        li_dist.append(list_dis_output_det.index(min(list_dis_output_det)))
    #     print(li_dist)
    kq = []
    for i in range(len(output)):
        i_li = output[i].tolist()
        i_li.append(int(det[li_dist[i]][-1]))
        kq.append(i_li)
    return kq

def assigned_center(obj, li_per):
    li_kq = []
    li_distance = []
    center_obj = get_center(obj)
    for i in li_per:
        center_per_i = get_center(i)
        distance_obj_peri = cal_distance(center_obj, center_per_i)
        li_distance.append(distance_obj_peri)

    min_distance = li_distance.index(min(li_distance))
    li_kq.append(li_per[min_distance])
    li_kq.append(obj)

    return li_kq


def bbox_rel(image_width, image_height,  *xyxy):
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


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0   
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


class ObjectOfPerson(object):

    def __init__(self):
        self.id_per = 0
        self.id_obj = 0
        self.list_obj = []
        self.list_obj_of_per = []
        self.id_distinct = []

    def get_id_track(self, id_class_track):

        if id_class_track not in self.list_obj:
            self.list_obj.append(id_class_track)


    def reset_id_track(self):
        self.list_obj = []

    def display_id_track(self):
        print(self.list_obj)
        return self.list_obj


nguoi_vat = ObjectOfPerson()

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    # device = select_device(opt.device)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use: {}".format(device))
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    list_obj_id = []
    li_out = []
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            text = ''
            list_obj = []

            list_per = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    text += '%g %ss, ' % (n, names[int(c)])
                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)


                # for outputi in outputs:
                #     center_i = get_center(outputi)
                #     if center_i[1] < 300:
                #         if list(outputi) not in list_per:
                #             list_per.append(list(outputi))
                #     else:
                #         if list(outputi) not in list_obj:
                #             list_obj.append(list(outputi))
                #
                #             if outputi[-1] not in list_obj_id:
                #                 list_obj_id.append(outputi[-1])

                # print("list_obj", list_obj)
                # print("list_per", list_per)
                # print("list_obj_id", list_obj_id)
                # print("-"*100)

                if len(outputs) == len(det):
                    output_classes = append_class(outputs, det)
                    print("*" * 100)
                    # print("outputs", outputs)
                    # print("output_classes", output_classes)
                    li_per = []
                    li_obj = []
                    for oc in output_classes:
                        if oc[-1] == 0:
                            if oc not in li_per:
                                li_per.append(oc)
                        else:
                            if oc not in li_obj:
                                li_obj.append(oc)
                    # print("li_per", li_per)
                    # print("li_obj", li_obj)
                    if len(li_per) != 0:
                        for obj in li_obj:
                            print("Vật đang xét: ", obj)
                            center_i = get_center(obj)
                            if center_i[1] < 300:
                                li_kq = assigned_center(obj, li_per)
                                print("Ra khỏi vùng: ", li_kq)
                            else:
                                print("Không ra khỏi vùng: ", li_per)
                                
                    
                #     print("det", det.tolist())
                #
                #     for i in output_classes:
                #         nguoi_vat.get_id_track([i[-2], i[-1]])
                #
                #     text_tracking = str(nguoi_vat.display_id_track())

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)





            # Stream results
            if view_img:
                cv2.rectangle(im0, (0, 300), (640, 480), [0, 0, 255], 5)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 67], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
