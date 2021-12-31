import argparse

import torch
import loadModel

from detect import Detect
from utils.general import  check_requirements

class loadingModel:
    def __init__(self):
        self.source = None
        self.device= None
        self.model= None
        self.imgsz= None
        self.dataset= None
        self.half= None
        self.opt= None
        self.classify= None
        self.modelc= None
        self.webcam= None
        self.save_dir= None
        self.names= None
        self.save_txt= None
        self.save_img= None
        self.view_img= None
        self.colors= None

        self.save_dir = None
        self.info_str= None


        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

        opt = parser.parse_args()
        print(opt)
        check_requirements()

        with torch.no_grad():
             self.source, self.device, self.model, self.imgsz, self.dataset, self.half, self.opt, self.classify, self.modelc, self.webcam, self.save_dir, self.names,self.save_txt,self.save_img, self.view_img, self.colors =  Detect(opt)

    def startDetect(self, data_dir):

        self.save_img, self.info_str = loadModel.detect(self.source, self.device, self.model, self.imgsz, self.dataset, self.half, self.opt, self.classify, self.modelc, self.webcam, self.save_dir, self.names,self.save_txt,self.save_img, self.view_img, self.colors,data_dir)

