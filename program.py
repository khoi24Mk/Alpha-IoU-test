import os

os.system(' python detect.py --weights runs/train/voc_yolov5s_3diou/weights/best.pt --img 640 --conf 0.009 --iou 0.45 --source img.jpg')
