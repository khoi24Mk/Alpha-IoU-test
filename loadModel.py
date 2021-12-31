import os
import pathlib

import torch
import time
import cv2
from pathlib import Path

from utils.datasets import  LoadImages
from utils.torch_utils import time_synchronized


from utils.general import  non_max_suppression, apply_classifier,  scale_coords, xyxy2xywh
from utils.plots import plot_one_box


def detect(source, device, model, imgsz, dataset, half, opt, classify, modelc, webcam, save_dir, names, save_txt,
           save_img, view_img, colors,data_dir):

    for xx in names:
        print(xx)

    save_detection = ""

    stride = int(model.stride.max())  # model stride
    dataset = LoadImages(data_dir, img_size=imgsz, stride=stride)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    info_str = []
    folder_index = 0
    print("PATH DIR")
    print(save_dir)
    while (os.path.isdir(str(save_dir / f'det{folder_index}'))):
        folder_index += 1
    pathlib.Path(str(save_dir / f'det{folder_index}')).mkdir(parents=True, exist_ok=True)
    save_detection = (save_dir / f'det{folder_index}')

    for path, img, im0s, vid_cap in dataset:
        print(11111111111)
        print(path)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print(2222222222)
        # Inference
        t1 = time_synchronized()
        print("--------------")
        pred = model(img, augment=opt.augment)[0]
        print(3333333333)
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        print(44444444444)
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        print(5555555555555555)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print("-------------------")
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            print(666666666666)
            p = Path(p)  # to Path





            save_path = str(save_detection / p.name)  # img.jpg
            txt_path = str(save_detection / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            print("save_pathhhhhhhhhhh "+save_path)
            print("txt_pathhhhhhhhh " + txt_path)

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    print("what n ", n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            info_str.append(s)
            print("NEXT")

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
    print("DONE")
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Dtone. ({time.time() - t0:.3f}s)')

    return save_detection, info_str









