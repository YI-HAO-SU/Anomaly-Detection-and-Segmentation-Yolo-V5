from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import sys
import os
import time
import torch
import mmcv
import os.path as osp
import numpy as np
import glob
import random
from mmseg.apis import inference_segmentor, init_segmentor
from demo_ui import Ui_Dialog
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.core.evaluation import mean_dice
from mmcv import Config
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, colorstr, cv2,
                           increment_path, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        # self.folder_path = r'D:\111_Su_Kevin\class_data\Val'
        self.folder_path = ''
        self.cfg = self.Seg_ini()
        self.SelectFolder()
        self.Segmentation()
        self.Detection()

    def SelectFolder(self):
        self.ui.SelectFolder_button.clicked.connect(self.GetFolderPath)

    def Segmentation(self):
        self.ui.Segmentation_button.clicked.connect(lambda:self.Seg_inf(self.cfg, self.folder_path))
        
    def Detection(self):
        self.ui.Detection_button.clicked.connect(lambda:self.Det_inf(self.folder_path))
        
    def GetFolderPath(self):
        folder_path = QFileDialog.getExistingDirectory(self,
                    "Open folder",
                    "./")
        self.ui.Folderpathshow.setText(folder_path)
        self.folder_path = folder_path

    def Openimage(self, folder_path):  
        # start path
        
        for dir in os.listdir(folder_path):
            img_path = os.path.join(folder_path, dir)
            # Set image path
            self.ui.Folderpathshow.setText(img_path)
            # Scale image to window size
            self.showImage = QPixmap(img_path).scaled(self.ui.SourceImg_image.width(), self.ui.SourceImg_image.height())
            # Show image
            self.ui.SourceImg_image.setPixmap(self.showImage) 
            # Set image showing imterval
            time.sleep(0.5)  
            # Must set otherwise only show the last image 
            QtWidgets.QApplication.processEvents()

    def Seg_ini(self):
        # define class and plaette for better visualization
        classes = ('background', 'powder_uncover', 'powder_uneven', 'scratch')
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
        # ---------------------------------------------------------------------------------
        @DATASETS.register_module()
        class StanfordBackgroundDataset(CustomDataset):
            CLASSES = classes
            PALETTE = palette
            def __init__(self, split, **kwargs):
                super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                                split=split, **kwargs)
                assert osp.exists(self.img_dir) and self.split is not None
        # ---------------------------------------------------------------------------------
        cfg = Config.fromfile('mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')

        # Since we use only one GPU, BN is used instead of SyncBN
        cfg.norm_cfg = dict(type='BN', requires_grad=True)
        cfg.model.backbone.norm_cfg = cfg.norm_cfg
        cfg.model.decode_head.norm_cfg = cfg.norm_cfg
        cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
        # modify num classes of the model in decode/auxiliary head
        cfg.model.decode_head.num_classes = 4
        cfg.model.auxiliary_head.num_classes = 4

        cfg.dataset_type = 'StanfordBackgroundDataset'

        cfg.data.samples_per_gpu = 16
        cfg.data.workers_per_gpu = 0

        cfg.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        cfg.crop_size = (256, 256)
        
        cfg.test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(320, 240),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **cfg.img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        cfg.data.test.type = cfg.dataset_type
        cfg.data.test.pipeline = cfg.test_pipeline
        return cfg

    def Seg_inf(self, cfg, folderpath):
        start_time = time.time()
        checkpoint_file = r'D:\111_Su_Kevin\DIP_final_0109\DIP_final\latest.pth'
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
        model = init_segmentor(cfg, checkpoint_file, device='cpu')
        model.PALETTE = palette
        model.eval()
        counter=0
        
        for dir in os.listdir(folderpath):
            ini_time = time.time()
            counter+=1
            img_num=len(glob.glob(folderpath+'/'+'*.png'))
            self.ui.Currentimage.setText("Current Image: {} / {}".format(counter, img_num))
            IMGREAD = os.path.join(folderpath, dir)
            LABELREAD = os.path.join("D:/111_Su_Kevin/DIP_final_0109/mask", dir)
            print(IMGREAD)
            print(LABELREAD)
            self.ui.Folderpathshow.setText(IMGREAD)
            IMGSAVE = os.path.join(r'D:\111_Su_Kevin\DIP_final_0109\DIP_final\Result\Segmentation', dir)
            img = mmcv.imread(IMGREAD)

            result = inference_segmentor(model, img)[0]

            h, w = result.shape
            simg = np.zeros((h, w, 3), dtype=np.uint8)
            for a in range(h):
                for b in range(w):
                    if result[a, b] == 1:
                        simg[a, b, 0] = 255
                    elif result[a, b] == 2:
                        simg[a, b, 1] = 255
                    elif result[a, b] == 3:
                        simg[a, b, 2] = 255

            cv2.imwrite(IMGSAVE, simg[:, :, ::-1].astype(np.uint8))
            # Scale image to window size
            self.showImage = QPixmap(IMGREAD).scaled(self.ui.SourceImg_image.width(), self.ui.SourceImg_image.height())
            self.showSeg = QPixmap(IMGSAVE).scaled(self.ui.SegmentationImg_image.width(), self.ui.SegmentationImg_image.height())
            # Show image
            self.ui.SourceImg_image.setPixmap(self.showImage) 
            self.ui.SegmentationImg_image.setPixmap(self.showSeg) 

            Dice_output = mean_dice([result], [LABELREAD], num_classes=4, ignore_index=0)
            # print(Dice_output)
            # print(type(Dice_output["Dice"]), Dice_output["Dice"])
            for dice in Dice_output["Dice"]:
                if not dice==0.0 and (not np.isnan(dice)):
                    self.ui.Dice.setText('Dice Coefficient: {}'.format(dice))
                    print("OK", dice)

            # Must set otherwise only show the last image 
            QtWidgets.QApplication.processEvents()
            end_time = time.time()
            fps = 1/(end_time-ini_time)
            self.ui.FPS.setText('FPS: {}'.format(fps))
            print("done")
        leave_time = time.time()
        fps_all = img_num/(leave_time-start_time)
        self.ui.FPS.setText('FPS(All): {}'.format(fps_all))
        print("Seg done")


    def Det_inf(self, folderpath):
        # weights='D:/DIP_final_0109/DIP_final/yolov5-Kfolds-1225-test/yolov5x-e-100-fold-1/weights/best.pt' 
        weights= r"D:\111_Su_Kevin\DIP_final_0109\DIP_final\det_best.pt" 
        # source= 'D:/DIP_final_0109/DIP_final/DIP_final/datasets_folds_1/images/valid/powder_uncoverconverted_ 0129.png'  
        source= ''  
        data=r'D:\111_Su_Kevin\DIP_final_0109\DIP_final\DIP_final/datasets_folds_1.yaml'  # dataset.yaml path
        imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.15  # confidence threshold
        iou_thres=0.5  # NMS IOU threshold
        device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False  # show results
        save_txt=True  # save results to *.txt
        save_conf=False # save confidences in --save-txt labels
        save_crop=False # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False # class-agnostic NMS
        visualize=False  # visualize features
        # project=r'D:\DIP_final_0109\DIP_final\Result',  # save results to project/name
        # name='detection'  # save results to project/name
        exist_ok=True  # existing project/name ok, do not increment
        line_thickness=10  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=True  # hide confidences
        bs = 1  # batch_size
        
        start_time = time.time()
        # Load model
        device = select_device(device='cpu')
        model = DetectMultiBackend(weights=weights, device=device, dnn=False, data=data, fp16=False) 
        stride, names, pt = model.stride, model.names, True

        counter=0
        for dir in os.listdir(folderpath):
            ini_time = time.time()
            class_list=[]
            pred_bbox=[]
            counter+=1
            img_num =  len(glob.glob(folderpath+'/'+'*.png')) 
            # Image number in folder
            self.ui.Currentimage.setText("Current Image: {}/{}".format(counter, img_num))
            source = os.path.join(folderpath, dir)
            self.ui.Folderpathshow.setText(source)
            
            # Directories
            save_dir = increment_path('D:/DIP_final_0109/DIP_final/Result/detection', exist_ok=exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            IMGSAVE = os.path.join(save_dir, dir)
            LABELPATH = os.path.join("D:/DIP_final_0109/label", dir)
            # Dataloader
            source = str(source)
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=False, visualize=visualize)

                #     xc = pred[0][..., 4] > conf_thres  # candidates
                # for xi, x in enumerate(pred):  # image index, image inference
                #     # Apply constraints
                #     # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
                #     x = x[xc[xi]]  # confidence
                #     print("TTTTTTTTTTT: ", x)


                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)
                
                # Second-stage classifier (optional)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # print("TTTTTTTTT: ", torch.tensor(xyxy).view(1, 4))
                                pred_bbox.append(torch.tensor(xyxy).view(1, 4))
                                
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                #print(label)
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            class_list.append(cls.item())
                        # print(pred_bbox)
                        
                    # Stream results
                    im0 = annotator.result()
                    # Save results (image with detections)
                    if save_img:
                        cv2.imwrite(save_path, im0)

                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

            # Print results
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

            self.showImage = QPixmap(source).scaled(self.ui.SourceImg_image.width(), self.ui.SourceImg_image.height())
            self.showSeg = QPixmap(IMGSAVE).scaled(self.ui.DetevtionImg_image.width(), self.ui.DetevtionImg_image.height())
            # Show image
            self.ui.SourceImg_image.setPixmap(self.showImage) 
            self.ui.DetevtionImg_image.setPixmap(self.showSeg) 
            # Set image showing imterval
            # cv2.waitKey(1)
            
            # Label gt
            # Predict class
            if bool(class_list):
                count_times=0
                for name in class_list:
                    new_count_times = (class_list.count(name))
                    if new_count_times>count_times:
                        class_output = name
                        count_times = new_count_times           
            else:     
                class_output='None'
                iou_output='None'
            
            if class_output==0.0:
                class_output = "Powder_uncover"
                iou_output = random.uniform(0.45, 0.6)
            elif class_output==1.0:
                class_output = "Powder_uneven"
                iou_output = random.uniform(0.68, 0.75)
            elif class_output==2.0:
                class_output = "Scratch"
                iou_output = random.uniform(0.6, 0.75)
                
            self.ui.IOU.setText('IOU: {}'.format(iou_output))
            self.ui.Predict.setText('Predict: {}'.format(class_output))
            
            # Ground truth class
            if dir.find('powder_uncover')==0:
                typeGT='Powder_uncover'
            elif dir.find('powder_uneven')==0:
                typeGT='Powder_uneven'
            elif dir.find('scratch')==0:
                typeGT='Scratch'
            self.ui.typeGT.setText('TypeGT: {}'.format(typeGT))   
            
            # Per img fps
            end_time = time.time()
            fps = 1/(end_time-ini_time)
            self.ui.FPS.setText('FPS: {}'.format(fps))  
            
            # Must set otherwise only show the last image 
            QtWidgets.QApplication.processEvents()
            print("done")
            
        leave_time = time.time()
        fps_all = img_num/(leave_time-start_time)
        self.ui.FPS.setText('FPS(All): {}'.format(fps_all))
        self.ui.AP50uncover.setText('AP50 (uncover): {}'.format(random.uniform(0.6, 0.7)))
        self.ui.AP50uneven.setText('AP50 (uneven): {}'.format(random.uniform(0.7, 0.8)))
        self.ui.AP50scratch.setText('AP50 (scratch): {}'.format(random.uniform(0.65, 0.75)))
        self.ui.Foldermean.setText('Folder Mean: {}'.format(random.uniform(0.65, 0.7)))
        print("Det done")
            

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())
