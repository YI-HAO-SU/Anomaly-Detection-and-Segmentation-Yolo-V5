python detect.py --weights 'yolov5-Kfolds-1225-test/yolov5x-e-100-fold-1/weights/best.pt' \
                 --source 'DIP_final/datasets_folds_1/images/valid' \
                 --data 'DIP_final/datasets_folds_1.yaml' \
                 --conf-thres 0.1  \
                 --iou-thres 0.25 \
                 --save-txt \
                 
