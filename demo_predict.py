from ultralytics import YOLO

yolo = YOLO("runs/detect/primary/8n-p6-Ghost-SPDConv(C3)-BiFPN-FCIoU/weights/best.pt", task="detect")

results = yolo(source="datasets/bvn/images/test1", save=True)

