from ultralytics import YOLO


# Load a model 加载预训练模型
# 添加注意力机制，SEAtt_yolov8.yaml 默认使用的是n

if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO("ultralytics/models/v8/Sugar-model/primary/yolov8-BiFPN-Ghost-SPDConv-C3.yaml")  # build a new model from scratch
#    model = YOLO("weights/yolov8s.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="mydata.yaml", workers=8, epochs=300, batch=16, cache=False, name='8n-Ghost-SPDConv-C3-BiFPN')  # train the model



