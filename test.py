from ultralytics import YOLO

# 加载训练好的模型（.pt 文件）
model = YOLO('D:/Churchill/TongJi/主要课程/工程智能基础/机器人小项目/my_data/runs/detect/train/weights/best_openvino_model/')

# 对一张图片进行预测
results = model('D:/Churchill/TongJi/主要课程/工程智能基础/机器人小项目/my_data/runs/detect/images/blue.jpg')  # 改成你的图片路径

# 显示结果
results[0].show()  # 会弹窗显示标注后的图片

# 打印检测到的目标信息
for box in results[0].boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    print(f"类别: {model.names[cls_id]}, 置信度: {conf:.2f}")