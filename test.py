from ultralytics import YOLO

# 加载训练好的模型（.pt 文件）
model = YOLO('weights文件夹之前的完整路径/weights/best_openvino_model/') #如果直接复制Windows给的路径要注意把"\"改为"/"，同时记得删掉多余的双引号

# 对一张图片进行预测
results = model('完整图片路径')  # 改成你的图片路径

# 显示结果
results[0].show()  # 会弹窗显示标注后的图片

# 打印检测到的目标信息
for box in results[0].boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    print(f"类别: {model.names[cls_id]}, 置信度: {conf:.2f}")