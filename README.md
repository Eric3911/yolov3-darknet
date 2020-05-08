
### Darknet版YOLOv3教程

### 温馨提示
如果出现了模型的label不显示可以修改主程序里面把加载coco那段代码改成voc就可以

该教程基于darknet训练的模型使用了cuda9.0和cudnn7.4用于加速张量计算这个框架在分类、检测、分割等计算机视觉任务上体现非凡工程价值和学术价值。
关于yolo框架下的使用整理出方便学习和工程使用文档及源代码部分注释方便研究和实践，如果相关问题请留言jungangan@outlook.com交流。

# relatework

keras 

  https://github.com/Eric3911/yolov3_keras
  
  keras原版本：https://github.com/Adamdad/keras-YOLOv3-mobilenet，
  
  keras改进版：https://github.com/Eric3911/YOLOv3-Mobilenet


pytorch

  https://github.com/ultralytics/yolov3

  https://github.com/eriklindernoren/PyTorch-YOLOv3

caffe

  linux的caffe：https://github.com/eric612/MobileNet-YOLO

  windows的caffe：https://github.com/eric612/Caffe-YOLOv3-Windows 


maxnet：

  https://gluon-cv.mxnet.io/model_zoo/detection.html#yolo-v3
  https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo
  https://github.com/sufeidechabei/gluon-mobilenet-yolov3


NCS2上部署yolo_tiny，这个模型是为了让工程化变得简单后续我们使用相关训练和在ARM上的部署方案供大家使用和研究。
https://mp.weixin.qq.com/s/kjii53YgOSFCA84Tbl6K0Q


![输入测试](https://github.com/Eric3911/Darknet-YOLOv3/blob/master/pred_input.jpg)

![识别结果](https://github.com/Eric3911/Darknet-YOLOv3/blob/master/pred_output.jpg)
