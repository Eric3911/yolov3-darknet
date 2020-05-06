
#Darknet版YOLOv3训练教程

温馨提示：如果出现了模型的label不显示可以修改主程序里面把加载coco那段代码改成voc就可以

该教程基于darknet训练的模型，使用了cuda9.0和cudnn7.4用于加速张量计算这个框架很有研究价值可以修改成为自己内部需要的计算框架。该框架可以做分类、检测、分割，
检测实现有问题可以留言互相学习，关于yolo不同框架下的版本笔记好的我看过一个知乎，下面是我自己整理的方便学习和工程使用，在这个项目中我和朋友把相关的源代码部分注释，这样方便与研究和工程实践，如果相关解释和代码存在问题请留言交流。

MobileNet-YOLOv3
linux的caffe：https://github.com/eric612/MobileNet-YOLO
windows的caffe：https://github.com/eric612/Caffe-YOLOv3-Windows 

keras版本：这个我训练时候出现val_loss错误暂时没解决https://github.com/Adamdad/keras-YOLOv3-mobilenet，后面我们通过对train文件的代码修改的数据切分的部分，目前改模型可以实现工程应用。想要修改版本的代码请您联系我。

maxnet：
https://gluon-cv.mxnet.io/model_zoo/detection.html#yolo-v3
https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo
https://github.com/sufeidechabei/gluon-mobilenet-yolov3

NCS2上部署yolo_tiny，这个模型是为了让工程化变得简单后续我们使用相关训练和在ARM上的部署方案供大家使用和研究。
https://mp.weixin.qq.com/s/kjii53YgOSFCA84Tbl6K0Q


![](https://github.com/Eric3911/Darknet-YOLOv3/blob/master/pred_input.jpg)
![](https://github.com/Eric3911/Darknet-YOLOv3/blob/master/pred_output.jpg)

