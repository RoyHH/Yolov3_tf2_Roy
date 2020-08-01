# YoloV3 Implemented in TensorFlow 2.1

## Key Features
- [x] TensorFlow 2.1 + Tensorboard 2.1
- [x] absl-py 0.9 + lxml + tqdm
- [x] Opencv 3.4.2
- [x] python 3.6 or 3.7
- [x] `yolov3` with pre-trained Weights
- [x] `yolov3-tiny` with pre-trained Weights
- [x] Inference example
- [x] Transfer learning example
- [x] Eager mode training with `tf.GradientTape`
- [x] Graph mode training with `model.fit`
- [x] Functional model with `tf.keras.layers`
- [x] Input pipeline using `tf.data`
- [x] Tensorflow Serving
- [x] Vectorized transformations
- [x] GPU accelerated
- [x] Fully integrated with `absl-py` from [abseil.io](https://abseil.io)

## To-do List
- [x] Complete
    - [x] detect function
        - [x] detect_picture：folder pictures detect && single picture detect
        - [x] detect_video：webcam && video file
    - [x] convert function：convert pre-trained Darknet weights
    - [x] train function
        - [x] transform Dataset
        - [x] VOC2012 show
        - [x] VOC2012 train
        - [x] tensorboard

- [ ] GTasks
    - [ ] detect functional optimization
        - [x] detect_picture：warning for specific detection object
        - [ ] detect_picture：the latest picture detect
    - [ ] train functional optimization
        - [x] presentation of acc data
        - [ ] tensorboard：real-time observation of training curve 
        - [ ] to anchor free ？
        - [ ] NMS loss optimization
        - [ ] IOU loss optimization
***
##1.Convert pre-trained Darknet weights  
```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights 
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf 

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```
***
##2.Detection
### 2.1 picture
- folder pictures detect
```
# yolov3
python detect_picture.py

# yolov3-tiny
python detect_picture.py --weights ./checkpoints/yolov3-tiny.tf --tiny
```
- single picture detect
```
# yolov3
python detect_picture.py --image ./data/girl.png

# yolov3-tiny
python detect_picture.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./data/girl.png
```
- warning for specific detection object -- take folder pictures detect for example
```
# yolov3
python detect_picture.py --warning bus

# yolov3-tiny
python detect_picture.py --weights ./checkpoints/yolov3-tiny.tf --tiny --warning bus
```
### 2.2 video
```
# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny

# video file with output
python detect_video.py --video path_to_file.mp4 --output ./output_vid/output.avi
```
***
##3.Training
### 3.1 VOC2012数据集训练
>关于如何使用VOC2012数据集从头开始训练的完整教程，参见如下文档（感谢GitHub的zzh8829用户）
```
https://github.com/zzh8829/yolov3-tf2/blob/master/docs/training_voc.md
```
这里我提供一个工具，可以可视化展示VOC2012数据集中所有数据原始的标注情况，文件输出在`./data/voc2012_showout/`
```
python tools/voc2012_show.py
```
具体训练步骤如下：
- (1) Download Dataset
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
mkdir -p ./data/voc2012_raw
tar -xf ./data/voc2012_raw.tar -C ./data/voc2012_raw
ls ./data/voc2012_raw/VOCdevkit/VOC2012 # Explore the dataset
```
- (2) Transform Dataset
```
python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split train \
  --output_file ./data/voc2012_train.tfrecord

python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split val \
  --output_file ./data/voc2012_val.tfrecord
```
>可以使用此工具可视化数据集：（它将输出一个带有标签的随机图像———output.jpg）
```
python tools/visualize_dataset.py --classes=./data/voc2012.names
```

- (3) Training

a. 迁移学习

这一步先需要加载预先训练好的darknet权重。
```
1.Convert pre-trained Darknet weights的内容
```
然后开始训练。（最初的yolov3有80个类，这里演示的是如何在20个类上进行迁移学习）
```
python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 10 \
	--weights ./checkpoints/yolov3.tf \
	--weights_num_classes 80 
```
b. 随机权重训练(不推荐)
```
python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer none \
	--batch_size 16 \
	--epochs 10 \
```
- (4) Inference

因为之前训练了10个epoch，所以checkpoint会有10个，这里是用第9个做演示。
```
# detect from images
python detect_picture.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3_train_9.tf \
	--image ./data/girl.png
```
- (5) tensorboard

观测训练中loss的变化曲线
```
tensorboard --logdir=logs
```

### 3.2 自定义数据集训练
对于自定义数据集的训练，需要将数据集转换成TensorFlow可识别的tfrecord格式。

>例如，可以使用[Microsoft VOTT](https://github.com/Microsoft/VoTT)生成这样的数据集，
或者使用[script](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py)此脚本创建pascal voc数据集。

具体训练过程可参考3.1 VOC2012数据集训练，过程大同小异。
***
##4.Tensorflow Serving
You can export the model to tf serving
```
python export_tfserving.py --output serving/yolov3/1/
# verify tfserving graph
saved_model_cli show --dir serving/yolov3/1/ --tag_set serve --signature_def serving_default
```
The inputs are preprocessed images (see `dataset.transform_iamges`)

outputs are
```
yolo_nms_0: bounding boxes
yolo_nms_1: scores
yolo_nms_2: classes
yolo_nms_3: numbers of valid detections
```
***
##5.GitHub
```
git log --oneline --all --graph
git add . && git commit -m"message"
git tag -a tagName -m"message"   #添加标签 tag
git push origin --tags   #上传所有标签 tag
git tag -d tagName   #删除标签 tag
git push origin :refs/tags/tagName   #强制删除GitHub上的标签 tag
git show tagName   #展示标签备注内容 tag
git branch Name   #创建分支
```
