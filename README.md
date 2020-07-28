# YoloV3 Implemented in TensorFlow 2.1

## Key Features

- [x] TensorFlow 2.0
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
- [x] Clean implementation
- [x] Following the best practices
- [x] MIT License

## 1.Convert pre-trained Darknet weights  
```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights 
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf 

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```

## 2.Detection
### 2.1 picture
```
# yolov3
python detect_picture.py

# yolov3-tiny
python detect_picture.py --weights ./checkpoints/yolov3-tiny.tf --tiny
```
### 2.2 video
```
# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny

# video file with output
python detect_video.py --video path_to_file.mp4 --output ./output.avi
```
