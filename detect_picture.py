import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

import os
# -1表示不用GPU，使用CPU，0、1、2、3表示GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 0、1、2表示显示信息的详细程度，2为最简洁显示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义FLAGS变量，便于后续代码使用，利于用户命令调用与赋值
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/input_pic/', 'path to input image')
flags.DEFINE_string('output', './output_pic/', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    # 获取GPU列表
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # 调用GPU
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    # FLAGS.tiny默认为False，判定是用yolo还是tiny
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    # 载入权重，其中logging.info表示日志输出，INFO：程序正常运行时使用
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    #载入类别名称，其中xx.strip()表示去除空格，xx.readlines()表示逐行读取
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    # 载入需检测的图片
    # 返回FLAGS.image的文件夹包含的文件或文件夹的名字的列表

    if FLAGS.image != './data/input_pic/':
        image = os.path.basename(FLAGS.image)
        image_pre, ext = os.path.splitext(image)
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)
        detect_image(img_raw, image_pre, ext, yolo, class_names)

    else:
        imagelist = os.listdir(FLAGS.image)
        for image in imagelist:
            # 分离文件名和拓展名（后缀）
            image_pre, ext = os.path.splitext(image)
            imgfile = FLAGS.image + image
            img_raw = tf.image.decode_image(
                open(imgfile, 'rb').read(), channels=3)
            # 进行检测
            detect_image(img_raw, image_pre, ext, yolo, class_names)

    # 绘制yolo的网络结构拓扑
    tf.keras.utils.plot_model(yolo, to_file='./logs/model_yolov3_tb.png', show_shapes=True, show_layer_names=True, rankdir='TB',
                              dpi=900, expand_nested=True)


def detect_image(img_raw, image_pre, ext, yolo, class_names):
    # 重构图片格式——416
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    # 日志中的时间标注，当前时间、运行时间
    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    # 日志中将检测出的目标逐个标注出类别、分数、boxes
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    # 图像上画出检测出的目标，并标注相关信息，但是nums是什么？
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output + image_pre + '_out' + ext, img)
    # 图像输出
    logging.info('output saved to: {}'.format(FLAGS.output + image_pre + '_out' + ext))
    return


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

