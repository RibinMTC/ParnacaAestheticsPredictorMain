"""
Module to test parnaca model initially before creating the classes: ParnaceModel and ParnacaAADBModel
"""

#!/usr/bin/env python
# coding: utf8

import json

import glob
import cv2
import caffe
import numpy as np
from caffe.proto import caffe_pb2

# AVA
from pathlib2 import Path

base_path = Path(__file__).parent.parent.parent
MODEL_RESOURCES_ROOT = str((base_path / 'model_resources').absolute()) + '/'
# str((base_path / 'test_images').absolute()) + '/'
# '/local/home/cribin/Documents/AestheticsBackup/Datasets/ava_evaluation_images_with_ground_truth/ava_test_images/'
IMAGES_ROOT = '/local/home/cribin/Documents/AestheticsBackup/Datasets/tx_group_images/'

stored_predictions_json_path = str(
    (base_path / 'model_evaluation/stored_predictions/tx_group_images_with_attributes.json').absolute())

# IMAGE_MEAN = MODEL_RESOURCES_ROOT + 'ava_trained_model/imagenet_mean.binaryproto'
# DEPLOY = MODEL_RESOURCES_ROOT + 'ava_trained_model/initNetArch.deploy'  #
# MODEL_FILE = MODEL_RESOURCES_ROOT + 'ava_trained_model/initModel_iter_16000.caffemodel'
# input_layer = 'data'
# score_layer = 'fc12_weightedSum'

IMAGE_MEAN = MODEL_RESOURCES_ROOT + 'mean_AADB_regression_warp256.binaryproto'
DEPLOY = MODEL_RESOURCES_ROOT + 'initModel.prototxt'
MODEL_FILE = MODEL_RESOURCES_ROOT + 'initModel.caffemodel'
input_layer = 'imgLow'
score_layer = 'fc11_score'

IMAGE_FILE = IMAGES_ROOT + "*jpg"

caffe.set_mode_gpu()
# caffe.set_mode_cpu()

# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    return img


def array_to_image(array):
    return np.transpose(array, axes=[1, 2, 0])


def image_to_array(array):
    return np.transpose(array, axes=[2, 0, 1])


def sum_up_all_attributes_together(out):
    out_sum = 0
    for attribute_name in predicted_attribute_names:
        out_sum += out[attribute_name][0][0]
        print(attribute_name + " " + str(out[attribute_name][0][0]))
    print("Total Sum of out: " + str(out_sum))


predicted_attribute_names = ['fc9_ColorHarmony', 'fc9_MotionBlur', 'fc9_Light', 'fc9_Content', 'fc9_Repetition',
                             'fc9_DoF', 'fc9_VividColor', 'fc9_Symmetry', 'fc9_Object', 'fc9_BalancingElement',
                             'fc9_RuleOfThirds']

'''
Reading mean image, caffe model and its weights
'''
# Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(IMAGE_MEAN) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

mean_image = array_to_image(mean_array)
mean_image = transform_img(mean_image)
mean_array = image_to_array(mean_image)
# Read model architecture and trained model's weights
net = caffe.Net(DEPLOY,
                MODEL_FILE,
                caffe.TEST)

# Define image transformers
print "Shape mean_array : ", mean_array.shape
print "Shape net : ", net.blobs[input_layer].data.shape
net.blobs[input_layer].reshape(1,  # batch size
                               3,  # channel
                               IMAGE_WIDTH, IMAGE_HEIGHT)  # image size
transformer = caffe.io.Transformer({input_layer: net.blobs[input_layer].data.shape})
transformer.set_mean(input_layer, mean_array)
transformer.set_transpose(input_layer, (2, 0, 1))

'''
Making predicitions
'''
# Reading image paths
test_img_paths = [img_path for img_path in glob.glob(IMAGE_FILE)]

# Making predictions
test_ids = []
preds = []
best_image = ''
best_score = 0.0
image_prediction_dict = {}
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_id = img_path.split('/')[-1][:-4]
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net.blobs[input_layer].data[...] = transformer.preprocess(input_layer, img)
    out = net.forward()
    # print out
    sum_up_all_attributes_together(out)
    # {'fc9_ColorHarmony': array([[ 0.22087261]], dtype=float32), 'fc9_MotionBlur': array([[-0.08059776]], dtype=float32), 'fc9_Light': array([[ 0.14933866]], dtype=float32), 'fc9_Content': array([[ 0.01467544]], dtype=float32), 'fc9_Repetition': array([[ 0.18157494]], dtype=float32), 'fc11_score': array([[ 0.55613178]], dtype=float32), 'fc9_DoF': array([[-0.05279735]], dtype=float32), 'fc9_VividColor': array([[ 0.13607402]], dtype=float32), 'fc9_Symmetry': array([[ 0.06802807]], dtype=float32), 'fc9_Object': array([[ 0.00289625]], dtype=float32), 'fc9_BalancingElement': array([[-0.04946293]], dtype=float32), 'fc9_RuleOfThirds': array([[-0.0477073]], dtype=float32)}

    pred_score = out[score_layer][0][0]*10
    image_prediction_dict[image_id] = str(pred_score)
    print img_path, '\t', pred_score
    if pred_score > best_score:
        # print "Better score !"
        best_score = pred_score
        best_image = img_path

with open(stored_predictions_json_path, 'w') as stored_predictions_json_path:
    json.dump(image_prediction_dict, stored_predictions_json_path, indent=2)
print "Best image, based only on fc11_score = ", best_image
