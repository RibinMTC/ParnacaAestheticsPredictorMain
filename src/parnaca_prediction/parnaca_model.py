import cv2
import caffe
import numpy as np
from caffe.proto import caffe_pb2


class ParnacaModel:
    """
    Class Responsibilities:
        1.  Initialize Parnaca model with a pretrained model parameters using the caffe framework.
        2.  Perform prediction for a given image using the initialized model.

    """
    def __init__(self, image_mean, deployment_file, model_file, input_layer_name, image_width, image_height):
        self.__input_layer_name = input_layer_name
        self.__image_width = image_width
        self.__image_height = image_height
        self.__load_model(image_mean, deployment_file, model_file)

    def predict(self, img_to_predict):
        resized_img = self.__transform_img(img_to_predict)

        self.__net.blobs[self.__input_layer_name].data[...] = self.__transformer.preprocess(self.__input_layer_name,
                                                                                            resized_img)
        out = self.__net.forward()
        return out

    def __load_model(self, image_mean, deployment_file, model_file):
        mean_blob = caffe_pb2.BlobProto()
        with open(image_mean) as f:
            mean_blob.ParseFromString(f.read())
        mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
            (mean_blob.channels, mean_blob.height, mean_blob.width))

        mean_image = self.__array_to_image(mean_array)
        mean_image = self.__transform_img(mean_image)
        mean_array = self.__image_to_array(mean_image)

        # Read model architecture and trained model's weights
        self.__net = caffe.Net(deployment_file,
                               model_file,
                               caffe.TEST)

        # Define image transformers
        print "Shape mean_array : ", mean_array.shape
        print "Shape net : ", self.__net.blobs[self.__input_layer_name].data.shape
        self.__net.blobs[self.__input_layer_name].reshape(1,  # batch size
                                                          3,  # channel
                                                          self.__image_width, self.__image_height)  # image size
        self.__transformer = caffe.io.Transformer(
            {self.__input_layer_name: self.__net.blobs[self.__input_layer_name].data.shape})
        self.__transformer.set_mean(self.__input_layer_name, mean_array)
        self.__transformer.set_transpose(self.__input_layer_name, (2, 0, 1))

    def __transform_img(self, img):
        # Image Resizing
        img = cv2.resize(img, (self.__image_width, self.__image_height), interpolation=cv2.INTER_CUBIC)
        return img

    def __array_to_image(self, array):
        return np.transpose(array, axes=[1, 2, 0])

    def __image_to_array(self, array):
        return np.transpose(array, axes=[2, 0, 1])
