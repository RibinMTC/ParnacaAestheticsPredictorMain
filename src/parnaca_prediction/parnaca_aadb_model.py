import cv2
from flask import jsonify
from pathlib2 import Path

from src.parnaca_prediction.parnaca_model import ParnacaModel


class ParnacaAADBModel:
    """
    Class Responsibilities:
    1.  Initialize a ParnacaModel class with a model pretrained on the AADB dataset.
    2.  Use ParnacaModel to perform prediction and return a json response with the predicted attributes(Interesting Content,
        Object Emphasis, Good Lighting, Color Harmony, Vivid Color, Shallow Depth Of Field, Motion Blur, Rule of Thirds,
        Balancing Element, Repetition, Symmetry)
    """

    def __init__(self):
        base_path = Path(__file__).parent.parent.parent
        model_resources_root_path_str = str((base_path / 'model_resources').absolute()) + '/'

        self.__predicted_attribute_names = ['fc9_ColorHarmony', 'fc9_MotionBlur', 'fc9_Light', 'fc9_Content',
                                            'fc9_Repetition',
                                            'fc9_DoF', 'fc9_VividColor', 'fc9_Symmetry', 'fc9_Object',
                                            'fc9_BalancingElement',
                                            'fc9_RuleOfThirds', 'fc11_score']
        self.__json_predicted_attribute_names = ['ColorHarmony', 'MotionBlur', 'GoodLighting', 'InterestingContent',
                                                 'Repetition', 'DoF', 'VividColor', 'Symmetry', 'ObjectEmphasis',
                                                 'BalancingElement', 'RuleOfThirds', 'TotalScore']
        self.__score_layer = 'TotalScore'

        image_mean = model_resources_root_path_str + 'mean_AADB_regression_warp256.binaryproto'
        deployment_file = model_resources_root_path_str + 'initModel.prototxt'
        model_file = model_resources_root_path_str + 'initModel.caffemodel'
        input_layer = 'imgLow'
        image_width = 227
        image_height = 227

        self.__parnaca_model = ParnacaModel(image_mean, deployment_file, model_file, input_layer, image_width,
                                            image_height)

    def predict(self, content_path, start_frame=0, end_frame=0):
        try:
            img_to_predict = cv2.imread(content_path, cv2.IMREAD_COLOR)
            out = self.__parnaca_model.predict(img_to_predict)
            predicted_attributes_dict = self.__create_predicted_attributes_dict(out)
            if predicted_attributes_dict is not None:
                print("Predicted: " + str(predicted_attributes_dict))
                return jsonify(predicted_attributes_dict)
            return "Error during prediction.", 400
        except Exception as e:
            print(str(e))
            return "Error during prediction.", 400

    def get_score_layer_name(self):
        return self.__score_layer

    def __create_predicted_attributes_dict(self, out):
        predicted_attributes = {}
        for attribute_name, json_attribute_name in zip(self.__predicted_attribute_names,
                                                       self.__json_predicted_attribute_names):
            predicted_attributes[json_attribute_name] = str(out[attribute_name][0][0])
        return predicted_attributes
