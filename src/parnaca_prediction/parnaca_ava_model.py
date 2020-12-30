from src.parnaca_prediction.parnaca_model import ParnacaModel


class ParnacaAVAModel:
    """
       Class Responsibilities:
       1.  Initialize a ParancaModel class with a model pretrained on the AVA dataset.
       2.  Use ParanacaModel to perform prediction and return a single score for the given image.
    """
    def __init__(self, model_resources_root_path_str):
        self.__score_layer = 'fc12_weightedSum'

        image_mean = model_resources_root_path_str +  'ava_trained_model/imagenet_mean.binaryproto'
        deployment_file = model_resources_root_path_str + 'ava_trained_model/initNetArch.deploy'
        model_file = model_resources_root_path_str + 'ava_trained_model/initModel_iter_16000.caffemodel'
        input_layer = 'data'
        image_width = 227
        image_height = 227

        self.__parnaca_model = ParnacaModel(image_mean, deployment_file, model_file, input_layer, image_width,
                                            image_height)

    def predict_score(self, img_to_predict):
        out = self.__parnaca_model.predict(img_to_predict)
        score = out[self.__score_layer][0][0]
        return score
