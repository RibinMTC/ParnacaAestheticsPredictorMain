"""
Module performs tests with the parnaca aadb model to evaluate given images
"""
import glob
import json

import cv2
from pathlib2 import Path

from src.parnaca_prediction.parnaca_aadb_model import ParnacaAADBModel

base_path = Path(__file__).parent.parent.parent
MODEL_RESOURCES_ROOT = str((base_path / 'model_resources').absolute()) + '/'
# str((base_path / 'test_images').absolute()) + '/'
# '/local/home/cribin/Documents/AestheticsBackup/Datasets/ava_evaluation_images_with_ground_truth/ava_test_images/'
IMAGES_ROOT = '/local/home/cribin/Documents/AestheticsBackup/Datasets/tx_group_images/'

score_predictions_path = str(
    (base_path / 'model_evaluation/stored_predictions/tx_group_image_aadb_predictions.json').absolute())
stored_predictions_json_path = str(
    (base_path / 'model_evaluation/stored_predictions/tx_group_images_with_attributes.json').absolute())

IMAGE_FILE = IMAGES_ROOT + "*jpg"


parnaca_aadb_model = ParnacaAADBModel(MODEL_RESOURCES_ROOT)

test_img_paths = [img_path for img_path in glob.glob(IMAGE_FILE)]

# Making predictions
test_ids = []
preds = []

image_predicted_attributes_dict = {}
image_score_dict = {}
for img_path in test_img_paths:
    image_id = img_path.split('/')[-1][:-4]

    predicted_attributes = parnaca_aadb_model.predict(img_path)
    score = predicted_attributes[parnaca_aadb_model.get_score_layer_name()]
    image_score_dict[image_id] = str(score * 10)
    image_predicted_attributes_dict[image_id] = predicted_attributes
    print img_path, '\t', score

# with open(stored_predictions_json_path, 'w') as stored_predictions_json_path:
#     json.dump(image_predicted_attributes_dict, stored_predictions_json_path, indent=2)
#
with open(score_predictions_path, 'w') as score_predictions_json_path:
    json.dump(image_score_dict, score_predictions_json_path, indent=2)