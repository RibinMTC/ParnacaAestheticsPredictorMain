import json
import math
import os

from PIL import Image, ImageDraw, ImageFont
from pathlib2 import Path
from tqdm import tqdm

predicted_attribute_names = ['fc9_ColorHarmony', 'fc9_MotionBlur', 'fc9_Light', 'fc9_Content',
                             'fc9_Repetition',
                             'fc9_DoF', 'fc9_VividColor', 'fc9_Symmetry', 'fc9_Object',
                             'fc9_BalancingElement',
                             'fc9_RuleOfThirds']


def plot_predicted_attribute_names():
    image_width = 400
    image_height = 500
    margin = [50, 10]
    text_width = image_width / 2 - margin[0]
    dst = Image.new('RGB', (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(dst)
    font = ImageFont.truetype("../../fonts/Roboto-Black.ttf", 14)
    test = image_height / float(11)
    counter = 0
    for predicted_attribute_name in predicted_attribute_names:
        draw.text((text_width, int(math.ceil(counter * test)) + margin[1]), predicted_attribute_name,
                  (0, 0, 0),
                  font=font)
        counter += 1
    return dst


def show_image_with_attributes(im_list, img_scores, img_attributes, margin, color):
    cross_img = Image.open("../../icons/cross.png")
    tick_img = Image.open("../../icons/tick.png")
    square_img = Image.open("../../icons/square.png")
    attribute_threshold = 0.1

    num_of_images = len(im_list)
    _im = list(im_list.values())[5]
    _im_id = list(im_list.keys())[5]
    width, height = _im.width, _im.height
    num_of_cols = 3
    num_of_rows = int(math.ceil(num_of_images / float(num_of_cols)))
    total_width = num_of_cols * width + (num_of_cols + 1) * margin[0]
    total_height = num_of_rows * height + (num_of_rows + 1) * margin[1]
    dst = Image.new('RGB', (total_width, total_height), color)
    draw = ImageDraw.Draw(dst)
    font = ImageFont.truetype("../../fonts/Roboto-Black.ttf", 50)
    dst.paste(_im, (margin[0], margin[1]))

    img_ids_sorted_by_scores = sorted(img_scores, key=img_scores.get)

    test = height / float(11)
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            if len(img_ids_sorted_by_scores) == 0:
                break
            curr_img_id = img_ids_sorted_by_scores.pop(0)
            img = im_list[curr_img_id]
            img_width = j * width + (j + 1) * margin[0] * 2
            img_height = i * height + (i + 1) * margin[1]
            dst.paste(img, (img_width, img_height))
            current_score_str = img_scores[curr_img_id][0:5]
            draw.text((img_width + width / 2 - margin[0] * 4, img_height - margin[1] * 0.75), current_score_str,
                      (0, 0, 0),
                      font=font)
            predicted_attributes = img_attributes[str(curr_img_id)]
            counter = 0
            for attribute_name, attribute_value in predicted_attributes.items():
                icon_img_width = img_width - margin[0]
                icon_img_height = img_height + int(math.ceil(counter * test))
                attribute_value_float = float(attribute_value)
                if abs(attribute_value_float) < attribute_threshold:
                    icon_img = square_img
                else:
                    if attribute_value_float > 0:
                        icon_img = tick_img
                    else:
                        icon_img = cross_img
                dst.paste(icon_img, (icon_img_width, icon_img_height))
                counter += 1

    return dst


def get_all_images_in_path(img_path):
    src_files = os.listdir(img_path)
    images = {}
    for file_name in tqdm(src_files):
        full_file_name = os.path.join(img_path, file_name)
        file_name_without_extension = os.path.splitext(file_name)[0]
        if os.path.isfile(full_file_name):
            img = Image.open(full_file_name)
            images[file_name_without_extension] = img
    return images


def get_img_ids_with_attributes(image_attributes_json_path):
    with open(image_attributes_json_path) as images_with_attributes:
        json_dict = json.load(images_with_attributes)
        return json_dict


def create_img_id_to_score_dict(prediction_path):
    with open(prediction_path) as json_file:
        data = json.load(json_file)
        return data


# with open(stored_image_attributes_json_path) as images_with_attributes:
#     json_dict = json.load(images_with_attributes)
#     for image_id, image_attributes in json_dict.items():
#         print(image_id)
#         for predicted_attribute in predicted_attribute_names:
#             print(" " + predicted_attribute + ": " + image_attributes[predicted_attribute])

base_path = Path(__file__).parent.parent.parent

tx_group_image_attributes_json_path = str(
    (base_path / 'model_evaluation/stored_predictions/tx_group_images_with_attributes.json').absolute())

tx_group_image_score_json_path = str(
    (base_path / 'model_evaluation/stored_predictions/tx_group_image_aadb_predictions.json').absolute())

tx_group_test_image_path = '/local/home/cribin/Documents/AestheticsBackup/Datasets/tx_group_images/'

img_id_to_score_dict = create_img_id_to_score_dict(tx_group_image_score_json_path)

image_list = get_all_images_in_path(tx_group_test_image_path)

image_id_to_attributes_dict = get_img_ids_with_attributes(tx_group_image_attributes_json_path)

img_name = 'parnaca_tx_group_images_with_attributes_and_scores.png'
# show_image_with_attributes(image_list, img_id_to_score_dict, image_id_to_attributes_dict, margin=(30, 100),
#                            color=(255, 255, 255)).save('../../model_evaluation/plots/' + img_name)
plot_predicted_attribute_names().save('../../model_evaluation/plots/predicted_attribute_names.png')