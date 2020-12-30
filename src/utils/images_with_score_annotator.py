import json
import math
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def get_concat_h_blank(im1, im2, margin, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width + margin, max(im1.height, im2.height) + 2 * margin), color)
    dst.paste(im1, (0, margin))
    dst.paste(im2, (im1.width + margin, margin))
    return dst


def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def get_concat_h_multi_blank(im_list, margin):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h_blank(_im, im, margin, color=(255, 255, 255))
    return _im


def get_all_images_with_prefix(img_path, prefix, prefix2):
    src_files = os.listdir(img_path)
    images = {}
    for file_name in tqdm(src_files):
        full_file_name = os.path.join(img_path, file_name)
        file_name_without_extension = os.path.splitext(file_name)[0]
        if os.path.isfile(full_file_name) and (
                prefix in file_name_without_extension or prefix2 in file_name_without_extension):
            img = Image.open(full_file_name)
            images[file_name_without_extension] = img
    return images


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


def create_mean_score_string_from_dicts(curr_img_id, img_id_mean_score_dict_list):
    assert len(img_id_mean_score_dict_list) > 0
    mean_score_str = "("
    for mean_score_dict in img_id_mean_score_dict_list:
        mean_score_str += str(round(float(mean_score_dict[curr_img_id]), 3)) + ", "

    mean_score_str = mean_score_str[:-2] + ")"

    return mean_score_str


def concatenate_images_into_grids_with_caption(im_list, margin, color, img_id_mean_score_dict_list):
    num_of_images = len(im_list)
    _im = list(im_list.values())[0]
    width, height = _im.width, _im.height
    num_of_cols = 3
    num_of_rows = int(math.ceil(num_of_images / 3))
    total_width = num_of_cols * width + (num_of_cols + 1) * margin[0]
    total_height = num_of_rows * height + (num_of_rows + 1) * margin[1]
    dst = Image.new('RGB', (total_width, total_height), color)
    draw = ImageDraw.Draw(dst)
    font = ImageFont.truetype("../../fonts/Roboto-Black.ttf", 50)

    img_id_to_mean_score_dict_main = img_id_mean_score_dict_list[0]

    img_ids_sorted_by_scores = sorted(img_id_to_mean_score_dict_main, key=img_id_to_mean_score_dict_main.get)

    for i in range(num_of_rows):
        for j in range(num_of_cols):
            if len(img_ids_sorted_by_scores) == 0:
                break
            curr_img_id = img_ids_sorted_by_scores.pop(0)
            img = im_list[curr_img_id]
            mean_score_str = create_mean_score_string_from_dicts(curr_img_id, img_id_mean_score_dict_list)
            img_width = j * width + (j + 1) * margin[0]
            img_height = i * height + (i + 1) * margin[1]
            dst.paste(img, (img_width, img_height))
            draw.text((img_width + width / 2 - margin[0] * 4, img_height - margin[1] * 0.75), mean_score_str,
                      (0, 0, 0),
                      font=font)
    return dst


def create_img_id_to_mean_score_dict(prediction_path):
    with open(prediction_path) as json_file:
        data = json.load(json_file)
        return data


tx_group_test_image_path = '/local/home/cribin/Documents/AestheticsBackup/Datasets/tx_group_images'

tx_group_parnaca_predictions_path = '../../model_evaluation/stored_predictions/tx_group_image_predictions.json'

image_list = get_all_images_in_path(tx_group_test_image_path)
img_id_to_parnaca_score_dict = create_img_id_to_mean_score_dict(tx_group_parnaca_predictions_path)

img_name = 'parnaca_tx_group_images_with_aesthetic_scores.png'
concatenate_images_into_grids_with_caption(image_list, margin=(30, 100), color=(255, 255, 255),
                                           img_id_mean_score_dict_list=[
                                               img_id_to_parnaca_score_dict]).save('../../model_evaluation/plots/' + img_name)
