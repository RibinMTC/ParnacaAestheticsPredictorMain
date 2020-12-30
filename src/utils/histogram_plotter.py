import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib2 import Path
from tqdm import tqdm


def plot_histogram(mean_scores, ground_truth, lcc, srcc, method_name, dataset_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    plt.xlim(1, 10)
    bins = np.linspace(1, 10, 100)
    plt.hist(mean_scores, bins, alpha=0.5, label='Predicted', histtype='bar', ec='black', color='red')
    plt.hist(ground_truth, bins, alpha=0.5, label='Ground Truth', histtype='bar', ec='black', color='blue')
    plt.gca().set(title=method_name + ' Pre-trained ' + dataset_name + ' Evaluation', xlabel='mean score',
                  ylabel='Frequency')

    # ava legend pos: 8.2, 600
    # tid legend pos: 8.2, 5
    ax.text(8.2, 600, 'LCC: ' + lcc + '\nSRCC: ' + srcc, style='normal',
            bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
    plt.legend(loc='upper right')
    plt.show()


def get_predicted_mean_scores_from_json(predictions_path):
    with predictions_path.open() as json_file:
        predicted_data = json.load(json_file)
        predicted_data_keys = sorted(predicted_data.keys())
        mean_scores = [float(predicted_data[key]) * 10 for key in predicted_data_keys]
        return mean_scores


def get_ground_truth_mean_scores_from_json(ground_truth_path, not_available_images=None):
    if not_available_images is None:
        not_available_images = []
    with ground_truth_path.open() as json_file:
        ground_truth_data = json.load(json_file)
        ground_truth_data = sorted(ground_truth_data, key=sort_by_image_id)
        img_ids = [int(img['image_id']) for img in ground_truth_data]
        ground_truth = [compute_mean_score_from_ratings(cur_rating["label"]) for cur_rating in ground_truth_data if
                        cur_rating["image_id"] not in not_available_images]
        return ground_truth, img_ids


def sort_by_image_id(data):
    return data['image_id']


def compute_mean_score_from_ratings(ratings):
    total_num_of_ratings = 0
    mean_score = 0
    for rating, frequency in enumerate(ratings):
        frequency = int(frequency)
        mean_score += (rating + 1) * frequency
        total_num_of_ratings += frequency

    mean_score /= float(total_num_of_ratings)
    return mean_score


def compute_linear_correlation_coefficient(mean_scores, ground_truth):
    lcc = round(pearsonr(mean_scores, ground_truth)[0], 3)
    srcc = round(spearmanr(mean_scores, ground_truth)[0], 3)

    return lcc, srcc


def print_img_not_in_test(predictions_path, ground_truth_path):
    with predictions_path.open() as predictions_json, ground_truth_path.open() as ground_truth_json:
        predicted_data = json.load(predictions_json)
        predicted_labels = predicted_data.keys()
        ground_truth_data = json.load(ground_truth_json)
        ground_truth_labels = [img['image_id'] for img in ground_truth_data]
    for image_id in tqdm(ground_truth_labels):
        if not any(s == image_id for s in predicted_labels):
            print("File not found: " + image_id)


base_path = Path(__file__).parent.parent.parent
model_evaluation_path = base_path / 'model_evaluation'
stored_predictions_path = model_evaluation_path / 'stored_predictions'
_ground_truth_path = model_evaluation_path / 'ground_truths'
ava_parnaca_predictions_path = stored_predictions_path / 'weighted_aesthetics_score_prediction.json'
ava_parnaca_ava_trained_predictions_path = stored_predictions_path / 'ava_model_weighted_aesthetics_score_prediction.json'
ava_ground_truth_path = _ground_truth_path / 'ava_evaluation_ground_truth_labels.json'

ground_truth_mean_score, ground_truth_img_ids = get_ground_truth_mean_scores_from_json(ava_ground_truth_path,
                                                                                       not_available_images=["848725",
                                                                                                             "953619",
                                                                                                             "953980"])
predicted_scores = get_predicted_mean_scores_from_json(ava_parnaca_ava_trained_predictions_path)
#
# print_img_not_in_test(ava_parnaca_predictions_path, ava_ground_truth_path)
assert (len(ground_truth_mean_score) == len(predicted_scores))
pearson, spearman = compute_linear_correlation_coefficient(predicted_scores, ground_truth_mean_score)
print("Pearson: " + str(pearson) + " spearman: " + str(spearman))
plot_histogram(predicted_scores, ground_truth_mean_score, str(pearson), str(spearman), "Parnaca", "Ava")