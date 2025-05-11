import skimage
import math
import src.display_utils as display_utils
import concurrent.futures
import numpy as np

def get_image_diagonal(image_shape):
    height, width = image_shape

    return math.sqrt(height**2 + width**2)

def get_sigma_range(image_shape):
    diagonal = get_image_diagonal(image_shape)

    min = round(0.001 * diagonal)
    max = round(0.01 * diagonal)
    step = math.floor((max - min) / 5)

    return range(min, max, step)

def get_noise_size(image_shape):
    diagonal = get_image_diagonal(image_shape)

    return round(0.08 * diagonal)

def classify(dataset_manager, entry_id):
    image = dataset_manager.load('image', entry_id)[:, :, 1]
    mask = dataset_manager.load('mask', entry_id)

    contrast_image = skimage.exposure.equalize_adapthist(image)
    smooth_image = skimage.filters.gaussian(contrast_image, sigma=2.0)

    classification = skimage.filters.frangi(smooth_image, get_sigma_range(image.shape))

    eroded_mask = ~skimage.morphology.binary_dilation(~mask, skimage.morphology.disk(5))
    classification[~eroded_mask] = 0.0

    classification = classification > skimage.filters.threshold_li(classification)

    classification = skimage.morphology.remove_small_objects(classification, min_size=get_noise_size(image.shape))

    return classification.astype(bool)

def classify_and_evaluate(dataset_manager, entry_id):
    classification = classify(dataset_manager, entry_id)
    gold = dataset_manager.load('gold', entry_id)

    tp, fn, fp, tn = display_utils.read_confusion_matrix(gold, classification)

    return entry_id,\
           display_utils.create_difference(gold, classification),\
           display_utils.build_metrics(tp, fn, fp, tn)

def evaluate_collection(dataset_manager, entry_ids):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(classify_and_evaluate, dataset_manager, entry_id) for entry_id in entry_ids]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    results.sort(key=lambda x: x[0])

    for entry_id, difference, evaluation in results:
        image = dataset_manager.load('image', entry_id)
        display_utils.display(np.hstack((image, difference)))
        print(evaluation)
