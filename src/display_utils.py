from PIL import Image
import numpy as np
import IPython
import sklearn
import src.dataset_utils as dataset_utils

def display(array: np.ndarray):
    assert array.ndim == 3
    assert array.shape[2] == 3
    assert array.min() >= 0.0
    assert array.max() <= 1.0
    assert array.dtype == np.float32

    image = Image.fromarray((array * 255.0).astype(np.uint8))

    IPython.display.display(image)

def create_difference(gold: np.ndarray, classification: np.ndarray) -> np.ndarray:
    difference = np.dstack([classification.astype(np.float32)] * 3)

    missing = gold & ~classification
    difference[missing] = [1.0, 0.0, 0.0]

    extra = classification & ~gold
    difference[extra] = [0.0, 1.0, 0.0]

    return difference

def read_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[1, 0])

    tp = confusion_matrix[0, 0]
    fn = confusion_matrix[0, 1]
    fp = confusion_matrix[1, 0]
    tn = confusion_matrix[1, 1]

    return tp, fn, fp, tn

def guarded_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_metrics(tp: int, fn: int, fp: int, tn: int) -> dict:
    metrics = {
        'accuracy': guarded_divide((tp + tn), (tp + tn + fp + fn)),
        'sensitivity': guarded_divide(tp, (tp + fn)),
        'specificity': guarded_divide(tn, (tn + fp))
    }

    metrics['g-mean(sens, spec)'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])

    return metrics

def build_metrics(tp: int, fn: int, fp: int, tn: int) -> str:
    metrics = calculate_metrics(tp, fn, fp, tn)

    rows = [f'| {key:19}: {value:.2f} |' for key, value in metrics.items()]
    floor = '\\' + '_' * 27 + '/'

    return '\n'.join(rows + [floor]) + '\n\n'
