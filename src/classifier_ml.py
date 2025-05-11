import sklearn
import numpy as np
import tqdm
import src.display_utils as display_utils

def create_model(**kwargs):
    return sklearn.ensemble.HistGradientBoostingClassifier(**kwargs)

def grid_search(batch_manager, batch_info):
    param_grid = {
        'max_iter': [500],
        'learning_rate': [0.1],
        'min_samples_leaf': [40],
        'max_leaf_nodes': [121],
        'max_depth': [None],
        'l2_regularization': [0.2],
        'scoring': ['loss']
    }

    combinations = list(sklearn.model_selection.ParameterGrid(param_grid))

    results = []

    for combination in tqdm.tqdm(combinations, desc='Grid search', unit='combination'):
        model = create_model(**combination)

        metrics = train_and_test(batch_manager, batch_info, model, verbose=False)
        results.append((combination, metrics))

    separator = ';'

    keys = list(results[0][0].keys()) + list(results[0][1].keys())
    print(separator.join(keys))

    for combination, metrics in results:
        values = list(combination.values()) + list(metrics.values())
        print(separator.join(map(str, values)))

def train_and_test(batch_manager, batch_info, model, verbose=True):
    if verbose:
        print('Training...')

    X_train = batch_manager.load_singleton_batch('X_train')
    y_train = batch_manager.load_singleton_batch('y_train')

    model.fit(X_train, y_train)

    tp, fn, fp, tn = 0, 0, 0, 0
    for entry_id in tqdm.tqdm(batch_info.entry_ids_train_test, desc='Testing', unit='entry', disable=not verbose):
        for batch_id in range(batch_info.bpi_test):
            X_test = batch_manager.load_batch('X_test', entry_id, batch_id)
            y_test = batch_manager.load_batch('y_test', entry_id, batch_id)

            y_pred = model.predict(X_test)

            tp_part, fn_part, fp_part, tn_part = display_utils.read_confusion_matrix(y_test, y_pred)
            tp += tp_part
            fn += fn_part
            fp += fp_part
            tn += tn_part

    if verbose:
        print(display_utils.build_metrics(tp, fn, fp, tn))
    else:
        return display_utils.calculate_metrics(tp, fn, fp, tn)

def evaluate_collection(dataset_manager, batch_manager, batch_info, model):
    for entry_id in batch_info.entry_ids_showcase:
        classification = np.zeros((dataset_manager.height, dataset_manager.width), dtype=np.bool)

        for batch_id in range(batch_info.bpi_showcase):
            batch = batch_manager.load_batch('intact', entry_id, batch_id)
            y_pred = model.predict(batch)

            coords = batch_manager.load_batch('coords', entry_id, batch_id)

            for k in range(len(y_pred)):
                y, x = coords[k]
                classification[y, x] = y_pred[k]

        gold = dataset_manager.load('gold', entry_id)
        difference = display_utils.create_difference(gold, classification)

        image = dataset_manager.load('image', entry_id)
        display_utils.display(np.hstack((image, difference)))

        tp, fn, fp, tn = display_utils.read_confusion_matrix(gold, classification)
        print(display_utils.build_metrics(tp, fn, fp, tn))
