import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score


def reconstruct_predict(y_pred, y_true, delay=7):
    """

    """
    change_points = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = (y_true[0] == 1)
    modified_y_pred = np.array(y_pred)
    current_pos = 0

    for point in change_points:
        if is_anomaly:
            if 1 in y_pred[current_pos: min(current_pos+delay+1, point)]:
                modified_y_pred[current_pos:point] = 1
            else:
                modified_y_pred[current_pos:point] = 0

        is_anomaly = not is_anomaly
        current_pos = point

    point = len(y_true)

    if is_anomaly:
        if 1 in y_pred[current_pos: min(current_pos+delay+1, point)]:
            modified_y_pred[current_pos:point] = 1
        else:
            modified_y_pred[current_pos:point] = 0

    return modified_y_pred


def reconstruct_label(timestamp, label):
    """

    """
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    label_reconstructed = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    label_reconstructed[idx] = label

    return label_reconstructed


def get_modified_predict_label(y_preds, y_trues, delay=7):
    """
    Compute modified labels
    Args:
        y_preds: shape (num_sample, num_series)
        y_trues: shape (num_sample, num_series)
        timestamp: 
        delay: 
    """
    assert(type(y_preds)==type(y_trues))
    if type(y_preds) is not list:
        y_pred = reconstruct_predict(y_preds, y_trues, delay)
        y_true = y_trues
    elif len(y_preds) == 1:
        y_pred = reconstruct_predict(y_preds[0], y_trues[0], delay)
        y_true = y_trues[0]
    else:
        y_pred = []
        for i, data in enumerate(y_preds):
            y_pred.append(reconstruct_predict(y_preds[i], y_trues[i], delay))
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_trues)

    return y_pred, y_true


def modified_precision(y_pred, y_true, delay=7):
    y_pred, y_true = get_modified_predict_label(y_pred, y_true, delay)

    return precision_score(y_true, y_pred)



def modified_recall(y_pred, y_true, timestamp, delay=7):
    y_pred, y_true = get_modified_predict_label(y_pred, y_true, delay)

    return recall_score(y_true, y_pred)


def modified_f1(y_pred, y_true, timestamp, delay=7):
    y_pred, y_true = get_modified_predict_label(y_pred, y_true, delay)

    return f1_score(y_true, y_pred)
