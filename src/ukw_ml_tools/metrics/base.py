from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import numpy as np

def get_classification_report(results_df, prediction_type):
    s1 = results_df._type == "annotations"
    y_true = results_df[s1].value.to_list()
    s1 = results_df._type == prediction_type
    y_pred = results_df[s1].value.to_list()
    _report = classification_report(y_true, y_pred, output_dict=True)
    return _report


def group_stats(group, drop, df):
    df = df.groupby(group).describe()
    df = df.drop(drop, axis=1)

    df = df.droplevel(0, 1)
    df = df.drop(["top", "freq"], axis=1)
    df = df.reset_index()
    for _ in drop:
        df[_] = "all"

    return df

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division = 0),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division = 0),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division = 0),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division = 0),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division = 0),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division = 0),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average=None, zero_division = 0),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average=None, zero_division = 0),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average=None, zero_division = 0),
            }