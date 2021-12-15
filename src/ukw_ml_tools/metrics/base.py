from sklearn.metrics import classification_report

def get_classification_report(results_df, prediction_type):
    s1 = results_df._type == "annotations"
    y_true = results_df[s1].value.to_list()
    s1 = results_df._type == prediction_type
    y_pred = results_df[s1].value.to_list()
    _report = classification_report(y_true, y_pred, output_dict = True)
    return _report