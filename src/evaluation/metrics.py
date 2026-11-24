from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    '''
    Binary metrics where "defective" (1) is the positive class.
    Recall matters most here: a missed defect (false negative) is the costly mistake.
    '''
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
    }
