from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np


def compute_metrics(evaluation_predictions):
    predictions, labels = evaluation_predictions
    preds = np.argmax(predictions, axis=1)
    f1_weighted = f1_score(labels, preds, average="weighted")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)
    confusion_mat = confusion_matrix(labels, preds)

    return {
        "f1_weighted": f1_weighted, 
        "f1_micro": f1_micro, 
        "f1_macro": f1_macro, 
        "accuracy": accuracy, 
        "confusion matrix": confusion_mat.tolist()
        }