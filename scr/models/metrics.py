import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(true_labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(np.int32)
    trues = true_labels.astype(np.int32)

    sample_acc = (preds == trues).mean(axis=1)
    multi_label_acc = sample_acc.mean()

    macro_f1 = f1_score(trues.reshape(-1), preds.reshape(-1),
                        average="macro", zero_division=0)

    attr_f1 = f1_score(trues, preds,
                       average=None, zero_division=0)

    return multi_label_acc, macro_f1, attr_f1
