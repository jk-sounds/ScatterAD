from metrics.f1_score_f1_pa import *
from metrics.fc_score import *
from metrics.precision_at_k import *
from metrics.customizable_f1_score import *
from metrics.AUC import *
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.models.feature import Window
from metrics.vus.metrics import get_range_vus_roc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
def combine_all_evaluation_scores(pred, gt):
    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(gt)
    Trange = (0, len(pred))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    P = affiliation['precision']
    R = affiliation['recall']
    affiliation_F = 2 * P * R / (P + R)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred, gt)
    ar_ap_metrics = calculate_ar_ap(gt, pred)


    vus_results = get_range_vus_roc(pred, gt, 110) # default slidingWindow = 100

    score_list_simple = {
                    "accuracy":accuracy,
                    "precision":precision,
                    "recall":recall,
                    "f_score":f_score,
                    "pa_accuracy":pa_accuracy,
                    "pa_precision":pa_precision,
                    "pa_recall":pa_recall,
                    "pa_f_score":pa_f_score,
                    "Affiliation precision": affiliation['precision'],
                    "Affiliation recall": affiliation['recall'],
                    "affiliation_F1" : affiliation_F,
                    "R_AUC_ROC": vus_results["R_AUC_ROC"],
                    "R_AUC_PR": vus_results["R_AUC_PR"],
                    "AUC_ROC" : ar_ap_metrics['AR'],
                    "AUC-PR" : ar_ap_metrics['AP'],
                    "VUS_ROC": vus_results["VUS_ROC"],
                    "VUS_PR": vus_results["VUS_PR"]
                  }
    # return score_list, score_list_simple
    return score_list_simple

def calculate_f1(predictions, gt):
    return f1_score(gt, predictions)

def calculate_ar_ap(true_labels, predicted_scores):
    """
    Calculates the AR (AUC-ROC) and AP (AUC-PR) metrics.

    Args:
        true_labels (array-like): Ground truth labels, shape (n_samples,), values are 0 or 1.
        predicted_scores (array-like): Model predicted scores (probability values), shape (n_samples,).

    Returns:
        dict: A dictionary containing AR and AP.
            - AR: AUC-ROC value
            - AP: AUC-PR value
    """
    # Calculate AR (AUC-ROC)
    ar = roc_auc_score(true_labels, predicted_scores)

    # Calculate AP (AUC-PR)
    ap = average_precision_score(true_labels, predicted_scores)

    # Return the results
    ar_ap_metrics = {
        'AR': ar,
        'AP': ap
    }
    return ar_ap_metrics