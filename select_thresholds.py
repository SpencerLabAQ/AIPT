from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd

from constants import THRESHOLDS_PATH, PREDICTION_VAL_PATH

def optimal_threshold_using_youden(y_true, y_prob):
    """
    Calculate the optimal threshold using Youden's J-statistic.

    Parameters:
    - y_true: List of true binary labels. 0 or 1.
    - y_prob: List of predicted probabilities for the positive class.

    Returns:
    - Optimal threshold value based on Youden's J-statistic.
    """
    
    # Calculate ROC curve. This returns three lists: 
    # False positive rates (fpr), true positive rates (tpr), and the thresholds.
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Compute Youden's J-statistic for each threshold
    J = tpr - fpr
    
    # Return the threshold for the maximum J-statistic
    return thresholds[np.argmax(J)]
    


res = pd.DataFrame(columns=["model", "fold", "threshold"])
res.set_index(["model", "fold"], inplace=True)

for clf in ["OSCNN", "FCN"]:
    for fold in range(5):
        df = pd.read_csv(f"{PREDICTION_VAL_PATH}/{clf}__{fold}.csv", index_col = [0])
        y_true = df["y"]
        y_prob = df["y_pred_proba"]
        optimal_threshold = optimal_threshold_using_youden(y_true, y_prob)
        res.loc[(clf, fold), "threshold"] = optimal_threshold
        print(f"Optimal Threshold for {clf} fold {fold}: {optimal_threshold}")


res.to_csv(THRESHOLDS_PATH, index=True)