import os
from datetime import datetime

import pandas as pd
import numpy as np

from rocket import Rocket
from oscnn import OSCNN
from fcn import FCN
from ml import CustomKFold
from loader import load_dataset
from constants import MODELS_PATH, PREDICTION_VAL_PATH, VALIDATION_SPLIT_SIZE
from ml import split, extract_features

def create_classifiers():
    # Initialize the classifiers
    rocket_classifier = Rocket()
    OS_CNN_classifier = OSCNN()
    FCN_classifier = FCN()

    return [FCN_classifier, OS_CNN_classifier, rocket_classifier]

def main():
    # Load the dataset
    print(f"[{datetime.now()}] Loading dataset ...")
    df = load_dataset()
    print(f"[{datetime.now()}] Dataset loaded")

    # Initialize the classifiers
    classifiers = create_classifiers()

    # Create predictions directory if it does not exist
    os.makedirs(PREDICTION_VAL_PATH, exist_ok=True)

    for clf in classifiers: 
    
        # initialize k-fold
        kf = CustomKFold(df, k=5)

        for fold, (df_train, _) in enumerate(kf.iter()):

            print(f"Working of fold {fold}")

            # get validation df
            df_train, df_val = split(df_train, test_size=VALIDATION_SPLIT_SIZE)
            
            # get the classifier name
            clf_name = clf.__class__.__name__

            # get the model
            model_path = f"{MODELS_PATH}/{clf_name}_{fold}"

            # load the best model
            clf.load(model_path)

            print(f"[{datetime.now()}] Model {model_path} loaded ...")
            
            # extract features
            x_val, _ = extract_features(df_val)
            # predict
            y_pred_proba = clf.predict_proba(x_val)

            print(f"[{datetime.now()}] Predictions completed ...")

            # append results
            res = df_val[['benchmark_id', 'no_fork','starts_at', 'y']].copy()
            res.loc[:, 'y_pred_proba'] = y_pred_proba

            res.to_csv(f"{PREDICTION_VAL_PATH}/{clf_name}__{fold}.csv", index=False)


if __name__ == "__main__":
    main()