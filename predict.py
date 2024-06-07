import os
from datetime import datetime

import pandas as pd

from rocket import Rocket
from oscnn import OSCNN
from fcn import FCN
from ml import CustomKFold
from loader import load_dataset
from constants import MODELS_PATH, PREDICTION_PATH
from ml import extract_features



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
    os.makedirs(PREDICTION_PATH, exist_ok=True)

    # initialize k-fold
    kf = CustomKFold(df, k=5)

    for clf in classifiers: 
        # initialize the results list
        results = []

        for fold, (_, df_test) in enumerate(kf.iter()):

            # get the classifier name
            clf_name = clf.__class__.__name__

            # get the model
            model_path = f"{MODELS_PATH}/{clf_name}_{fold}"

            # load the best model
            clf.load(model_path)

            print(f"[{datetime.now()}] Model {model_path} loaded ...")
            
            
            # extract features
            x_test, _ = extract_features(df_test)
            # predict
            y_pred = clf.predict_proba(x_test)

            print(f"[{datetime.now()}] Predictions completed ...")

            # append results
            res = df_test[['benchmark_id', 'no_fork','starts_at', 'y']].copy()
            res.loc[:, 'y_pred'] = y_pred
            results.append(res)

        # concatenate results
        results = pd.concat(results)
        # save results
        results.to_csv(f"{PREDICTION_PATH}/{clf_name}.csv", index=False)

        print(f"[{datetime.now()}] Predictions saved to {PREDICTION_PATH}/{clf_name}.csv")



if __name__ == "__main__":
    main()