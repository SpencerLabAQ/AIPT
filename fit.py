from datetime import datetime
import time
import os

import pandas as pd

from rocket import Rocket
from oscnn import OSCNN
from fcn import FCN
from loader import load_dataset
from ml import split, extract_features
from ml import CustomKFold
from constants import FOLDS_PATH, FIT_MODELS_METRICS_PATH, MODELS_PATH, RES_PATH

def create_classifiers():
    # Initialize the classifiers
    rocket_classifier = Rocket()
    OS_CNN_classifier = OSCNN()
    FCN_classifier = FCN()
    
    return [FCN_classifier, OS_CNN_classifier, rocket_classifier]

def train(clf, df_train):
    t1 = time.time()
    if isinstance(clf, FCN) or isinstance(clf, OSCNN):
        # split the dataset
        df_train, df_val = split(df_train, test_size=0.25)
        
        x_train, y_train = extract_features(df_train)
        x_val, y_val = extract_features(df_val)   
        
        # fit the model
        clf.fit(x_train, y_train, x_val, y_val)
    elif isinstance(clf, Rocket):
        x_train, y_train = extract_features(df_train)
        
        # fit the model
        clf.fit(x_train, y_train)
    t2 = time.time()
    train_time = t2 - t1

    return train_time


def main():
    # init results and moder folder
    for folder in [RES_PATH, MODELS_PATH]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # load the dataset
    print(f"[{datetime.now()}] Loading dataset...")
    df = load_dataset(steady_state_only=True, stratify=True)
    print(f"[{datetime.now()}] Dataset loaded")

    # initialize the results list
    res =[]

    # initialize k-fold
    kf = CustomKFold(df, k=5)
    # save folds
    kf.save_folds(FOLDS_PATH)

    for fold, (df_train, _test) in enumerate(kf.iter()):
        # create the classifiers
        classifiers = create_classifiers()
        for clf in classifiers:
            # get the classifier name
            clf_name = clf.__class__.__name__

            # init metrics dict
            metrics = {"fold": fold, "clf": clf_name}

            # train the model
            print(f"[{datetime.now()}] Training {clf_name} on fold {fold} ...")
            train_time = train(clf, df_train)
            metrics["train_time"] = train_time
            print(f"[{datetime.now()}] Training completed")

            # save the model
            save_to = f"{MODELS_PATH}/{clf_name}_{fold}"
            clf.dump(save_to)
            print(f"[{datetime.now()}] Model saved to {save_to}")

            # append the metrics to the results list
            res.append(metrics)

    # Save the results
    pd.DataFrame(res).to_csv(FIT_MODELS_METRICS_PATH, index=False)
    print(f"[{datetime.now()}] Metrics saved to {FIT_MODELS_METRICS_PATH}")


if __name__ == "__main__":
    main()
