import json
import os

import pandas as pd
import numpy as np

from constants import WINDOW_SIZE, CFG_PATH, PREDICTION_PATH, THRESHOLDS_PATH, FOLDS_PATH

MIN_WARMUP_ITERS = 0
MAX_WARMUP_ITERS = 500

def save_configuration(benchmark, clf_name, res):
    filename = f"{CFG_PATH}/{clf_name}/{benchmark}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(res, f)


def infer_cfg(benchmark, no_fork, df, threshold):
        for offset in np.arange(MIN_WARMUP_ITERS, MAX_WARMUP_ITERS):
            y = df.loc[(benchmark, no_fork, offset), 'y_pred']
            if y >= threshold:
                last_warmup_it = int(offset - 1)
                last_measure_it = last_warmup_it + WINDOW_SIZE
                return [last_warmup_it, last_measure_it]
        last_warmup_it = MAX_WARMUP_ITERS - 1
        last_measure_it = last_warmup_it + WINDOW_SIZE
        return [last_warmup_it, last_measure_it]


if __name__ == '__main__':
    # load thresholds and folds
    thresholds = pd.read_csv(THRESHOLDS_PATH, index_col=["model","fold"])
    folds = pd.read_csv(FOLDS_PATH, index_col=["benchmark_id"])

    for clf in  ["OSCNN", "FCN", "Rocket"]:
        # load predictions
        print(f"Loading predictions for {clf}")
        df = pd.read_csv(f"{PREDICTION_PATH}/{clf}.csv")
        
        # get benchmark, forks and offsets
        benchmarks = df['benchmark_id'].unique()
        forks = df['no_fork'].unique()

        # use benchmark, forks and offsets as index
        df.set_index(['benchmark_id', 'no_fork', 'starts_at'], inplace=True)

        # sort forks and offsets to ensure consistency
        forks.sort()

        # infer configurations for each benchmark
        for b in benchmarks:
            print("Processing {}".format(b))
            # get fold the benchmark belongs to
            fold = folds.loc[b, 'fold']

            if clf == "OSCNN" or clf == "FCN":
                # get threshold for the fold
                threshold = thresholds.loc[(clf, fold), 'threshold']
            else:
                # set threshold to 0.5 for Rocket (predictions are already binary)
                threshold = 0.5

            for f in forks:
                # infer configuration for each fork
                cfg = [infer_cfg(b, f, df, threshold)  for f in forks]
            
            # save configurations
            save_configuration(b, clf, cfg)

