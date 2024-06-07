#!/bin/bash

mkdir -p logs

# train models
# --- we provide dumps of previously trained models in ./results/models/ in order to avoid the need to re-run the training from scratch ---
# --- uncomment the following line to perform model training ---
# python fit.py &> logs/fit.py

# generate predictions 
python predict_val.py &> logs/predict_val.log

# generate best thresholds
python select_thresholds.py &> logs/select_thresholds.log

# generate evaluation metrics for testing set
python evaluate.py &> logs/evaluate.log