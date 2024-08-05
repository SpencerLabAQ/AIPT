#!/bin/bash

mkdir -p logs

## Run inference

# generate predictions for each model
python predict.py &> logs/predict.log

# generate json configurations
python configuration.py &> logs/configuration.log

## Application evaluation metrics

# generate Warm-Up Estimation Error
python wee.py &> logs/wee.log

# generate Time Costs
python time_cost.py &> logs/time_cost.log

# generate Measurements Deviations
python performance_deviation.py &> logs/performance_deviation.log
