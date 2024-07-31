#!/bin/bash

mkdir -p tables
mkdir -p figures

python rq1.py
python rq2_wee.py
python rq2_improvement.py
python rq23_estimation.py
python rq3_improvement.py
python rq3_wee.py
python rq3_tc_pd.py