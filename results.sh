#!/bin/bash

mkdir -p tables
mkdir -p figures

python rq1.py # Table 2 --> tables/rq1.tex

python rq2_wee.py # Table 3 --> tables/rq2_wee.tex
python rq2_improvement.py # Table 4 --> tables/rq2_impr.tex
python rq23_estimation.py # Figure 4 --> figures/rq23_estimates.pdf

python rq3_tc_pd.py # Table 5 --> tables/rq23_tc_pd.tex
python rq3_wee.py # Table 6 --> tables/rq3_wee_FCN.tex - tables/rq3_wee_OSCNN.tex - tables/rq3_wee_Rocket.tex
python rq3_improvement.py # Table 7 --> tables/rq3_impr_FCN.tex - tables/rq3_impr_OSCNN.tex - tables/rq3_impr_Rocket.tex