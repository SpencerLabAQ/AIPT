import random
import math
import json
from multiprocessing import Pool

import pandas as pd

import kalibera
from util import load_timeseries, load_classification, get_benchmark_list
from constants import PD_PATH, CFG_PATH


def load_cfg(benchmark, technique):
    with open(f"{CFG_PATH}/{technique}/{benchmark}.json") as f:
        cfg = json.load(f)
        return cfg

def estimate_time(measurements):
    time = 0
    for avgt in measurements:
        no_ops = math.ceil(0.1 / avgt)
        time += no_ops * avgt
    return time

def process_benchmark(benchmark):
    print(f'Processing { benchmark }...')
    forks = pd.read_csv("./data/benchmarks.csv", index_col="benchmark_id")

    baselines = ['SOP', 'CV', 'RCIW', 'KLD']
    ai_techniques = ['OSCNN', 'FCN', 'Rocket']

    techniques = [ f"{t}_{bl}" for t in ai_techniques for bl in baselines]
    techniques += baselines

   # load timeseries
    timeseries = load_timeseries(benchmark)
    # load classification
    classification = load_classification(benchmark)

    results = pd.DataFrame(columns=['technique', 'benchmark','lower_bound', 'upper_bound'])
    results.set_index(['technique', 'benchmark'], inplace=True)

    # create steady measurements
    steady_measurements = []
    for ts, clas, st in zip(timeseries, classification['forks'], classification['steady_state_starts']):      
            if clas == 'steady state':
                # append the steady measurements
                steady_measurements.append(ts[st + 1:])

    for technique in techniques:
        print(f'Processing { technique }...')

        # if technique is a baseline then load associated configuration
        if technique in baselines:
            cfg = load_cfg(benchmark, technique)
            if technique == "SOP":
                no_fork = forks.loc[benchmark, "forks"]
                cfg = cfg[:no_fork]
        else:
        # if technique is an AI technique then adapt the configuration relative to the baseline for fair comparison
            technique_, baseline = technique.split("_")
            # load cfg
            cfg = load_cfg(benchmark, technique_)
            bl_cfg = load_cfg(benchmark, baseline)

            if technique.endswith("SOP"):
                no_fork = forks.loc[benchmark, "forks"]
                bl_cfg = bl_cfg[:no_fork]

            cfg = cfg[:len(bl_cfg)]
            
            for i, (cfg_, bl_cfg_) in enumerate(zip(cfg, bl_cfg)):
                last_warmup_it, last_measure_it = cfg_
                measurement_iters = last_measure_it - last_warmup_it
                bl_measurement_iters = bl_cfg_[1] - bl_cfg_[0]
                if measurement_iters > bl_measurement_iters:
                    last_warmup_it = last_measure_it - bl_measurement_iters
                    assert last_warmup_it >= -1
                elif measurement_iters < bl_measurement_iters:
                    last_measure_it = last_warmup_it + bl_measurement_iters
                
                cfg[i] = (last_warmup_it, last_measure_it)
                
            


        # Create configuration measurements 
        cfg_measurements = []
        for (last_warmup_it, last_measure_it), ts in zip(cfg, timeseries): 
                # append the measurements
                cfg_measurements.append(ts[last_warmup_it + 1: last_measure_it + 1])
                

        # compute the confidence interval
        random.seed(42)
        lower_bound, upper_bound = kalibera.confidence_interval(cfg_measurements, steady_measurements, hierarchical=True)

        # store the results
        results.loc[(technique, benchmark), 'lower_bound'] = lower_bound
        results.loc[(technique, benchmark), 'upper_bound'] = upper_bound

    return results



if __name__ == '__main__':
    # get benchmark list
    benchmarks = get_benchmark_list()

    with Pool(30) as pool:
        df_list = pool.map(process_benchmark, get_benchmark_list())     
        df = pd.concat(df_list)
        df.to_csv(PD_PATH, index=True)