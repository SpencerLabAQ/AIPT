import math
import json
from multiprocessing import Pool

import pandas as pd

from util import load_timeseries, load_classification, foreach_fork , get_benchmark_list
from constants import WEE_PATH, CFG_PATH


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


def compute_warmup_time(ts, last_warmup_it):
    warmup_time = estimate_time(ts[:last_warmup_it + 1])
    return warmup_time



def process_benchmark(benchmark):
    print(f'Processing { benchmark }...')

    techniques = ['OSCNN', 'FCN', 'Rocket', 'SOP', 'CV', 'RCIW', 'KLD']

    # load timeseries
    timeseries = load_timeseries(benchmark)
    # load classification
    classification = load_classification(benchmark)

    results = []

    for technique in techniques:  
        # load cfg      
        cfg = load_cfg(benchmark, technique)
        # for each steady fork
        for no_fork, ts, st in foreach_fork(timeseries, classification, steady_state_only=True):
            if no_fork < len(cfg):
                actual_warmup_time = compute_warmup_time(ts, st)
                estimated_warmup_time = compute_warmup_time(ts, cfg[no_fork][0])
                results.append((technique, benchmark, no_fork, actual_warmup_time, estimated_warmup_time))
    
    return results



if __name__ == '__main__':
    # get benchmark list
    benchmarks = get_benchmark_list()

    with Pool(2) as pool:
        columns = ['technique', 'benchmark', 'no_fork', 'actual_warmup_time', 'estimated_warmup_time']
        results = pool.map(process_benchmark, get_benchmark_list())
        results = sum(results, start=[])

        df = pd.DataFrame(results, columns=columns)
        df['wee'] = abs(df['estimated_warmup_time'] - df['actual_warmup_time'])

        df.to_csv(WEE_PATH, index=False)
