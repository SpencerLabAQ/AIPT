import math
import json
from multiprocessing import Pool

import pandas as pd

from util import load_timeseries, get_benchmark_list
from constants import TC_PATH, CFG_PATH


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

    results = pd.DataFrame(columns=['technique', 'benchmark', 'time_cost'])
    results.set_index(['technique', 'benchmark'], inplace=True)


    for technique in techniques:
        print(f'Processing { technique }...')
        # if technique is a baseline then load associated configuration
        if technique in baselines:
            cfg = load_cfg(benchmark, technique)
        # if technique is an AI technique then adapt the configuration relative to the baseline for fair comparison
        else:                
            technique_, baseline = technique.split("_")
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

        # Create configuration measurements and compute time cost
        time_cost = 0
        for (last_warmup_it, last_measure_it), ts in zip(cfg, timeseries): 
                # increment the time cost
                time_cost += estimate_time(ts[: last_measure_it + 1])
                
        # store the results
        results.loc[(technique, benchmark), 'time_cost'] = time_cost

    return results


if __name__ == '__main__':
    # get benchmark list
    benchmarks = get_benchmark_list()

    with Pool(10) as pool:
        df_list = pool.map(process_benchmark, get_benchmark_list())        
        df = pd.concat(df_list)
        df.to_csv(TC_PATH, index=True)
