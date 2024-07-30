### Replication Package for the paper

*AI-driven Java Performance Testing: Balancing Result Quality with Testing Time*

---
### Dataset

The dataset used in this paper has been presented in the paper:

L. Traini et al., ***Towards effective assessment of steady state performance in Java software: are we there yet?*** Empirical Software Engineering 28, 1 (2022), 13. [DOI: 10.1007/s10664-022-10247-x](https://doi.org/10.1007/s10664-022-10247-x)

It includes performance measurements from 586 JMH microbenchmarks across 30 Java software systems. 
The `data.zip` archive contains dataset used for this work. Once extracted, the total size of the data archived in this repository is less than 500 Mb.
We distilled the timeseries data from the metadata related to the steady state and warm-up phases.
Specifically, the archive contains the following folders:
- *timeseries*: contains all the timeseries used in this work in JSON format. All the forks of a same microbenchmark have been included as a list in the same file.
- *classification*: contains all the details related to the warm-up phases and steady state iterations in JSON format. Each file contains the '*forks*' field including a string indicating if the fork reached the steady state (*steady state*) or not (*no steady state*). In addition, the field '*steady_state_starts*' contains a list of datapoint indexes indicating for each fork the first iteration identified as *steady*, -1 if there are none.

The archive also contains the *benchmarks.csv* file including some additional details on JHM micrombenchmarks.

---
### Requirements

#### Recommended Hardware

- **CPU**: Intel(R) Xeon(R), 2.30 GHz or faster
- **Memory (RAM)**: 64 GB or more
- **Storage**: >= 30 GB free
- **Operating System**: Linux Ubuntu 18.04

#### Software Requirements
- Python 3.9.1
- Additional python dependencies are listed in *requirements.txt* file. Follow the experimental setup instruction to install all of them.

---
### Experimental Setup
 
Install the recommended/latest version of **Python3**.

Initialize the python execution environment:
```shell
git clone https://github.com/SpencerLabAQ/AIPT.git
cd AIPT

python3 -m venv .venv
source .venv/bin/activate
```

Install all the dependencies:
```shell
pip install --upgrade pip
pip install -r requirements.txt
```

Extract the dataset from the `data.zip` archive.
Ensure that all the files contained in the archived are extracted in the `./data/` folder:
```shell
unzip data.zip
```

---
### Usage

#### Experiment pipeline
Run *preprocessing* and *training* phases:
```shell
bash run_training.sh
```

Run *application* phase:
```shell
bash run_application.sh
```

### Generate all the figures and tables contained in this paper

For sake of clarity, we present here how to generate individual results for each research question:

#### RQ1: Classification of segments
```shell
python rq1.py # Table 2
```

#### RQ2: AI-based framework compared to the state-of-practice (SOP)
```shell
python rq2_wee.py # Table 3
python rq2_improvement.py # Table 4
python rq23_estimation.py # Figure 4
```

#### RQ3: AI-based framework compared to the state-of-the-art (SOTA)
```shell
python rq3_tc_pd.py # Table 5
python rq3_wee.py # Table 6
python rq3_improvement.py # Table 7
```

[Shortcut] The following command can generate all the tables and figures in the paper:
```shell
bash results.sh 
```

---
### Credits

The dataset used in this study is based on the data provided in <a href="https://doi.org/10.1007/s10664-022-10247-x">Towards effective assessment of steady state performance in Java software: are we there yet?</a> article. 

The models used in the paper have been implemented using:
- OmniScale-CNN (<a href="https://doi.org/10.48550/arXiv.2002.10061">Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification</a>)
- aeon ToolKit (https://github.com/aeon-toolkit/aeon)

---
*The software and the data included in this repository are released under the MIT License.*