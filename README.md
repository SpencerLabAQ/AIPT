### Replication Package for the paper

*AI-driven Java Performance Testing: Balancing Result Quality with Testing Time*

---
### Dataset

The dataset utilized in this paper is introduced in https://doi.org/10.1007/s10664-022-10247-x.
It includes performance measurements from 500+ JMH microbenchmarks across 30 Java software systems.

The dataset has been directly included in this repository (`data.zip` archive). Once extracted, the total size of the data archived in this repository is less than 500 Mb.
We distilled the timeseries data from the metadata related to the steady state and warm-up phases.
Specifically, the archive contains the following folders:
- *timeseries*: contains all the timeseries used in this work in JSON format. A single file pertains to a specific microbenchmark and includes time series data for all the associated forks.
- *classification*: each file is structured in JSON format and holds comprehesive details about the warm-up phases and steady state iterations. The '*forks*' field in each file includes a string that indicates whether the fork reached the steady state ('*steady state*') or not ('*no steady state*'). 
Additionally, the field '*steady_state_starts*' comprises a list of data point indexes; for each fork, it marks the first iteration identified as *steady*, -1 if no such iteration exists.

The archive also contains the *benchmarks.csv* file including some additional details about the JHM micrombenchmarks.

> :exclamation: Due to the extensive duration required to run experiments with the full dataset, we provide a reduced version of the dataset to facilitate quicker replication and testing. This subset includes all representative features of the full dataset but with a smaller volume, enabling users to achieve similar results without the prolonged runtime. In the *Usage* section, we describe how to conduct the experiment using the reduced dataset.

---
### Requirements

#### Recommended Hardware

- **CPU**: n=40, Intel(R) Xeon(R), 2.30 GHz or faster
- **Memory (RAM)**: 64 GB or more
- **Storage**: >= 30 GB free
- **Operating System**: Linux Ubuntu 18.04

The time required to complete the experiments may vary based on the hardware used. The estimated time using the recommended hardware configuration is of two days.

*Note: Using hardware with lower specifications may result in significantly longer times.*

#### Software Requirements
- Python 3.9.1
- Additional python dependencies are listed in *requirements.txt* file. 

---
### Experimental Setup
 
Install the recommended version of **Python 3**.

Open your terminal and run the following command to clone the repository:
```shell
git clone https://github.com/SpencerLabAQ/AIPT.git
cd AIPT
```

Initialize the python execution environment:
```shell
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```shell
pip install --upgrade pip
pip install -r requirements.txt
```

Test the environment, including Python and some essential packages:
```shell
python --version
pip show pandas
pip show torch
pip show aeon
```

Extract the dataset from the `data.zip` archive.
Ensure that all the files contained in the archived are extracted in the `./data/` folder:
```shell
unzip data.zip
```

---
### Usage

#### Experiment pipeline

Given the extensive time required to complete the training phase, we provide a dump of the already trained models, which are ready for use in running evaluations and future studies.

You can train the models from scratch by uncommenting the line below in the [run_training.sh](run_training.sh) file:
```bash
python fit.py &> logs/fit.py
```

#### Running

Run *training* and *evaluation*:
```shell
bash run_training.sh
```

Run *application* phase:
```shell
bash run_application.sh
```

### Generate all the figures and tables contained in this paper

For sake of clarity, we outline the method for generating specific results for each research question:

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

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.