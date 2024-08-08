## AI-driven Java Performance Testing: Balancing Result Quality with Testing Time

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

> :exclamation: Due to the extensive duration required to run experiments with the full dataset, we provide a reduced version of the dataset to facilitate quicker replication and testing. This subset includes all representative features of the full dataset but with a smaller volume, enabling users to achieve similar results without the prolonged runtime. Specifically, the small dataset includes a total of 15 JHM microbenchmarks related to 3 projects. The reduced version of the dataset has been uploaded in the `data_reduced.zip` archive.

To run the experiments, extract the dataset with the following command:
```shell
unzip data.zip          # For the full dataset
unzip data_reduced.zip  # For the reduced version of the dataset
```

---
### Requirements

#### Recommended Hardware
- **Memory (RAM)**: >= 64 GB
- **Storage**: >= 30 GB free

The experiment can be executed entirely on a CPU. However, utilizing a GPU with CUDA can significantly speed up the process.

> The estimated time using the recommended hardware configuration is of **two days**. Using hardware with lower specifications may result in significantly longer times or potential errors. If the specified hardware configuration is unavailable, we recommend running the experiment with the reduced settings as outlined below. This may lead in oucomes that differ from those reported in the paper.

#### Software Requirements
We do not provide a VM/Docker image for the working execution environment as our experiments don't require any non-trivial piece of software. Software required to run the experiments are:

- Python 3.9.1. Alternatively, you can use the *python:3.9-bookworm* docker image containing python 3.9.19.
- Additional python dependencies are listed in *requirements.txt* file. 

---
### Experimental Setup
 
Install the recommended version of **Python 3**.

Open your terminal and run the following commands:
```shell
# Clone the repository
git clone https://github.com/SpencerLabAQ/AIPT.git
cd AIPT

# Initialize the python execution environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test Python and some essential packages
python --version
pip show pandas
pip show torch
pip show aeon
```

---
### Running the Experiment with Reduced Settings

- Ensure you extract the reduced dataset (`unzip data_reduced.zip`) instead of the full version.
- Reduce the number of training epochs by modifying the `constants.py` file: change `EPOCHS = 500` to `EPOCHS = 10`.
- Reduce the number of bootstrap iterations of in `kalibera.py` file: change `BOOTSTRAP_ITERATIONS = 10000` to `BOOTSTRAP_ITERATIONS = 100`.

The reduced version of the experiment took approximately 2 hours on a *MacBook Air (M1, 2020)*, **Processor:** Apple M1 chip with 8-core CPU, **Memory:** 16GB, **Operating System:** macOS Sonoma 14.5

---
### Usage

#### Experiment pipeline

Given the extensive time required to complete the training phase, we provide a dump of the already trained models ([results/models/](./results/models/)), which are ready for use in running evaluations and future studies.

You can train the models from scratch by uncommenting the line below in the [run_training.sh](run_training.sh) file:
```bash
python fit.py &> logs/fit.log
```
and removing the [results/models/](./results/models/) folder. Otherwise, the provided checkpoints will be used to perform the evaluation avoiding to run the training from scratch.

#### Running
```shell
# Training and evaluation phases
bash run_training.sh

# Application phase
bash run_application.sh

# Generate results
bash results.sh 
```

The successfull execution of the `run_training.sh` and `run_application.sh` scripts can be verified by the proper generation of the corresponding log files in the `./logs/` folder since the standard output (stdout) and standard error (stderr) from Python scripts are redirected to that folder. The `results.sh` is expected to generate all the related tables and figures, as specified in the comments of the script itself. 

### Generate all the figures and tables contained in this paper

The `bash results.sh` command generates all the results of this experiment in a single step. However, for clarity, we outline in this section how to generate specific results for each research question:

#### RQ1: Classification of segments
```shell
python rq1.py # Table 2 --> tables/rq1.tex
```

#### RQ2: AI-based framework compared to the state-of-practice (SOP)
```shell
python rq2_wee.py # Table 3 --> tables/rq2_wee.tex
python rq2_improvement.py # Table 4 --> tables/rq2_impr.tex
python rq23_estimation.py # Figure 4 --> figures/rq23_estimates.pdf
```

#### RQ3: AI-based framework compared to the state-of-the-art (SOTA)
```shell
python rq3_tc_pd.py # Table 5 --> tables/rq23_tc_pd.tex
python rq3_wee.py # Table 6 --> tables/rq3_wee_{model}.tex
python rq3_improvement.py # Table 7 --> tables/rq3_impr_{model}.tex
```

---
### Credits

The dataset used in this study is based on the data provided in <a href="https://doi.org/10.1007/s10664-022-10247-x">Towards effective assessment of steady state performance in Java software: are we there yet?</a> article. 

The models used in the paper have been implemented using:
- OmniScale-CNN (<a href="https://doi.org/10.48550/arXiv.2002.10061">Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification</a>)
- aeon ToolKit (https://github.com/aeon-toolkit/aeon)

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.