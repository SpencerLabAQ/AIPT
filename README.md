### Replication Package for the paper

*AI-driven Java Performance Testing: Balancing Result Quality with Testing Time*

---

### How to generate the tables and figures in the paper

#### Experimental Setup 
Install the latest version of Python3 (*tested with version 3.9.1*)

Initialize the python execution environment:
```shell
git clone https://github.com/SpencerLabAQ/AIPT.git
cd AIPT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Dataset
Extract the data:
```shell
unzip data.zip
```

#### Replicate paper experiments
Run *preprocessing* and *training* phases:
```shell
bash run_training.sh
```

Run *application* phase:
```shell
bash run_application.sh
```

Generate tables and figures:
```shell
bash results.sh
```

### Credits

The dataset used in this study is based on the data provided in <a href="https://doi.org/10.1007/s10664-022-10247-x">Towards effective assessment of steady state performance in Java software: are we there yet?</a> article. 

The models used in the paper have been implemented using:
- OmniScale-CNN (https://github.com/Wensi-Tang/OS-CNN)
- aeon ToolKit (https://github.com/aeon-toolkit/aeon)
