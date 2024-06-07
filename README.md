### Replication Package for the paper

*AI-driven Java Performance Testing: Balancing Result Quality with Testing Time*

---

### How to generate the tables and figures in the paper
Initialize the python execution environment:
```shell
git clone https://github.com/SpencerLabAQ/AIPT.git
cd AIPT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

### Experiments

The models used in the paper have been implemented using:
- OmniScale-CNN (https://github.com/Wensi-Tang/OS-CNN)
- aeon ToolKit (https://github.com/aeon-toolkit/aeon)
