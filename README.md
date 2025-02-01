# C-HMCNN

Code and data adapted from the paper "[Coherent Hierarchical Multi-label Classification Networks](https://proceedings.neurips.cc//paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)". I thank the authors for making their code freely available.

## Adaptations to Experiment with Increasing Amounts of Training Data

(For more detailed instructions, please refer to the original README: [C-HMCNN](https://github.com/EGiunchiglia/C-HMCNN))

This document provides guidelines for interacting with my integrations, including running the main code and setting up experimental runs.

**Note:** I only worked on **cellcycle_FUN**; the script will not handle other datasets appropriately without proper modifications.

### Running a Single Experiment
To execute a single run on **cellcycle_FUN** using incremental cumulative splits, use the following command:

```sh
python main_split.py --dataset cellcycle_FUN --seed <any int> --device <0 for CPU, 1 for GPU> --num_splits <number of desired splits>
```

### Running Multiple Experiments
For multiple runs, use **metric_analysis.py** from the terminal. This script allows tuning key parameters such as the range of seeds and the number of splits. Once finished, it will output a **visualization** summarizing the experiment.

### Where do I find my results?
Results are always saved in /results folder

### Code Modifications
The functional modifications (i.e., those essential for running the experiment) were implemented in:
- **main_split.py** (adapted from `main.py`)
- **metric_analysis.py**

You may find additional modifications in other files, but these are not active by default.


