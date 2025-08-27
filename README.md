# Decentralized-Trustworthy-Radio-Map

ChainXim-based simulation of Decentralized Trustworthy Radio Map

Source code for the simulations in the paper "Decentralized Trustworthy Radio Map:A Blockchain Approach". This project is built upon a previous version of [ChainXim](https://github.com/XinLab-SEU/ChainXim).

## Quick Start
### Download

You can clone the Github repository with git.

Git clone command: `git clone https://github.com/Erling-Shelby/Decentralized-Trustworthy-Radio-Map.git`

Or you can download the master branch from the code repository: [master.zip](https://github.com/Erling-Shelby/Decentralized-Trustworthy-Radio-Map/archive/refs/heads/master.zip)

### Environment Setup
1. Install Anaconda. [Anaconda download link](https://www.anaconda.com/download)
2. Open Anaconda Prompt from the Start menu.
3. Create a conda environment and activate it, choosing Python version 3.10.
```
conda create -n RadioMap python=3.10 python-graphviz
activate RadioMap
```
4. Install the required packages via pip.
```
cd <project_directory>
pip install -r requirements.txt
```

### Datasets

Our dataset is located in `npz/`. You can also use the commercial ray tracing software [Remcom Wireless Insite](https://www.remcom.com/wireless-insite-em-propagation-software) to build your own dataset.
