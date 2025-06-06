# Free-rider Attacking on Federated Learning: A Quantitative Perspective

This repository contains the source code for **GradPred**, a unified framework for simulating free-rider attacks in Federated Learning (FL) by predicting future gradients. GradPred includes multiple forgery strategies such as:

- **Random**
- **FedAVG**
- **ARIMA**
- **Moirai**

The framework also evaluates their quantitative impact using **Shapley Value-based contribution estimation**. Our implementation supports a variety of datasets and analyzes attacker effectiveness, impact on honest clients, and runtime performance.


---
## Datasets

All datasets used (Adult, TicTacToe, Dota2, etc.) are publicly available and included under `data/raw/`.  
Original sources:

- Adult: [UCI Census Income](https://archive.ics.uci.edu/ml/datasets/adult)
- TicTacToe: [UCI Tic-Tac-Toe](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
- Dota2: [UCI Dota2](https://archive.ics.uci.edu/dataset/367/dota2+games+results)


---
## Installation Requirements

All required Python packages are listed in [`requirement_attack.txt`](./requirement_attack.txt).  
We recommend using a **dedicated Conda environment**.

### Step 1: Create a new Conda environment

```bash
conda create -n flce_attack python=3.10 -y
conda activate flce_attack
pip install -r requirement_attack.txt
```

> **Note:** If the installation fails due to `@ file://` paths in the `requirement_attack.txt` file,  
> it may be because these packages were originally installed from local Conda cache or internal builds.  
> You can remove such lines or simply delete the file and install packages manually (especially `uni2ts` and its dependencies).

---

### Step 2: Install additional packages for Moirai

Moirai-based attacks use Salesforce's [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts), which requires:

```bash
pip install uni2ts==1.2.0 gluonts==0.14.4
```

Or (preferred) clone and install the full package locally:

```bash
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
pip install -e ".[notebook]"
```

---
## How to Run

To launch all attack experiments across all datasets, methods, and configurations, you can run the provided Bash script:

```bash
bash run_parallel.sh
```

This script will:
- Sweep across datasets: `tictactoe`, `adult`, `dota2`
- Use all four attack methods: `random`, `fedavg`, `arima`, `moirai`
- Test multiple client numbers: `2, 4, 6, 8`
- Sweep corresponding `alpha` values for each dataset
- Perform trials `0` and `1`
- Run both with and without attack (`--use_attack`)
- Save logs to the `logs_sv_parallel/` directory

It supports basic parallelism with `MAX_PARALLEL=4` by default. You can adjust this value at the top of `run_parallel.sh`.


### Compatibility

Our implementation is fully compatible with **CPU-only environments**  
No CUDA or GPU support is required for baseline experiments.

