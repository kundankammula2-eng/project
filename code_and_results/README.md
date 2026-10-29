# Comparative Study of Community Detection Algorithms in Complex Networks
**Author:** Kundan Kammula  
**Project:** Network Science & Graph Analytics — Project 4

This repository contains a comprehensive empirical evaluation of five core community detection algorithms (Louvain, Greedy Modularity, Label Propagation, Spectral Clustering, and Girvan-Newman) evaluated across real-world social networks and synthetic benchmark topologies.

| ID | Experiment | Question answered |
|----|-----------|-------------------|
| E1 | Karate club comparison | How do the algorithms compare on a labelled real network? |
| E2 | Les Miserables | Are the detected communities meaningful to a human reader? |
| E3 | LFR robustness | How does accuracy fall as community structure becomes fuzzy (mu = 0.1 ... 0.8)? |
| E4 | Edge-removal stability | Do communities survive missing data? |
| E5 | Scalability | How does runtime grow from 200 to 8,000 nodes? |
| E6 | Community-aware node roles | Which nodes are hubs / bridges? (centralities + within-module z + participation coefficient) |
| E7 | Null model | Is the community structure statistically significant? |
| E8 | Your own SNAP dataset | Optional, run with `--edgelist` |

## Setup (macOS, Homebrew Python)

```bash
cd community_detection_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py                  # E1-E7, about 35 seconds, everything is saved to results/
python3 main.py --quick          # smaller/faster version for a smoke test
```

### Using a Stanford SNAP dataset (the dataset suggested in the project sheet)

1. Download `facebook_combined.txt.gz` from https://snap.stanford.edu/data/ego-Facebook.html
2. Unzip it into this folder, then:

```bash
python3 main.py --edgelist facebook_combined.txt
```

Add `--with-greedy` to also run Greedy Modularity (slow on large graphs).
Results go to `results/E8_custom_dataset.csv` and `results/fig9_custom_dataset_sizes.png`.

## Files

```
cd_lib.py       algorithms, metrics, benchmarks, node-role analysis
main.py         all experiments, tables and figures
results/        generated CSV tables, figures (fig1..fig9) and summary.json
```

All random seeds are fixed (`SEED = 42`), so results are reproducible.
