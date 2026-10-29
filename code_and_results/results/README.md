# Experimental Results Directory
**Project:** Comparative Study of Community Detection Algorithms in Complex Networks  
**Author:** Kundan Kammula  

This directory contains the numerical data, empirical logs, and high-resolution figures produced by the experimental pipeline (`main.py`).

---

## 1. Directory Contents & Summary

| File | Type | Description |
| :--- | :--- | :--- |
| **`E1_karate_comparison.csv`** | CSV | Performance metrics (Q, NMI, ARI, Conductance, Time) on Zachary's Karate Club. |
| **`E2_lesmis_communities.csv`** | CSV | 6 detected narrative story arcs and lead characters from *Les Misérables*. |
| **`E3_robustness_summary.csv`** | CSV | NMI and ARI accuracy means and std deviations across LFR mixing parameters ($\mu = 0.1 \dots 0.8$). |
| **`E3_robustness_raw.csv`** | CSV | Full raw trial runs across all random seeds for Experiment 3. |
| **`E4_stability.csv`** | CSV | Partition stability under 0% to 50% random edge deletion. |
| **`E5_scalability.csv`** | CSV | Wall-clock runtimes and modularity scores on networks from $N=200$ to $N=8,000$ nodes. |
| **`E6_node_roles_karate.csv`** | CSV | Centrality metrics, within-module degree $z$-scores, participation coefficients $P$, and classified roles for all 34 karate club members. |
| **`E6_spearman.csv`** | CSV | Spearman rank correlation matrix linking centrality metrics to community roles. |
| **`E7_null_model.csv`** | CSV | Real modularity vs. 30 degree-preserving randomized configuration graphs ($z$-score significance). |
| **`summary.json`** | JSON | Machine-readable aggregated results across all experiments. |
| **`run_log.txt`** | Text | Console stdout log from the execution run. |

---

## 2. High-Resolution Visualizations

* **`fig1_karate_partitions.png`**: Side-by-side comparison of Zachary's Karate Club ground truth vs. partitions identified by Louvain, Greedy Modularity, Label Propagation, Spectral, and Girvan-Newman.
* **`fig2_lesmis_communities.png`**: Network visualization of the *Les Misérables* co-appearance graph showing the 6 narrative subplots discovered by Weighted Louvain.
* **`fig3_robustness_lfr.png`**: Dual-panel plot of NMI accuracy and cluster count degradation as the mixing parameter $\mu$ increases, illustrating the phase transition and LPA epidemic collapse.
* **`fig4_stability.png`**: Partition retention curves under progressive random edge removal ($0\% \to 50\%$).
* **`fig5_scalability.png`**: Log-log execution runtime scaling from $200$ to $8,000$ nodes.
* **`fig6_roles_karate.png`**: Guimerà & Amaral cartographic map displaying node distribution across the $(P_i, z_i)$ plane with node sizes proportional to Betweenness Centrality.
* **`fig7_centrality_correlation.png`**: Annotated heatmap of Spearman rank correlations across centrality metrics and community role indices.
* **`fig8_null_model.png`**: Null hypothesis distributions contrasting real graph modularity against 30 degree-preserving configuration models.
