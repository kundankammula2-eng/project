# Community Detection in Social Networks: Presentation Guide & Defense Script
**Author:** Kundan Kammula  
**Project:** Network Science & Graph Analytics (Project 4)

---

## 1. The 30-Second Elevator Pitch
> *"In this project, I performed an empirical comparative study of five core community detection algorithms—Louvain, Greedy Modularity, Label Propagation, Spectral Clustering, and Girvan-Newman. Beyond simply running the algorithms, I stress-tested them across seven experiments evaluating ground-truth accuracy, semantic interpretability in literature networks, noise robustness, missing data stability, computational scalability up to 8,000 nodes, cartographic node roles, and statistical significance against randomized null models. Overall, the Louvain method emerged as the optimal workhorse across quality, speed, and noise resilience."*

---

## 2. Slide-by-Slide Presentation Script

### Slide 1: Motivation & Problem Definition
* **What to Show**: Title slide with high-level network diagram.
* **Talking Points**:
  * In complex real-world systems (social circles, biological pathways, citation webs), entities naturally cluster into **communities**—subgraphs with dense internal edges and sparse cross-module links.
  * Unlike supervised classification, community detection is **unsupervised**: we must infer latent social structures solely from graph connectivity $A_{ij}$.
  * We evaluate five algorithms representing three major paradigms: optimization-based (Louvain, CNM), dynamic/diffusion-based (LPA), spectral/algebraic (Spectral), and divisive centrality-based (Girvan-Newman).

---

### Slide 2: Experiment E1 — Zachary's Karate Club Ground Truth
* **What to Show**: `code_and_results/results/fig1_karate_partitions.png`
* **Key Numbers**:
  * Real 2-way club split modularity: $Q = 0.358$
  * Louvain & Spectral: $Q = 0.420$, $\text{NMI} = 0.588$ (4 communities)
  * Greedy Modularity: $Q = 0.381$, $\text{ARI} = 0.568$ (3 communities)
* **Talking Points**:
  * Zachary's Karate Club is the canonical benchmark in network science ($N=34, M=78$), representing a club split into two factions.
  * Notice that all algorithms achieved higher modularity ($0.38 - 0.42$) than the ground-truth binary split ($0.358$).
  * This is not an error: modularity optimization identifies that both rival factions are further subdivided into cohesive, tightly-knit social sub-cliques.

---

### Slide 3: Experiment E2 — Semantic Interpretability on Les Misérables
* **What to Show**: `code_and_results/results/fig2_lesmis_communities.png`
* **Key Numbers**: Weighted Louvain modularity $Q = 0.567$, 6 communities.
* **Talking Points**:
  * We evaluated whether network topology reflects human narrative structure on the Les Misérables character co-appearance network ($N=77, M=254$).
  * Purely from who shared scenes together, Weighted Louvain cleanly isolated **6 narrative story arcs**:
    1. *Parisian Core & Main Plot* (Valjean, Marius, Cosette, Javert)
    2. *The ABC Student Revolution* (Enjolras, Courfeyrac, Gavroche)
    3. *Fantine's Early Life & Suitors* (Fantine, Tholomyès)
    4. *The Criminal Underworld* (Thénardier, Patron-Minette gang)
    5. *Digne & Spiritual Redemption* (Bishop Myriel, Mme Magloire)
    6. *The Arras Courtroom Trial* (Champmathieu, Judge)
  * This demonstrates that unsupervised graph partitioning captures genuine semantic contexts.

---

### Slide 4: Experiments E3 & E4 — Noise Robustness & Missing Data Stability
* **What to Show**: `code_and_results/results/fig3_robustness_lfr.png` & `code_and_results/results/fig4_stability.png`
* **Key Numbers**:
  * LFR Benchmark sweep: mixing parameter $\mu \in [0.1, 0.8]$.
  * At $\mu = 0.1$: Louvain achieves $\text{NMI} = 0.997$, Spectral $\text{NMI} = 0.980$.
  * At $\mu \ge 0.4$: Label Propagation collapses to $\text{NMI} = 0.000$ (1 giant community).
  * Missing edges: Spectral retains $0.933$ stability at 10% dropout, $0.551$ at 50% dropout.
* **Talking Points**:
  * Real-world networks are noisy and incompletely sampled.
  * In E3, as external noise $\mu$ increases, a sharp phase transition occurs around $\mu \approx 0.4 - 0.5$. Label Propagation suffers an epidemic failure mode where a single label sweeps the graph. Louvain and Spectral are significantly more robust.
  * In E4, Spectral Clustering proved most stable under missing data because Laplacian eigenvectors smooth localized edge perturbations.

---

### Slide 5: Experiment E5 — Computational Scalability (up to 8,000 Nodes)
* **What to Show**: `code_and_results/results/fig5_scalability.png` (log-log runtime plot)
* **Key Numbers**:
  * $N=8,000$ nodes ($M = 46,052$ edges):
    * Label Propagation: **0.24 s**
    * Louvain: **0.60 s** (achieving highest $Q = 0.865$)
    * Greedy Modularity: **4.81 s** ($8\times$ slower)
    * Girvan-Newman: omitted ($O(M^2 N)$ is intractable)
* **Talking Points**:
  * Louvain demonstrated near-linear empirical scaling, completing 8,000 nodes in just 0.6 seconds while maximizing partition quality ($Q = 0.865$).
  * Greedy Modularity scales poorly due to maintaining global priority queues.

---

### Slide 6: Experiment E6 — Cartographic Node Roles & Centrality
* **What to Show**: `code_and_results/results/fig6_roles_karate.png` & `fig7_centrality_correlation.png`
* **Key Framework**: Guimerà & Amaral cartography:
  * Within-module degree z-score ($z_i$): Local hub status.
  * Participation coefficient ($P_i$): Breadth of inter-community connectivity.
* **Key Findings**:
  * Out of 34 nodes, the algorithm mathematically classified exactly **2 Connector Hubs**: **Node 0** (Mr. Hi, $z=2.14, P=0.54$) and **Node 33** (John A., $z=2.51, P=0.51$).
  * Discovered 11 bridging connectors and 15 internal peripherals.
  * Spearman correlation between Participation Coefficient and Betweenness Centrality was **$\rho = 0.73$**, validating that high-participation nodes control inter-modular traffic.

---

### Slide 7: Experiment E7 — Statistical Significance via Null Models
* **What to Show**: `code_and_results/results/fig8_null_model.png`
* **Key Numbers**:
  * Karate Club: Real $Q = 0.420$ vs Null Mean $Q = 0.303$ ($\mathbf{z = 6.99}, p < 10^{-11}$).
  * Les Misérables: Real $Q = 0.531$ vs Null Mean $Q = 0.287$ ($\mathbf{z = 30.27}, p \ll 10^{-50}$).
* **Talking Points**:
  * Can degree heterogeneity alone create artificial modularity?
  * We benchmarked real modularity against 30 degree-preserving randomized graphs (Chung-Lu configuration model via double-edge swaps).
  * With z-scores of $6.99$ and $30.27$, the observed community structures are statistically significant and reject the random null hypothesis with near certainty.

---

### Slide 8: Synthesis & Practical Takeaways
* **Summary Matrix**:
  * **Louvain**: Undisputed general-purpose choice (top modularity, fast, noise-tolerant).
  * **Spectral**: Optimal when cluster count $k$ is known or when graph edges are missing.
  * **Label Propagation**: Best suited for massive, clean streaming graphs where latency is the only priority.
  * **Greedy Modularity & Girvan-Newman**: Useful pedagogical baselines, but outclassed on modern large-scale graphs.

---

## 3. Defense Q&A: Master Answers for Tough Questions

#### Q1: "What exactly is Modularity ($Q$) mathematically?"
> **Answer:** *"Modularity compares the fraction of edges falling inside communities against the expected fraction if connections were made purely at random preserving node degrees. Mathematically, $Q = \frac{1}{2m} \sum_{ij} [A_{ij} - \frac{k_i k_j}{2m}] \delta(c_i, c_j)$. Values above 0.3 indicate strong non-random community organization."*

#### Q2: "What is the resolution limit of modularity optimization?"
> **Answer:** *"Discovered by Fortunato and Barthélemy (2007), modularity optimization has an inherent scale below which it fails to resolve small, distinct communities in large graphs (typically subgraphs with fewer than $\approx \sqrt{2m}$ edges). In very large networks, modularity may artificially merge small cliques."*

#### Q3: "Why did Label Propagation collapse at $\mu = 0.4$ in Experiment E3?"
> **Answer:** *"Because LPA relies on local majority consensus without a global objective function. When the mixing parameter $\mu$ exceeds $\approx 0.4$, inter-community edges outnumber intra-community links. A dominant label can cross the boundary and cause an epidemic cascade, resulting in a single trivial component."*

#### Q4: "How does Spectral Clustering choose the number of clusters $k$?"
> **Answer:** *"In our pipeline, we computed the top $k_{max}$ eigenvectors of the normalized Laplacian $N = D^{-1/2} A D^{-1/2}$, ran $k$-Means across $k \in [2, k_{max}]$, and chose the $k$ that maximized graph modularity on the resulting partition."*

#### Q5: "What is the significance of the double-edge swap null model in E7?"
> **Answer:** *"A simple Erdos-Renyi random graph is a flawed null model because real networks have heavy-tailed degree distributions. Double-edge swaps ($u-v, x-y \to u-y, x-v$) preserve the exact degree of every single vertex while destroying community correlation, providing a rigorous test of statistical significance."*
