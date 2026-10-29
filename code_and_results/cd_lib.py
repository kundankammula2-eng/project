"""
cd_lib.py -- Core Library for Community Detection Analysis
Author: Kundan Kammula
Course: Project 4 - Network Science & Complex Systems

Provides:
  * Dataset loaders and benchmark network generators (LFR, SBM)
  * Implementations & wrappers for 5 community detection algorithms:
      1. Louvain (Multi-scale heuristic modularity optimization)
      2. Greedy Modularity / CNM (Agglomerative bottom-up optimization)
      3. Label Propagation / LPA (Asynchronous majority voting)
      4. Spectral Clustering (Normalized Laplacian embedding + k-Means)
      5. Girvan-Newman (Divisive edge-betweenness centrality removal)
  * Quantitative validation metrics:
      - Modularity (Q), Conductance, Coverage, Performance, NMI, ARI
  * Cartographic node-role analysis (Guimerà & Amaral framework):
      - Within-module degree z-score (z_i)
      - Participation coefficient (P_i)
  * Degree-preserving randomized null model generation (Chung-Lu configuration)
"""
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from networkx.algorithms import community as nxc
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEED: int = 42


# ====================================================================== DATA LOADERS
def load_karate() -> Tuple[nx.Graph, List[int]]:
    """
    Zachary's Karate Club network (unweighted) + ground-truth faction split.
    
    Nodes (34 members) represent individuals, and edges represent social interactions.
    The ground-truth reflects the historical split: Mr. Hi (0) vs. John A. (1).
    """
    raw = nx.karate_club_graph()
    g = nx.Graph()
    g.add_nodes_from(raw.nodes)
    g.add_edges_from(raw.edges)                     # strip weights -> standard literature baseline
    truth = [0 if raw.nodes[v]["club"] == "Mr. Hi" else 1 for v in g.nodes]
    return g, truth


def load_lesmis() -> nx.Graph:
    """
    Les Misérables character co-appearance network (weighted).
    
    Nodes (77 characters) connected by weighted edges indicating the number of
    chapters in Victor Hugo's novel where characters co-occur in the same scene.
    """
    return nx.les_miserables_graph()


def load_edgelist(path: str) -> nx.Graph:
    """
    Loads a SNAP-formatted edge list (ignoring '#' comments) as a simple undirected graph.
    Removes any self-loops to maintain simple graph topology.
    """
    g = nx.read_edgelist(path, comments="#", nodetype=int, create_using=nx.Graph)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def lfr_graph(
    n: int,
    mu: float,
    seed: int,
    avg_deg: int = 8,
    max_deg: int = 40,
    min_comm: int = 20
) -> Tuple[nx.Graph, List[Set[Any]], List[int]]:
    """
    Generates an LFR (Lancichinetti-Fortunato-Radicchi) benchmark graph with power-law degree
    and community size distributions. Retries with sequential seeds upon convergence failure.
    
    Args:
        n: Number of nodes.
        mu: Mixing parameter (fraction of inter-community edges, 0.0 to 1.0).
        seed: Random seed for reproducibility.
    """
    for attempt in range(25):
        try:
            g = nx.LFR_benchmark_graph(
                n, tau1=3, tau2=1.5, mu=mu, average_degree=avg_deg,
                max_degree=max_deg, min_community=min_comm, seed=seed + 1000 * attempt
            )
            break
        except Exception:                           # generator occasionally fails to converge
            continue
    else:
        raise RuntimeError(f"LFR generation failed after 25 attempts for mu={mu}")
        
    g.remove_edges_from(nx.selfloop_edges(g))
    comms = {frozenset(g.nodes[v]["community"]) for v in g}
    comms_list = [set(c) for c in comms]
    truth = truth_labels(g, comms_list)
    return nx.Graph(g), comms_list, truth


def sbm_graph(
    n: int,
    block: int = 50,
    k_in: float = 10.0,
    k_out: float = 1.5,
    seed: int = SEED
) -> nx.Graph:
    """
    Planted-partition (Stochastic Block Model) graph used for controlled scalability benchmarking.
    """
    nb = n // block
    sizes = [block] * nb
    p_in = k_in / (block - 1)
    p_out = k_out / (n - block)
    probs = [[p_in if i == j else p_out for j in range(nb)] for i in range(nb)]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def truth_labels(g: nx.Graph, comms: List[Set[Any]]) -> List[int]:
    """Maps ground-truth community membership sets to an ordered list of integer labels."""
    lab = {v: i for i, c in enumerate(comms) for v in c}
    return [lab[v] for v in g.nodes]


def labels_from(g: nx.Graph, comms: List[Set[Any]]) -> List[int]:
    """Generates a partition label array matching the graph's node order."""
    lab = {v: i for i, c in enumerate(comms) for v in c}
    return [lab[v] for v in g.nodes]


# ================================================================== ALGORITHMS
def alg_louvain(
    g: nx.Graph,
    seed: int = SEED,
    weight: Optional[str] = None,
    restarts: int = 1
) -> List[Set[Any]]:
    """
    Louvain algorithm for modularity maximization.
    Performs multi-scale heuristic optimization with local vertex movements
    followed by network coarsening.
    """
    best, best_q = None, -2.0
    for r in range(restarts):
        comms = [set(c) for c in nxc.louvain_communities(g, weight=weight, seed=seed + r)]
        q = nxc.modularity(g, comms, weight=weight)
        if q > best_q:
            best, best_q = comms, q
    return best if best is not None else []


def alg_greedy(
    g: nx.Graph,
    seed: int = SEED,
    weight: Optional[str] = None,
    **kw: Any
) -> List[Set[Any]]:
    """
    Greedy Modularity (Clauset-Newman-Moore) algorithm.
    Agglomerative bottom-up hierarchical clustering repeatedly merging the community
    pair that yields the largest positive modularity gain (Delta Q).
    """
    return [set(c) for c in nxc.greedy_modularity_communities(g, weight=weight)]


def alg_lpa(
    g: nx.Graph,
    seed: int = SEED,
    weight: Optional[str] = None,
    **kw: Any
) -> List[Set[Any]]:
    """
    Asynchronous Label Propagation Algorithm (LPA).
    Simulates label diffusion where each node asynchronously adopts the most
    frequent label among its neighbors in near-linear O(M) time.
    """
    return [set(c) for c in nxc.asyn_lpa_communities(g, weight=weight, seed=seed)]


def alg_girvan_newman(
    g: nx.Graph,
    seed: int = SEED,
    weight: Optional[str] = None,
    max_k: int = 15,
    **kw: Any
) -> List[Set[Any]]:
    """
    Girvan-Newman divisive algorithm.
    Progressively severs edges with highest edge betweenness centrality.
    Evaluates modularity at each dendrogram level and returns the optimal partition.
    """
    best, best_q = None, -2.0
    for part in nxc.girvan_newman(g):
        q = nxc.modularity(g, part, weight=weight)
        if q > best_q:
            best, best_q = part, q
        if len(part) >= max_k:
            break
    return [set(c) for c in best] if best is not None else []


def alg_spectral(
    g: nx.Graph,
    seed: int = SEED,
    weight: Optional[str] = None,
    kmax: int = 10,
    **kw: Any
) -> List[Set[Any]]:
    """
    Normalized Spectral Clustering.
    Computes the normalized Laplacian operator N = D^(-1/2) A D^(-1/2),
    projects nodes onto the subspace of leading eigenvectors, and optimizes k in [2..kmax]
    via k-Means clustering to maximize modularity.
    """
    nodes = list(g.nodes)
    n = len(nodes)
    kmax = min(kmax, n - 1)
    A = nx.to_scipy_sparse_array(g, nodelist=nodes, weight=weight, format="csr", dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    N = A.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()   # D^-1/2 A D^-1/2
    
    if n <= 600:
        vals, vecs = np.linalg.eigh(N.toarray())
        vecs = vecs[:, ::-1][:, :kmax]                              # largest eigenvalues
    else:
        vals, vecs = eigsh(N, k=kmax, which="LA")
        vecs = vecs[:, ::-1]
        
    best, best_q = None, -2.0
    for k in range(2, kmax + 1):
        X = vecs[:, :k]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        groups = defaultdict(set)
        for node, l in zip(nodes, lab):
            groups[l].add(node)
        comms = list(groups.values())
        q = nxc.modularity(g, comms, weight=weight)
        if q > best_q:
            best, best_q = comms, q
    return best if best is not None else []


ALGORITHMS: Dict[str, Callable] = {
    "Louvain": alg_louvain,
    "Greedy Modularity": alg_greedy,
    "Label Propagation": alg_lpa,
    "Spectral": alg_spectral,
    "Girvan-Newman": alg_girvan_newman,
}
FAST_ALGS: List[str] = ["Louvain", "Greedy Modularity", "Label Propagation", "Spectral"]


def run(
    name: str,
    g: nx.Graph,
    seed: int = SEED,
    weight: Optional[str] = None,
    **kw: Any
) -> Tuple[List[Set[Any]], float]:
    """Executes a community detection algorithm and records runtime in seconds."""
    t = time.perf_counter()
    comms = ALGORITHMS[name](g, seed=seed, weight=weight, **kw)
    return comms, time.perf_counter() - t


# ==================================================================== METRICS
def conductance_avg(g: nx.Graph, comms: List[Set[Any]]) -> float:
    """
    Computes average community conductance.
    Conductance measures the fraction of cut edges leaving a cluster relative to its total volume.
    Lower values reflect higher internal density and better cluster isolation.
    """
    vals = []
    for c in comms:
        if 0 < len(c) < g.number_of_nodes():
            vals.append(nx.conductance(g, c))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(
    g: nx.Graph,
    comms: List[Set[Any]],
    truth: Optional[List[int]] = None,
    weight: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates both internal structural quality (Modularity, Coverage, Performance, Conductance)
    and external ground-truth agreement (NMI, ARI).
    """
    out: Dict[str, Any] = {
        "n_communities": len(comms),
        "modularity": float(nxc.modularity(g, comms, weight=weight)),
        "coverage": float(nxc.partition_quality(g, comms)[0]),
        "performance": float(nxc.partition_quality(g, comms)[1]),
        "conductance": conductance_avg(g, comms),
    }
    if truth is not None:
        pred = labels_from(g, comms)
        out["nmi"] = float(normalized_mutual_info_score(truth, pred))
        out["ari"] = float(adjusted_rand_score(truth, pred))
    return out


# ============================================================== NODE ROLES (GUIMERÀ & AMARAL)
def role_metrics(g: nx.Graph, comms: List[Set[Any]]) -> Dict[Any, Tuple[float, float]]:
    """
    Computes Guimerà-Amaral cartographic metrics:
      1. Within-module degree z-score (z_i): Normalized intra-community connectivity.
         z_i = (k_i^{in} - mean(k_{s_i})) / std(k_{s_i})
      2. Participation coefficient (P_i): Link dispersion across disparate modules.
         P_i = 1 - sum_{s} (k_{is} / k_i)^2
    """
    lab = {v: i for i, c in enumerate(comms) for v in c}
    k_in: Dict[Any, int] = {v: 0 for v in g}
    k_to: Dict[Any, Dict[int, int]] = {v: defaultdict(int) for v in g}
    
    for u, v in g.edges():
        k_to[u][lab[v]] += 1
        k_to[v][lab[u]] += 1
        if lab[u] == lab[v]:
            k_in[u] += 1
            k_in[v] += 1
            
    stats = {}
    for i, c in enumerate(comms):
        vals = np.array([k_in[v] for v in c], dtype=float)
        stats[i] = (vals.mean(), vals.std())
        
    rows = {}
    for v in g:
        mu, sd = stats[lab[v]]
        z = (k_in[v] - mu) / sd if sd > 0 else 0.0
        k = g.degree(v)
        P = 1.0 - sum((kk / k) ** 2 for kk in k_to[v].values()) if k > 0 else 0.0
        rows[v] = (float(z), float(P))
    return rows


def classify_role(z: float, P: float, hub_z: float = 2.0) -> str:
    """
    Categorizes nodes into distinct functional roles on the (P, z) cartographic plane:
      - Connector Hub: Dominant within-cluster hub with broad inter-cluster bridging.
      - Provincial Hub: Dominant within-cluster hub focused purely internally.
      - Connector (bridge): Non-hub with high cross-community communication.
      - Peripheral: Standard member with predominantly local connections.
    """
    if z >= hub_z:
        return "Connector hub" if P > 0.30 else "Provincial hub"
    if P > 0.40:
        return "Connector (bridge)"
    if P > 0.05:
        return "Peripheral (some external links)"
    return "Peripheral (internal only)"


def centralities(g: nx.Graph) -> Dict[str, Dict[Any, float]]:
    """Calculates canonical structural centrality metrics for comparative correlation analysis."""
    return {
        "degree": nx.degree_centrality(g),
        "closeness": nx.closeness_centrality(g),
        "betweenness": nx.betweenness_centrality(g),
        "eigenvector": nx.eigenvector_centrality(g, max_iter=2000),
        "pagerank": nx.pagerank(g),
    }


# ============================================================ NULL MODEL (CHUNG-LU CONFIGURATION)
def null_modularity(
    g: nx.Graph,
    alg: str,
    n_samples: int = 30,
    seed: int = SEED,
    **kw: Any
) -> np.ndarray:
    """
    Evaluates modularity distribution on degree-preserving randomized null models
    using double-edge swap rewiring (10*M successful swaps).
    Tests the null hypothesis that detected modularity is an artifact of degree heterogeneity.
    """
    rng = np.random.RandomState(seed)
    qs = []
    m = g.number_of_edges()
    for _ in range(n_samples):
        h = g.copy()
        nx.double_edge_swap(h, nswap=10 * m, max_tries=200 * m, seed=int(rng.randint(1 << 30)))
        comms = ALGORITHMS[alg](h, seed=SEED, **kw)
        qs.append(nxc.modularity(h, comms))
    return np.array(qs)
