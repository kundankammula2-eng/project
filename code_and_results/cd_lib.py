"""
cd_lib.py  --  core library for the Community Detection project.

Contains
  * dataset loaders / synthetic benchmark generators
  * six community detection algorithms behind one common interface
  * evaluation metrics (internal + external)
  * node-role analysis (within-module z-score and participation coefficient)
"""
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from networkx.algorithms import community as nxc
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEED = 42


# ====================================================================== DATA
def load_karate():
    """Zachary's karate club (unweighted) + ground truth = the real club split."""
    raw = nx.karate_club_graph()
    g = nx.Graph()
    g.add_nodes_from(raw.nodes)
    g.add_edges_from(raw.edges)                     # strip weights -> literature values
    truth = [0 if raw.nodes[v]["club"] == "Mr. Hi" else 1 for v in g.nodes]
    return g, truth


def load_lesmis():
    """Les Miserables character co-appearance network (weighted)."""
    return nx.les_miserables_graph()


def load_edgelist(path):
    """Load a SNAP-style edge list ('#' comment lines allowed) as a simple undirected graph."""
    g = nx.read_edgelist(path, comments="#", nodetype=int, create_using=nx.Graph)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def lfr_graph(n, mu, seed, avg_deg=8, max_deg=40, min_comm=20):
    """LFR benchmark graph with planted communities. Retries with new seeds on failure."""
    for attempt in range(25):
        try:
            g = nx.LFR_benchmark_graph(
                n, tau1=3, tau2=1.5, mu=mu, average_degree=avg_deg,
                max_degree=max_deg, min_community=min_comm, seed=seed + 1000 * attempt)
            break
        except Exception:                           # generator occasionally fails to converge
            continue
    else:
        raise RuntimeError("LFR generation failed")
    g.remove_edges_from(nx.selfloop_edges(g))
    comms = {frozenset(g.nodes[v]["community"]) for v in g}
    comms = [set(c) for c in comms]
    truth = truth_labels(g, comms)
    return nx.Graph(g), comms, truth


def sbm_graph(n, block=50, k_in=10.0, k_out=1.5, seed=SEED):
    """Planted-partition (stochastic block model) graph, used for scalability tests."""
    nb = n // block
    sizes = [block] * nb
    p_in = k_in / (block - 1)
    p_out = k_out / (n - block)
    probs = [[p_in if i == j else p_out for j in range(nb)] for i in range(nb)]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def truth_labels(g, comms):
    lab = {v: i for i, c in enumerate(comms) for v in c}
    return [lab[v] for v in g.nodes]


def labels_from(g, comms):
    lab = {v: i for i, c in enumerate(comms) for v in c}
    return [lab[v] for v in g.nodes]


# ================================================================== ALGORITHMS
def alg_louvain(g, seed=SEED, weight=None, restarts=1):
    """Louvain method. With restarts > 1 the best-modularity run is kept (Louvain is stochastic)."""
    best, best_q = None, -2.0
    for r in range(restarts):
        comms = [set(c) for c in nxc.louvain_communities(g, weight=weight, seed=seed + r)]
        q = nxc.modularity(g, comms, weight=weight)
        if q > best_q:
            best, best_q = comms, q
    return best


def alg_greedy(g, seed=SEED, weight=None, **kw):
    return [set(c) for c in nxc.greedy_modularity_communities(g, weight=weight)]


def alg_lpa(g, seed=SEED, weight=None, **kw):
    return [set(c) for c in nxc.asyn_lpa_communities(g, weight=weight, seed=seed)]


def alg_girvan_newman(g, seed=SEED, weight=None, max_k=15, **kw):
    """Girvan-Newman edge-betweenness removal; return the dendrogram level with best modularity."""
    best, best_q = None, -2.0
    for part in nxc.girvan_newman(g):
        q = nxc.modularity(g, part, weight=weight)
        if q > best_q:
            best, best_q = part, q
        if len(part) >= max_k:
            break
    return [set(c) for c in best]


def alg_spectral(g, seed=SEED, weight=None, kmax=10, **kw):
    """Normalised spectral clustering. k is chosen (2..kmax) by maximising modularity."""
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
    return best


ALGORITHMS = {
    "Louvain": alg_louvain,
    "Greedy Modularity": alg_greedy,
    "Label Propagation": alg_lpa,
    "Spectral": alg_spectral,
    "Girvan-Newman": alg_girvan_newman,
}
FAST_ALGS = ["Louvain", "Greedy Modularity", "Label Propagation", "Spectral"]


def run(name, g, seed=SEED, weight=None, **kw):
    """Run an algorithm; return (communities, seconds)."""
    t = time.perf_counter()
    comms = ALGORITHMS[name](g, seed=seed, weight=weight, **kw)
    return comms, time.perf_counter() - t


# ==================================================================== METRICS
def conductance_avg(g, comms):
    vals = []
    for c in comms:
        if 0 < len(c) < g.number_of_nodes():
            vals.append(nx.conductance(g, c))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(g, comms, truth=None, weight=None):
    out = {
        "n_communities": len(comms),
        "modularity": nxc.modularity(g, comms, weight=weight),
        "coverage": nxc.partition_quality(g, comms)[0],
        "performance": nxc.partition_quality(g, comms)[1],
        "conductance": conductance_avg(g, comms),
    }
    if truth is not None:
        pred = labels_from(g, comms)
        out["nmi"] = normalized_mutual_info_score(truth, pred)
        out["ari"] = adjusted_rand_score(truth, pred)
    return out


# ============================================================== NODE ROLES
def role_metrics(g, comms):
    """Guimera-Amaral within-module degree z-score and participation coefficient P."""
    lab = {v: i for i, c in enumerate(comms) for v in c}
    k_in = {v: 0 for v in g}
    k_to = {v: defaultdict(int) for v in g}
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
        rows[v] = (z, P)
    return rows


def classify_role(z, P, hub_z=2.0):
    """Simplified Guimera-Amaral role classes (thresholds adapted for small networks:
    hub if z >= 2.0 [paper: 2.5]; connector if P > 0.40 [paper: 0.62])."""
    if z >= hub_z:
        return "Connector hub" if P > 0.30 else "Provincial hub"
    if P > 0.40:
        return "Connector (bridge)"
    if P > 0.05:
        return "Peripheral (some external links)"
    return "Peripheral (internal only)"


def centralities(g):
    return {
        "degree": nx.degree_centrality(g),
        "closeness": nx.closeness_centrality(g),
        "betweenness": nx.betweenness_centrality(g),
        "eigenvector": nx.eigenvector_centrality(g, max_iter=2000),
        "pagerank": nx.pagerank(g),
    }


# ============================================================ NULL MODEL
def null_modularity(g, alg, n_samples=30, seed=SEED, **kw):
    """Algorithm's modularity on degree-preserving rewired versions of g."""
    rng = np.random.RandomState(seed)
    qs = []
    m = g.number_of_edges()
    for _ in range(n_samples):
        h = g.copy()
        nx.double_edge_swap(h, nswap=10 * m, max_tries=200 * m, seed=int(rng.randint(1 << 30)))
        comms = ALGORITHMS[alg](h, seed=SEED, **kw)
        qs.append(nxc.modularity(h, comms))
    return np.array(qs)
