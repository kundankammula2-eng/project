"""
main.py  --  runs every experiment of the Community Detection project.

    python3 main.py                              # all built-in experiments
    python3 main.py --edgelist facebook_combined.txt   # + analysis of a SNAP dataset
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nxc

import cd_lib as C

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "font.size": 10, "axes.titleweight": "bold",
})
PALETTE = plt.cm.tab10.colors
ALG_COLOR = {"Louvain": "#1f77b4", "Greedy Modularity": "#2ca02c",
             "Label Propagation": "#ff7f0e", "Spectral": "#9467bd", "Girvan-Newman": "#d62728"}
SUMMARY = {}


def log(msg):
    print(msg, flush=True)


def draw_partition(g, labels, pos, ax, title, node_size=170, font=7, sizes=None, edge_alpha=0.35):
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=edge_alpha, width=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[PALETTE[l % 10] for l in labels],
                           node_size=sizes if sizes is not None else node_size,
                           edgecolors="white", linewidths=0.8)
    if font:
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=font, font_color="white", font_weight="bold")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# ===================================================== E1  Zachary karate club
def exp_karate():
    log("\n[E1] Karate club -- algorithm comparison")
    g, truth = C.load_karate()
    rows, parts = [], {}
    for name in C.ALGORITHMS:
        kw = {"restarts": 30} if name == "Louvain" else {}      # best of 30 Louvain runs
        comms, t = C.run(name, g, **kw)
        m = C.evaluate(g, comms, truth)
        m.update(algorithm=name, time_s=t)
        rows.append(m)
        parts[name] = comms
    df = pd.DataFrame(rows).set_index("algorithm")
    df.round(4).to_csv(f"{OUT}/E1_karate_comparison.csv")
    log(df.round(3).to_string())

    pos = nx.spring_layout(g, seed=7, k=0.55)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    draw_partition(g, truth, pos, axes[0], "Ground truth (actual club split)")
    for ax, name in zip(axes[1:], C.ALGORITHMS):
        m = df.loc[name]
        draw_partition(g, C.labels_from(g, parts[name]), pos, ax,
                       f"{name}\n{int(m.n_communities)} communities | Q={m.modularity:.3f} | NMI={m.nmi:.3f}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_karate_partitions.png")
    plt.close()
    truth_comms = [{v for v, t in zip(g.nodes, truth) if t == k} for k in (0, 1)]
    SUMMARY["karate_truth_modularity"] = round(nxc.modularity(g, truth_comms), 4)
    log(f"modularity of the real 2-way club split = {SUMMARY['karate_truth_modularity']}")
    SUMMARY["karate"] = df.round(4).reset_index().to_dict("records")
    SUMMARY["karate_graph"] = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                               "density": round(nx.density(g), 4),
                               "avg_clustering": round(nx.average_clustering(g), 4),
                               "diameter": nx.diameter(g),
                               "avg_path": round(nx.average_shortest_path_length(g), 3)}
    return g, truth, parts["Louvain"]


# ===================================================== E2  Les Miserables
def exp_lesmis():
    log("\n[E2] Les Miserables -- interpretable communities (weighted Louvain)")
    g = C.load_lesmis()
    comms, _ = C.run("Louvain", g, weight="weight", restarts=30)
    comms = sorted(comms, key=len, reverse=True)
    q = nxc.modularity(g, comms, weight="weight")
    deg = dict(g.degree(weight="weight"))
    rows = []
    for i, c in enumerate(comms):
        top = sorted(c, key=lambda v: deg[v], reverse=True)[:5]
        rows.append({"community": i + 1, "size": len(c), "top_characters": ", ".join(top)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/E2_lesmis_communities.csv", index=False)
    log(df.to_string(index=False))
    log(f"weighted modularity = {q:.4f}")

    lab = C.labels_from(g, comms)
    pos = nx.spring_layout(g, seed=3, k=0.35, weight="weight")
    sizes = [60 + 14 * g.degree(v) for v in g]
    fig, ax = plt.subplots(figsize=(11, 9))
    draw_partition(g, lab, pos, ax, f"Les Miserables: {len(comms)} communities found by Louvain (Q={q:.3f})",
                   sizes=sizes, font=0, edge_alpha=0.18)
    top_nodes = sorted(g, key=lambda v: g.degree(v), reverse=True)[:14]
    nx.draw_networkx_labels(g, pos, labels={v: v for v in top_nodes}, ax=ax, font_size=8, font_weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_lesmis_communities.png")
    plt.close()
    SUMMARY["lesmis"] = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                         "n_communities": len(comms), "modularity": round(q, 4),
                         "communities": rows}
    return g, comms


# ============================================= E3  robustness to community fuzziness
def exp_robustness(n=250, mus=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), n_seeds=5):
    log("\n[E3] LFR robustness: NMI vs mixing parameter mu")
    recs = []
    for mu in mus:
        for s in range(n_seeds):
            g, planted, truth = C.lfr_graph(n, mu, seed=s)
            for name in C.FAST_ALGS:
                comms, t = C.run(name, g, seed=s)
                m = C.evaluate(g, comms, truth)
                recs.append({"mu": mu, "seed": s, "algorithm": name, "nmi": m["nmi"], "ari": m["ari"],
                             "modularity": m["modularity"], "n_found": m["n_communities"],
                             "n_planted": len(planted), "time_s": t})
        log(f"  mu={mu} done")
    df = pd.DataFrame(recs)
    df.to_csv(f"{OUT}/E3_robustness_raw.csv", index=False)
    agg = df.groupby(["algorithm", "mu"]).agg(nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
                                              ari_mean=("ari", "mean"), n_found=("n_found", "mean"),
                                              n_planted=("n_planted", "mean")).reset_index()
    agg.round(4).to_csv(f"{OUT}/E3_robustness_summary.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for name in C.FAST_ALGS:
        d = agg[agg.algorithm == name]
        ax[0].plot(d.mu, d.nmi_mean, "-o", color=ALG_COLOR[name], label=name, ms=4)
        ax[0].fill_between(d.mu, (d.nmi_mean - d.nmi_std).clip(lower=0), (d.nmi_mean + d.nmi_std).clip(upper=1),
                           color=ALG_COLOR[name], alpha=0.12)
        ax[1].plot(d.mu, d.n_found, "-o", color=ALG_COLOR[name], label=name, ms=4)
    pl = agg[agg.algorithm == "Louvain"]
    ax[1].plot(pl.mu, pl.n_planted, "k--", label="Planted (true)")
    ax[0].set_xlabel("Mixing parameter $\\mu$ (fraction of edges leaving a community)")
    ax[0].set_ylabel("NMI vs planted communities")
    ax[0].set_title("Accuracy as community structure fades")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("Mixing parameter $\\mu$")
    ax[1].set_ylabel("Number of communities found")
    ax[1].set_title("Over/under-segmentation")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_robustness_lfr.png")
    plt.close()
    SUMMARY["robustness"] = agg.round(4).to_dict("records")
    return agg


# ============================================= E4  stability under edge removal
def exp_stability(n=250, mu=0.2, fracs=(0, 0.1, 0.2, 0.3, 0.4, 0.5), trials=5):
    log("\n[E4] Stability under random edge removal")
    g, _, _ = C.lfr_graph(n, mu, seed=11)
    base = {name: C.labels_from(g, C.run(name, g)[0]) for name in C.FAST_ALGS}
    rng = np.random.RandomState(0)
    edges = list(g.edges())
    recs = []
    from sklearn.metrics import normalized_mutual_info_score as nmi
    for f in fracs:
        for t in range(trials):
            h = g.copy()
            drop = rng.choice(len(edges), int(f * len(edges)), replace=False)
            h.remove_edges_from([edges[i] for i in drop])
            for name in C.FAST_ALGS:
                lab = C.labels_from(h, C.run(name, h, seed=t)[0])
                recs.append({"removed": f, "algorithm": name, "nmi_vs_original": nmi(base[name], lab)})
    df = pd.DataFrame(recs)
    agg = df.groupby(["algorithm", "removed"]).nmi_vs_original.agg(["mean", "std"]).reset_index()
    agg.round(4).to_csv(f"{OUT}/E4_stability.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name in C.FAST_ALGS:
        d = agg[agg.algorithm == name]
        ax.errorbar(d.removed * 100, d["mean"], yerr=d["std"], fmt="-o", ms=4, capsize=3,
                    color=ALG_COLOR[name], label=name)
    ax.set_xlabel("Edges removed at random (%)")
    ax.set_ylabel("NMI vs partition of the intact graph")
    ax.set_title("Stability of detected communities")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_stability.png")
    plt.close()
    SUMMARY["stability"] = agg.round(4).to_dict("records")


# ========================================================== E5  scalability
def exp_scalability(sizes=(200, 500, 1000, 2000, 4000, 8000), cap=60.0):
    log("\n[E5] Scalability on planted-partition graphs")
    recs, dead = [], set()
    for n in sizes:
        g = C.sbm_graph(n)
        for name in C.FAST_ALGS:
            if name in dead or (name == "Spectral" and n > 4000):
                continue
            comms, t = C.run(name, g)
            recs.append({"n": n, "m": g.number_of_edges(), "algorithm": name, "time_s": t,
                         "modularity": nxc.modularity(g, comms)})
            if t > cap:
                dead.add(name)
        log(f"  n={n} done")
    df = pd.DataFrame(recs)
    df.round(4).to_csv(f"{OUT}/E5_scalability.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name in C.FAST_ALGS:
        d = df[df.algorithm == name]
        ax.plot(d.n, d.time_s, "-o", ms=4, color=ALG_COLOR[name], label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime scaling")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_scalability.png")
    plt.close()
    SUMMARY["scalability"] = df.round(4).to_dict("records")


# ======================================== E6  centrality + community-aware roles
def exp_roles(g, comms):
    log("\n[E6] Centrality and community-aware node roles (karate club)")
    cen = C.centralities(g)
    zp = C.role_metrics(g, comms)
    lab = C.labels_from(g, comms)
    df = pd.DataFrame({k: pd.Series(v) for k, v in cen.items()})
    df["z_within"] = pd.Series({v: zp[v][0] for v in g})
    df["participation"] = pd.Series({v: zp[v][1] for v in g})
    df["community"] = pd.Series({v: l + 1 for v, l in zip(g.nodes, lab)})
    df["role"] = [C.classify_role(df.z_within[v], df.participation[v]) for v in df.index]
    df.index.name = "node"
    df.round(4).to_csv(f"{OUT}/E6_node_roles_karate.csv")
    log(df.sort_values("betweenness", ascending=False).head(8).round(3).to_string())
    log(df.role.value_counts().to_string())

    pos = nx.spring_layout(g, seed=7, k=0.55)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sizes = [150 + 3500 * df.betweenness[v] for v in g]
    draw_partition(g, lab, pos, ax[0], "Louvain communities  (node size = betweenness)", sizes=sizes, font=7)
    ax[0].set_title("Louvain communities (node size = betweenness centrality)")
    cols = {"Connector hub": "#d62728", "Provincial hub": "#1f77b4", "Connector (bridge)": "#ff7f0e",
            "Peripheral (some external links)": "#7f7f7f", "Peripheral (internal only)": "#c7c7c7"}
    for role, c in cols.items():
        d = df[df.role == role]
        if len(d):
            ax[1].scatter(d.participation, d.z_within, s=55, color=c, label=f"{role} ({len(d)})",
                          edgecolors="k", linewidths=0.4)
    for v, r in df.iterrows():
        if r.z_within >= 1.0 or r.participation >= 0.45:
            ax[1].annotate(str(v), (r.participation, r.z_within), xytext=(4, 4),
                           textcoords="offset points", fontsize=8)
    ax[1].axhline(2.0, color="k", ls=":", lw=0.8)
    ax[1].axvline(0.30, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("Participation coefficient P (spread of links across communities)")
    ax[1].set_ylabel("Within-community degree z-score")
    ax[1].set_title("Role map of every member")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig6_roles_karate.png")
    plt.close()

    metrics = ["degree", "closeness", "betweenness", "eigenvector", "pagerank", "z_within", "participation"]
    corr = df[metrics].corr(method="spearman")
    corr.round(3).to_csv(f"{OUT}/E6_spearman.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics)), metrics, rotation=40, ha="right")
    ax.set_yticks(range(len(metrics)), metrics)
    ax.grid(False)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Spearman correlation: centralities vs community roles")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_centrality_correlation.png")
    plt.close()
    SUMMARY["roles"] = df.round(4).reset_index().to_dict("records")
    SUMMARY["role_counts"] = df.role.value_counts().to_dict()
    SUMMARY["spearman_p_vs_betw"] = round(float(corr.loc["participation", "betweenness"]), 3)


# ============================================================== E7  null model
def exp_null(graphs, n_samples=30):
    log("\n[E7] Significance: real Q vs degree-preserving random graphs")
    fig, axes = plt.subplots(1, len(graphs), figsize=(6.2 * len(graphs), 4.2))
    res = {}
    for ax, (name, g) in zip(np.atleast_1d(axes), graphs.items()):
        q_real = nxc.modularity(g, C.alg_louvain(g, restarts=10))
        qs = C.null_modularity(g, "Louvain", n_samples=n_samples, restarts=10)
        z = (q_real - qs.mean()) / qs.std()
        res[name] = {"Q_real": round(q_real, 4), "Q_null_mean": round(float(qs.mean()), 4),
                     "Q_null_std": round(float(qs.std()), 4), "z_score": round(float(z), 2)}
        log(f"  {name}: {res[name]}")
        ax.hist(qs, bins=12, color="#bbbbbb", edgecolor="white")
        ax.axvline(q_real, color="#d62728", lw=2, label=f"real graph  Q={q_real:.3f}")
        ax.set_title(f"{name}: z = {z:.1f}")
        ax.set_xlabel("Louvain modularity")
        ax.set_ylabel(f"Count ({n_samples} rewired graphs)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig8_null_model.png")
    plt.close()
    pd.DataFrame(res).T.to_csv(f"{OUT}/E7_null_model.csv")
    SUMMARY["null_model"] = res


# ========================================================= E8  user SNAP dataset
def exp_edgelist(path, with_greedy=False):
    log(f"\n[E8] Custom dataset: {path}")
    g = C.load_edgelist(path)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    log(f"  largest component: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    algs = ["Louvain", "Label Propagation", "Spectral"] + (["Greedy Modularity"] if with_greedy else [])
    rows, parts = [], {}
    for name in algs:
        comms, t = C.run(name, g, **({"kmax": 40} if name == "Spectral" else {}))
        m = C.evaluate(g, comms)
        m.update(algorithm=name, time_s=t)
        rows.append(m)
        parts[name] = comms
        log(f"  {name}: {m['n_communities']} communities, Q={m['modularity']:.4f}, {t:.1f}s")
    df = pd.DataFrame(rows).set_index("algorithm")
    df.round(4).to_csv(f"{OUT}/E8_custom_dataset.csv")
    best = df.modularity.idxmax()
    sizes = sorted((len(c) for c in parts[best]), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(1, len(sizes) + 1), sizes, color=ALG_COLOR[best])
    ax.set_xlabel("Community rank")
    ax.set_ylabel("Size (nodes)")
    ax.set_title(f"Community size distribution ({best})")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig9_custom_dataset_sizes.png")
    plt.close()
    SUMMARY["custom"] = {"file": os.path.basename(path), "nodes": g.number_of_nodes(),
                         "edges": g.number_of_edges(), "results": df.round(4).reset_index().to_dict("records")}


# ================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgelist", help="SNAP-style edge list to analyse in addition")
    ap.add_argument("--with-greedy", action="store_true", help="also run (slow) greedy modularity on --edgelist")
    ap.add_argument("--quick", action="store_true", help="fewer seeds / smaller sizes for a fast run")
    args = ap.parse_args()
    t0 = time.time()

    g_k, truth_k, louvain_k = exp_karate()
    g_l, _ = exp_lesmis()
    exp_robustness(n_seeds=2 if args.quick else 5)
    exp_stability(trials=2 if args.quick else 5)
    exp_scalability(sizes=(200, 500, 1000, 2000) if args.quick else (200, 500, 1000, 2000, 4000, 8000))
    exp_roles(g_k, louvain_k)
    exp_null({"Karate club": g_k, "Les Miserables": nx.Graph(g_l)}, n_samples=10 if args.quick else 30)
    if args.edgelist:
        exp_edgelist(args.edgelist, args.with_greedy)

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=1, default=str)
    log(f"\nAll experiments finished in {time.time() - t0:.0f}s. Outputs are in ./{OUT}/")
