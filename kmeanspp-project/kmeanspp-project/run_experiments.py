"""
Reproduce every number and figure of the report.

    python run_experiments.py            # full run (a few minutes on a laptop)
    python run_experiments.py --quick    # fewer trials, for a smoke test

Outputs
    results/*.json      raw per-trial measurements
    results/summary.csv one line per (dataset, k, method)
    report/figures/*    figures included in the LaTeX report
    report/tables.tex   tables included in the LaTeX report (auto-generated)
"""

import argparse
import csv
import json
import os
import time
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

from src.datasets import blobs_2d, digits, norm_clusters, overlapping_clusters  # noqa: E402
from src.kmeans import adjusted_rand_index, kmeans  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

METHODS = ["random", "kmeans++"]
LABEL = {"random": "k-means (random init)", "kmeans++": "k-means++"}
COLOR = {"random": "#2a78d6", "kmeans++": "#eb6834"}   # validated, CVD-safe pair
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "lines.linewidth": 2,
})


def plain_log(axis, ticks=None):
    """Readable labels on a log axis (no 1.4x10^0 style ticks)."""
    if ticks is not None:
        axis.set_major_locator(FixedLocator(ticks))
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Core experiment: many paired trials of both methods on one dataset
# --------------------------------------------------------------------------- #
def run_trials(X, k, n_trials, y=None, base_seed=0):
    """Trial t uses the same seed for both methods (paired design)."""
    out = {}
    for m in METHODS:
        rows, labelings, histories = [], [], []
        for t in range(n_trials):
            t0 = time.perf_counter()
            r = kmeans(X, k, init=m, seed=base_seed + t)
            elapsed = time.perf_counter() - t0
            row = {"seed_inertia": r["seed_inertia"], "inertia": r["inertia"],
                   "n_iter": r["n_iter"], "time": elapsed}
            if y is not None:
                row["ari_truth"] = adjusted_rand_index(r["labels"], y)
            rows.append(row)
            labelings.append(r["labels"])
            histories.append(r["history"])
        # stability: agreement between the partitions of different runs
        pair_ari = [adjusted_rand_index(a, b)
                    for a, b in combinations(labelings, 2)]
        out[m] = {"trials": rows, "pairwise_ari": pair_ari,
                  "histories": histories}
    best = min(min(r["inertia"] for r in out[m]["trials"]) for m in METHODS)
    out["best_inertia"] = best
    return out


def summarise(name, k, res):
    """One summary line per method."""
    lines = []
    best = res["best_inertia"]
    for m in METHODS:
        tr = res[m]["trials"]
        inn = np.array([r["inertia"] for r in tr])
        seed = np.array([r["seed_inertia"] for r in tr])
        it = np.array([r["n_iter"] for r in tr])
        tm = np.array([r["time"] for r in tr])
        line = {
            "dataset": name, "k": k, "method": m, "trials": len(tr),
            "seed_mean": seed.mean(), "final_mean": inn.mean(),
            "final_min": inn.min(), "final_max": inn.max(), "final_std": inn.std(),
            "cv_percent": 100 * inn.std() / inn.mean(),
            "ratio_mean": (inn / best).mean(), "ratio_max": (inn / best).max(),
            "hit_best_percent": 100 * np.mean(inn <= best * 1.001),
            "iter_mean": it.mean(), "time_mean_ms": 1000 * tm.mean(),
            "pair_ari_mean": float(np.mean(res[m]["pairwise_ari"])),
        }
        if "ari_truth" in tr[0]:
            line["ari_truth_mean"] = float(np.mean([r["ari_truth"] for r in tr]))
        lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def boxplot_ratios(results, ks, title, fname):
    """Distribution of final inertia / best inertia found, per k and method."""
    fig, axes = plt.subplots(1, len(ks), figsize=(2.4 * len(ks) + 0.6, 2.8))
    rng = np.random.default_rng(0)
    for ax, k in zip(np.atleast_1d(axes), ks):
        res = results[k]
        data = [np.array([r["inertia"] for r in res[m]["trials"]])
                / res["best_inertia"] for m in METHODS]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.2})
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLOR[m] + "40")
            patch.set_edgecolor(COLOR[m])
        for i, (d, m) in enumerate(zip(data, METHODS), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=9,
                       color=COLOR[m], alpha=0.7, linewidths=0)
        ax.set_xticks([1, 2], ["random", "k-means++"])
        ax.set_title(f"k = {k}", color=INK)
        if max(d.max() for d in data) > 10:
            ax.set_yscale("log")
            plain_log(ax.yaxis)
            ax.set_ylabel("final inertia / best found (log)")
        else:
            ax.set_ylabel("final inertia / best found")
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, fname)


def plot_convergence(res, title, fname):
    """Objective vs Lloyd iteration (iteration 0 = right after seeding)."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    best = res["best_inertia"]
    for m in METHODS:
        for h in res[m]["histories"]:
            ax.plot(np.arange(len(h)), np.array(h) / best, color=COLOR[m],
                    alpha=0.15, linewidth=1)
        L = max(len(h) for h in res[m]["histories"])
        padded = np.array([h + [h[-1]] * (L - len(h)) for h in res[m]["histories"]])
        ax.plot(np.arange(L), padded.mean(0) / best, color=COLOR[m],
                label=LABEL[m] + " (mean)")
    top = max(max(h) for m in METHODS for h in res[m]["histories"]) / best
    if top > 10:
        ax.set_yscale("log")
        plain_log(ax.yaxis)
        ax.set_ylabel("inertia / best found (log)")
    else:
        ax.set_ylabel("inertia / best found")
    ax.set_xlabel("Lloyd iteration")
    ax.set_title(title, color=INK)
    ax.legend()
    save(fig, fname)


def plot_pairwise(stab, fname):
    """Agreement (ARI) between partitions found by different runs."""
    names = list(stab)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    pos = np.arange(len(names))
    for j, m in enumerate(METHODS):
        data = [stab[n][m] for n in names]
        bp = ax.boxplot(data, positions=pos + (j - 0.5) * 0.36, widths=0.3,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.2})
        for p in bp["boxes"]:
            p.set_facecolor(COLOR[m] + "40")
            p.set_edgecolor(COLOR[m])
        ax.plot([], [], color=COLOR[m], label=LABEL[m], linewidth=6, alpha=0.5)
    ax.set_xticks(pos, names)
    ax.set_ylabel("ARI between two runs")
    ax.set_title("Stability: how similar are partitions from different seeds?",
                 color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, fname)


def plot_2d_examples(X, k, fname, n_trials):
    """Worst run of each method on the 2-D grid of 9 clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, m in zip(axes, METHODS):
        runs = [kmeans(X, k, init=m, seed=s) for s in range(n_trials)]
        worst = max(runs, key=lambda r: r["inertia"])
        best = min(runs, key=lambda r: r["inertia"])["inertia"]
        ax.scatter(X[:, 0], X[:, 1], c=worst["labels"], cmap="tab10", s=4,
                   alpha=0.6, linewidths=0)
        ax.scatter(*worst["centres"].T, marker="X", s=80, color=INK,
                   edgecolors="white", linewidths=1.2)
        ax.set_title(f"{LABEL[m]}: worst of {n_trials} runs\n"
                     f"inertia = {worst['inertia']:.0f} (best run {best:.0f})",
                     color=INK, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save(fig, fname)


def plot_extension(xs, res_by_x, xlabel, title, fname, logx=False):
    """Mean inertia ratio and mean ARI vs ground truth, as a parameter varies."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for m in METHODS:
        ratio = [np.mean([r["inertia"] for r in res_by_x[x][m]["trials"]])
                 / res_by_x[x]["best_inertia"] for x in xs]
        ari = [np.mean([r["ari_truth"] for r in res_by_x[x][m]["trials"]])
               for x in xs]
        axes[0].plot(xs, ratio, marker="o", markersize=5, color=COLOR[m],
                     label=LABEL[m])
        axes[1].plot(xs, ari, marker="o", markersize=5, color=COLOR[m],
                     label=LABEL[m])
    axes[0].set_ylabel("mean inertia / best found")
    axes[1].set_ylabel("mean ARI vs true clusters")
    for ax in axes:
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
            plain_log(ax.xaxis, xs)
    if max(max(np.mean([r["inertia"] for r in res_by_x[x][m]["trials"]])
               / res_by_x[x]["best_inertia"] for x in xs) for m in METHODS) > 20:
        axes[0].set_yscale("log")
        plain_log(axes[0].yaxis)
    axes[1].legend()
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, fname)


# --------------------------------------------------------------------------- #
# LaTeX table writer (keeps the report in sync with the code)
# --------------------------------------------------------------------------- #
def latex_tables(summary, path):
    def fmt(x, nd=2):
        return f"{x:,.{nd}f}".replace(",", "{,}")

    out = []
    for ds in dict.fromkeys(s["dataset"] for s in summary):
        rows = [s for s in summary if s["dataset"] == ds]
        has_truth = any("ari_truth_mean" in r for r in rows)
        out.append(r"\begin{table}[ht]\centering\small")
        out.append(r"\caption{Results on \texttt{" + ds + r"} over "
                   + str(rows[0]["trials"]) + r" paired trials per method. "
                   r"Ratio = final inertia divided by the best inertia found "
                   r"by any run; \emph{hit} = runs within 0.1\% of the best; "
                   r"pair ARI = mean agreement between partitions of two runs.}")
        out.append(r"\label{tab:" + ds.replace(" ", "").replace("-", "") + "}")
        cols = "ll" + "r" * 9 + ("r" if has_truth else "")
        out.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{" + cols + "}\\toprule")
        head = (r"$k$ & method & seed inertia & final mean & final std & "
                r"CV\% & ratio mean & ratio max & hit\% & iters & pair ARI")
        if has_truth:
            head += r" & ARI truth"
        out.append(head + r"\\\midrule")
        for s in rows:
            def sci(v):
                if abs(v) < 1e-6 * max(abs(s["final_mean"]), 1.0):
                    return "0"
                m, e = f"{v:.3e}".split("e")
                return m + r"\mathrm{e}" + str(int(e))
            line = (f"{s['k']} & {'random' if s['method']=='random' else 'k-means++'} & "
                    f"${sci(s['seed_mean'])}$ & ${sci(s['final_mean'])}$ & "
                    f"${sci(s['final_std'])}$ & {fmt(s['cv_percent'])} & "
                    f"{fmt(s['ratio_mean'], 3)} & {fmt(s['ratio_max'], 3)} & "
                    f"{fmt(s['hit_best_percent'], 0)} & {fmt(s['iter_mean'], 1)} & "
                    f"{fmt(s['pair_ari_mean'], 3)}")
            if has_truth:
                line += (f" & {fmt(s['ari_truth_mean'], 3)}"
                         if "ari_truth_mean" in s else " & --")
            out.append(line + r"\\")
            if s["method"] == "kmeans++" and s is not rows[-1]:
                out.append(r"\midrule")
        out.append(r"\bottomrule\end{tabular}}\end{table}")
        out.append("")
    with open(path, "w") as f:
        f.write("\n".join(out))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    T = 10 if args.quick else 50          # trials per configuration
    TE = 5 if args.quick else 20          # trials per point in extensions
    t_start = time.time()
    summary, stab = [], {}

    # ---- 1. Synthetic Norm-25 (replica of the paper's synthetic set) -------
    X, y, _ = norm_clusters(seed=0)
    print(f"Norm-25: X {X.shape}")
    norm_res = {}
    for k in (10, 25, 50):
        norm_res[k] = run_trials(X, k, T, y=y if k == 25 else None)
        summary += summarise("Norm-25", k, norm_res[k])
        print(f"  k={k} done")
    boxplot_ratios(norm_res, (10, 25, 50),
                   "Norm-25 (synthetic, 25 true clusters, d = 15)", "norm25_box")
    plot_convergence(norm_res[25], "Norm-25, k = 25", "norm25_convergence")
    stab["Norm-25\nk=25"] = {m: norm_res[25][m]["pairwise_ari"] for m in METHODS}

    # ---- 2. Real-world: UCI digits ---------------------------------------
    Xd, yd = digits()
    print(f"Digits: X {Xd.shape}")
    dig_res = {}
    for k in (10, 25, 50):
        dig_res[k] = run_trials(Xd, k, T, y=yd if k == 10 else None)
        summary += summarise("Digits", k, dig_res[k])
        print(f"  k={k} done")
    boxplot_ratios(dig_res, (10, 25, 50),
                   "UCI digits (real, 10 classes, d = 64)", "digits_box")
    plot_convergence(dig_res[10], "Digits, k = 10", "digits_convergence")
    stab["Digits\nk=10"] = {m: dig_res[10][m]["pairwise_ari"] for m in METHODS}
    stab["Digits\nk=25"] = {m: dig_res[25][m]["pairwise_ari"] for m in METHODS}

    # held-out check on digits: centres learnt on 80%, cost measured on 20%
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(Xd))
    tr, te = perm[: int(0.8 * len(Xd))], perm[int(0.8 * len(Xd)):]
    heldout = {}
    from src.kmeans import inertia as cost
    for m in METHODS:
        tr_c, te_c = [], []
        for s in range(T):
            r = kmeans(Xd[tr], 10, init=m, seed=s)
            tr_c.append(r["inertia"] / len(tr))
            te_c.append(cost(Xd[te], r["centres"]) / len(te))
        heldout[m] = {"train_per_point": float(np.mean(tr_c)),
                      "test_per_point": float(np.mean(te_c)),
                      "test_per_point_std": float(np.std(te_c))}
    print("  held-out:", heldout)

    # ---- 3. 2-D pictures ---------------------------------------------------
    X2, y2 = blobs_2d(seed=1)
    plot_2d_examples(X2, 9, "blobs2d_worst", T)
    blob_res = run_trials(X2, 9, T, y=y2)
    summary += summarise("Blobs-2D", 9, blob_res)
    stab["Blobs-2D\nk=9"] = {m: blob_res[m]["pairwise_ari"] for m in METHODS}
    plot_pairwise(stab, "stability_pairwise_ari")

    # ---- 4. Extension: overlapping clusters ------------------------------
    sigmas = [0.25, 0.5, 1.0, 2.0, 4.0]
    ov = {}
    for s in sigmas:
        Xo, yo, _ = overlapping_clusters(sigma=s, seed=2)
        ov[s] = run_trials(Xo, 10, TE, y=yo)
    plot_extension(sigmas, ov, r"cluster std $\sigma$ (centres in a cube of side 10)",
                   "Extension: overlapping clusters (k = 10, d = 10)",
                   "ext_overlap", logx=True)

    # ---- 5. Extension: dimensionality ------------------------------------
    dims = [2, 5, 15, 50, 100]
    dm = {}
    for d in dims:
        Xh, yh, _ = norm_clusters(dim=d, n_per_cluster=100, seed=3)
        dm[d] = run_trials(Xh, 25, TE, y=yh)
    plot_extension(dims, dm, "dimension d", "Extension: dimensionality "
                   "(Norm-25 generator, k = 25)", "ext_dimension", logx=True)

    # ---- save everything ---------------------------------------------------
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for s in summary for k in s}))
        w.writeheader()
        w.writerows(summary)
    ext = {
        "overlap": {str(s): {m: {"ratio": float(np.mean([r["inertia"] for r in ov[s][m]["trials"]]) / ov[s]["best_inertia"]),
                                 "ari": float(np.mean([r["ari_truth"] for r in ov[s][m]["trials"]]))}
                             for m in METHODS} for s in sigmas},
        "dimension": {str(d): {m: {"ratio": float(np.mean([r["inertia"] for r in dm[d][m]["trials"]]) / dm[d]["best_inertia"]),
                                   "ari": float(np.mean([r["ari_truth"] for r in dm[d][m]["trials"]]))}
                               for m in METHODS} for d in dims},
        "heldout_digits_k10": heldout,
    }
    with open(os.path.join(RES, "extensions.json"), "w") as f:
        json.dump(ext, f, indent=2)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    latex_tables(summary, os.path.join(ROOT, "report", "tables.tex"))
    print(f"All done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
