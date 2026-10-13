"""
make_figures.py -- regenerate every figure in the revised manuscript.

Run after run_cv.py, aggregate_metrics.py and run_holdout_variability.py.
Outputs to out/:
    fig_multilabel_overlap.png            Figure 1
    revised_fig_roc_pr.png                Figure 2
    revised_fig_f1_mcc_distribution.png   Figure 3
    revised_fig_confusion.png             Figure 4
    revised_fig_holdout_variability.png   Figure 5
    fig_redundancy.png                    Supplementary Figure S1

Styling (font sizes, frame) is applied by `apply_figure_style()` if available; the figures
are otherwise identical.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TASKS = ["apoptosis", "necrosis", "pyroptosis"]
MODELS = ["RF", "SVM", "XGB", "STACK_SVM_RF", "STACK_SVM_RF_XGB"]
LBL = {"RF": "RF", "SVM": "SVM", "XGB": "XGBoost",
       "STACK_SVM_RF": "SVM+RF", "STACK_SVM_RF_XGB": "SVM+RF+XGB"}
MC = {"RF": "#b2182b", "SVM": "#4393c3", "XGB": "#1a9850",
      "STACK_SVM_RF": "#984ea3", "STACK_SVM_RF_XGB": "#e08214"}
TC = {"apoptosis": "#1b6ca8", "necrosis": "#d95f02", "pyroptosis": "#7570b3"}
PUB_F1 = {"apoptosis": (0.75, 0.80), "necrosis": (0.82, 0.76), "pyroptosis": (0.73, 0.77)}


def style(sizes=(8, 7, 6)):
    base, mid, small = sizes
    mpl.rcParams.update({
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": mid, "xtick.labelsize": small, "ytick.labelsize": small,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white"})


def roc_points(y, p):
    o = np.argsort(-p); y = np.asarray(y)[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    P, N = y.sum(), len(y) - y.sum()
    return np.r_[0, fp / N], np.r_[0, tp / P]


def pr_points(y, p):
    o = np.argsort(-p); y = np.asarray(y)[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    return tp / y.sum(), tp / (tp + fp)


def fig1_overlap(dfs, out="out/fig_multilabel_overlap.png"):
    long = pd.concat([d.assign(Task=t) for t, d in dfs.items()])
    wide = long.pivot_table(index="Sequence", columns="Task", values="Label", aggfunc="first")
    pos = wide == 1
    combo = pos.apply(lambda r: "+".join(t.capitalize() for t in TASKS if r[t]), axis=1)
    cc = combo.value_counts()
    M = pd.DataFrame(0, index=[t.capitalize() for t in TASKS], columns=[t.capitalize() for t in TASKS])
    for a in TASKS:
        pa = set(dfs[a][dfs[a].Label == 1].Sequence)
        for b in TASKS:
            M.loc[a.capitalize(), b.capitalize()] = (
                len(pa) if a == b else len(pa & set(dfs[b][dfs[b].Label == 0].Sequence)))
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.42))
    ax = axes[0]
    order = ["Apoptosis", "Necrosis", "Pyroptosis", "Apoptosis+Necrosis",
             "Necrosis+Pyroptosis", "Apoptosis+Pyroptosis", "Apoptosis+Necrosis+Pyroptosis"]
    short = ["Apo", "Nec", "Pyr", "Apo+Nec", "Nec+Pyr", "Apo+Pyr", "Apo+Nec+Pyr"]
    vals = [int(cc[o]) for o in order]
    cols = ["#b8c4cc" if o.count("+") == 0 else ("#e08214" if o.count("+") == 1 else "#b2182b")
            for o in order]
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, color=cols, height=0.68)
    for yi, v in zip(y, vals):
        ax.text(v + 4, yi, str(v), va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels(short)
    ax.set_xlabel("Peptides (n)"); ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title(f"{sum(v for o, v in zip(order, vals) if '+' in o)} of {len(wide)} peptides "
                 "induce more than one death mode", fontsize=8, loc="left")
    ax.legend(handles=[Patch(color="#b8c4cc", label="1 mode"),
                       Patch(color="#e08214", label="2 modes"),
                       Patch(color="#b2182b", label="3 modes")],
              frameon=False, fontsize=6, loc="lower right")
    ax = axes[1]
    Mv = M.values.astype(float); off = Mv.copy(); np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap="Oranges", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            if i == j:
                ax.text(j, i, f"{int(Mv[i, j])}\npositives", ha="center", va="center",
                        fontsize=6, color="#555")
            else:
                ax.text(j, i, int(Mv[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if Mv[i, j] > 60 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Apo", "Nec", "Pyr"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["Apo", "Nec", "Pyr"])
    ax.set_xlabel("labelled negative in task"); ax.set_ylabel("labelled positive in task")
    ax.set_title("100% of every negative set is\npositive for another death mode",
                 fontsize=8, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("shared peptides (n)", fontsize=6); cb.ax.tick_params(labelsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out, dpi=320, bbox_inches="tight"); plt.close(fig)


def figS1_redundancy(red, ident, csd, out="out/fig_redundancy.png"):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7), gridspec_kw=dict(wspace=0.46))
    ax = axes[0]
    for t in TASKS:
        ax.hist(ident[t], bins=np.linspace(0, 1, 51), histtype="step", lw=1.3,
                color=TC[t], label=t.capitalize(), density=True)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 30)
    for th, ls in [(0.9, ":"), (0.8, "--")]:
        ax.axvline(th, color="#444", ls=ls, lw=0.9)
    ax.text(0.80, 3e-3, "0.8", fontsize=6, ha="right", va="bottom", color="#444")
    ax.text(0.92, 3e-3, "0.9", fontsize=6, ha="left", va="bottom", color="#444")
    ax.set_xlabel("Pairwise sequence identity"); ax.set_ylabel("Density")
    ax.set_title("Near-identical pairs in every task", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.02, 0.80))
    ax = axes[1]; w = 0.26
    for k, t in enumerate(TASKS):
        sub = red[red.Task == t.capitalize()].sort_values("Threshold")
        x = np.arange(len(sub)) + (k - 1) * w
        ax.bar(x, sub.Reduction_pct, width=w, color=TC[t])
        for xi, vv in zip(x, sub.Reduction_pct):
            ax.text(xi, vv + 0.8, f"{vv:.0f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.7", "0.8", "0.9"])
    ax.set_xlabel("CD-HIT identity threshold"); ax.set_ylabel("Sequences removed (%)")
    ax.set_title("28-40% redundancy", fontsize=8, loc="left"); ax.set_ylim(0, 48)
    ax = axes[2]
    for t in TASKS:
        d = csd[t]; ks = sorted(int(k) for k in d)
        ax.plot(ks, [d[str(k)] for k in ks], "o-", ms=3.5, lw=1.2, color=TC[t])
    ax.set_yscale("log"); ax.set_xlabel("Cluster size (peptides)"); ax.set_ylabel("Clusters (n)")
    ax.set_title("Cluster sizes at 0.8 identity", fontsize=8, loc="left")
    for a, l in zip(axes, "abc"):
        a.text(-0.22, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=10)
    fig.savefig(out, dpi=320, bbox_inches="tight"); plt.close(fig)


def fig2_roc_pr(preds, summ, out="out/revised_fig_roc_pr.png"):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2), gridspec_kw=dict(hspace=0.45, wspace=0.34))
    sub = preds[(preds.Regime == "cluster08") & (preds.Repeat == 0)]
    for j, task in enumerate(TASKS):
        axr, axp = axes[0, j], axes[1, j]
        s = sub[sub.Task == task]; prev = s[s.Model == "RF"].y_true.mean()
        for m in MODELS:
            d = s[s.Model == m]
            q = summ.query("Task==@task and Regime=='cluster08' and Model==@m and Threshold=='fixed'")
            fpr, tpr = roc_points(d.y_true.values, d.y_prob.values)
            axr.plot(fpr, tpr, lw=1.3, color=MC[m], label=f"{LBL[m]} {q.ROC_AUC_mean.iloc[0]:.2f}")
            rc, pc = pr_points(d.y_true.values, d.y_prob.values)
            axp.plot(rc, pc, lw=1.3, color=MC[m], label=f"{LBL[m]} {q.PR_AUC_mean.iloc[0]:.2f}")
        axr.plot([0, 1], [0, 1], ls=":", color="#999", lw=0.9)
        axp.axhline(prev, ls=":", color="#999", lw=0.9)
        axp.text(0.98, prev + 0.015, f"prevalence {prev:.2f}", fontsize=5.5, color="#666", ha="right")
        axr.set_title(task.capitalize(), fontsize=8, loc="left")
        axr.set_xlabel("False positive rate"); axp.set_xlabel("Recall")
        if j == 0:
            axr.set_ylabel("True positive rate"); axp.set_ylabel("Precision")
        axr.legend(frameon=False, fontsize=5.5, loc="lower right",
                   title="mean ROC-AUC", title_fontsize=5.5)
        axp.legend(frameon=False, fontsize=5.5, loc="lower left",
                   title="mean PR-AUC", title_fontsize=5.5, bbox_to_anchor=(-0.01, -0.02))
        axr.set_xlim(-0.02, 1.02); axr.set_ylim(-0.02, 1.02)
        axp.set_xlim(-0.02, 1.02); axp.set_ylim(0.35, 1.02)
    axes[0, 0].text(-0.34, 1.16, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=10)
    axes[1, 0].text(-0.34, 1.16, "b", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=10)
    fig.savefig(out, dpi=320, bbox_inches="tight"); plt.close(fig)


def fig3_distributions(per_rep, out="out/revised_fig_f1_mcc_distribution.png"):
    style()
    f = per_rep[per_rep.Threshold == "fixed"]
    rc = {"random": "#8c8c8c", "cluster08": "#b2182b"}
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.9),
                             gridspec_kw=dict(hspace=0.30, wspace=0.26), sharex=True)
    for j, task in enumerate(TASKS):
        for i, met in enumerate(["F1", "MCC"]):
            ax = axes[i, j]
            for k, reg in enumerate(["random", "cluster08"]):
                for xi, m in enumerate(MODELS):
                    v = f[(f.Task == task) & (f.Model == m) & (f.Regime == reg)][met].values
                    x = xi + (k - 0.5) * 0.30
                    ax.scatter(np.full(len(v), x) + rng.uniform(-0.055, 0.055, len(v)), v,
                               s=5, color=rc[reg], alpha=0.5, lw=0, zorder=2)
                    mu = v.mean(); h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
                    ax.plot([x - 0.11, x + 0.11], [mu, mu], color=rc[reg], lw=1.8, zorder=3)
                    ax.plot([x, x], [mu - h, mu + h], color=rc[reg], lw=0.9, zorder=3)
            if met == "F1":
                ax.axhline(PUB_F1[task][1], color="#1b6ca8", ls="--", lw=0.9, zorder=1)
                ax.axhline(PUB_F1[task][0], color="#1b6ca8", ls=":", lw=0.9, zorder=1)
                ax.set_title(task.capitalize(), fontsize=8, loc="left")
                ax.set_ylim(0.58, 0.86)
            else:
                ax.set_ylim(-0.12, 0.56); ax.axhline(0, color="#bbb", lw=0.7)
            ax.set_xticks(range(len(MODELS)))
            ax.set_xticklabels([LBL[m] for m in MODELS], rotation=40, ha="right")
            if j == 0:
                ax.set_ylabel("F1-score" if met == "F1" else "MCC")
            ax.set_xlim(-0.55, len(MODELS) - 0.45)
    axes[0, 2].legend(handles=[
        Line2D([], [], color=rc["random"], lw=1.8, label="random folds"),
        Line2D([], [], color=rc["cluster08"], lw=1.8, label="identity-aware folds"),
        Line2D([], [], color="#1b6ca8", ls="--", lw=0.9, label="published RF, single hold-out"),
        Line2D([], [], color="#1b6ca8", ls=":", lw=0.9, label="published RF, single 5-fold CV")],
        frameon=False, fontsize=5.5, loc="lower right", bbox_to_anchor=(1.02, -0.02))
    axes[0, 0].text(-0.30, 1.12, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=10)
    axes[1, 0].text(-0.30, 1.04, "b", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=10)
    fig.savefig(out, dpi=320, bbox_inches="tight"); plt.close(fig)


def fig4_confusion(summ, out="out/revised_fig_confusion.png"):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), gridspec_kw=dict(hspace=0.46, wspace=0.40))
    for j, task in enumerate(TASKS):
        for i, thr in enumerate(["fixed", "f1opt"]):
            ax = axes[i, j]
            r = summ.query("Task==@task and Regime=='cluster08' and Model=='RF' and Threshold==@thr").iloc[0]
            M = np.array([[r.TN_mean, r.FP_mean], [r.FN_mean, r.TP_mean]])
            Mn = M / M.sum(axis=1, keepdims=True)
            ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{M[a, b]:.1f}\n{Mn[a, b]*100:.0f}%", ha="center", va="center",
                            fontsize=6.5, color="white" if Mn[a, b] > 0.55 else "#222")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred. neg", "pred. pos"], fontsize=6)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["true neg", "true pos"], fontsize=6)
            head = (task.capitalize() + "\n") if i == 0 else ""
            ax.set_title(head + ("threshold 0.5" if thr == "fixed" else "F1-optimal threshold"),
                         fontsize=7.5, loc="left")
            ax.set_xlabel(f"MCC {r.MCC_mean:.2f}  |  Spec. {r.Specificity_mean:.2f}", fontsize=6)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.savefig(out, dpi=320, bbox_inches="tight"); plt.close(fig)


def fig5_holdout(hv, out="out/revised_fig_holdout_variability.png"):
    style()
    cho, ccv = "#1b6ca8", "#999999"
    rng = np.random.RandomState(1)
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.8),
                             gridspec_kw=dict(hspace=0.34, wspace=0.24), sharex=True)
    for j, task in enumerate(TASKS):
        ax = axes[0, j]
        for xi, m in enumerate(MODELS):
            d = hv[(hv.Task == task) & (hv.Model == m)]
            for k, (col, c) in enumerate([("F1_holdout", cho), ("F1_cv", ccv)]):
                x = xi + (k - 0.5) * 0.32; v = d[col].values
                ax.scatter(np.full(len(v), x) + rng.uniform(-0.06, 0.06, len(v)), v,
                           s=3, color=c, alpha=0.35, lw=0)
                ax.plot([x - 0.12, x + 0.12], [v.mean()] * 2, color=c, lw=1.7, zorder=3)
            s42 = d[d.Seed == 42]
            ax.scatter([xi - 0.16], [s42.F1_holdout.iloc[0]], marker="D", s=14,
                       facecolor="none", edgecolor="#b2182b", lw=1.1, zorder=4)
        ax.set_title(task.capitalize(), fontsize=8, loc="left"); ax.set_ylim(0.50, 0.90)
        if j == 0:
            ax.set_ylabel("F1-score")
        ax = axes[1, j]
        for xi, m in enumerate(MODELS):
            d = hv[(hv.Task == task) & (hv.Model == m)]
            g = (d.F1_holdout - d.F1_cv).values
            parts = ax.violinplot([g], positions=[xi], widths=0.62, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor("#c7d9e8"); b.set_edgecolor("#4a7ba6"); b.set_alpha(0.9); b.set_lw(0.7)
            ax.plot([xi - 0.2, xi + 0.2], [g.mean()] * 2, color="#1b6ca8", lw=1.6)
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([LBL[m] for m in MODELS], rotation=40, ha="right")
        ax.set_ylim(-0.26, 0.22)
        if j == 0:
            ax.set_ylabel("hold-out F1 minus CV F1")
    axes[0, 0].legend(handles=[
        Line2D([], [], color=cho, lw=1.7, label="single 80/20 hold-out"),
        Line2D([], [], color=ccv, lw=1.7, label="5-fold CV on the training part"),
        Line2D([], [], marker="D", color="none", markerfacecolor="none",
               markeredgecolor="#b2182b", label="seed 42 (used in the paper)")],
        frameon=False, fontsize=5.5, loc="lower left")
    axes[0, 0].text(-0.26, 1.10, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=10)
    axes[1, 0].text(-0.26, 1.04, "b", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=10)
    fig.savefig(out, dpi=320, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs("out", exist_ok=True)
    dfs = {t: pd.read_csv(f"data/{t}_clusters.csv") for t in TASKS}
    preds = pd.read_csv("out/per_fold_predictions.csv.gz")
    summ = pd.read_csv("out/summary_metrics.csv")
    per_rep = pd.read_csv("out/per_repeat_metrics.csv")
    fig1_overlap(dfs)
    fig2_roc_pr(preds, summ)
    fig3_distributions(per_rep)
    fig4_confusion(summ)
    if os.path.exists("out/holdout_seed_variability.csv"):
        fig5_holdout(pd.read_csv("out/holdout_seed_variability.csv"))
    if os.path.exists("out/redundancy_summary.csv") and os.path.exists("out/ident_apoptosis.npy"):
        figS1_redundancy(pd.read_csv("out/redundancy_summary.csv"),
                         {t: np.load(f"out/ident_{t}.npy") for t in TASKS},
                         json.load(open("out/cluster_size_dist_08.json")))
    print("figures written to out/")


if __name__ == "__main__":
    main()
