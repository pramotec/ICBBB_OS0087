"""
prepare_data.py -- dataset audit, pairwise identity, CD-HIT-style clustering.

Reads data/{Apoptosis,Necrosis,Pyroptosis}_Data_Set.csv and writes:
    data/<task>_clusters.csv        original rows + cluster_09 / cluster_08 / cluster_07
    data/clusters_union.csv         cluster assignment for all unique peptides
    out/dataset_audit.csv           counts, lengths, duplicates, non-standard residues
    out/label_combinations.csv      how many peptides induce 1, 2 or 3 death modes
    out/multilabel_overlap.csv      negative-set composition per task
    out/pos_in_A_neg_in_B_matrix.csv
    out/redundancy_summary.csv      clusters and redundant pairs at 0.9 / 0.8 / 0.7
    out/identity_matrix_union.npy   497 x 497 identity matrix (cached; ~80 s to compute)
    out/ident_<task>.npy            within-task identity values, for Supplementary Figure S1
    out/cluster_size_dist_08.json

Runtime: about 80 s for the all-against-all alignment of 497 peptides.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import acp_cluster as ac

TASKS = ["apoptosis", "necrosis", "pyroptosis"]
FILES = {"apoptosis": "Apoptosis_Data_Set.csv",
         "necrosis": "Necrosis_Data_Set.csv",
         "pyroptosis": "Pyroptosis_Data_Set.csv"}
THRESHOLDS = [0.9, 0.8, 0.7]
MANUSCRIPT_N = {"apoptosis": (352, 204), "necrosis": (357, 204), "pyroptosis": (318, 185)}


def load():
    dfs = {}
    for t in TASKS:
        d = pd.read_csv(f"data/{FILES[t]}")
        d["Sequence"] = d.Sequence.astype(str).str.strip().str.upper()
        dfs[t] = d
    return dfs


def audit(dfs):
    rows = []
    for t in TASKS:
        d = dfs[t]
        L = d.Sequence.str.len()
        n_man, p_man = MANUSCRIPT_N[t]
        rows.append(dict(
            Task=t.capitalize(), N_csv=len(d), N_manuscript=n_man,
            Positive=int((d.Label == 1).sum()), Negative=int((d.Label == 0).sum()),
            Pos_manuscript=p_man,
            Ratio_pos_neg=round((d.Label == 1).sum() / (d.Label == 0).sum(), 2),
            Unique_seq=d.Sequence.nunique(), Duplicates=int(d.Sequence.duplicated().sum()),
            Len_min=int(L.min()), Len_median=float(L.median()), Len_max=int(L.max()),
            Len_mean=round(float(L.mean()), 1),
            Nonstandard_aa=int((~d.Sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")).sum())))
    return pd.DataFrame(rows)


def overlap(dfs):
    long = pd.concat([d.assign(Task=t) for t, d in dfs.items()])
    wide = long.pivot_table(index="Sequence", columns="Task", values="Label", aggfunc="first")
    pos = wide == 1
    combo = pos.apply(lambda r: "+".join(t.capitalize() for t in TASKS if r[t]) or "none", axis=1)
    combos = combo.value_counts().rename_axis("Label_combination").reset_index(name="N_peptides")
    rows = []
    for a in TASKS:
        neg = set(dfs[a][dfs[a].Label == 0].Sequence)
        other = set().union(*[set(dfs[b][dfs[b].Label == 1].Sequence) for b in TASKS if b != a])
        r = dict(Task=a.capitalize(), N_negative=len(neg),
                 Neg_that_are_positive_elsewhere=len(neg & other),
                 Pct=round(100 * len(neg & other) / len(neg), 1))
        for b in TASKS:
            if b != a:
                r[f"neg_pos_in_{b}"] = len(neg & set(dfs[b][dfs[b].Label == 1].Sequence))
        rows.append(r)
    M = pd.DataFrame(0, index=[t.capitalize() for t in TASKS],
                     columns=[t.capitalize() for t in TASKS])
    for a, b in itertools.product(TASKS, TASKS):
        pa = set(dfs[a][dfs[a].Label == 1].Sequence)
        M.loc[a.capitalize(), b.capitalize()] = (
            len(pa) if a == b else len(pa & set(dfs[b][dfs[b].Label == 0].Sequence)))
    return combos, pd.DataFrame(rows), M, wide


def main():
    os.makedirs("out", exist_ok=True)
    dfs = load()
    audit(dfs).to_csv("out/dataset_audit.csv", index=False)
    combos, negcomp, M, wide = overlap(dfs)
    combos.to_csv("out/label_combinations.csv", index=False)
    negcomp.to_csv("out/multilabel_overlap.csv", index=False)
    M.to_csv("out/pos_in_A_neg_in_B_matrix.csv")

    allseq = sorted(set().union(*[set(d.Sequence) for d in dfs.values()]))
    cache = "out/identity_matrix_union.npy"
    if os.path.exists(cache) and len(np.load(cache)) == len(allseq):
        I = np.load(cache)
    else:
        print(f"aligning {len(allseq)} peptides all-against-all ...", flush=True)
        I = ac.pairwise_identity_matrix(allseq)
        np.save(cache, I)
    json.dump(allseq, open("out/union_sequences.json", "w"))
    idx = {s: i for i, s in enumerate(allseq)}

    rows, clusters = [], {}
    for th in THRESHOLDS:
        lab, _ = ac.greedy_cluster(allseq, I, th)
        clusters[th] = lab
        for t in TASKS:
            ii = [idx[s] for s in dfs[t].Sequence]
            sub = pd.Series(lab[ii]); sizes = sub.value_counts()
            y = dfs[t].Label.values
            Isub = I[np.ix_(ii, ii)]; u = np.triu_indices(len(ii), 1)
            cross = y[u[0]] != y[u[1]]
            rows.append(dict(Threshold=th, Task=t.capitalize(), N=len(ii),
                             N_clusters=int(sub.nunique()),
                             Reduction_pct=round(100 * (1 - sub.nunique() / len(ii)), 1),
                             Largest_cluster=int(sizes.max()),
                             Singletons=int((sizes == 1).sum()),
                             Redundant_pairs=int((Isub[u] >= th).sum()),
                             Redundant_pairs_crosslabel=int((Isub[u][cross] >= th).sum())))
    pd.DataFrame(rows).to_csv("out/redundancy_summary.csv", index=False)

    for t in TASKS:
        ii = [idx[s] for s in dfs[t].Sequence]
        out = dfs[t].copy()
        for th in THRESHOLDS:
            out[f"cluster_{str(th).replace('.', '')}"] = clusters[th][ii]
        out.to_csv(f"data/{t}_clusters.csv", index=False)
        Isub = I[np.ix_(ii, ii)]; u = np.triu_indices(len(ii), 1)
        np.save(f"out/ident_{t}.npy", Isub[u])
    pd.DataFrame({"Sequence": allseq,
                  **{f"cluster_{str(th).replace('.', '')}": clusters[th] for th in THRESHOLDS}}
                 ).to_csv("data/clusters_union.csv", index=False)
    csd = {}
    for t in TASKS:
        ii = [idx[s] for s in dfs[t].Sequence]
        vc = pd.Series(clusters[0.8][ii]).value_counts().value_counts().sort_index()
        csd[t] = {str(k): int(v) for k, v in vc.items()}
    json.dump(csd, open("out/cluster_size_dist_08.json", "w"))
    print("prepare_data complete:", len(allseq), "unique peptides", flush=True)


if __name__ == "__main__":
    main()
