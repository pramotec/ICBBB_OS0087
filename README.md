# ICBBB_OS0087
Machine learning prediction of the mode of cancer-cell death induced by anticancer peptides: apoptosis and necrosis
Machine-learning prediction of the mode of cancer-cell death induced by anticancer peptides
Code, data and results for the revised manuscript (ICBBB_OS0087). Two binary tasks —
apoptosis and necrosis — over a curated collection of 497 anticancer peptides, evaluated
under identity-aware splitting, length matching and repeated nested cross-validation.
Pyroptosis was curated and evaluated but is not reported as a prediction task. The
measurements behind that decision are in `docs/two_task_scope_note.md` and the
Limitations section; the data and results are kept here so the decision is checkable.
---
Quick start
```bash
conda env create -f env/environment.yml
conda activate acp-reanalysis
python src/run_tiered.py            # ~3 h on 5 workers; resumes if interrupted
python src/aggregate_tiered.py      # writes the summary tables
```
Every script writes its own outputs; nothing depends on a manual step.
What is where
Path	Contents
`data/raw/`	the three curated death-mode datasets as analysed
`data/designs/`	the tiered modelling designs — the original 12, the six-peptide `_v2` variants and the `_v3` variants that merge the authors' manual curation — each with `Sequence,Label,length,cluster_08`
`data/clusters/`	0.8-identity cluster assignment over the union of all peptide pools, and the cross-pool redundancy tables
`data/curation/`	provisional additions, excluded entries, label-provenance audit, mined-candidate audits, curation worklist
`src/`	descriptors, clustering, models, every run script, the two literature tools
`results/tables/`	every table in the manuscript and supplement, as CSV
`results/figures/`	every figure, at 300 dpi
`results/predictions/`	per-fold out-of-fold predictions, so any metric can be recomputed without retraining
`curation/runs/`	one folder per literature run, each with its manifest
`docs/`	manuscript sections, response to reviewers, methods, curation protocol
`supplementary/`	the supplementary PDF/DOCX with all figures, tables and legends
Reproducing the numbers
To reproduce	Run	Output
Identity clusters	`src/acp_cluster.py`	`data/clusters/clusters_union_all_pools.csv`
Descriptors (and the R-parity test)	`src/acp_features.py`, `tests/test_parity.py`	438 columns, max abs. difference from the R implementation 1.1e-13
Main tiered results	`src/run_tiered.py` → `src/aggregate_tiered.py`	`results/tables/tiered_*.csv`
Expanded-positive comparison	`src/run_expanded.py`	`results/tables/expanded_*.csv`
Hold-out variability (100 seeds)	`src/run_holdout_variability.py`	`results/tables/holdout_*.csv`
Descriptor ablation	`src/run_ablation.py`	`results/tables/descriptor_ablation_*.csv`
Literature mining	`src/mine_death_mode.py --mode apoptosis`	`curation/runs/<date>_<mode>/`
Targeted mechanism lookup	`src/lookup_mechanism.py --sequences <csv>`	`curation/runs/<date>_lookup_*/`
Seeds: master 42; outer repeat r uses 42+r; fold f in repeat r uses 42+100r+f;
the stacking inner split uses the fold seed + 1000. Package versions are in
`env/environment.yml` and in each run manifest.
Design decisions a reader should know before using these files
Folds respect identity clusters, not rows. Clustering is computed over the union
of the death-mode peptides and both external pools, because per-dataset clustering leaves
391 near-duplicate pairs spanning 74 clusters between our peptides and the external
negatives — a larger leakage path than the one being closed.
Length is matched, and the matched variant is primary. Peptide length is recoverable
from the descriptors and differs between classes. Matching costs the necrosis task
0.144 MCC, so unmatched numbers are not interpretable as mechanism prediction.
Negatives come in three tiers, reported side by side. Tier 1 = anticancer peptides
acting through another mode; tier 2 = anticancer peptides not annotated for the mode;
tier 3 = non-anticancer peptides. The gap between tiers measures how much of the
performance is "is this an anticancer peptide" rather than "which mechanism".
MCC is the primary metric. At a prevalence of 0.57 a classifier that labels nearly
everything positive scores a high F1 at MCC near zero, and F1 ranks it first.
Three dataset versions are kept, not one. The original designs, a `_v2` variant adding
six peptides from our own search, and a `_v3` variant adding 58 from the authors' manual
curation. Neither enlargement changed any length-matched result (Figures S5 and S6), and
keeping all three makes that checkable rather than asserted. The necrosis data are
identical in all three versions: the curation establishes apoptosis evidence for the added
peptides, not the absence of necrosis evidence, so adding them to the necrosis negative set
would have manufactured labels from positive-only evidence.
Mined and looked-up peptides are proposals, not labels. `data/curation/` separates
what was confirmed by reading the assay from what was retrieved automatically. Six
peptides are marked `provisional` and are included in the `*_v2` designs only.
Data sources
Death-mode sequences: https://github.com/TaoHu888/ACP_data_set.
External anticancer-peptide benchmarks: the UniDL4BioPep distribution (`main` and
`alternative` ACP sets). Literature: Europe PMC open-access subset, queried as recorded
in each run manifest.
Not included
The HTTP response cache from the literature runs (about 2 GB). It is fully regenerable
from the `papers_scanned.csv` in each run folder and is excluded by `.gitignore`.
Citation
Add the DOI of the archived release here on acceptance.
Licence
Code: MIT (`LICENSE`). Curated data and derived tables: CC BY 4.0
(`LICENSE-DATA`). Upstream datasets remain under their original terms.
