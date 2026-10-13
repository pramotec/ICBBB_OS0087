# ICBBB_OS0087 — machine-learning prediction of the mode of cancer-cell death induced by anticancer peptides

[![Code licence: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data licence: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-green.svg)](LICENSE-DATA)

Code, data, trained-model inputs and every result behind the revised manuscript
**"Machine learning-based prediction of anticancer peptides inducing apoptosis and necrosis
in cancer cells"** (ICBBB_OS0087).

Two binary tasks — apoptosis and necrosis — over a curated collection of anticancer
peptides, evaluated under identity-aware splitting, length matching and repeated nested
cross-validation. Everything reported in the manuscript and the supplement can be
regenerated from this repository.

- **Repository:** <https://github.com/pramotec/ICBBB_OS0087>
- **Corresponding author:** Pramote Chumnanpuen, Department of Zoology, Faculty of Science,
  Kasetsart University
- **Authors:** Khunanon Seehamat¹, Pitawat Piruksa², Pramote Chumnanpuen¹
  (¹ Department of Zoology, Kasetsart University; ² Department of Computer Science,
  Thammasat University)

Pyroptosis was curated and evaluated but is **not** reported as a prediction task. The
measurements behind that decision are in `docs/two_task_scope_note.md` and in the
Limitations section; the data and results are kept here so the decision is checkable
rather than asserted.

---

## Quick start

```bash
conda env create -f env/environment.yml
conda activate acp-reanalysis

python src/run_tiered.py        # main tiered analysis, ~3 h on 8 workers; resumes if interrupted
python src/aggregate_tiered.py  # writes results/tables/tiered_*.csv
```

Every script writes its own outputs and its own run manifest; no step depends on manual
editing. Long runs write one file per repeat and skip completed ones on restart, so an
interrupted job resumes instead of starting over.

## What is where

| Path | Contents |
|---|---|
| `data/raw/` | the three curated death-mode datasets exactly as analysed |
| `data/designs/` | every modelling design as `Sequence,Label,length,cluster_08` — the original 12, plus the `_v2`, `_v3` and `_v4` variants described below |
| `data/clusters/` | identity-cluster assignment at 0.9 / 0.8 / 0.7 over the union of all peptide pools, and the cross-pool redundancy tables |
| `data/curation/` | the per-peptide curation table, the audit of the authors' manual curation, provisional additions, excluded entries, label-provenance audit, mined-candidate audits and the reference verification |
| `src/` | descriptors, clustering, models, every run script, the two literature tools |
| `results/tables/` | every table in the manuscript and supplement, as CSV |
| `results/figures/` | every figure, 300 dpi |
| `results/predictions/` | per-fold out-of-fold predictions, so any metric can be recomputed without retraining |
| `curation/runs/` | one folder per literature run, each with its manifest, query and script checksum |
| `docs/` | manuscript (tracked and clean), response to reviewers, methods, verified references, resubmission checklist |
| `supplementary/` | `Supplementary_Material.docx` with every figure and table and its legend |
| `env/` | `environment.yml` and `requirements.txt` with pinned versions |
| `tests/` | the R-parity test for the descriptor implementation |

## Reproducing the numbers

| To reproduce | Run | Output |
|---|---|---|
| Identity clusters | `src/acp_cluster.py` | `data/clusters/clusters_union_all_pools_v4.csv` |
| Descriptors (and the R-parity test) | `src/acp_features.py`, `tests/test_parity.py` | 438 columns; max absolute difference from the original R implementation 1.1e-13 |
| Main tiered results | `src/run_tiered.py` → `src/aggregate_tiered.py` | `results/tables/tiered_*.csv` |
| Six-peptide expansion | `src/run_expanded.py` | `results/tables/expanded_*.csv` |
| Manual-curation merge (v3) | `src/run_v3.py` | `results/tables/v3_*.csv` |
| Checklist resolution (v4) | `src/run_v4.py` | `results/tables/v4_*.csv` |
| Hold-out variability, 100 seeds | `src/run_holdout_variability.py` | `results/tables/holdout_*.csv` |
| Descriptor ablation | `src/run_ablation.py` | `results/tables/descriptor_ablation_*.csv` |
| Literature mining | `src/mine_death_mode.py --mode apoptosis` | `curation/runs/<date>_<mode>/` |
| Targeted mechanism lookup | `src/lookup_mechanism.py --sequences <csv>` | `curation/runs/<date>_lookup_*/` |

Seeds: master 42; outer repeat *r* uses 42 + *r*; fold *f* within repeat *r* uses
42 + 100*r* + *f*; the stacking inner split uses the fold seed + 1000. Package versions are
pinned in `env/environment.yml` and recorded again in every run manifest.

## Design decisions a reader should know before using these files

**Folds respect identity clusters, not rows.** Clustering is computed over the *union* of
the death-mode peptides and both external pools. Per-dataset clustering leaves 391
near-duplicate pairs spanning 74 clusters between our peptides and the external negatives —
a larger leakage path than the one being closed.

**Length is matched, and the matched variant is primary.** Peptide length is recoverable
from composition descriptors and differs between classes. Matching costs the necrosis task
0.144 MCC, so unmatched numbers are not interpretable as mechanism prediction.

**Negatives come in three tiers, reported side by side.** Tier 1 = anticancer peptides
acting through another mode; Tier 2 = anticancer peptides not annotated for the mode;
Tier 3 = non-anticancer peptides. The gap between tiers measures how much of the
performance answers "is this an anticancer peptide" rather than "which mechanism".

**MCC is the primary metric.** At a prevalence near 0.6 a classifier that labels almost
everything positive scores a high F1 at MCC near zero, and F1 ranks it first. F1 is
retained for comparability with the literature, never as a selection criterion.

**Four dataset versions are kept, not one.**

| Version | What it adds | Why it is kept |
|---|---|---|
| original | the curated collection as received | the baseline every comparison is paired against |
| `_v2` | six peptides recovered by our own literature search and confirmed by reading the cited assay | tests whether a small, carefully read expansion moves anything |
| `_v3` | 58 peptides from the authors' independent manual curation | tests whether a larger expansion of the apoptosis positives moves anything |
| `_v4` | the authors' resolution of the resubmission checklist: the six provisional peptides confirmed, and nine peptides relabelled from negative to positive on the mechanism their cited assay reports | tests whether correcting the labels we ourselves flagged changes the conclusion |

No length-matched comparison between versions reached significance after Holm correction.
Keeping all four makes that checkable rather than asserted, and lets a reader re-run any
analysis against whichever labelling they find most defensible.

**Mined and looked-up peptides are proposals, not labels.** `data/curation/` separates what
a curator confirmed by reading the assay from what an automated search merely retrieved.
The distinction is carried per row in `per_peptide_curation_table.csv` as
`Evidence_strength`, and it matters: in validation against known labels, paper-level
mechanism co-occurrence fired on 95 % of true positives *and* 78 % of true negatives. It
retrieves; it does not attribute.

**Nine entries in the source collection are not anticancer peptides** — four-residue caspase
cleavage motifs, substance P and penetratin among them — and are excluded from every
negative tier. They are listed with reasons in `data/curation/`.

## Data sources

Death-mode sequences were taken from <https://github.com/TaoHu888/ACP_data_set>, the
collection assembled by the first author. This repository is the archival copy maintained
by the corresponding author and is the version the manuscript cites.

External anticancer-peptide benchmarks: the UniDL4BioPep distribution (`main` and
`alternative` ACP sets). Literature: the Europe PMC open-access subset, queried exactly as
recorded in each run manifest. Reference metadata was verified against CrossRef; the
results are in `data/curation/reference_verification.csv` and `docs/verified_references.md`.

## Not included

The HTTP response cache from the literature runs, about 2 GB. It is fully regenerable from
the `papers_scanned.csv` in each run folder and is excluded by `.gitignore`.

## Citation

If you use this code or these data, please cite the manuscript and the archived release.
Replace the placeholder below with the Zenodo DOI once the release is minted (see
`docs/ZENODO_GUIDE.md`):

```bibtex
@article{Seehamat_ICBBB_OS0087,
  author  = {Seehamat, Khunanon and Piruksa, Pitawat and Chumnanpuen, Pramote},
  title   = {Machine learning-based prediction of anticancer peptides inducing
             apoptosis and necrosis in cancer cells},
  year    = {2026},
  note    = {Code and data: https://github.com/pramotec/ICBBB_OS0087},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

`CITATION.cff` carries the same metadata in machine-readable form; update its `doi:` field
at the same time.

## Licence

Code: MIT (`LICENSE`). Curated data and derived tables: CC BY 4.0 (`LICENSE-DATA`).
Upstream datasets remain under their original terms.

## Contact

Questions about the data or the analysis: Pramote Chumnanpuen (corresponding author),
Department of Zoology, Faculty of Science, Kasetsart University. Please open an issue on
<https://github.com/pramotec/ICBBB_OS0087/issues> for anything that can be discussed in
public, so the answer is available to the next reader.
