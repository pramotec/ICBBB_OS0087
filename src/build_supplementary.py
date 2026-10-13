#!/usr/bin/env python3
"""Build the supplementary material: every figure and table with its legend.

Run after the result tables exist. Figures are embedded at 6.4 inches; tables
that are short enough to read are rendered inline, longer ones are referenced by
filename and shipped as CSV in results/tables/.
"""
from __future__ import annotations

import os
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = "gh_repo"
FIG = f"{ROOT}/results/figures"
TAB = f"{ROOT}/results/tables"
MAX_INLINE_ROWS = 28

FIGURES = [
    ("fig_multilabel_overlap.png", "Overlap between the three death-mode annotations.",
     "Every one of the 497 peptides in the pooled collection is positive for at least one "
     "mode, so each task's negative set consists entirely of positives of another task. "
     "Seventy-four peptides (14.9 %) are annotated for more than one mode. This is the "
     "structural reason a negative in this design means \u201cacts by another mechanism\u201d "
     "rather than \u201cinactive\u201d, and why the negative-set tiers of Figure S4 are needed to "
     "interpret any performance figure."),
    ("fig_redundancy.png", "Sequence redundancy within the death-mode collection.",
     "All-against-all global alignment (BLOSUM62, gap open \u221210, extend \u22120.5), identity "
     "normalised by the shorter sequence following the CD-HIT convention. Of 123,256 pairs, "
     "448 reach 0.8 identity. Greedy longest-first clustering at 0.8 reduces each task to "
     "69\u201372 % of its nominal size. Cross-label redundant pairs \u2014 near-identical sequences "
     "carrying opposite labels \u2014 number 76 for apoptosis and 82 for necrosis and set a "
     "ceiling on attainable specificity that no modelling choice can lift."),
    ("fig_debias_qc.png", "Length matching across the three negative tiers.",
     "Peptide length distributions for positives (blue) and negatives (orange) in each task "
     "and tier, before and after 1:1 optimal length matching with a two-residue caliper. "
     "P-values are Mann\u2013Whitney tests of the between-class length difference; red marks "
     "p < 0.05. Matching inside tier 1 retains 69 % of apoptosis and 64 % of necrosis "
     "positives because that negative pool is smaller than the positive set; against the "
     "external reservoirs of tiers 2 and 3 it retains 99\u2013100 %. Note the unmatched tier-3 "
     "panels, where the negatives are markedly longer than the positives \u2014 the confound "
     "reverses sign rather than disappearing when external negatives are substituted."),
    ("fig_tier_comparison.png",
     "Performance across negative-set tiers, with and without length matching.",
     "Random Forest, 438 descriptors, identity-aware outer folds on the union 0.8-identity "
     "clusters; mean Matthews correlation coefficient over repeats with a t-based 95 % "
     "confidence interval. Dashes mark the 38-descriptor composition-and-property subset. "
     "Tier 1 (negatives are anticancer peptides acting through another mode) is the lowest "
     "of the three in both tasks once length is matched. The apoptosis gap between tier 1 "
     "and tier 2 is 0.167 MCC and measures how much of the apparent performance reflects "
     "distinguishing anticancer peptides from other peptides rather than distinguishing "
     "mechanisms. n = 10 repeats for tier 1 and 5 for tiers 2\u20133; the unit of replication "
     "is the repeat, each pooling five outer folds."),
    ("revised_fig_roc_pr.png", "Receiver-operating and precision\u2013recall curves.",
     "Curves pooled across outer folds for each task and model under identity-aware "
     "splitting. Shown for the original tier-1 design. The precision\u2013recall panels carry "
     "the prevalence baseline, without which the PR curves of a 0.57-prevalence problem "
     "are easy to over-read."),
    ("revised_fig_f1_mcc_distribution.png",
     "Distribution of F1 and MCC across repeats, by model.",
     "Each point is one repeat of the nested cross-validation, so the spread is resampling "
     "uncertainty under a fixed protocol. The panels show why F1 cannot be the selection "
     "criterion on these data: the support vector machine attains the highest F1 of any "
     "model on one task while its MCC is indistinguishable from zero, because at a "
     "prevalence of 0.57 a near-all-positive classifier scores well on F1 by construction."),
    ("revised_fig_confusion.png", "Confusion matrices at the fixed 0.5 threshold.",
     "Counts pooled over all outer folds and repeats, per task and model. Reported in "
     "response to the reviewer's request for confusion matrices alongside the "
     "high-recall/low-specificity operating characteristic."),
    ("revised_fig_holdout_variability.png",
     "Variability of a single 80/20 hold-out across 100 random seeds.",
     "The original protocol was re-run with 100 different split seeds. Hold-out F1 has a "
     "standard deviation of 0.036\u20130.058 and a range of 0.19\u20130.29, while cross-validated F1 "
     "varies 2\u20133 times less (SD 0.014\u20130.024). For 14 of 15 task-by-model combinations the "
     "mean gap between hold-out and cross-validation lies between \u22120.016 and +0.013, and "
     "the hold-out is higher in 38\u201363 % of seeds. Differences of 0.01\u20130.03 between models "
     "measured on one hold-out are therefore not interpretable, and cross-validation is "
     "not systematically pessimistic."),
    ("revised_fig_descriptor_ablation.png",
     "Descriptor-block ablation against trivial baselines.",
     "Random Forest MCC under identity-aware folds for six feature sets: peptide length "
     "alone, length plus molecular weight, the 18 physicochemical properties, the 20 "
     "amino-acid composition features, the 400 dipeptide-composition features, and all "
     "438. Error bars are 95 % confidence intervals across repeats. The 400 "
     "dipeptide-composition features are the weakest block in every task."),
    ("revised_fig_length_matched.png",
     "Effect of length matching on the original tier-1 design.",
     "Grey bars are the unmatched design, blue bars the 1:1 length-matched design, red "
     "diamonds the 38-descriptor composition-and-property set on matched data. For "
     "necrosis, peptide length alone reaches MCC 0.402 unmatched and 0.105 matched, which "
     "is the clearest statement of the size confound in these data."),
    ("fig_v3_comparison.png",
     "Effect of adding 58 manually curated peptides to the apoptosis positive set.",
     "Paired difference in MCC between the v3 and original apoptosis designs, by tier, "
     "descriptor set and model; bars are 95 % confidence intervals across 10 repeats and "
     "the shaded band is the smallest difference those repeats could resolve. Left: the "
     "length-matched designs, where no comparison is significant after Holm correction and "
     "the largest absolute change is 0.067 MCC. Right: the unmatched designs, which are not "
     "interpretable for this comparison because adding positives without negatives raised "
     "prevalence from 0.592 to 0.651; all eight Tier-1 differences are positive, the "
     "signature of that shift. The 58 peptides form 28 new identity clusters, and Tier 1 "
     "gains only one cluster because its matched design is capped by the negative pool."),
    ("fig_expanded_comparison.png",
     "Effect of adding six peptides recovered by our own literature search.",
     "Paired difference in MCC between the expanded and original tier-1 designs across 32 "
     "comparisons; none is significant after Holm correction and the largest absolute "
     "change is 0.032 MCC. The shaded band is the smallest difference 10 repeats could "
     "resolve, which is about 0.09 MCC - the null result excludes a large change, not a "
     "small one."),
    ("fig_mining_yield.png",
     "Yield of open-ended literature mining for all three death modes.",
     "Left: the fate of every string extracted from full text, as a share of all strings. "
     "Machinery fragments are rejected only for pyroptosis; for apoptosis and necrosis a "
     "machinery match is flagged and demoted rather than dropped, because 11 of the 173 "
     "apoptosis positives in the mined length range are themselves BH3-domain or "
     "p53-derived peptides. Right: candidates surviving each evidence requirement, per "
     "1,000 papers opened, on a symmetric-log scale. Requiring a specific assay readout "
     "together with evidence that the peptide was applied to cancer cells reduces the "
     "yield by three orders of magnitude, and to zero for pyroptosis."),
    ("fig_lookup_funnel.png",
     "Targeted mechanism lookup over 873 known anticancer peptides.",
     "Left: attrition from the queried list to the peptides with usable evidence. Right: "
     "the resulting peptide-by-mode proposals, ranked. Priority D rests on paper-level "
     "co-occurrence only and was shown, against known labels, to fire on 94.7 % of true "
     "positives and 78.3 % of true negatives \u2014 it retrieves papers but does not attribute "
     "mechanism. The tiers are retrieval ranks for a curator, not labels."),
]

TABLES = [
    ("dataset_audit.csv", "Composition of the three curated datasets as analysed.",
     "Row counts, class balance, length range and residue alphabet for each file. The "
     "counts differ by one from those in the original submission, where the header line "
     "was counted as data."),
    ("multilabel_overlap.csv", "Cross-tabulation of the three death-mode annotations.",
     "For every ordered pair of modes, the number of peptides positive for the first that "
     "are positive, negative or absent for the second."),
    ("redundancy_summary.csv", "Sequence redundancy at four identity thresholds.",
     "Pair counts and resulting cluster counts per task at 0.95, 0.9, 0.8 and 0.7 identity, "
     "with the number of cross-label redundant pairs at each threshold."),
    ("cross_pool_redundancy.csv",
     "Redundancy between the death-mode peptides and the external pools.",
     "Near-duplicate pair and shared-cluster counts between our peptides and each external "
     "pool at three thresholds. These are the numbers that make union-level clustering "
     "mandatory once external negatives are used."),
    ("dataset_design_qc.csv", "The twelve tiered modelling designs.",
     "Sample sizes, class balance, prevalence, reservoir size, matching retention, median "
     "lengths, mean absolute length difference per matched pair, the Mann\u2013Whitney p-value "
     "for the between-class length difference, and the number of identity clusters."),
    ("tiered_rf_table.csv",
     "Random Forest performance across tiers, matching states and descriptor sets.",
     "Mean over repeats with a t-based 95 % confidence interval for MCC, and means for F1, "
     "ROC-AUC, PR-AUC, recall and specificity, at the fixed 0.5 threshold."),
    ("tiered_model_comparison_stats.csv",
     "Paired model, descriptor-set and matching comparisons.",
     "Mean paired difference with a 95 % confidence interval and a Holm-adjusted p-value "
     "from a paired t-test on per-repeat values, within each family of comparisons. Random "
     "Forest and the SVM+RF+XGBoost stack differ significantly in 1 of 24 comparisons."),
    ("expanded_design_qc.csv",
     "Original and provisionally expanded tier-1 designs.",
     "The v2 designs add the six peptides in provisional_additions.csv that survived a "
     "hand audit of the mined and looked-up candidates. Cluster assignment for the added "
     "peptides was computed against the existing union clustering."),
    ("v3_design_comparison.csv",
     "Design sizes before and after the manual curation was merged.",
     "Positives, identity clusters, matching retention and prevalence for each apoptosis "
     "design in the original and v3 versions. Tier 1 gains one cluster; Tiers 2 and 3 gain "
     "74 and 79, because their matched designs are not capped by the negative pool."),
    ("v3_v1_vs_v3_tests.csv",
     "Paired tests for the manually curated additions.",
     "Mean paired difference with a 95 % confidence interval, raw and Holm-adjusted "
     "p-values, the standard deviation of the paired difference and the minimum detectable "
     "difference, for each of the 48 comparisons."),
    ("curated_additions_v3.csv",
     "The 58 peptides added by manual curation.",
     "Sequence, peptide name, representative DOI, length, independent anticancer "
     "corroboration and assigned identity cluster for every peptide merged into the "
     "apoptosis positive set."),
    ("curated_apoptosis_audit.csv",
     "Audit of the manual curation against the existing labels.",
     "All 78 curated entries with their status against the current apoptosis file, nearest "
     "sequence in the union with its identity and pool, and a recommended action. Eighteen "
     "were already positive, two contradicted existing negative labels, five record a "
     "sequence that is not the tested agent, and three are four-residue peptides."),
    ("expanded_summary_metrics.csv",
     "Performance on the original and six-peptide expanded positive sets.",
     "Mean over 10 repeats with 95 % confidence intervals for both tier-1 designs, both "
     "descriptor sets and four models."),
    ("expanded_v1_vs_v2_tests.csv",
     "Paired tests for the six-peptide expansion.",
     "Mean paired difference with confidence interval, Holm-adjusted p-value and minimum "
     "detectable difference for each of the 32 comparisons."),
    ("length_confounding.csv", "Between-class length difference per task.",
     "Median lengths, Mann\u2013Whitney U, p-value and rank-biserial correlation. Necrosis "
     "positives are longer than their negatives (22 against 15 residues, p = 3.1e-15); "
     "apoptosis classes have identical median length."),
    ("descriptor_ablation_summary.csv", "Descriptor-block ablation.",
     "Mean MCC with a 95 % confidence interval for each feature subset under identity-aware "
     "folds."),
    ("descriptor_ablation_tests.csv", "Paired tests for the descriptor ablation.",
     "Holm-corrected paired comparisons between feature subsets within each task."),
    ("holdout_variability_summary.csv",
     "Hold-out variability across 100 split seeds.",
     "Mean, standard deviation and range of hold-out and cross-validated F1 per task and "
     "model, with the mean gap between them and the fraction of seeds in which the hold-out "
     "is the higher of the two."),
    ("threshold_analysis.csv", "Effect of F1-optimal threshold selection.",
     "Metrics at the fixed 0.5 threshold against a threshold chosen to maximise F1 on inner "
     "out-of-fold predictions. The tuned threshold raises F1 but lowers MCC and roughly "
     "halves specificity in every task."),
    ("split_and_preprocessing_summary.csv",
     "Split sizes and fold-internal preprocessing.",
     "Train and test counts with positive and negative numbers per fold, features entering "
     "and leaving the preprocessor, principal components retained for the "
     "dipeptide-composition block, and the variance retained."),
    ("tuned_hyperparameters.csv", "Hyperparameters selected inside the training folds.",
     "One row per task, regime, repeat, fold and model. Tuning never saw the outer test "
     "fold."),
    ("label_provenance_audit.csv",
     "Literature verification of every positive label.",
     "For each of the 590 positive labels, the retrieved record count, the verdict, the "
     "mechanisms discussed in the retrieved literature, and the top matching DOI, PMID and "
     "title. Caveat: this is a sequence-string search, so a paper that names the peptide "
     "without printing its sequence is invisible \u2014 \u201cno hit\u201d is not evidence of a wrong "
     "label."),
    ("query_pilot.csv", "Query selection for literature mining.",
     "Hit count, papers opened, strings extracted, distinct sequences, sequences per 100 "
     "papers and independently corroborated anticancer peptides, for four query "
     "formulations per mode. Requiring a peptide class phrase rather than the bare word "
     "\u201cpeptide\u201d raised yield seventeen-fold for apoptosis."),
    ("filter_validation.csv", "Validation of the mining filters against curated peptides.",
     "The fraction of curated positives within the mined length range that survive every "
     "filter, and the number flagged as fragments of the death pathway itself."),
    ("mining_yield_summary.csv", "Open-ended mining yield for all three modes.",
     "Papers opened, strings extracted, rejections by reason, distinct peptide sequences, "
     "sequences new to the datasets, independent anticancer corroboration, and candidates "
     "meeting the top evidence requirement."),
    ("lookup_validation_summary.csv",
     "Calibration of the targeted lookup against known labels.",
     "Firing rates on true positives and true negatives for each evidence scope and tier "
     "threshold, using the curated apoptosis and necrosis labels of 186 peptides as ground "
     "truth."),
    ("lookup_window_experiment.csv",
     "Effect of the evidence window on the targeted lookup.",
     "Firing rates on true positives and true negatives for evidence windows of 0, 1, 2 and "
     "5 sentences around each sequence mention. No window both recalls and discriminates."),
    ("unidl4biopep_audit.csv", "Characterisation of the external peptide pools.",
     "Size, median length, mean GRAVY, median net charge and mean lysine-plus-arginine and "
     "cysteine fractions for our peptides and for each UniDL4BioPep pool. The "
     "\u201calternative\u201d non-anticancer pool has a median net charge of 0 against +3.1 for every "
     "anticancer pool and was therefore not used."),
]


def add_csv_table(doc, path, max_rows=MAX_INLINE_ROWS):
    d = pd.read_csv(path)
    note = ""
    if len(d) > max_rows:
        note = f"First {max_rows} of {len(d)} rows; the full table is in the repository."
        d = d.head(max_rows)
    if d.shape[1] > 10:
        d = d.iloc[:, :10]
        note = (note + " " if note else "") + "First 10 columns shown."
    t = doc.add_table(rows=1, cols=d.shape[1])
    t.style = "Light Grid Accent 1"
    for j, c in enumerate(d.columns):
        cell = t.rows[0].cells[j]
        cell.text = str(c)
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(7)
    for _, row in d.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
            for r in cells[j].paragraphs[0].runs:
                r.font.size = Pt(7)
    return note


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
        s.top_margin = s.bottom_margin = Inches(0.8)

    doc.add_heading("Supplementary Material", 0)
    p = doc.add_paragraph()
    p.add_run("Machine-learning prediction of the mode of cancer-cell death induced by "
              "anticancer peptides: apoptosis and necrosis").italic = True
    doc.add_paragraph(
        "All figures are at 300 dpi and all tables are supplied as CSV in the accompanying "
        "repository under results/figures/ and results/tables/. Unless stated otherwise, "
        "outer folds respect 0.8-identity clusters computed over the union of the "
        "death-mode peptides and both external peptide pools; hyperparameters are tuned "
        "inside training folds only; the operating threshold is 0.5; and intervals are "
        "t-based 95 % confidence intervals across repeats, each repeat pooling five outer "
        "folds.")

    doc.add_page_break()
    doc.add_heading("Supplementary Figures", 1)
    n = 0
    for fname, title, legend in FIGURES:
        path = os.path.join(FIG, fname)
        if not os.path.exists(path):
            continue
        n += 1
        doc.add_picture(path, width=Inches(6.4))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.add_run(f"Figure S{n}. ").bold = True
        cap.add_run(title + " ").bold = True
        cap.add_run(legend)
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for r in cap.runs:
            r.font.size = Pt(9)
        src = doc.add_paragraph(f"Source file: results/figures/{fname}")
        src.runs[0].font.size = Pt(8)
        src.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        doc.add_page_break()

    doc.add_heading("Supplementary Tables", 1)
    m = 0
    for fname, title, legend in TABLES:
        path = os.path.join(TAB, fname)
        if not os.path.exists(path):
            continue
        m += 1
        cap = doc.add_paragraph()
        cap.add_run(f"Table S{m}. ").bold = True
        cap.add_run(title + " ").bold = True
        cap.add_run(legend)
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for r in cap.runs:
            r.font.size = Pt(9)
        note = add_csv_table(doc, path)
        foot = doc.add_paragraph(
            (note + "  " if note else "") + f"Source file: results/tables/{fname}")
        foot.runs[0].font.size = Pt(8)
        foot.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        doc.add_paragraph()

    out = f"{ROOT}/supplementary/Supplementary_Material.docx"
    doc.save(out)
    print(f"wrote {out}: {n} figures, {m} tables")


if __name__ == "__main__":
    main()
