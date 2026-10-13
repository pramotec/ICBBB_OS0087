#!/usr/bin/env python3
"""
lookup_mechanism.py — targeted search for death-mode evidence, one peptide at a time.

This inverts the direction of mine_death_mode.py. That script asks "which papers
mention a death mode, and what sequences do they contain?", which is efficient at
finding papers but inefficient at finding peptides: an exhaustive sweep of 10,305
papers returned 10,274 distinct strings and only 14 with usable mechanism evidence.

Here the starting point is a list of peptides already established as anticancer by an
independent source. "Is this an anticancer peptide?" is therefore settled before the
search begins, and the only open question is which death programme it induces. Each
sequence is used as an exact-phrase query, so every retrieved paper is about that
peptide, and evidence is read only from the sentences and table rows that contain the
sequence itself.

What this buys, and what it does not — measured, not assumed
------------------------------------------------------------
The pipeline was calibrated against 186 peptides whose apoptosis status is already
curated (see lookup_validation_summary.csv and lookup_window_experiment.csv). Two
numbers decide how its output should be used.

RETRIEVAL WORKS. The exact-sequence query located 103 of 186 peptides (55 %) in
open-access full text, and for 95 % of those it surfaced at least one paper carrying a
death-mode assay term. Papers that refer to a peptide only by name - melittin, LL-37,
lactoferricin B - are invisible to a sequence query, so a peptide with no hit is NOT
evidence of a missing annotation; it is an unmeasured peptide. The two outcomes are
recorded separately and must never be conflated.

ATTRIBUTION DOES NOT. Within the retrieved papers, mechanism terms do not discriminate
between death modes. Using the necrosis labels of those same peptides as ground truth,
paper-scope evidence at tier <= 3 fired for 94.7 % of true necrosis positives and 78.3 %
of true necrosis negatives; at tier <= 2 the rates were 39.5 % and 28.3 %. Restricting
the evidence to a window of 0, 1, 2 or 5 sentences around the sequence raised precision
only by destroying recall: at a five-sentence window, tier <= 2 fired for 2.6 % of
positives and 4.3 % of negatives. There is no setting at which this method both recalls
and discriminates, because a paper reporting an anticancer peptide mentions apoptosis
almost regardless of which mechanism it demonstrates.

The tiers in the output are therefore RETRIEVAL RANKS FOR A CURATOR, NOT LABELS. What
the pipeline delivers is a bounded reading list: for each peptide, the open-access
papers that print its sequence, with the assay vocabulary each one contains, so that
assigning a death mode means reading two or three papers instead of hunting for them.

A peptide may receive evidence for more than one mode. That is reported as it stands,
because multi-mechanism behaviour is real and collapsing it would repeat the problem
this project set out to fix.

Nothing here assigns a label. Every row is a proposal carrying the matched assay term,
the sentence it came from, and a DOI, for a curator who reads the assay.

Usage
-----
    python3 lookup_mechanism.py --sequences peptides.csv --outdir lookup_out
    python3 lookup_mechanism.py --sequences peptides.csv --outdir lookup_out \
            --cache mine_apo/cache          # reuse full text already downloaded

`--sequences` accepts a CSV with a Sequence column, or a FASTA file. Results are written
one JSON file per peptide under <outdir>/parts/, so an interrupted run resumes instead
of starting over.

Outputs
-------
    mechanism_evidence.csv   one row per (peptide, mode, evidence unit)
    peptide_summary.csv      one row per peptide: best tier per mode, hit counts
    lookup_manifest.json     query template, script checksum, counts, package versions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import mine_death_mode as M

MIN_SEQ_LEN = 8          # shorter strings match by chance in unrelated text
MAX_PAPERS_PER_PEPTIDE = 25


def read_sequences(path):
    if path.lower().endswith((".fa", ".fasta", ".faa")):
        seqs, cur = [], []
        for line in open(path, encoding="utf-8"):
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            else:
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
    else:
        d = pd.read_csv(path, encoding="utf-8-sig")
        col = next((c for c in d.columns if "seq" in c.lower()), d.columns[0])
        seqs = d[col].astype(str).tolist()
    out = []
    for s in seqs:
        s = re.sub(r"[^A-Za-z]", "", s).upper()
        if len(s) >= MIN_SEQ_LEN and set(s) <= set(M.AA1):
            out.append(s)
    return sorted(set(out))


def evidence_units(xml, seq):
    """Sentences and table rows that contain the sequence verbatim."""
    sents, rows = M.parse_fulltext(xml)
    units = [("body", s) for s in sents] + [("table_row", t) for t in rows]
    hits = []
    for kind, text in units:
        if seq in text.upper().replace(" ", "").replace("-", ""):
            hits.append((kind, text))
        elif seq in text.upper():
            hits.append((kind, text))
    return hits


_TERM_RX = {mode: [(re.compile(p, re.I), name, spec)
                   for p, (name, spec) in cfg["terms"].items()]
            for mode, cfg in M.MODES.items()}


def score_unit(text, mode):
    """Mechanism terms, specific readouts and treatment context in one evidence unit."""
    hits = [(n, sp) for rx, n, sp in _TERM_RX[mode] if rx.search(text)]
    mechs = sorted({n for n, _ in hits})
    specific = sorted({n for n, sp in hits if sp})
    treat = bool(M.TREATMENT_CONTEXT.search(text) and M.CANCER_CONTEXT.search(text))
    return mechs, specific, treat


def lookup_one(seq, args, cachedir):
    """Evidence rows for one peptide: one row per (paper, mode).

    Two scopes are scored and reported separately, because they carry different
    strengths of attribution.

    UNIT scope is the same rule the open-ended miner uses: the assay term must appear
    in the same sentence or table row as the sequence. It is the stricter reading, and
    on the positive control it fired for only a minority of peptides, because papers
    routinely print a sequence in a table and describe the assay in the Results text.

    PAPER scope uses the fact that this search is targeted: the paper was retrieved by
    an exact-sequence query, so it is a paper about this peptide. Mechanism terms and
    treatment context anywhere in such a paper are evidence about this peptide —
    provided the paper is not discussing a large panel. The number of distinct
    peptide-like sequences in the paper is recorded so a curator can judge that
    directly, and paper-scope evidence from a many-peptide paper is demoted.
    """
    d = M.http_get(f"{M.EPMC}/search",
                   dict(query=f'"{seq}"', format="json", pageSize=MAX_PAPERS_PER_PEPTIDE,
                        resultType="core"),
                   cachedir, "seq", email=args.email)
    res = (d.get("resultList") or {}).get("result") or []
    hit_count = d.get("hitCount", 0)
    papers = [r for r in res if r.get("pmcid")]
    rows, n_found = [], 0
    for r in papers:
        meta = dict(pmcid=r.get("pmcid"), pmid=r.get("pmid"), doi=r.get("doi"),
                    year=r.get("pubYear"), title=r.get("title"))
        xml = M.http_get(f"{M.EPMC}/{r['pmcid']}/fullTextXML", {}, cachedir, "ft",
                         raw=True, email=args.email)
        units = evidence_units(xml, seq)
        if not units:
            continue
        n_found += 1
        sents, trows = M.parse_fulltext(xml)
        all_units = [("body", s) for s in sents] + [("table_row", t) for t in trows]
        full = " ".join(t for _, t in all_units)
        n_seqs_in_paper = len({s for _, t in all_units
                               for s, _, _ in M.extract_sequences(t)})
        for mode in M.MODES:
            # ---- unit scope
            best = None
            for kind, text in units:
                mechs, specific, treat = score_unit(text, mode)
                if not mechs:
                    continue
                tier = 4 if not treat else (3 if not specific else
                                            (1 if kind == "table_row" else 2))
                if best is None or tier < best[0]:
                    best = (tier, kind, mechs, specific, treat, text)
            # ---- paper scope
            p_mechs, p_specific, p_treat = score_unit(full, mode)
            if p_mechs:
                if not p_treat:
                    p_tier = 4
                elif not p_specific:
                    p_tier = 3
                else:
                    p_tier = 2 if n_seqs_in_paper <= 3 else 3
            else:
                p_tier = None
            if best is None and p_tier is None:
                continue
            rows.append(dict(
                sequence=seq, length=len(seq), mode=mode,
                unit_tier=best[0] if best else None,
                unit_kind=best[1] if best else None,
                unit_assay_terms=";".join(best[2]) if best else "",
                unit_specific_terms=";".join(best[3]) if best else "",
                paper_tier=p_tier,
                paper_assay_terms=";".join(p_mechs),
                paper_specific_terms=";".join(p_specific),
                paper_treatment_context=p_treat,
                n_peptide_sequences_in_paper=n_seqs_in_paper,
                best_tier=min([t for t in (best[0] if best else None, p_tier)
                               if t is not None]),
                evidence_text=(best[5][:500] if best else ""), **meta))
    return dict(sequence=seq, hit_count=int(hit_count), n_papers_opened=len(papers),
                n_papers_with_sequence=n_found, n_evidence_rows=len(rows), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", required=True, help="CSV with a Sequence column, or FASTA")
    ap.add_argument("--outdir", default="lookup_out")
    ap.add_argument("--cache", default=None,
                    help="existing cache directory to reuse (e.g. mine_apo/cache)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="process at most N peptides")
    ap.add_argument("--email", default=os.environ.get("CONTACT_EMAIL"))
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    parts = os.path.join(a.outdir, "parts")
    os.makedirs(parts, exist_ok=True)
    cachedir = a.cache or os.path.join(a.outdir, "cache")
    os.makedirs(cachedir, exist_ok=True)

    import shutil as _shutil
    _self = os.path.abspath(__file__)
    with open(_self, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    _shutil.copy2(_self, os.path.join(a.outdir, "script_used.py"))

    seqs = read_sequences(a.sequences)
    if a.limit:
        seqs = seqs[:a.limit]
    todo = [s for s in seqs
            if not os.path.exists(os.path.join(parts, hashlib.sha1(s.encode()).hexdigest()[:16] + ".json"))]
    print(f"peptides: {len(seqs)} | already done: {len(seqs) - len(todo)} | to do: {len(todo)}",
          flush=True)

    def work(seq):
        try:
            out = lookup_one(seq, a, cachedir)
        except Exception as exc:
            out = dict(sequence=seq, error=str(exc), hit_count=0, n_papers_opened=0,
                       n_evidence_rows=0, rows=[])
        p = os.path.join(parts, hashlib.sha1(seq.encode()).hexdigest()[:16] + ".json")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        os.replace(tmp, p)
        return out

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for _ in ex.map(work, todo):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}", flush=True)

    allrows, summary = [], []
    for s in seqs:
        p = os.path.join(parts, hashlib.sha1(s.encode()).hexdigest()[:16] + ".json")
        if not os.path.exists(p):
            continue
        rec = json.load(open(p, encoding="utf-8"))
        allrows += rec.get("rows", [])
        r = dict(sequence=s, length=len(s), hit_count=rec.get("hit_count", 0),
                 n_papers_opened=rec.get("n_papers_opened", 0),
                 n_papers_with_sequence=rec.get("n_papers_with_sequence", 0),
                 sequence_found_in_full_text=bool(rec.get("n_papers_with_sequence", 0)))
        for mode in M.MODES:
            rs = [x for x in rec.get("rows", []) if x["mode"] == mode]
            ut = [x["unit_tier"] for x in rs if x.get("unit_tier") is not None]
            pt = [x["paper_tier"] for x in rs if x.get("paper_tier") is not None]
            r[f"{mode}_unit_tier"] = min(ut) if ut else None
            r[f"{mode}_paper_tier"] = min(pt) if pt else None
            r[f"{mode}_best_tier"] = min(ut + pt) if (ut or pt) else None
        summary.append(r)

    E = pd.DataFrame(allrows)
    S = pd.DataFrame(summary)
    if len(E):
        E = (E.sort_values("best_tier")
               .drop_duplicates(subset=["sequence", "mode", "pmcid"]))
        E.to_csv(os.path.join(a.outdir, "mechanism_evidence.csv"), index=False)
    S.to_csv(os.path.join(a.outdir, "peptide_summary.csv"), index=False)

    manifest = dict(
        finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), script=os.path.basename(_self),
        script_sha256=sha, query_template='"<SEQUENCE>"',
        sequences_file=os.path.abspath(a.sequences), n_peptides=len(seqs),
        max_papers_per_peptide=MAX_PAPERS_PER_PEPTIDE,
        results=dict(
            n_with_any_hit=int((S.hit_count > 0).sum()),
            n_with_sequence_in_full_text=int(S.sequence_found_in_full_text.sum()),
            n_papers_with_sequence=int(S.n_papers_with_sequence.sum()),
            n_evidence_rows=int(len(E)),
            by_mode_best_tier={
                mode: {str(k): int(v) for k, v in
                       S[f"{mode}_best_tier"].value_counts().sort_index().items()}
                for mode in M.MODES}),
        packages={p: __import__(p).__version__ for p in ("pandas", "requests")})
    with open(os.path.join(a.outdir, "lookup_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n================ SUMMARY ================")
    print(f"peptides queried                   : {len(S)}")
    print(f"  with at least one indexed paper  : {int((S.hit_count > 0).sum())}")
    print(f"  sequence located in full text    : {int(S.sequence_found_in_full_text.sum())}")
    print(f"evidence rows                      : {len(E)}")
    for mode in M.MODES:
        col = S[f"{mode}_best_tier"]
        print(f"\n{mode}: best tier per peptide")
        print("  " + col.value_counts().sort_index().to_string().replace("\n", "\n  "))
    print(f"\nwrote {a.outdir}/mechanism_evidence.csv and peptide_summary.csv")
    print("\nEVERY ROW IS A PROPOSAL. A peptide with no hit is unmeasured, not negative.")


if __name__ == "__main__":
    main()
