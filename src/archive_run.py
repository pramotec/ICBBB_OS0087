#!/usr/bin/env python3
"""
archive_run.py — package one literature-mining run so it can be cited and repeated.

A mining run is only useful later if three things travel with its output: the exact query
that produced it, the exact script that ran it, and the list of papers that were opened.
Without those, the numbers in the paper ("11 candidates from 3,000 papers") cannot be
checked by anyone, including you in six months.

This script takes a finished --outdir and writes a self-contained, dated run folder:

    curation/runs/<YYYY-MM-DD>_<label>/
        run_manifest.json          query, script checksum, date, counts, versions
        pyroptosis_candidates.csv
        pyroptosis_rejected.csv
        papers_scanned.csv
        mine_pyroptosis.py         the exact script used, copied verbatim

The API cache is deliberately NOT copied. It is large and fully regenerable, and it stays
where it is so re-runs remain instant. Its size and location are recorded in the manifest.

Usage
-----
    python3 archive_run.py --rundir pyro_out --script mine_pyroptosis.py \\
                           --label pyroptosis_p30 --archive curation/runs
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import sys

import pandas as pd

COPY = ["pyroptosis_candidates.csv", "pyroptosis_rejected.csv", "papers_scanned.csv"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(path):
    total = files = 0
    for root, _, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
                files += 1
            except OSError:
                pass
    return total, files


def extract_query(script_path):
    """Read SEARCH_QUERY out of the script source, so the manifest cannot drift from it."""
    src = open(script_path, encoding="utf-8").read()
    m = re.search(r"SEARCH_QUERY\s*=\s*\((.*?)\n\)", src, re.S)
    if not m:
        return None
    parts = re.findall(r"'([^']*)'", m.group(1))
    return "".join(parts).strip()


def summarise(rundir):
    out = {}
    cand = os.path.join(rundir, "pyroptosis_candidates.csv")
    if os.path.exists(cand):
        d = pd.read_csv(cand)
        out["n_candidate_rows"] = len(d)
        out["n_distinct_sequences"] = int(d.sequence.nunique())
        if "already_in_dataset" in d:
            out["n_not_already_in_dataset"] = int((~d.already_in_dataset).sum())
        if "tier_meaning" in d:
            out["by_tier"] = d.tier_meaning.value_counts().sort_index().to_dict()
        if "assay_terms" in d:
            terms = (d.assay_terms.dropna().astype(str).str.split(";").explode()
                     .str.strip().replace("", pd.NA).dropna())
            out["assay_terms_seen"] = terms.value_counts().to_dict()
    rej = os.path.join(rundir, "pyroptosis_rejected.csv")
    if os.path.exists(rej):
        d = pd.read_csv(rej)
        out["n_rejected_rows"] = len(d)
        out["rejection_reasons"] = d.reason.value_counts().to_dict()
    pap = os.path.join(rundir, "papers_scanned.csv")
    if os.path.exists(pap):
        d = pd.read_csv(pap)
        out["n_papers_opened"] = int(d.pmcid.nunique()) if "pmcid" in d else len(d)
        if "year" in d:
            yr = pd.to_numeric(d.year, errors="coerce").dropna()
            if len(yr):
                out["paper_years"] = {"min": int(yr.min()), "max": int(yr.max())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True, help="the --outdir of the finished run")
    ap.add_argument("--script", default="mine_pyroptosis.py")
    ap.add_argument("--label", required=True, help="short name, e.g. pyroptosis_p30")
    ap.add_argument("--archive", default="curation/runs")
    ap.add_argument("--pages", type=int, default=None, help="--pages value used, for the record")
    ap.add_argument("--note", default="", help="one line on why this run was made")
    a = ap.parse_args()

    if not os.path.isdir(a.rundir):
        sys.exit(f"no such run directory: {a.rundir}")
    if not os.path.exists(a.script):
        sys.exit(f"no such script: {a.script}")

    stamp = dt.date.today().isoformat()
    dest = os.path.join(a.archive, f"{stamp}_{a.label}")
    if os.path.exists(dest):
        sys.exit(f"{dest} already exists — pick another --label or remove it first")
    os.makedirs(dest)

    copied = []
    for name in COPY:
        src = os.path.join(a.rundir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
            copied.append(dict(file=name, bytes=os.path.getsize(src), sha256=sha256(src)))
        else:
            print(f"  ! missing from the run: {name}")
    shutil.copy2(a.script, os.path.join(dest, os.path.basename(a.script)))

    cache = os.path.join(a.rundir, "cache")
    cache_bytes, cache_files = dir_size(cache) if os.path.isdir(cache) else (0, 0)

    # prefer the configuration the run itself recorded; fall back to --script
    cfg_path = os.path.join(a.rundir, "run_config.json")
    run_config = None
    if os.path.exists(cfg_path):
        run_config = json.load(open(cfg_path, encoding="utf-8"))
        if run_config.get("script_sha256") != sha256(a.script):
            print("  ! the script you passed is NOT the one that produced this run; "
                  "the manifest records the run's own copy (script_used.py)")
        used = os.path.join(a.rundir, "script_used.py")
        if os.path.exists(used):
            shutil.copy2(used, os.path.join(dest, "script_used.py"))

    manifest = dict(
        label=a.label,
        run_config=run_config,
        archived=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        note=a.note,
        source_run_directory=os.path.abspath(a.rundir),
        pages_requested=a.pages,
        script=dict(name=os.path.basename(a.script), sha256=sha256(a.script),
                    search_query=(run_config or {}).get("search_query")
                    or extract_query(a.script),
                    query_source="run_config.json" if run_config else "parsed from --script"),
        results=summarise(a.rundir),
        files=copied,
        cache=dict(path=os.path.abspath(cache) if cache_files else None,
                   files=cache_files, bytes=cache_bytes,
                   copied=False,
                   note="regenerable; kept in place so re-runs reuse it"),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         pandas=pd.__version__),
    )
    with open(os.path.join(dest, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"archived to {dest}")
    for k in ("n_papers_opened", "n_candidate_rows", "n_distinct_sequences",
              "n_not_already_in_dataset", "n_rejected_rows"):
        if k in manifest["results"]:
            print(f"  {k:28s} {manifest['results'][k]}")
    if "by_tier" in manifest["results"]:
        print("  by tier:")
        for k, v in manifest["results"]["by_tier"].items():
            print(f"    {k:55s} {v}")
    if cache_files:
        print(f"  cache left in place: {cache_files} files, {cache_bytes/1e6:.0f} MB")


if __name__ == "__main__":
    main()
