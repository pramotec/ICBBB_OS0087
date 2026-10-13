#!/usr/bin/env python3
"""
mine_death_mode.py — mine Europe PMC open-access full text for peptide sequences
reported to kill cancer cells by a named mechanism.

Generalises mine_pyroptosis.py v3 to apoptosis and necrosis. Every filter from v3 is
carried over unchanged, because each was validated against real data:

  * JATS XML is parsed and <ref-list>, <back>, <ack>, <fn-group>, <table-wrap-foot>,
    <notes>, <author-notes>, <funding-group>, <contrib-group> and <journal-meta> are
    removed before any text is read.
  * Tables are mined row by row, so "sequence and assay term in the same row" is real
    evidence rather than an artefact of flattening.
  * Runs of five or more three-letter codes (Gly-Ile-Gly-Ala-Val-Leu-Lys-Val ...) are
    detected in hyphen-, space- and non-separated form, any case, and converted to
    one-letter form.
  * Nucleotide strings and primer/PCR context are rejected. On the pyroptosis run this
    removed 4,185 strings, all verified to be genuine DNA.
  * English words are rejected by a system dictionary and by an amino-acid composition
    log-odds score. The composition test is soft: a candidate that fails it is rescued
    when the surrounding text is clearly discussing peptides, so that designed
    low-complexity sequences (KLAKLAKKLAKLAK, RLKRLLRRLLRR) survive.
  * Homopolymers are rejected.

Two things are NEW, and both come from what the pyroptosis run taught us.

1. SPECIFIC READOUTS ARE DISTINGUISHED FROM THE BARE MECHANISM WORD.
   "Apoptosis" appears in roughly a fifth of the cancer literature, so the presence of
   the word next to a sequence is close to worthless as evidence. Each assay term is
   therefore tagged as specific (annexin V, caspase-3/7 activation, PARP cleavage,
   TUNEL, sub-G1, cytochrome c release, JC-1; LDH release, MLKL phosphorylation,
   necrostatin-1 rescue, HMGB1 release) or generic (the word itself). The top two
   evidence tiers require a SPECIFIC readout.

2. THE PATHWAY-PROTEIN SCREEN IS MODE-DEPENDENT, AND FOR GOOD REASON.
   In the pyroptosis run, 9 of the 11 best-ranked candidates were fragments of NLRP3,
   GSDMD, GSDME or IL-1beta — tryptic peptides, antibody antigens and predicted
   epitopes of the machinery being studied, never the agent that triggered it. Those
   are rejected outright.

   For apoptosis and necrosis the same rule would be WRONG, because several real
   anticancer peptides ARE derived from the machinery: Smac/DIABLO N-terminal mimetics
   (AVPI...), BH3-domain peptides from BIM/BAD/BID, and p53-derived peptides such as
   PNC-27. Rejecting machinery fragments would discard exactly these true positives.
   For these modes a machinery match is therefore recorded as a FLAG
   (pathway_fragment_of) and the candidate is demoted rather than dropped: it can still
   reach the top tiers if the text shows the peptide was applied to cancer cells and a
   specific readout was measured — which is what distinguishes a Smac mimetic from a
   tryptic peptide of caspase-3 in a proteomics table.

Nothing here assigns a final label. Every row is a proposal to be confirmed against the
cited paper by a curator who reads the assay.

Quick start
-----------
    pip install requests pandas openpyxl
    python3 mine_death_mode.py --mode apoptosis --indir . --outdir mine_apo --pages 40
    python3 mine_death_mode.py --mode necrosis  --indir . --outdir mine_nec --pages 40

Outputs (in --outdir)
---------------------
    <mode>_candidates.csv   ranked proposals, one row per distinct sequence
    <mode>_rejected.csv     everything filtered out, with the reason — always check this
    papers_scanned.csv      every paper opened, so the search is reproducible
    run_config.json         query, script SHA-256, parameters, start time
    script_used.py          verbatim copy of this file
    cache/                  raw API responses; delete to force a refetch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SLEEP = 0.3
MIN_LEN, MAX_LEN = 10, 60
MIN_FRAGMENT = 8          # shorter matches to a 1000-residue protein happen by chance

# =============================================================== mode definitions
# term regex -> (readout recorded in the curation table, is it a SPECIFIC assay?)
# Recording the readout matters: a reviewer asks which assay supported the label, not
# whether the mechanism word appeared.

APOPTOSIS_TERMS = {
    r"annexin[-\s]?v": ("annexin V / PI staining", True),
    r"caspase[-\s]?3\b|caspase[-\s]?7\b|caspase[-\s]?3/7": ("caspase-3/7 activation", True),
    r"caspase[-\s]?8\b": ("caspase-8 (extrinsic)", True),
    r"caspase[-\s]?9\b": ("caspase-9 (intrinsic)", True),
    r"\bparp\b.{0,30}cleav|cleav.{0,30}\bparp\b": ("PARP cleavage", True),
    r"\btunel\b": ("TUNEL", True),
    r"sub[-\s]?g1|sub[-\s]?g0": ("sub-G1 DNA content", True),
    r"dna ladder|dna fragmentation": ("DNA laddering", True),
    r"cytochrome c relea|release of cytochrome c": ("cytochrome c release", True),
    r"\bjc-?1\b|mitochondrial membrane potential|Δψm|delta\s?psi": ("mitochondrial membrane potential", True),
    r"\bbax\b|\bbcl-?2\b|bax/bcl|\bbak\b|\bbid\b|\bbim\b|\bbh3\b": ("Bcl-2 family / BH3", True),
    r"nuclear condensation|chromatin condensation|apoptotic bod|membrane blebbing": ("apoptotic morphology", True),
    r"phosphatidylserine externalis|phosphatidylserine externaliz": ("PS externalisation", True),
    r"apoptos|apoptotic": ("apoptosis (named)", False),
}

NECROSIS_TERMS = {
    r"\bldh\b.{0,30}relea|lactate dehydrogenase.{0,30}relea": ("LDH release", True),
    r"propidium iodide.{0,40}(uptake|positive|influx)|\bpi\b[-\s]?positive": ("PI uptake", True),
    r"\bmlkl\b": ("MLKL (necroptosis)", True),
    r"\bripk?[-\s]?1\b|\brip1\b": ("RIPK1", True),
    r"\bripk?[-\s]?3\b|\brip3\b": ("RIPK3", True),
    r"necrostatin|\bnec-?1\b": ("necrostatin-1 rescue", True),
    r"\bhmgb1\b": ("HMGB1 release", True),
    r"membrane rupture|membrane lysis|plasma membrane permeabili|cell lysis": ("membrane rupture", True),
    # Trypan blue exclusion is a VIABILITY assay, not a necrosis readout: membrane
    # integrity is also lost in late apoptosis. Treating it as specific put six
    # wheat-germ peptides from a single review table into the top tier on the first
    # necrosis run. Kept as a recorded term, demoted to generic.
    r"trypan blue": ("trypan blue exclusion", False),
    r"atp depletion|cellular atp.{0,20}deplet": ("ATP depletion", True),
    r"\bnecroptos|necroptotic": ("necroptosis (named)", False),
    r"\bnecrosis\b|\bnecrotic\b": ("necrosis (named)", False),
}

PYROPTOSIS_TERMS = {
    r"gasdermin[-\s]?d\b|\bgsdmd\b": ("GSDMD", True),
    r"gasdermin[-\s]?e\b|\bgsdme\b|\bdfna5\b": ("GSDME", True),
    r"\bgasdermin\b|\bgsdm[ab]\b": ("gasdermin (unspecified)", True),
    r"caspase[-\s]?1\b": ("caspase-1", True),
    r"caspase[-\s]?(4|5|11)\b": ("caspase-4/5/11", True),
    r"inflammasome|\bnlrp3\b|\basc speck": ("inflammasome / ASC speck", True),
    r"il[-\s]?1\s?(beta|β)\b|\bil1b\b": ("IL-1beta release", True),
    r"il[-\s]?18\b": ("IL-18 release", True),
    r"membrane ballooning|cell swelling and rupture": ("pyroptotic morphology", True),
    r"pyroptos|pyroptotic": ("pyroptosis (named)", False),
}

CANCER_CLAUSE = ('(cancer OR tumour OR tumor OR carcinoma OR leukemia OR leukaemia '
                 'OR melanoma OR glioma)')
PEPTIDE_CLAUSE = '(peptide OR polypeptide OR "amino acid sequence")'
# The decisive clause. Requiring the paper to name a peptide CLASS, rather than merely
# using the word "peptide", is what separates papers that print sequences from papers
# that mention peptides in passing. It raised sequence yield 17-fold in the pilot.
PEPTIDE_CLASS_CLAUSE = ('("anticancer peptide" OR "anticancer peptides" OR '
                        '"antitumor peptide" OR "antitumour peptide" OR '
                        '"cytotoxic peptide" OR "antimicrobial peptide" OR '
                        '"peptide sequence" OR "amino acid sequence")')
ASSAY_CLAUSE = '(IC50 OR MTT OR "cell viability" OR cytotoxicity OR "CCK-8")'
SPECIFIC_APOPTOSIS_READOUTS = ('("annexin V" OR "caspase-3" OR "caspase-9" OR TUNEL '
                               'OR "PARP cleavage" OR "sub-G1" OR "cytochrome c release")')
OA = 'OPEN_ACCESS:Y AND HAS_FT:Y'

MODES = {
    "apoptosis": dict(
        terms=APOPTOSIS_TERMS,
        pathway_policy="flag",
        # Apoptosis machinery. Matches are FLAGGED, not rejected — Smac/DIABLO, BH3 and
        # p53-derived peptides are genuine anticancer agents drawn from these proteins.
        accessions={
            "P42574": "CASP3 human", "P55210": "CASP7 human", "Q14790": "CASP8 human",
            "P55211": "CASP9 human", "P55212": "CASP6 human", "P70677": "CASP3 mouse",
            "Q07812": "BAX human", "Q16611": "BAK1 human", "P10415": "BCL2 human",
            "Q07817": "BCL2L1 human", "P55957": "BID human", "O43521": "BCL2L11/BIM human",
            "O14727": "APAF1 human", "P99999": "CYCS human", "P09874": "PARP1 human",
            "P04637": "TP53 human", "P02340": "TP53 mouse", "Q9NR28": "DIABLO/Smac human",
            "O95831": "AIFM1 human", "P25445": "FAS human", "P48023": "FASLG human",
        },
        # Chosen by measurement, not by taste — see query_pilot.csv. Against a broad
        # "apoptosis AND peptide AND cancer" query this returns 17x more peptide
        # sequences per paper opened (172.5 vs 10.1 per 100 papers) and matches 6,325
        # open-access papers rather than 131,261, so the whole matching corpus can be
        # scanned exhaustively instead of sampled.
        query=(f'{SPECIFIC_APOPTOSIS_READOUTS} AND {PEPTIDE_CLASS_CLAUSE} '
               f'AND {CANCER_CLAUSE} AND {ASSAY_CLAUSE} AND {OA}'),
    ),
    "necrosis": dict(
        terms=NECROSIS_TERMS,
        pathway_policy="flag",
        accessions={
            "Q13546": "RIPK1 human", "Q9Y572": "RIPK3 human", "Q9QZL0": "RIPK3 mouse",
            "Q8NB16": "MLKL human", "Q9D2Y4": "MLKL mouse", "P09429": "HMGB1 human",
            "P63158": "HMGB1 mouse", "Q9NQC7": "CYLD human", "P01375": "TNF human",
            "P19438": "TNFRSF1A human", "Q13158": "FADD human", "Q96HS1": "PGAM5 human",
        },
        # Also chosen by measurement (query_pilot.csv): 204 sequences per 100 papers
        # against 30.5 for the broad form, over 10,638 papers rather than 79,749.
        # Unlike apoptosis, adding the specific readouts to the QUERY lowered yield
        # here, so the readouts are used only for tiering.
        query=('(necrosis OR necrotic OR necroptosis) '
               f'AND {PEPTIDE_CLASS_CLAUSE} AND {CANCER_CLAUSE} AND {ASSAY_CLAUSE} '
               f'AND {OA}'),
    ),
    "pyroptosis": dict(
        terms=PYROPTOSIS_TERMS,
        pathway_policy="reject",
        accessions={
            "P57764": "GSDMD human", "Q9D8T2": "GSDMD mouse", "O60443": "GSDME human",
            "Q9Z2D3": "GSDME mouse", "Q96P20": "NLRP3 human", "Q8R4B8": "NLRP3 mouse",
            "P29466": "CASP1 human", "P29452": "CASP1 mouse", "Q9UHD4": "CASP5 human",
            "Q9NPP4": "NLRC4 human", "Q9ULZ3": "PYCARD/ASC human", "P01584": "IL1B human",
            "P10749": "IL1B mouse", "Q14116": "IL18 human", "Q9HC29": "NOD2 human",
            "P57730": "GSDMA human", "Q8TAB3": "GSDMB human",
        },
        query=('(pyroptosis OR pyroptotic OR gasdermin OR GSDMD OR GSDME OR inflammasome '
               'OR "caspase-1" OR "caspase-4" OR "caspase-11") '
               f'AND {PEPTIDE_CLAUSE} AND {CANCER_CLAUSE} AND {OA}'),
    ),
}

# ----------------------------------------------------------------- sequence patterns
AA1 = "ACDEFGHIKLMNPQRSTVWY"
ONE_RX = re.compile(rf"\b[{AA1}]{{{MIN_LEN},{MAX_LEN}}}\b")

AA3_TO_1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
            "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
            "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
            "TYR": "Y", "VAL": "V"}
_ALT = "|".join(AA3_TO_1)
TRIPLET_RX = re.compile(rf"\b(?:(?:{_ALT})[-–—·\s]?){{5,}}(?![A-Za-z])", re.I)
CODE_RX = re.compile(rf"(?:{_ALT})", re.I)

NUCLEIC_CONTEXT = re.compile(
    r"\bprimer|\bpcr\b|\bqpcr\b|\brt-pcr\b|oligonucleotide|\bsirna\b|\bshrna\b|\bsgrna\b|"
    r"\bcdna\b|\bmrna\b|5['\u2032]|3['\u2032]|\bbp\b|base pair|forward and reverse|"
    r"\bgapdh\b.{0,40}\bprimer|nucleotide sequence", re.I)

TREATMENT_CONTEXT = re.compile(
    r"treat(ed|ment)\s+with|incubat|expos(ed|ure)\s+to|administ|\bin vitro\b|"
    r"\bic\s?50|\bec\s?50|\blc\s?50|\bgi\s?50|cell viability|\bmtt\b|\bcck-?8\b|"
    r"induced? (?:apoptos|necros|necroptos|pyroptos|cell death|gasdermin)|"
    r"dose[-\s]dependent", re.I)
CANCER_CONTEXT = re.compile(
    r"\bcancer\b|\btumou?r|carcinoma|leukaemi|leukemi|melanoma|glioma|sarcoma|"
    r"\bhela\b|\bmcf-?7\b|\ba549\b|\bhct-?116\b|\bsw480\b|\bhepg2\b|\bpc-?3\b|"
    r"\bb16\b|\bmda-?mb|\bu87\b|\bsk-?ov", re.I)
PEPTIDE_CONTEXT = re.compile(
    r"\bpeptide|polypeptide|amino acid sequence|\bresidues?\b|analogue|analog\b|"
    r"\bnh2\b|-nh\s?2|amidat|\bac-\b|antimicrobial|\bamp\b|\bacp\b|synthesi[sz]ed|"
    r"\bic\s?50|\blc\s?50|\bmic\b|\bec\s?50|\bhelix\b|amphipathic|cationic", re.I)

ENG_FREQ = dict(A=8.2, C=2.8, D=4.3, E=12.7, F=2.2, G=2.0, H=6.1, I=7.0, K=0.8, L=4.0,
                M=2.4, N=6.7, P=1.9, Q=0.1, R=6.0, S=6.3, T=9.1, V=1.0, W=2.4, Y=2.0)

DATASETS = ("Apoptosis", "Necrosis", "Pyroptosis")


# ===================================================================== helpers
def http_get(url, params, cachedir, tag, raw=False, email=None):
    if email:
        params = dict(params, email=email)
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]
    path = os.path.join(cachedir, f"{tag}_{key}." + ("txt" if raw else "json"))
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read() if raw else json.load(fh)
    ua = "acp-mechanism-curation/4.0" + (f" (mailto:{email})" if email else "")
    out = "" if raw else {}
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=60, headers={"User-Agent": ua})
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                break
            out = r.text if raw else r.json()
            break
        except Exception as exc:
            if attempt == 4:
                print(f"  ! {exc}", file=sys.stderr)
            else:
                time.sleep(3 * (attempt + 1))
    tmp = path + f".tmp{os.getpid()}_{id(out)}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(out if raw else json.dumps(out))
    os.replace(tmp, path)
    time.sleep(SLEEP)
    return out


def _read_datasets(indir):
    frames = []
    for name in DATASETS:
        path = os.path.join(indir, f"{name}_Data_Set.csv")
        if os.path.exists(path):
            d = pd.read_csv(path, encoding="utf-8-sig")
            d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]
            frames.append(d)
    return frames


def build_composition_model(indir):
    """Mean per-residue log-odds: curated peptide composition vs English letters."""
    counts = {a: 1.0 for a in AA1}
    n_seqs = 0
    seqs_all = []
    for d in _read_datasets(indir):
        for s in d.Sequence.astype(str).str.strip().str.upper():
            n_seqs += 1
            seqs_all.append(s)
            for c in s:
                if c in counts:
                    counts[c] += 1
    total = sum(counts.values())
    p_pep = {a: counts[a] / total for a in AA1}
    et = sum(ENG_FREQ.values())
    p_eng = {a: ENG_FREQ[a] / et for a in AA1}
    lo = {a: math.log(p_pep[a] / p_eng[a]) for a in AA1}

    def score(seq):
        chars = [c for c in seq if c in lo]
        return sum(lo[c] for c in chars) / len(chars) if chars else -9.0

    cutoff = 0.09
    vals = sorted(score(s) for s in seqs_all if len(s) >= MIN_LEN)
    if vals:
        cutoff = vals[int(0.05 * len(vals))]
    print(f"composition model built from {n_seqs} curated peptides; cutoff = {cutoff:+.3f}")
    return score, cutoff


def load_dictionary(path):
    if not path:
        return set()
    if not os.path.exists(path):
        print(f"  ! dictionary not found at {path} — skipping the dictionary check")
        return set()
    with open(path, encoding="utf-8", errors="ignore") as fh:
        words = {w.strip().upper() for w in fh if w.strip()}
    print(f"dictionary loaded: {len(words)} words")
    return words


def known_sequences(indir):
    known = set()
    for d in _read_datasets(indir):
        known |= set(d.Sequence.astype(str).str.strip().str.upper())
    print(f"known sequences loaded: {len(known)}")
    return known


def load_acp_reference(indir):
    """UniDL4BioPep ACP positives, if the spreadsheets are present.

    A mined sequence that already appears in an independent ACP benchmark is
    corroborated as an anticancer peptide by a second source — which does not give it a
    death mode, but does separate it from an arbitrary string that passed the filters.
    """
    acp = set()
    for stem in ("main_ACP_train", "main_ACP_test",
                 "alternative_ACP_train", "alternative_ACP_test"):
        for ext in (".xlsx", ".csv"):
            path = os.path.join(indir, stem + ext)
            if not os.path.exists(path):
                continue
            try:
                d = pd.read_excel(path) if ext == ".xlsx" else pd.read_csv(path)
            except Exception as exc:
                print(f"  ! could not read {path}: {exc}")
                continue
            cols = {c.lower(): c for c in d.columns}
            scol = next((cols[c] for c in cols if "seq" in c), d.columns[0])
            lcol = next((cols[c] for c in cols if "label" in c or "class" in c), None)
            s = d[d[lcol] == 1][scol] if lcol is not None else d[scol]
            acp |= set(s.astype(str).str.strip().str.upper())
    if acp:
        print(f"ACP reference loaded: {len(acp)} UniDL4BioPep positives")
    else:
        print("  ! no UniDL4BioPep spreadsheets found — acp_corroborated will be blank")
    return acp


def load_pathway_proteins(cachedir, accessions, mode):
    path = os.path.join(cachedir, f"pathway_proteins_{mode}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            seqs = json.load(fh)
        print(f"pathway-protein screen ({mode}): {len(seqs)} sequences (cached)")
        return seqs
    seqs = {}
    for acc in accessions:
        try:
            r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30)
            if r.status_code == 200:
                seqs[acc] = "".join(r.text.split("\n")[1:]).strip().upper()
        except Exception:
            pass
        time.sleep(0.2)
    if seqs:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(seqs, fh)
        print(f"pathway-protein screen ({mode}): {len(seqs)} sequences fetched from UniProt")
    else:
        print(f"  ! could not reach UniProt — the pathway screen is DISABLED for this run")
    return seqs


def is_pathway_fragment(seq, pathway):
    probe = seq.lstrip("H") if seq.startswith("H" * 6) else seq     # drop a His-tag
    if len(probe) < MIN_FRAGMENT:
        return None
    for acc, prot in pathway.items():
        if probe in prot:
            return acc
    return None


# ===================================================================== XML handling
DROP_TAGS = {"ref-list", "back", "ack", "fn-group", "table-wrap-foot", "notes",
             "author-notes", "funding-group", "contrib-group", "journal-meta"}


def _strip(elem):
    for parent in elem.iter():
        for child in list(parent):
            if child.tag in DROP_TAGS:
                parent.remove(child)
    return elem


def parse_fulltext(xml):
    """Return (body_sentences, table_rows). Tables are kept separate on purpose."""
    if not xml.strip():
        return [], []
    try:
        root = ET.fromstring(xml.encode("utf-8"))
    except ET.ParseError:
        return [], []
    body = root.find(".//body")
    if body is None:
        return [], []
    _strip(body)
    rows = []
    for tw in body.iter("table-wrap"):
        for tr in tw.iter("tr"):
            cells = [" ".join(c.itertext()).strip() for c in list(tr)]
            text = " | ".join(x for x in cells if x)
            if text:
                rows.append(re.sub(r"\s+", " ", text))
    for parent in body.iter():
        for child in list(parent):
            if child.tag == "table-wrap":
                parent.remove(child)
    text = re.sub(r"\s+", " ", " ".join(body.itertext()))
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sents if s.strip()], rows


# ===================================================================== extraction
def extract_sequences(text):
    """Return [(sequence, form, raw_match)] from one sentence or table row."""
    found = []
    for m in TRIPLET_RX.finditer(text):
        raw = m.group(0)
        seq = "".join(AA3_TO_1[c.upper()] for c in CODE_RX.findall(raw))
        if MIN_LEN <= len(seq) <= MAX_LEN:
            sep = "hyphen" if "-" in raw else ("space" if " " in raw.strip() else "none")
            found.append((seq, f"triplet:{sep}", raw[:120]))
    masked = TRIPLET_RX.sub(" ", text)                 # do not re-read triplet regions
    for m in ONE_RX.finditer(masked):
        found.append((m.group(0), "single", m.group(0)))
    return found


def classify(seq, context, score, cutoff, dictionary, pathway, policy):
    """None if acceptable, otherwise the rejection reason."""
    if len(seq) < MIN_LEN or len(seq) > MAX_LEN:
        return "length_out_of_range"
    if sum(c in "ACGT" for c in seq) / len(seq) > 0.90:
        return "looks_like_nucleotide"
    if NUCLEIC_CONTEXT.search(context):
        return "primer_or_nucleic_acid_context"
    if dictionary and seq in dictionary:
        return "english_word_in_dictionary"
    if pathway and policy == "reject":
        acc = is_pathway_fragment(seq, pathway)
        if acc:
            return f"pathway_protein_fragment:{acc}"
    if len(set(seq)) < 2:
        return "low_complexity"
    if score(seq) < cutoff and not PEPTIDE_CONTEXT.search(context):
        return "composition_not_peptide_like_and_no_peptide_context"
    return None


# ===================================================================== main mining
def mine(args, cfg, score, cutoff, dictionary, known, acp, pathway):
    term_rx = [(re.compile(p, re.I), name, spec) for p, (name, spec) in cfg["terms"].items()]
    policy = cfg["pathway_policy"]
    cachedir = os.path.join(args.outdir, "cache")
    kept, rejected, papers = [], [], []
    cursor, seen = "*", set()

    for page in range(args.pages):
        d = http_get(f"{EPMC}/search",
                     dict(query=cfg["query"], format="json", pageSize=100,
                          resultType="core", cursorMark=cursor),
                     cachedir, "srch", email=args.email)
        res = (d.get("resultList") or {}).get("result") or []
        if not res:
            break
        batch = []
        for r in res:
            pmc = r.get("pmcid")
            if not pmc or pmc in seen:
                continue
            seen.add(pmc)
            batch.append((pmc, dict(
                pmcid=pmc, pmid=r.get("pmid"), doi=r.get("doi"), year=r.get("pubYear"),
                title=r.get("title"),
                journal=(r.get("journalInfo") or {}).get("journal", {}).get("title"))))
        print(f"page {page + 1}/{args.pages}: {len(batch)} new records "
              f"(hitCount={d.get('hitCount')})", flush=True)

        def fetch(item):
            pmc, meta = item
            return meta, http_get(f"{EPMC}/{pmc}/fullTextXML", {}, cachedir, "ft",
                                  raw=True, email=args.email)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for meta, xml in ex.map(fetch, batch):
                papers.append(meta)
                sents, rows = parse_fulltext(xml)
                units = [("body", s) for s in sents] + [("table_row", t) for t in rows]
                for kind, text in units:
                    seqs = extract_sequences(text)
                    if not seqs:
                        continue
                    hits = [(n, sp) for rx, n, sp in term_rx if rx.search(text)]
                    mechs = sorted({n for n, _ in hits})
                    specific = sorted({n for n, sp in hits if sp})
                    n_in_unit = len({s for s, _, _ in seqs})
                    treat = bool(TREATMENT_CONTEXT.search(text)
                                 and CANCER_CONTEXT.search(text))
                    for seq, form, raw in seqs:
                        reason = classify(seq, text, score, cutoff, dictionary,
                                          pathway, policy)
                        frag = is_pathway_fragment(seq, pathway) if pathway else None
                        row = dict(sequence=seq, length=len(seq), form=form, raw_match=raw,
                                   unit=kind, n_sequences_in_unit=n_in_unit,
                                   assay_terms=";".join(mechs),
                                   specific_assay_terms=";".join(specific),
                                   mechanism_in_same_unit=bool(mechs),
                                   specific_in_same_unit=bool(specific),
                                   treatment_context=treat,
                                   pathway_fragment_of=frag or "",
                                   evidence_text=text[:500],
                                   peptide_likeness=round(score(seq), 3),
                                   already_in_dataset=seq in known,
                                   acp_corroborated=seq in acp, **meta)
                        (rejected if reason else kept).append(
                            dict(row, reason=reason) if reason else row)
        cursor = d.get("nextCursorMark")
        if not cursor:
            break
    return kept, rejected, papers


def rank(df):
    """1 is best.

    A mechanism word alone is not evidence. The pyroptosis run showed that a mechanism
    term beside a sequence usually means the sequence belongs to the pathway machinery
    being studied, not to a compound that was applied. The top tiers therefore require a
    SPECIFIC assay readout, evidence the peptide was administered to cancer cells, and
    an unambiguous sequence-to-evidence mapping. A machinery match demotes a candidate
    to tier 3 at best unless both of those are satisfied.
    """
    def tier(r):
        if not r.mechanism_in_same_unit:
            return 5
        if not r.treatment_context:
            return 4
        if (not r.specific_in_same_unit) or r.n_sequences_in_unit > 1:
            return 3
        if r.pathway_fragment_of:
            return 3
        return 1 if r.unit == "table_row" else 2

    df = df.copy()
    df["tier"] = df.apply(tier, axis=1)
    df["tier_meaning"] = df.tier.map({
        1: "1_table_row_single_sequence_specific_assay_and_treatment",
        2: "2_sentence_single_sequence_specific_assay_and_treatment",
        3: "3_treatment_present_but_generic_assay_or_ambiguous_or_machinery",
        4: "4_assay_term_only_no_evidence_the_peptide_was_applied",
        5: "5_no_assay_term_in_unit"})
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--indir", default=".", help="folder holding the *_Data_Set.csv files")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pages", type=int, default=30, help="100 papers per page")
    ap.add_argument("--workers", type=int, default=6, help="parallel full-text fetches")
    ap.add_argument("--email", default=os.environ.get("CONTACT_EMAIL"),
                    help="contact address sent to Europe PMC; optional but polite")
    ap.add_argument("--dict", dest="dictpath", default="/usr/share/dict/words",
                    help="word list for the dictionary check; pass '' to disable")
    a = ap.parse_args()
    a.outdir = a.outdir or f"mine_{a.mode}"
    cfg = MODES[a.mode]
    os.makedirs(a.outdir, exist_ok=True)
    cachedir = os.path.join(a.outdir, "cache")
    os.makedirs(cachedir, exist_ok=True)

    import shutil as _shutil
    _self = os.path.abspath(__file__)
    with open(_self, "rb") as _fh:
        _sha = hashlib.sha256(_fh.read()).hexdigest()
    _shutil.copy2(_self, os.path.join(a.outdir, "script_used.py"))
    with open(os.path.join(a.outdir, "run_config.json"), "w", encoding="utf-8") as _fh:
        json.dump(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), mode=a.mode,
                       argv=sys.argv, pages=a.pages, workers=a.workers,
                       indir=os.path.abspath(a.indir), script=os.path.basename(_self),
                       script_sha256=_sha, search_query=cfg["query"],
                       pathway_policy=cfg["pathway_policy"],
                       pathway_accessions=cfg["accessions"],
                       min_len=MIN_LEN, max_len=MAX_LEN,
                       dictionary=a.dictpath or None), _fh, indent=2)

    print(f"=== mode: {a.mode} | pathway policy: {cfg['pathway_policy']} ===")
    score, cutoff = build_composition_model(a.indir)
    dictionary = load_dictionary(a.dictpath)
    known = known_sequences(a.indir)
    acp = load_acp_reference(a.indir)
    pathway = load_pathway_proteins(cachedir, cfg["accessions"], a.mode)

    kept, rejected, papers = mine(a, cfg, score, cutoff, dictionary, known, acp, pathway)
    if not kept and not rejected:
        print("nothing retrieved — check the network and the Europe PMC query")
        return

    R = pd.DataFrame(rejected)
    if len(R):
        R.to_csv(os.path.join(a.outdir, f"{a.mode}_rejected.csv"), index=False)
    pd.DataFrame(papers).to_csv(os.path.join(a.outdir, "papers_scanned.csv"), index=False)
    if not kept:
        print(f"every extracted string was filtered out; see {a.mode}_rejected.csv")
        return

    D = rank(pd.DataFrame(kept))
    agg = (D.sort_values("tier")
             .groupby("sequence", as_index=False)
             .agg(length=("length", "first"), form=("form", "first"),
                  tier=("tier", "min"), tier_meaning=("tier_meaning", "first"),
                  treatment_context=("treatment_context", "max"),
                  assay_terms=("assay_terms", "first"),
                  specific_assay_terms=("specific_assay_terms", "first"),
                  pathway_fragment_of=("pathway_fragment_of", "first"),
                  peptide_likeness=("peptide_likeness", "first"),
                  already_in_dataset=("already_in_dataset", "first"),
                  acp_corroborated=("acp_corroborated", "first"),
                  n_papers=("pmcid", "nunique"),
                  doi=("doi", "first"), pmid=("pmid", "first"), pmcid=("pmcid", "first"),
                  year=("year", "first"), title=("title", "first"),
                  evidence_text=("evidence_text", "first"))
             .sort_values(["tier", "acp_corroborated", "n_papers", "peptide_likeness"],
                          ascending=[True, False, False, False]))
    agg.to_csv(os.path.join(a.outdir, f"{a.mode}_candidates.csv"), index=False)

    manifest = dict(
        mode=a.mode, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        query=cfg["query"], script_sha256=_sha, pages_requested=a.pages,
        pathway_policy=cfg["pathway_policy"],
        results=dict(
            n_papers_opened=len(papers),
            n_strings_extracted=int(len(D) + len(R)),
            n_kept=int(len(D)), n_rejected=int(len(R)),
            n_distinct_sequences=int(len(agg)),
            n_new=int((~agg.already_in_dataset).sum()),
            n_acp_corroborated=int(agg.acp_corroborated.sum()),
            n_machinery_flagged=int((agg.pathway_fragment_of != "").sum()),
            by_tier={k: int(v) for k, v in agg.tier_meaning.value_counts().items()},
            by_rejection={k: int(v) for k, v in (R.reason.value_counts().items()
                                                 if len(R) else [])}),
        packages={p: __import__(p).__version__ for p in ("pandas", "requests")})
    with open(os.path.join(a.outdir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n================ SUMMARY ================")
    print(f"papers opened                 : {len(papers)}")
    print(f"strings extracted             : {len(D) + len(R)}")
    print(f"strings rejected              : {len(R)}")
    if len(R):
        print("\nrejection reasons:")
        print(R.reason.value_counts().to_string())
    print("\ncandidates by evidence tier:")
    print(agg.tier_meaning.value_counts().sort_index().to_string())
    new = agg[~agg.already_in_dataset]
    print(f"\ndistinct sequences            : {len(agg)}")
    print(f"not already in your dataset   : {len(new)}")
    print(f"corroborated as ACP elsewhere : {int(agg.acp_corroborated.sum())}")
    print(f"flagged as machinery fragment : {int((agg.pathway_fragment_of != '').sum())}")
    for t in (1, 2, 3):
        print(f"  tier {t}                      : {int((new.tier == t).sum())}")
    print(f"\nwrote {a.outdir}/{a.mode}_candidates.csv and run_manifest.json")
    print("\nEVERY ROW IS A PROPOSAL. Open the DOI and confirm the assay before labelling.")


if __name__ == "__main__":
    main()
