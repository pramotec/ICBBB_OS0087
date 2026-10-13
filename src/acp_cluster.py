"""
acp_cluster.py -- sequence-identity clustering for leakage-aware data splitting.

Implements the CD-HIT greedy incremental clustering algorithm in pure Python so
that the analysis is reproducible without an external binary on any platform.

Identity definition follows the CD-HIT convention:
    identity = (# identical aligned residue pairs) / (length of the SHORTER sequence)
computed from a Needleman-Wunsch global alignment (BLOSUM62, gap open -10,
gap extend -0.5) via Bio.Align.PairwiseAligner.

Greedy incremental clustering (CD-HIT, Li & Godzik 2006):
    1. sort sequences by decreasing length
    2. the first unassigned sequence becomes a cluster representative
    3. every later sequence with identity >= threshold to that representative
       joins its cluster
This reproduces `cd-hit -c <threshold> -n 2 -g 1` behaviour for short peptides.
"""
import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices


def _make_aligner():
    al = Align.PairwiseAligner()
    al.mode = "global"
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score = -10.0
    al.extend_gap_score = -0.5
    al.target_end_gap_score = 0.0
    al.query_end_gap_score = 0.0
    return al


def pairwise_identity_matrix(seqs, aligner=None, progress=None):
    """Symmetric n x n matrix of global-alignment identity (shorter-seq normalised)."""
    al = aligner or _make_aligner()
    n = len(seqs)
    I = np.eye(n, dtype=np.float32)
    for i in range(n):
        if progress and i % 50 == 0:
            progress(i, n)
        for j in range(i + 1, n):
            a, b = seqs[i], seqs[j]
            aln = al.align(a, b)[0]
            ta, qa = aln[0], aln[1]
            ident = sum(1 for x, y in zip(ta, qa) if x == y and x != "-")
            I[i, j] = I[j, i] = ident / min(len(a), len(b))
    return I


def greedy_cluster(seqs, ident_matrix, threshold):
    """CD-HIT-style greedy incremental clustering. Returns (labels, representatives)."""
    order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]), seqs[i]))
    labels = np.full(len(seqs), -1, dtype=int)
    reps = []
    for i in order:
        if labels[i] != -1:
            continue
        cid = len(reps)
        reps.append(i)
        labels[i] = cid
        for j in order:
            if labels[j] == -1 and ident_matrix[i, j] >= threshold:
                labels[j] = cid
    return labels, reps
