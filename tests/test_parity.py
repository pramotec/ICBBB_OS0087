"""
test_parity.py -- assert that the Python descriptors reproduce the original R code exactly.

The reference values in `tests/r_reference_features.csv` were produced by sourcing the
descriptor block of `app.R` (extracted verbatim as `r_descriptors.R`) in R 4.5.3:

    source("r_descriptors.R")
    seqs <- readLines("tests/r_test_seqs.txt")
    write.csv(cbind(calc_AAC(seqs), calc_DPC(seqs), calc_PCP(seqs)),
              "tests/r_reference_features.csv", row.names = FALSE)

Run with `python tests/test_parity.py` or `pytest tests/`.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import acp_features as af

TOL = 1e-9


def test_descriptor_parity():
    seqs = [s.strip() for s in open(os.path.join(HERE, "r_test_seqs.txt")) if s.strip()]
    ref = pd.read_csv(os.path.join(HERE, "r_reference_features.csv"))
    X, cols = af.build_features(seqs, "ALL")
    assert X.shape == ref.shape, f"shape mismatch: {X.shape} vs {ref.shape}"
    assert list(cols) == list(ref.columns), "descriptor name or order mismatch"
    d = np.abs(ref.values - X)
    worst = np.unravel_index(d.argmax(), d.shape)
    assert d.max() < TOL, (
        f"max |difference| = {d.max():.3e} at descriptor {cols[worst[1]]} "
        f"(R {ref.values[worst]} vs Python {X[worst]})")
    return d.max()


def test_dipeptide_order():
    """app.R builds the index with the FIRST residue varying most slowly."""
    assert af.DIPEP[:3] == ["AA", "AC", "AD"]
    assert af.DIPEP[20:23] == ["CA", "CC", "CD"]
    assert len(af.DIPEP) == 400 and len(set(af.DIPEP)) == 400


if __name__ == "__main__":
    test_dipeptide_order()
    m = test_descriptor_parity()
    print(f"descriptor parity OK: max |R - Python| = {m:.3e} over 438 descriptors")
