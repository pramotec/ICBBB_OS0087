"""
acp_features.py -- sequence descriptors for ACP death-mode prediction.

Direct Python port of the descriptor block in app.R (PepDeathBench v2.2):
AAC (20), DPC (400) and PCP (18). Values are numerically identical to the R
implementation; see tests/test_parity.py.

Author-facing note: this module replaces the undocumented `sanitize_matrix()`
behaviour of the Shiny app. Dimensionality reduction is NOT applied here -- it
is a fold-internal step of the scikit-learn Pipeline (see acp_models.py).
"""
import numpy as np

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}
DIPEP = [a + b for a in AA for b in AA]          # matches R: as.vector(t(outer(AA,AA,paste0)))
DIPEP_IDX = {d: i for i, d in enumerate(DIPEP)}

KD = dict(A=1.8, C=2.5, D=-3.5, E=-3.5, F=2.8, G=-0.4, H=-3.2, I=4.5, K=-3.9,
          L=3.8, M=1.9, N=-3.5, P=-1.6, Q=-3.5, R=-4.5, S=-0.8, T=-0.7, V=4.2,
          W=-0.9, Y=-1.3)
EISEN = dict(A=0.62, C=0.29, D=-0.90, E=-0.74, F=1.19, G=0.48, H=-0.40, I=1.38,
             K=-1.50, L=1.06, M=0.64, N=-0.78, P=0.12, Q=-0.85, R=-2.53,
             S=-0.18, T=-0.05, V=1.08, W=0.81, Y=0.26)
MW_RES = dict(A=71.079, C=103.144, D=115.089, E=129.116, F=147.177, G=57.052,
              H=137.141, I=113.160, K=128.174, L=113.160, M=131.198, N=114.104,
              P=97.117, Q=128.131, R=156.188, S=87.078, T=101.105, V=99.133,
              W=186.213, Y=163.176)
POLARITY = dict(A=8.1, C=5.5, D=13.0, E=12.3, F=5.2, G=9.0, H=10.4, I=5.2,
                K=11.3, L=4.9, M=5.7, N=11.6, P=8.0, Q=10.5, R=10.5, S=9.2,
                T=8.6, V=5.9, W=5.4, Y=6.2)
BULK = dict(A=11.50, C=13.46, D=11.68, E=13.57, F=19.80, G=3.40, H=13.69,
            I=21.40, K=15.71, L=21.40, M=16.25, N=12.82, P=17.43, Q=14.45,
            R=14.28, S=9.47, T=15.77, V=21.57, W=21.67, Y=18.03)
FLEX = dict(A=0.357, C=0.346, D=0.511, E=0.497, F=0.314, G=0.544, H=0.323,
            I=0.462, K=0.466, L=0.365, M=0.295, N=0.463, P=0.509, Q=0.493,
            R=0.529, S=0.507, T=0.444, V=0.386, W=0.305, Y=0.420)
BOMAN = dict(A=1.81, C=1.28, D=-8.72, E=-6.81, F=2.98, G=0.94, H=-0.64, I=4.92,
             K=-5.55, L=4.92, M=2.35, N=-1.91, P=0.00, Q=-1.30, R=-14.92,
             S=-0.75, T=-0.29, V=4.04, W=2.33, Y=-0.14)
PKA = dict(Nterm=9.69, Cterm=2.34, C=8.18, D=3.65, E=4.25, H=6.00, K=10.53,
           R=12.48, Y=10.07)

PCP_NAMES = ["Length", "MW", "GRAVY", "NetCharge_pH7", "ChargeDensity", "pI",
             "Aromaticity", "AliphaticIndex", "HydrophobicMoment",
             "Frac_Hydrophobic", "Frac_Charged", "Frac_Polar", "Frac_Tiny",
             "Frac_Pro_Gly", "MeanPolarity", "MeanBulkiness",
             "MeanFlexibility", "BomanIndex"]

AAC_NAMES = [f"AAC_{a}" for a in AA]
DPC_NAMES = [f"DPC_{d}" for d in DIPEP]
PCP_COLS = [f"PCP_{n}" for n in PCP_NAMES]


def clean_seq(s):
    return "".join(ch for ch in str(s).upper() if ch in AA_IDX)


def calc_AAC(seqs):
    M = np.zeros((len(seqs), 20))
    for i, s in enumerate(seqs):
        ch = clean_seq(s)
        if not ch:
            continue
        for a in ch:
            M[i, AA_IDX[a]] += 1
        M[i] /= len(ch)
    return M


def calc_DPC(seqs):
    M = np.zeros((len(seqs), 400))
    for i, s in enumerate(seqs):
        ch = clean_seq(s)
        if len(ch) < 2:
            continue
        for j in range(len(ch) - 1):
            M[i, DIPEP_IDX[ch[j:j + 2]]] += 1
        M[i] /= (len(ch) - 1)
    return M


def _net_charge(cnt, pH):
    pos = (1 / (1 + 10 ** (pH - PKA["Nterm"]))
           + cnt["K"] / (1 + 10 ** (pH - PKA["K"]))
           + cnt["R"] / (1 + 10 ** (pH - PKA["R"]))
           + cnt["H"] / (1 + 10 ** (pH - PKA["H"])))
    neg = (1 / (1 + 10 ** (PKA["Cterm"] - pH))
           + cnt["D"] / (1 + 10 ** (PKA["D"] - pH))
           + cnt["E"] / (1 + 10 ** (PKA["E"] - pH))
           + cnt["C"] / (1 + 10 ** (PKA["C"] - pH))
           + cnt["Y"] / (1 + 10 ** (PKA["Y"] - pH)))
    return pos - neg


def _pI(cnt):
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _net_charge(cnt, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _hydrophobic_moment(ch, angle_deg=100.0):
    n = len(ch)
    if n < 2:
        return 0.0
    h = np.array([EISEN[a] for a in ch])
    ang = np.deg2rad(angle_deg) * np.arange(n)
    return float(np.sqrt(np.sum(h * np.cos(ang)) ** 2 + np.sum(h * np.sin(ang)) ** 2) / n)


def calc_PCP(seqs):
    M = np.zeros((len(seqs), len(PCP_NAMES)))
    for i, s in enumerate(seqs):
        ch = clean_seq(s)
        n = len(ch)
        if n == 0:
            continue
        cnt = {a: 0.0 for a in AA}
        for a in ch:
            cnt[a] += 1
        frac = {a: cnt[a] / n for a in AA}
        q7 = _net_charge(cnt, 7.0)
        M[i] = [
            n,
            sum(cnt[a] * MW_RES[a] for a in AA) + 18.015,
            sum(cnt[a] * KD[a] for a in AA) / n,
            q7,
            q7 / n,
            _pI(cnt),
            frac["F"] + frac["W"] + frac["Y"],
            (frac["A"] + 2.9 * frac["V"] + 3.9 * (frac["I"] + frac["L"])) * 100,
            _hydrophobic_moment(ch),
            sum(frac[a] for a in "AVLIMFWC"),
            sum(frac[a] for a in "DEKR"),
            sum(frac[a] for a in "STNQYH"),
            sum(frac[a] for a in "AGS"),
            sum(frac[a] for a in "PG"),
            sum(cnt[a] * POLARITY[a] for a in AA) / n,
            sum(cnt[a] * BULK[a] for a in AA) / n,
            sum(cnt[a] * FLEX[a] for a in AA) / n,
            sum(cnt[a] * BOMAN[a] for a in AA) / n,
        ]
    return M


def build_features(seqs, kind="ALL"):
    """Return (X, column_names). kind in {AAC, DPC, PCP, ALL}."""
    if kind == "AAC":
        return calc_AAC(seqs), list(AAC_NAMES)
    if kind == "DPC":
        return calc_DPC(seqs), list(DPC_NAMES)
    if kind == "PCP":
        return calc_PCP(seqs), list(PCP_COLS)
    if kind == "ALL":
        return (np.hstack([calc_AAC(seqs), calc_DPC(seqs), calc_PCP(seqs)]),
                AAC_NAMES + DPC_NAMES + PCP_COLS)
    raise ValueError(f"unknown feature set: {kind}")
