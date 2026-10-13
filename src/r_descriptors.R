AA    <- strsplit("ACDEFGHIKLMNPQRSTVWY", "")[[1]]
DIPEP <- as.vector(t(outer(AA, AA, paste0)))

KD <- c(A = 1.8, C = 2.5, D = -3.5, E = -3.5, F = 2.8, G = -0.4, H = -3.2,
        I = 4.5, K = -3.9, L = 3.8, M = 1.9, N = -3.5, P = -1.6, Q = -3.5,
        R = -4.5, S = -0.8, T = -0.7, V = 4.2, W = -0.9, Y = -1.3)

EISEN <- c(A = 0.62, C = 0.29, D = -0.90, E = -0.74, F = 1.19, G = 0.48,
           H = -0.40, I = 1.38, K = -1.50, L = 1.06, M = 0.64, N = -0.78,
           P = 0.12, Q = -0.85, R = -2.53, S = -0.18, T = -0.05, V = 1.08,
           W = 0.81, Y = 0.26)

MW_RES <- c(A = 71.079, C = 103.144, D = 115.089, E = 129.116, F = 147.177,
            G = 57.052, H = 137.141, I = 113.160, K = 128.174, L = 113.160,
            M = 131.198, N = 114.104, P = 97.117, Q = 128.131, R = 156.188,
            S = 87.078, T = 101.105, V = 99.133, W = 186.213, Y = 163.176)

POLARITY <- c(A = 8.1, C = 5.5, D = 13.0, E = 12.3, F = 5.2, G = 9.0, H = 10.4,
              I = 5.2, K = 11.3, L = 4.9, M = 5.7, N = 11.6, P = 8.0, Q = 10.5,
              R = 10.5, S = 9.2, T = 8.6, V = 5.9, W = 5.4, Y = 6.2)

BULK <- c(A = 11.50, C = 13.46, D = 11.68, E = 13.57, F = 19.80, G = 3.40,
          H = 13.69, I = 21.40, K = 15.71, L = 21.40, M = 16.25, N = 12.82,
          P = 17.43, Q = 14.45, R = 14.28, S = 9.47, T = 15.77, V = 21.57,
          W = 21.67, Y = 18.03)

FLEX <- c(A = 0.357, C = 0.346, D = 0.511, E = 0.497, F = 0.314, G = 0.544,
          H = 0.323, I = 0.462, K = 0.466, L = 0.365, M = 0.295, N = 0.463,
          P = 0.509, Q = 0.493, R = 0.529, S = 0.507, T = 0.444, V = 0.386,
          W = 0.305, Y = 0.420)

BOMAN <- c(A = 1.81, C = 1.28, D = -8.72, E = -6.81, F = 2.98, G = 0.94,
           H = -0.64, I = 4.92, K = -5.55, L = 4.92, M = 2.35, N = -1.91,
           P = 0.00, Q = -1.30, R = -14.92, S = -0.75, T = -0.29, V = 4.04,
           W = 2.33, Y = -0.14)

PKA <- list(Nterm = 9.69, Cterm = 2.34, C = 8.18, D = 3.65, E = 4.25,
            H = 6.00, K = 10.53, R = 12.48, Y = 10.07)

DATASET_URLS <- c(
  Apoptosis  = "https://raw.githubusercontent.com/TaoHu888/ACP_data_set/main/Apoptosis_Data_Set.csv",
  Necrosis   = "https://raw.githubusercontent.com/TaoHu888/ACP_data_set/main/Necrosis_Data_Set.csv",
  Pyroptosis = "https://raw.githubusercontent.com/TaoHu888/ACP_data_set/main/Pyroptosis_Data_Set.csv"
)

MODEL_CHOICES <- c(
  "Random Forest (RF)"           = "RF",
  "Support Vector Machine (SVM)" = "SVM",
  "XGBoost"                      = "XGB",
  "Stacked: SVM + RF"            = "STACK_SVM_RF",
  "Stacked: SVM + RF + XGBoost"  = "STACK_SVM_RF_XGB"
)

FEATURE_CHOICES <- c(
  "AAC (Amino Acid Composition, 20)" = "AAC",
  "DPC (Dipeptide Composition, 400)" = "DPC",
  "PCP (Physicochemical Properties)" = "PCP",
  "ALL (AAC + DPC + PCP)"            = "ALL"
)

MODEL_LABEL <- c(RF = "RF", SVM = "SVM", XGB = "XGBoost",
                 STACK_SVM_RF = "Stacked (SVM+RF)",
                 STACK_SVM_RF_XGB = "Stacked (SVM+RF+XGB)")

#
# 2) Feature extraction
#

clean_seq   <- function(s) toupper(gsub("[^A-Za-z]", "", as.character(s)))
split_valid <- function(s) { ch <- strsplit(s, "")[[1]]; ch[ch %in% AA] }

calc_AAC <- function(seqs) {
  m <- matrix(0, length(seqs), 20, dimnames = list(NULL, paste0("AAC_", AA)))
  for (i in seq_along(seqs)) {
    ch <- split_valid(seqs[i]); if (!length(ch)) next
    m[i, ] <- as.numeric(table(factor(ch, levels = AA))) / length(ch)
  }
  m
}

calc_DPC <- function(seqs) {
  m <- matrix(0, length(seqs), 400, dimnames = list(NULL, paste0("DPC_", DIPEP)))
  for (i in seq_along(seqs)) {
    ch <- split_valid(seqs[i]); if (length(ch) < 2) next
    dp <- paste0(ch[-length(ch)], ch[-1])
    m[i, ] <- as.numeric(table(factor(dp, levels = DIPEP))) / length(dp)
  }
  m
}

net_charge_at <- function(cnt, pH) {
  pos <- 1 / (1 + 10^(pH - PKA$Nterm)) +
    cnt[["K"]] / (1 + 10^(pH - PKA$K)) +
    cnt[["R"]] / (1 + 10^(pH - PKA$R)) +
    cnt[["H"]] / (1 + 10^(pH - PKA$H))
  neg <- 1 / (1 + 10^(PKA$Cterm - pH)) +
    cnt[["D"]] / (1 + 10^(PKA$D - pH)) +
    cnt[["E"]] / (1 + 10^(PKA$E - pH)) +
    cnt[["C"]] / (1 + 10^(PKA$C - pH)) +
    cnt[["Y"]] / (1 + 10^(PKA$Y - pH))
  as.numeric(pos - neg)
}

calc_pI <- function(cnt) {
  lo <- 0; hi <- 14
  for (k in 1:60) {
    mid <- (lo + hi) / 2
    if (net_charge_at(cnt, mid) > 0) lo <- mid else hi <- mid
  }
  (lo + hi) / 2
}

hydrophobic_moment <- function(ch, angle_deg = 100) {
  n <- length(ch); if (n < 2) return(0)
  h <- as.numeric(EISEN[ch])
  ang <- (angle_deg * pi / 180) * (seq_len(n) - 1)
  sqrt(sum(h * cos(ang))^2 + sum(h * sin(ang))^2) / n
}

PCP_NAMES <- c("Length", "MW", "GRAVY", "NetCharge_pH7", "ChargeDensity", "pI",
               "Aromaticity", "AliphaticIndex", "HydrophobicMoment",
               "Frac_Hydrophobic", "Frac_Charged", "Frac_Polar", "Frac_Tiny",
               "Frac_Pro_Gly", "MeanPolarity", "MeanBulkiness",
               "MeanFlexibility", "BomanIndex")

calc_PCP <- function(seqs) {
  m <- matrix(0, length(seqs), length(PCP_NAMES),
              dimnames = list(NULL, paste0("PCP_", PCP_NAMES)))
  for (i in seq_along(seqs)) {
    ch <- split_valid(seqs[i]); n <- length(ch)
    if (n == 0) next
    cnt  <- setNames(as.numeric(table(factor(ch, levels = AA))), AA)
    frac <- cnt / n
    q7   <- net_charge_at(cnt, 7.0)
    vals <- c(
      n,
      sum(cnt * MW_RES[AA]) + 18.015,
      sum(cnt * KD[AA]) / n,
      q7,
      q7 / n,
      calc_pI(cnt),
      sum(frac[c("F", "W", "Y")]),
      (frac[["A"]] + 2.9 * frac[["V"]] + 3.9 * (frac[["I"]] + frac[["L"]])) * 100,
      hydrophobic_moment(ch),
      sum(frac[c("A", "V", "L", "I", "M", "F", "W", "C")]),
      sum(frac[c("D", "E", "K", "R")]),
      sum(frac[c("S", "T", "N", "Q", "Y", "H")]),
      sum(frac[c("A", "G", "S")]),
      sum(frac[c("P", "G")]),
      sum(cnt * POLARITY[AA]) / n,
      sum(cnt * BULK[AA]) / n,
      sum(cnt * FLEX[AA]) / n,
      sum(cnt * BOMAN[AA]) / n
    )
    m[i, ] <- as.numeric(vals)
  }
  m
}

build_features <- function(seqs, type) {
  switch(type,
         AAC = calc_AAC(seqs),
         DPC = calc_DPC(seqs),
         PCP = calc_PCP(seqs),
         ALL = cbind(calc_AAC(seqs), calc_DPC(seqs), calc_PCP(seqs)),
         stop("\u0e44\u0e21\u0e48\u0e23\u0e39\u0e49\u0e08\u0e31\u0e01\u0e0a\u0e19\u0e34\u0e14\u0e04\u0e38\u0e13\u0e25\u0e31\u0e01\u0e29\u0e13\u0e30: ", type))
}