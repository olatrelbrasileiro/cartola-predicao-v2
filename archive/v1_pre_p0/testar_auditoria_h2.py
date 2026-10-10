#!/usr/bin/env python3
"""Testes H2A e H2B."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
REP = ROOT/"reports"
H2A_FOLDS = REP/"auditoria_h2a_folds.csv"
H2A_ALPHAS = REP/"auditoria_h2a_alphas.csv"
H2A_SUM = REP/"auditoria_h2a_summary.md"
H2B_FOLDS = REP/"auditoria_h2b_folds.csv"
H2B_BUCKETS = REP/"auditoria_h2b_buckets.csv"
H2B_SUM = REP/"auditoria_h2b_summary.md"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
B2 = REP/"baseline_b2_folds.csv"; B3 = REP/"baseline_b3_folds.csv"
H_PQ = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []
    for p in [H2A_FOLDS, H2A_ALPHAS, H2A_SUM, H2B_FOLDS, H2B_BUCKETS, H2B_SUM]:
        if not p.exists():
            falhas.append(f"falta {p.name}")
    if falhas:
        print("=== H2 TESTS ==="); print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)

    dfr = pd.read_csv(H2A_FOLDS)
    dfa = pd.read_csv(H2A_ALPHAS)
    dfb = pd.read_csv(H2B_FOLDS)
    bk = pd.read_csv(H2B_BUCKETS)

    if len(dfr) != 76: falhas.append(f"H2A folds: {len(dfr)}")
    if (dfr["alpha_star"] < 0).any() or (dfr["alpha_star"] > 1).any():
        falhas.append("H2A alpha_star fora de [0,1]")
    if (dfr["mae_b2"] < 0).any(): falhas.append("H2A MAE B2 negativo")
    if (dfr["mae_alpha_star"] < 0).any(): falhas.append("H2A MAE alpha* negativo")
    if (dfr["mae_alpha_04_fixo"] < 0).any(): falhas.append("H2A MAE fixo negativo")

    # 21 alphas x 76 = 1596
    if len(dfa) != 76 * 21:
        falhas.append(f"H2A alphas linhas: {len(dfa)} (esperado {76*21})")

    # H2B
    if len(bk) != 5: falhas.append(f"H2B buckets: {len(bk)} (esperado 5)")

    # Parquet e baselines
    if sha256(PARQUET) != H_PQ: falhas.append("Parquet SHA mudou")
    for name, p in [("B2",B2),("B3",B3)]:
        if len(pd.read_csv(p)) != 76: falhas.append(f"{name} alterado")

    print("=== H2 TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" H2A folds: {len(dfr)}  alphas: {dfa['alpha'].nunique()}")
    print(f" H2B folds: {len(dfb)}  buckets: {len(bk)}")


if __name__ == "__main__":
    main()
