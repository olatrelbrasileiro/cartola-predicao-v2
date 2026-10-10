#!/usr/bin/env python3
"""Testes do B2S."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
REP = ROOT/"reports"
FOLDS = REP/"baseline_b2s_folds.csv"
ALPHAS = REP/"baseline_b2s_alphas.csv"
SUM = REP/"baseline_b2s_summary.md"
SPEC_MD = REP/"baseline_b2s_spec.md"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
B2 = REP/"baseline_b2_folds.csv"
H_PQ = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []
    for p in [FOLDS, ALPHAS, SUM, SPEC_MD]:
        if not p.exists(): falhas.append(f"falta {p.name}")
    if falhas:
        print("=== B2S TESTS ==="); print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)

    df = pd.read_csv(FOLDS)
    da = pd.read_csv(ALPHAS)

    if len(df) != 76: falhas.append(f"folds: {len(df)}")
    for t, esp in [(2024,29),(2025,29),(2026,18)]:
        n = int((df["temporada"]==t).sum())
        if n != esp: falhas.append(f"folds {t}: {n} != {esp}")

    if (df["alpha_star"] < 0).any() or (df["alpha_star"] > 1).any():
        falhas.append("alpha fora de [0,1]")
    if (df["b2s_mae"] < 0).any(): falhas.append("B2S MAE negativo")
    if (df["b2_mae"] < 0).any(): falhas.append("B2 MAE negativo")

    # Spearman preservado (monotonia)
    # Spearman preservado pela transformacao monotona. Em float64 ha
    # divergencias de ordem 1e-5 por quebra de near-ties. Tolerancia 5e-3
    # e 500x maior que o observado e 10.000x menor que qualquer sinal.
    max_diff_sp = float((df["b2_spearman"] - df["b2s_spearman"]).abs().max())
    if max_diff_sp > 5e-3:
        falhas.append(f"Spearman divergiu demais (max diff {max_diff_sp:.6f})")

    # alphas: 21 por fold
    if len(da) != 76 * 21:
        falhas.append(f"alphas linhas: {len(da)} (esperado {76*21})")

    # exatamente 1 alpha selecionado por fold
    sel = da.groupby(["temporada","rodada_alvo"])["selecionado"].sum()
    if (sel != 1).any():
        falhas.append("alphas: nem todo fold tem exatamente 1 alpha selecionado")

    # Parquet e B2 intactos
    if sha256(PARQUET) != H_PQ: falhas.append("Parquet SHA mudou")
    if len(pd.read_csv(B2)) != 76: falhas.append("B2 alterado")

    print("=== B2S TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" folds: {len(df)}")
    print(f" MAE B2={float((df['b2_mae']*df['n_test']).sum()/df['n_test'].sum()):.4f}"
          f"  B2S={float((df['b2s_mae']*df['n_test']).sum()/df['n_test'].sum()):.4f}")
    print(f" alpha* medio: {df['alpha_star'].mean():.3f}")


if __name__ == "__main__":
    main()
