#!/usr/bin/env python3
"""Testes do Ridge v1."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path.home()/"cartola-predicao-v2"
REP = ROOT/"reports"
FOLDS = REP/"ridge_v1_folds.csv"
ALPHAS = REP/"ridge_v1_alphas.csv"
SUM = REP/"ridge_v1_summary.md"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
SPEC = ROOT/"config"/"ridge_v1_spec.yaml"
H_PQ = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []
    for p in [FOLDS, ALPHAS, SUM]:
        if not p.exists(): falhas.append(f"falta {p.name}")
    if falhas:
        print("=== RIDGE v1 TESTS ==="); print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)

    df = pd.read_csv(FOLDS)
    da = pd.read_csv(ALPHAS)
    spec = yaml.safe_load(open(SPEC))
    grid = spec["ridge"]["grade_alpha"]

    if len(df) != 76: falhas.append(f"folds: {len(df)}")
    for t, esp in [(2024,29),(2025,29),(2026,18)]:
        n = int((df["temporada"]==t).sum())
        if n != esp: falhas.append(f"folds {t}: {n} != {esp}")

    if (df["mae"] < 0).any(): falhas.append("MAE negativo")
    if (df["rmse"] < 0).any(): falhas.append("RMSE negativo")
    if df["mae"].isna().any(): falhas.append("MAE NaN")
    if df["spearman"].isna().any(): falhas.append("Spearman NaN")

    # alpha escolhido pertence a grade
    if not set(df["alpha_star"].unique()).issubset(set(grid)):
        falhas.append("alpha_star fora da grade")

    # alphas: 76 * 5
    if len(da) != 76 * len(grid):
        falhas.append(f"alphas linhas: {len(da)} (esperado {76*len(grid)})")

    # exatamente 1 alpha por fold
    sel = da.groupby(["temporada","rodada_alvo"])["selecionado"].sum()
    if (sel != 1).any():
        falhas.append("alphas: nem todo fold tem exatamente 1 alpha selecionado")

    if sha256(PARQUET) != H_PQ: falhas.append("Parquet SHA mudou")
    dp = pd.read_parquet(PARQUET)
    if len(dp) != 77253: falhas.append(f"Parquet linhas: {len(dp)}")

    print("=== RIDGE v1 TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" folds: {len(df)}")
    print(f" Ridge MAE: {float((df['mae']*df['n_test']).sum()/df['n_test'].sum()):.4f}")
    print(f" Ridge Spearman medio: {float(df['spearman'].mean()):.4f}")
    print(f" alpha distribuicao: {df['alpha_star'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
