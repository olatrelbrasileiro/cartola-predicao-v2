#!/usr/bin/env python3
"""Testes do baseline B0."""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "cartola-predicao-v2"
CSV_FOLDS = ROOT / "reports" / "baseline_b0_folds.csv"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"

PARQUET_SHA_ESPERADO = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
ESPERADO_FOLDS = {2024: 29, 2025: 29, 2026: 18}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []

    if not CSV_FOLDS.exists():
        print("=== B0 TESTS ===")
        print("FAIL")
        print(f" - {CSV_FOLDS} nao existe")
        sys.exit(1)

    df = pd.read_csv(CSV_FOLDS)

    if len(df) != 76:
        falhas.append(f"folds totais: {len(df)} != 76")
    for t, esp in ESPERADO_FOLDS.items():
        n = int((df["temporada"] == t).sum())
        if n != esp:
            falhas.append(f"folds {t}: {n} != {esp}")

    if (df["train_max_round"] >= df["rodada_alvo"]).any():
        falhas.append("train_max_round >= rodada_alvo")
    if (df["test_round"] != df["rodada_alvo"]).any():
        falhas.append("test_round != rodada_alvo")

    if (df["n_test_metric"] + df["n_test_ambiguous_B"] != df["n_test_raw"]).any():
        falhas.append("n_test_metric + n_test_ambiguous_B != n_test_raw")

    if (df["mae"] < 0).any():
        falhas.append("MAE negativo")
    if (df["rmse"] < 0).any():
        falhas.append("RMSE negativo")
    if not df["train_mean"].apply(lambda x: (x == x) and abs(x) != float("inf")).all():
        falhas.append("train_mean nao finito")

    if (df["rodada_alvo"] == 1).any():
        falhas.append("fold com rodada_alvo==1 (2025/R1 nao deveria virar fold)")

    # 2025: nenhum fold deve comecar em R1
    f25 = df[df["temporada"] == 2025]
    if (f25["train_min_round"] == 1).any():
        falhas.append("2025 com train_min_round==1 (R1 deveria estar excluida)")

    # Parquet
    if not PARQUET.exists():
        falhas.append("Parquet nao existe")
    else:
        dfp = pd.read_parquet(PARQUET)
        if len(dfp) != 77253:
            falhas.append(f"Parquet: {len(dfp)} linhas != 77253")
        h = sha256(PARQUET)
        if h != PARQUET_SHA_ESPERADO:
            falhas.append(f"Parquet SHA-256 mudou: {h}")

    print("=== B0 TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas:
            print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" folds: {len(df)}")
    print(f" MAE medio simples: {df['mae'].mean():.4f}")
    total_n = int(df["n_test_metric"].sum())
    mae_pond = float((df["mae"] * df["n_test_metric"]).sum() / total_n)
    print(f" MAE global ponderado: {mae_pond:.4f}")
    print(f" Parquet sha256: {sha256(PARQUET)[:16]}...")


if __name__ == "__main__":
    main()
