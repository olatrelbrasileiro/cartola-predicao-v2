#!/usr/bin/env python3
"""Testes do baseline B2."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "cartola-predicao-v2"
CSV_FOLDS = ROOT / "reports" / "baseline_b2_folds.csv"
CSV_SPOT = ROOT / "reports" / "baseline_b2_spotcheck.csv"
SUMMARY = ROOT / "reports" / "baseline_b2_summary.md"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
B0_CSV = ROOT / "reports" / "baseline_b0_folds.csv"
B1_CSV = ROOT / "reports" / "baseline_b1_folds.csv"

PARQUET_SHA = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
ESPERADO_FOLDS = {2024: 29, 2025: 29, 2026: 18}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []
    for p in [CSV_FOLDS, CSV_SPOT, SUMMARY]:
        if not p.exists():
            falhas.append(f"falta {p.name}")

    if falhas:
        print("=== B2 TESTS ==="); print("FAIL"); [print(f" - {f}") for f in falhas]; sys.exit(1)

    df = pd.read_csv(CSV_FOLDS)
    spot = pd.read_csv(CSV_SPOT)

    if len(df) != 76: falhas.append(f"folds: {len(df)} != 76")
    for t, esp in ESPERADO_FOLDS.items():
        n = int((df["temporada"] == t).sum())
        if n != esp: falhas.append(f"folds {t}: {n} != {esp}")

    if (df["train_max_round"] >= df["rodada_alvo"]).any(): falhas.append("T1: train_max >= alvo")
    if (df["test_round"] != df["rodada_alvo"]).any(): falhas.append("T2: test_round != alvo")
    if ((df["temporada"] == 2025) & (df["train_max_round"] < 2)).any(): falhas.append("T4: 2025 com treino < R2")
    if (df["n_test_com_historico"] + df["n_test_fallback_b0"] != df["n_test_metric"]).any():
        falhas.append("n_com_hist + n_fallback != n_metric")
    if not ((df["coverage_historico"] >= 0) & (df["coverage_historico"] <= 1)).all():
        falhas.append("coverage fora de [0,1]")
    if (df["mae"] < 0).any(): falhas.append("MAE negativo")
    if (df["rmse"] < 0).any(): falhas.append("RMSE negativo")

    # B2 - B1 = delta
    b1 = pd.read_csv(B1_CSV)
    m = df.merge(b1[["temporada","rodada_alvo","mae"]], on=["temporada","rodada_alvo"], suffixes=("","_b1real"))
    if (abs((m["mae"] - m["mae_b1real"]) - m["delta_mae_vs_b1"]) > 1e-9).any():
        falhas.append("T12: delta_mae_vs_b1 inconsistente")

    # B0/B1 intactos
    if len(pd.read_csv(B0_CSV)) != 76: falhas.append("B0 alterado")
    if len(pd.read_csv(B1_CSV)) != 76: falhas.append("B1 alterado")

    # Parquet
    if sha256(PARQUET) != PARQUET_SHA: falhas.append("Parquet SHA mudou")
    dfp = pd.read_parquet(PARQUET)
    if len(dfp) != 77253: falhas.append(f"Parquet linhas: {len(dfp)}")
    if dfp.shape[1] != 38: falhas.append(f"Parquet colunas: {dfp.shape[1]}")

    # spotcheck: predictions consistentes com formula
    dfp = dfp.sort_values(["temporada","rodada","atleta_id"]).reset_index(drop=True)
    dfp["flag_ambigua_B"] = (dfp["entrou_em_campo"] == False) & (dfp["pontos_num"] != 0)
    dfp["elegivel"] = ~((dfp["temporada"] == 2025) & (dfp["rodada"] == 1))

    n_ok_hist = 0
    n_ok_fb = 0
    for _, row in spot.iterrows():
        t, r, aid = int(row["temporada"]), int(row["rodada_alvo"]), int(row["atleta_id"])
        mask = dfp["temporada"] == t
        train = dfp[mask & (dfp["rodada"] < r) & dfp["elegivel"]]
        sub = train[train["atleta_id"] == aid].sort_values("rodada")
        if row["used_fallback_b0"]:
            if len(sub) != 0:
                falhas.append(f"T11 spot: fallback com historico em {t}/R{r}/{aid}")
            tg = float(train["pontos_num"].mean())
            if abs(row["prediction"] - tg) > 1e-9:
                falhas.append(f"T11 spot: fallback != train_mean em {t}/R{r}/{aid}")
            n_ok_fb += 1
        else:
            if len(sub) == 0:
                falhas.append(f"T13 spot: sem historico mas used_fallback=False {t}/R{r}/{aid}")
                continue
            h_manual = float(sub["pontos_num"].mean())
            r_manual = float(sub.tail(5)["pontos_num"].mean())
            if abs(h_manual - row["historico_mean"]) > 1e-9:
                falhas.append(f"T7 spot: H difere em {t}/R{r}/{aid}")
            if abs(r_manual - row["recente_mean"]) > 1e-9:
                falhas.append(f"T8 spot: R difere em {t}/R{r}/{aid}")
            if abs(r_manual - row["prediction"]) > 1e-9:
                falhas.append(f"T13 spot: prediction != R em {t}/R{r}/{aid}")
            # janela <= 5
            if len(sub) > 5:
                # confere que pegamos apenas ultimas 5
                r5 = float(sub.tail(5)["pontos_num"].mean())
                if abs(r5 - row["recente_mean"]) > 1e-9:
                    falhas.append(f"T17 spot: recente != tail(5) em {t}/R{r}/{aid}")
            n_ok_hist += 1

    print(f"spot checks com historico: {n_ok_hist}")
    print(f"spot checks fallback:      {n_ok_fb}")

    print("=== B2 TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" folds: {len(df)}")
    print(f" MAE global ponderado: "
          f"{float((df['mae']*df['n_test_metric']).sum()/df['n_test_metric'].sum()):.4f}")
    print(f" Parquet sha256: {sha256(PARQUET)[:16]}...")


if __name__ == "__main__":
    main()
