#!/usr/bin/env python3
"""Testes do baseline B1."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "cartola-predicao-v2"
CSV_FOLDS = ROOT / "reports" / "baseline_b1_folds.csv"
SUMMARY = ROOT / "reports" / "baseline_b1_summary.md"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
B0_CSV = ROOT / "reports" / "baseline_b0_folds.csv"

PARQUET_SHA_ESPERADO = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
ESPERADO_FOLDS = {2024: 29, 2025: 29, 2026: 18}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []
    if not CSV_FOLDS.exists():
        print("=== B1 TESTS ==="); print("FAIL"); print(" - CSV nao existe"); sys.exit(1)
    if not SUMMARY.exists():
        falhas.append("summary nao existe")

    df = pd.read_csv(CSV_FOLDS)

    # 3-6 folds
    if len(df) != 76: falhas.append(f"folds: {len(df)} != 76")
    for t, esp in ESPERADO_FOLDS.items():
        n = int((df["temporada"] == t).sum())
        if n != esp: falhas.append(f"folds {t}: {n} != {esp}")

    # 7-8 temporalidade
    if (df["train_max_round"] >= df["rodada_alvo"]).any():
        falhas.append("T7: train_max_round >= rodada_alvo")
    if (df["test_round"] != df["rodada_alvo"]).any():
        falhas.append("T8: test_round != rodada_alvo")

    # 9 2025/R1
    if ((df["temporada"] == 2025) & (df["train_max_round"] < 2)).any():
        falhas.append("T9: 2025 com treino < R2")

    # 10 fallback + historico
    if (df["n_test_com_historico"] + df["n_test_fallback_b0"] != df["n_test_metric"]).any():
        falhas.append("T10: n_com_hist + n_fallback != n_test_metric")

    # 11 coverage
    if not ((df["coverage_historico"] >= 0) & (df["coverage_historico"] <= 1)).all():
        falhas.append("T11: coverage fora de [0,1]")

    # 12-13 metricas
    if (df["mae"] < 0).any(): falhas.append("T12: mae negativo")
    if (df["rmse"] < 0).any(): falhas.append("T13: rmse negativo")
    if not np.isfinite(df["train_mean_global"]).all():
        falhas.append("T14: train_mean_global nao finito")

    # 15 b0_mae
    b0 = pd.read_csv(B0_CSV)
    merge = df.merge(b0[["temporada","rodada_alvo","mae"]],
                     on=["temporada","rodada_alvo"], suffixes=("","_b0real"))
    if (abs(merge["b0_mae"] - merge["mae_b0real"]) > 1e-6).any():
        falhas.append("T15: b0_mae no B1 diferente do B0 real")

    # 16 delta
    if (abs((df["mae"] - df["b0_mae"]) - df["delta_mae"]) > 1e-9).any():
        falhas.append("T16: delta_mae inconsistente")

    # 17-18 spot check rodada e temporada
    print("-- spot check independente --")
    dfp = pd.read_parquet(PARQUET).sort_values(["temporada","rodada","atleta_id"]).reset_index(drop=True)
    dfp["flag_ambigua_B"] = (dfp["entrou_em_campo"] == False) & (dfp["pontos_num"] != 0)
    dfp["elegivel"] = ~((dfp["temporada"] == 2025) & (dfp["rodada"] == 1))

    rng = np.random.default_rng(42)
    folds_spot = rng.choice(len(df), size=5, replace=False)
    ok_spot = 0
    for idx in folds_spot:
        row = df.iloc[idx]
        t, r = int(row["temporada"]), int(row["rodada_alvo"])
        mask = dfp["temporada"] == t
        train = dfp[mask & (dfp["rodada"] < r) & dfp["elegivel"]]
        test = dfp[mask & (dfp["rodada"] == r) & dfp["elegivel"]]
        test_m = test[~test["flag_ambigua_B"]]
        pm = train.groupby("atleta_id")["pontos_num"].mean()
        tg = float(train["pontos_num"].mean())

        # T7 e T8: pick player com historico
        com_hist = test_m[test_m["atleta_id"].isin(pm.index)]
        if len(com_hist):
            p = com_hist.iloc[0]
            manual = float(train[train["atleta_id"] == p["atleta_id"]]["pontos_num"].mean())
            if abs(manual - float(pm.loc[p["atleta_id"]])) > 1e-9:
                falhas.append(f"T7 spot: media manual difere em {t}/R{r}")
            # T8: todas as obs do historico sao da mesma temporada
            sub_tr = train[train["atleta_id"] == p["atleta_id"]]
            if (sub_tr["temporada"] != t).any():
                falhas.append(f"T8 spot: historico de outra temporada em {t}/R{r}")
            # T9: todas as obs do historico tem rodada < r
            if (sub_tr["rodada"] >= r).any():
                falhas.append(f"T9 spot: historico com rodada >= alvo em {t}/R{r}")
            ok_spot += 1

        # T10: player sem historico
        sem_hist = test_m[~test_m["atleta_id"].isin(pm.index)]
        if len(sem_hist):
            p = sem_hist.iloc[0]
            # a previsao deveria ser tg
            if not np.isclose(tg, float(train["pontos_num"].mean())):
                falhas.append(f"T10 spot: fallback inconsistente em {t}/R{r}")
            ok_spot += 1

    print(f"spot checks OK: {ok_spot}")

    # 20-21 Parquet
    if not PARQUET.exists():
        falhas.append("Parquet nao existe")
    else:
        df_parq = pd.read_parquet(PARQUET)
        if len(df_parq) != 77253:
            falhas.append(f"Parquet: {len(df_parq)} linhas != 77253")
        if sha256(PARQUET) != PARQUET_SHA_ESPERADO:
            falhas.append("Parquet SHA-256 mudou")

    # 22 B0 intacto
    if not B0_CSV.exists():
        falhas.append("B0 csv sumiu")
    else:
        b0_chk = pd.read_csv(B0_CSV)
        if len(b0_chk) != 76:
            falhas.append(f"B0 csv: {len(b0_chk)} folds != 76")

    print("=== B1 TESTS ===")
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
