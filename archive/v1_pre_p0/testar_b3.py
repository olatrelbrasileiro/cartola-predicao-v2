#!/usr/bin/env python3
"""Testes do baseline B3 (corrigido)."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path.home() / "cartola-predicao-v2"
CSV_FOLDS = ROOT / "reports" / "baseline_b3_folds.csv"
CSV_SPOT = ROOT / "reports" / "baseline_b3_spotcheck.csv"
SUMMARY = ROOT / "reports" / "baseline_b3_summary.md"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
AUDIT = ROOT / "config" / "pontuacao_scouts_auditoria.yaml"
B0_CSV = ROOT / "reports" / "baseline_b0_folds.csv"
B1_CSV = ROOT / "reports" / "baseline_b1_folds.csv"
B2_CSV = ROOT / "reports" / "baseline_b2_folds.csv"

PARQUET_SHA = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
ESPERADO_FOLDS = {2024: 29, 2025: 29, 2026: 18}
SCOUTS_B3 = ["DS","FT","FF","FD","DE","FC","FS","I","CV","CA"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def apply_cumulative_diff(df, scout_cols):
    out = df.sort_values(["temporada","atleta_id","rodada"],
                         kind="mergesort").reset_index(drop=True)
    for col in scout_cols:
        ff = out.groupby(["temporada","atleta_id"], sort=False)[col].ffill()
        chave = out[["temporada","atleta_id"]].copy()
        chave["_ff"] = ff.values
        prev = chave.groupby(["temporada","atleta_id"], sort=False)["_ff"].shift(1)
        out[f"diff_{col}"] = ff.fillna(0).to_numpy() - prev.fillna(0).to_numpy()
    return out


def main():
    falhas = []
    for p in [CSV_FOLDS, CSV_SPOT, SUMMARY]:
        if not p.exists():
            falhas.append(f"falta {p.name}")
    if falhas:
        print("=== B3 TESTS ==="); print("FAIL"); [print(f" - {f}") for f in falhas]; sys.exit(1)

    df = pd.read_csv(CSV_FOLDS)
    spot = pd.read_csv(CSV_SPOT)

    if len(df) != 76: falhas.append(f"folds: {len(df)} != 76")
    for t, esp in ESPERADO_FOLDS.items():
        n = int((df["temporada"] == t).sum())
        if n != esp: falhas.append(f"folds {t}: {n} != {esp}")

    if (df["train_max_round"] >= df["rodada_alvo"]).any(): falhas.append("T1")
    if (df["test_round"] != df["rodada_alvo"]).any(): falhas.append("T2")
    if ((df["temporada"] == 2025) & (df["train_max_round"] < 2)).any(): falhas.append("T4")
    if (df["n_test_com_basic_history"] + df["n_test_fallback_b0"] != df["n_test_metric"]).any():
        falhas.append("n_hist + n_fallback != n_metric")
    if not ((df["coverage_basic_history"] >= 0) & (df["coverage_basic_history"] <= 1)).all():
        falhas.append("coverage fora [0,1]")
    if (df["mae"] < 0).any(): falhas.append("MAE negativo")
    if (df["rmse"] < 0).any(): falhas.append("RMSE negativo")

    audit = yaml.safe_load(open(AUDIT))
    if set(audit["scouts_basicos"]) != set(SCOUTS_B3):
        falhas.append(f"scouts_basicos != esperado")
    for s in ["G","A","SG"]:
        if s in audit["scouts_basicos"]:
            falhas.append(f"{s} no basico (proibido)")

    if len(pd.read_csv(B0_CSV)) != 76: falhas.append("B0 alterado")
    if len(pd.read_csv(B1_CSV)) != 76: falhas.append("B1 alterado")
    if len(pd.read_csv(B2_CSV)) != 76: falhas.append("B2 alterado")

    if sha256(PARQUET) != PARQUET_SHA: falhas.append("Parquet SHA mudou")
    dfp = pd.read_parquet(PARQUET)
    if len(dfp) != 77253: falhas.append(f"Parquet linhas: {len(dfp)}")
    if dfp.shape[1] != 38: falhas.append(f"Parquet colunas: {dfp.shape[1]}")

    # checagem independente: recomputa basic_score com diff e compara com spotcheck
    dfp = dfp.sort_values(["temporada","atleta_id","rodada"]).reset_index(drop=True)
    dfp["flag_ambigua_B"] = (dfp["entrou_em_campo"] == False) & (dfp["pontos_num"] != 0)
    dfp["elegivel"] = ~((dfp["temporada"] == 2025) & (dfp["rodada"] == 1))

    scouts = audit["scouts_basicos"]
    scout_cols = [f"scout_{s}" for s in scouts]
    dfp = apply_cumulative_diff(dfp, scout_cols)

    pesos = audit["pesos_por_temporada"]
    dfp["basic_score"] = 0.0
    for s in scouts:
        col = f"diff_scout_{s}"
        peso_linha = dfp["temporada"].map({int(t): pesos[t][s] for t in pesos})
        dfp["basic_score"] += dfp[col].to_numpy() * peso_linha.to_numpy()

    n_ok_hist = 0; n_ok_fb = 0
    for _, row in spot.iterrows():
        t, r, aid = int(row["temporada"]), int(row["rodada_alvo"]), int(row["atleta_id"])
        train = dfp[(dfp["temporada"]==t) & (dfp["rodada"]<r) & dfp["elegivel"]]
        sub = train[train["atleta_id"] == aid].sort_values("rodada")
        if row["used_fallback_b0"]:
            if len(sub) != 0:
                falhas.append(f"T12: fallback com historico {t}/R{r}/{aid}")
            n_ok_fb += 1
            continue
        if len(sub) == 0:
            falhas.append(f"T13: sem historico mas sem fallback {t}/R{r}/{aid}")
            continue
        manual = float(sub["basic_score"].mean())
        if abs(manual - row["basic_mean"]) > 1e-6:
            falhas.append(f"T18: basic_mean difere {t}/R{r}/{aid}: {manual} vs {row['basic_mean']}")
        if abs(row["prediction"] - row["basic_mean"]) > 1e-6:
            falhas.append(f"T13: prediction != basic_mean {t}/R{r}/{aid}")
        if (sub["rodada"] >= r).any():
            falhas.append(f"T7: historico com rodada >= alvo {t}/R{r}/{aid}")
        if (sub["temporada"] != t).any():
            falhas.append(f"T8: historico de outra temporada {t}/R{r}/{aid}")
        n_ok_hist += 1

    print(f"spot checks historico: {n_ok_hist}")
    print(f"spot checks fallback:  {n_ok_fb}")

    print("=== B3 TESTS ===")
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
