#!/usr/bin/env python3
"""Testes da auditoria B2 x B3."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
REP = ROOT/"reports"
CSV_FOLDS = REP/"auditoria_b2_b3_folds.csv"
CSV_ERROS = REP/"auditoria_b2_b3_erros.csv"
CSV_TARGET = REP/"auditoria_b2_b3_por_target.csv"
CSV_TEMP = REP/"auditoria_b2_b3_por_temporada.csv"
CSV_HIST = REP/"auditoria_b2_b3_por_historico.csv"
SUMMARY = REP/"auditoria_b2_b3_summary.md"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
B0 = REP/"baseline_b0_folds.csv"; B1 = REP/"baseline_b1_folds.csv"
B2 = REP/"baseline_b2_folds.csv"; B3 = REP/"baseline_b3_folds.csv"
H_PARQUET = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []
    for p in [CSV_FOLDS, CSV_ERROS, CSV_TARGET, CSV_TEMP, CSV_HIST, SUMMARY]:
        if not p.exists():
            falhas.append(f"falta {p.name}")
    if falhas:
        print("=== AUDITORIA B2×B3 TESTS ==="); print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)

    df = pd.read_csv(CSV_FOLDS)
    obs = pd.read_csv(CSV_ERROS)
    tgt = pd.read_csv(CSV_TARGET)
    tmp = pd.read_csv(CSV_TEMP)
    hist = pd.read_csv(CSV_HIST)

    if len(df) != 76: falhas.append(f"folds: {len(df)}")
    for t, esp in [(2024,29),(2025,29),(2026,18)]:
        n = int((df["temporada"]==t).sum())
        if n != esp: falhas.append(f"folds {t}: {n} != {esp}")

    # B2/B3 oficiais intactos
    for name, p in [("B0",B0),("B1",B1),("B2",B2),("B3",B3)]:
        if len(pd.read_csv(p)) != 76:
            falhas.append(f"{name} alterado")

    # Parquet
    if sha256(PARQUET) != H_PARQUET: falhas.append("Parquet SHA mudou")
    dp = pd.read_parquet(PARQUET)
    if len(dp) != 77253: falhas.append(f"Parquet linhas: {len(dp)}")

    # deltas consistentes
    if not np.allclose(df["b3_mae"] - df["b2_mae"], df["delta_mae"], atol=1e-9):
        falhas.append("delta_mae inconsistente")
    if (df["b2_mae"] < 0).any() or (df["b3_mae"] < 0).any():
        falhas.append("MAE negativo")
    if (df["b2_rmse"] < 0).any() or (df["b3_rmse"] < 0).any():
        falhas.append("RMSE negativo")

    # buckets de target cobrem tudo
    if obs["bucket_target"].isna().any() if "bucket_target" in obs.columns else False:
        pass
    # obs CSV nao tem bucket_target (por design); valida via soma de n do target
    if int(tgt["n"].sum()) != len(obs):
        falhas.append(f"soma n target ({tgt['n'].sum()}) != obs ({len(obs)})")

    # buckets historico
    if int(hist["n"].sum()) != len(obs):
        falhas.append(f"soma n hist ({hist['n'].sum()}) != obs ({len(obs)})")

    # fallback
    if (obs["fallback_b2"] != obs["fallback_b3"]).any():
        falhas.append("fallback divergente B2/B3 (esperado identico)")

    # nenhuma combinacao de previsao
    if "prediction_combo" in obs.columns or "prediction_blend" in obs.columns:
        falhas.append("coluna de blend detectada")

    # por temporada
    if len(tmp) != 3: falhas.append(f"por_temporada: {len(tmp)}")

    print("=== AUDITORIA B2×B3 TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" folds: {len(df)}")
    print(f" obs: {len(obs)}")
    print(f" MAE B2={float((df['b2_mae']*df['n_test']).sum()/df['n_test'].sum()):.4f}"
          f"  B3={float((df['b3_mae']*df['n_test']).sum()/df['n_test'].sum()):.4f}")
    print(f" Parquet: {sha256(PARQUET)[:16]}...")


if __name__ == "__main__":
    main()
