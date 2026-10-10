#!/usr/bin/env python3
"""Testes do CatBoost v1."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

def _detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz do projeto nao encontrada")

ROOT = _detect_root()
REP = ROOT/"reports"
INTERIM = ROOT/"data"/"interim"

FOLDS = REP/"catboost_v1_folds.csv"
SUMMARY = REP/"catboost_v1_summary.md"
FI = REP/"catboost_v1_feature_importance.csv"
GRP = REP/"catboost_v1_group_importance.csv"
ABL = REP/"catboost_v1_ablation.csv"
HARNESS = REP/"harness_folds_v2.csv"
B2S = REP/"baseline_b2s_folds.csv"
CANON = INTERIM/"cartola_jogador_rodada_v2.parquet"
TEMPORAL = INTERIM/"features_confronto_v1_temporal_safe.parquet"

H_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
H_TEMPORAL = "0fcc9fe7e5ec16dbdf4df2477aa5e45dc620f19708b5baf7f36b863947f5a1d2"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []

    for p in [FOLDS, SUMMARY, FI, GRP, ABL]:
        if not p.exists():
            falhas.append(f"FALTA {p.name}")
    if falhas:
        print("=== CATBOOST v1 TESTS ==="); print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)

    dfr = pd.read_csv(FOLDS)
    man = pd.read_csv(HARNESS)
    b2s = pd.read_csv(B2S)

    # T13: 76 folds
    if len(dfr) != 76: falhas.append(f"folds: {len(dfr)} != 76")
    for t, esp in [(2024,29),(2025,29),(2026,18)]:
        n = int((dfr["temporada"]==t).sum())
        if n != esp: falhas.append(f"folds {t}: {n} != {esp}")

    # T: chaves identicas ao harness
    k1 = set(map(tuple, dfr[["temporada","rodada_alvo"]].values.tolist()))
    k2 = set(map(tuple, man[["temporada","rodada_alvo"]].values.tolist()))
    if k1 != k2: falhas.append("chaves divergem do harness")

    # T: sem 2025/R1
    if ((dfr["temporada"]==2025) & (dfr["rodada_alvo"]==1)).any():
        falhas.append("2025/R1 presente")

    # T: metricas nao negativas
    if (dfr["cb_mae"] < 0).any(): falhas.append("cb_mae negativo")
    if (dfr["cb_rmse"] < 0).any(): falhas.append("cb_rmse negativo")

    # T: B2S inalterado
    if len(b2s) != 76: falhas.append(f"B2S: {len(b2s)} != 76")

    # T: hashes
    if sha256(CANON) != H_CANON: falhas.append("Parquet canonico alterado")
    if sha256(TEMPORAL) != H_TEMPORAL: falhas.append("Temporal-safe alterado")

    # T: feature importance cobre 21 features
    fi = pd.read_csv(FI)
    if len(fi) != 21: falhas.append(f"feature_importance: {len(fi)} != 21")
    grp = pd.read_csv(GRP)
    if len(grp) != 7: falhas.append(f"group_importance: {len(grp)} != 7")

    # T: ablation com 76 folds
    abl = pd.read_csv(ABL)
    if len(abl) != 76: falhas.append(f"ablation: {len(abl)} != 76")

    # T: nenhuma coluna proibida no conjunto de features
    proibidas = {"HG","AG","Res","hg","ag","res"}
    if proibidas.intersection(set(fi["feature"])):
        falhas.append(f"features proibidas presentes: {proibidas.intersection(set(fi['feature']))}")

    print("=== CATBOOST v1 TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" folds: {len(dfr)}")
    print(f" CB MAE global: {float((dfr['cb_mae']*dfr['n_test_metric']).sum()/dfr['n_test_metric'].sum()):.4f}")
    print(f" B2S MAE global: {float((dfr['b2s_mae']*dfr['n_test_metric']).sum()/dfr['n_test_metric'].sum()):.4f}")
    print(f" hashes: OK")


if __name__ == "__main__":
    main()
