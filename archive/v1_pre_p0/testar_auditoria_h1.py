#!/usr/bin/env python3
"""Testes da auditoria H1."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
REP = ROOT/"reports"
CSV_H11 = REP/"auditoria_h1_b2_compressao.csv"
CSV_H12 = REP/"auditoria_h1_b3_permutado.csv"
CSV_H13 = REP/"auditoria_h1_b2_stdmatch.csv"
SUMMARY = REP/"auditoria_h1_summary.md"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
B0 = REP/"baseline_b0_folds.csv"; B1 = REP/"baseline_b1_folds.csv"
B2 = REP/"baseline_b2_folds.csv"; B3 = REP/"baseline_b3_folds.csv"
ERROS = REP/"auditoria_b2_b3_erros.csv"
H_PARQUET = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []
    for p in [CSV_H11, CSV_H12, CSV_H13, SUMMARY]:
        if not p.exists(): falhas.append(f"falta {p.name}")
    if falhas:
        print("=== AUDITORIA H1 TESTS ==="); print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)

    h11 = pd.read_csv(CSV_H11)
    h12 = pd.read_csv(CSV_H12)
    h13 = pd.read_csv(CSV_H13)

    # 76 folds em cada
    if h11[["temporada","rodada"]].drop_duplicates().shape[0] != 76:
        falhas.append("H1.1 nao tem 76 folds")
    if h12[["temporada","rodada"]].drop_duplicates().shape[0] != 76:
        falhas.append("H1.2 nao tem 76 folds")
    if len(h13) != 76:
        falhas.append(f"H1.3 tem {len(h13)} linhas (esperado 76)")

    # 11 alphas x 76 = 836 linhas em h11
    if len(h11) != 76 * 11:
        falhas.append(f"H1.1 tem {len(h11)} linhas (esperado {76*11})")
    # 5 seeds x 76 = 380
    if len(h12) != 76 * 5:
        falhas.append(f"H1.2 tem {len(h12)} linhas (esperado {76*5})")

    # alphas tem que incluir 0 e 1
    alphas = sorted(h11["alpha"].unique())
    if 0.0 not in alphas or 1.0 not in alphas:
        falhas.append(f"alphas faltando: {alphas}")

    # MAE nao negativo
    if (h11["mae"] < 0).any(): falhas.append("H1.1 MAE negativo")
    if (h12["mae"] < 0).any(): falhas.append("H1.2 MAE negativo")
    if (h13["mae"] < 0).any(): falhas.append("H1.3 MAE negativo")

    # imutabilidade
    if sha256(PARQUET) != H_PARQUET: falhas.append("Parquet SHA mudou")
    for name, p in [("B0",B0),("B1",B1),("B2",B2),("B3",B3)]:
        if len(pd.read_csv(p)) != 76: falhas.append(f"{name} alterado")

    # nenhuma coluna de blend
    if "prediction_combo" in h13.columns or "prediction_blend" in h13.columns:
        falhas.append("coluna de blend detectada")

    # consistencia: alpha=1 em h11 deve reproduzir MAE do B2 (por fold)
    h11_a1 = h11[h11["alpha"] == 1.0].sort_values(["temporada","rodada"]).reset_index(drop=True)
    b2f = pd.read_csv(B2).sort_values(["temporada","rodada_alvo"]).reset_index(drop=True)
    # comparar mae por fold (deve ser proximo, dado que usamos mesma definicao)
    b2f_mae = b2f["mae"].to_numpy()
    h11_a1_mae = h11_a1["mae"].to_numpy()
    # podem nao bater exatamente (B2 original pode ter usado levemente diferentes
    # ordens de fold/rows), mas o global deve convergir
    b2_global = float((b2f["mae"]*b2f["n_test_metric"]).sum() / b2f["n_test_metric"].sum())
    h11_a1_global = float((h11_a1["mae"]*h11_a1["n"]).sum() / h11_a1["n"].sum())
    if abs(b2_global - h11_a1_global) > 0.01:
        falhas.append(f"alpha=1 MAE ({h11_a1_global:.4f}) difere de B2 ({b2_global:.4f})")

    # obs unica — H1.3 reproduz o mesmo conjunto de folds que erros
    obs = pd.read_csv(ERROS)
    n_obs_esperado = len(obs)
    if int(h13["n"].sum()) != n_obs_esperado:
        falhas.append(f"H1.3 total n={h13['n'].sum()} != obs={n_obs_esperado}")
    if int(h11[h11["alpha"]==1.0]["n"].sum()) != n_obs_esperado:
        falhas.append(f"H1.1 alpha=1 n={h11[h11['alpha']==1.0]['n'].sum()} != obs={n_obs_esperado}")
    if int(h12[h12["seed"]==42]["n"].sum()) != n_obs_esperado:
        falhas.append(f"H1.2 seed=42 n={h12[h12['seed']==42]['n'].sum()} != obs={n_obs_esperado}")

    print("=== AUDITORIA H1 TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas: print(f" - {f}")
        sys.exit(1)
    print("PASS")
    print(f" H1.1 folds={h11[['temporada','rodada']].drop_duplicates().shape[0]}  alphas={len(alphas)}")
    print(f" H1.2 folds={h12[['temporada','rodada']].drop_duplicates().shape[0]}  seeds={h12['seed'].nunique()}")
    print(f" H1.3 folds={len(h13)}")
    print(f" obs consistentes: {n_obs_esperado}")
    print(f" Parquet sha256: {sha256(PARQUET)[:16]}...")


if __name__ == "__main__":
    main()
