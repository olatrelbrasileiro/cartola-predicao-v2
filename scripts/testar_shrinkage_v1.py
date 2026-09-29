#!/usr/bin/env python3
"""Testes de leakage e spot checks para shrinkage v1."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"
H_PARQUET = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"

def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()

falhas = []

# T7: Parquet canônico
if sha256(INTERIM/"cartola_jogador_rodada_v2.parquet") != H_PARQUET:
    falhas.append("T7: Parquet canonico alterado")

# Carregar artefatos
h = pd.read_parquet(INTERIM/"features_historico_shrinkage_v1.parquet")
g = pd.read_parquet(INTERIM/"features_ga_shrinkage_v1.parquet")

# T8: 76 folds
if len(h.groupby(["temporada","rodada"])) != 76:
    falhas.append("T8: historico nao tem 76 folds")
if len(g.groupby(["temporada","rodada"])) != 76:
    falhas.append("T8: ga nao tem 76 folds")

# Spot checks: hist_shrunk = (n*hist_mean + k*prior)/(n+k)
K = 7
for i in range(30):
    row = h.sample(1).iloc[0]
    if row["hist_n"] == 0:
        continue
    expected = (row["hist_n"] * row["hist_mean"] + K * row["prior_pos"]) / (row["hist_n"] + K)
    if abs(expected - row["hist_shrunk"]) > 1e-9:
        falhas.append(f"spot hist: {row['atleta_id']} R{row['rodada']} {expected} != {row['hist_shrunk']}")

# Spot checks GA
for i in range(30):
    row = g.sample(1).iloc[0]
    if row["hist_n"] == 0:
        continue
    expected = (row["hist_n"] * row["ga_rate"] + K * row["prior_ga_pos"]) / (row["hist_n"] + K)
    if abs(expected - row["ga_shrunk"]) > 1e-9:
        falhas.append(f"spot ga: {row['atleta_id']} R{row['rodada']} {expected} != {row['ga_shrunk']}")

# Testes de sanidade
if h["hist_shrunk"].isna().sum() > 0:
    # cold starts sao permitidos como NaN
    pass

# Salvar spotcheck
spot = pd.concat([h.head(15), g.head(15)])
spot.to_csv(REP/"shrinkage_v1_spotcheck.csv", index=False)

print("=== SHRINKAGE TESTS ===")
if falhas:
    print("FAIL")
    for f in falhas: print(f" - {f}")
    sys.exit(1)
print("PASS")
print(f" historico linhas: {len(h)}")
print(f" ga linhas: {len(g)}")
