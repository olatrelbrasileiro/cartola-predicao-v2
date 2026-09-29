#!/usr/bin/env python3
"""Shrinkage historico por posicao, calculado dentro de cada fold.

Especificacao pre-registrada:
- k = 7 (hiperparametro fixo, nao otimizado)
- prior_pos = media de pontos_num da posicao, calculada APENAS no treino do fold
- hist_shrunk = (n * hist_mean + k * prior_pos) / (n + k)
- cold start: mantido como fallback B0 (train_mean_global)
"""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path.home()/"cartola-predicao-v2"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
CONFIG = ROOT/"config.yaml"
MANIFEST = ROOT/"reports"/"harness_folds_v2.csv"
REP = ROOT/"reports"
OUT = ROOT/"data"/"interim"/"features_historico_shrinkage_v1.parquet"

K_SHRINK = 7   # pre-registrado

def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()

cfg = yaml.safe_load(open(CONFIG))
mtr = cfg["validation"]["min_train_rounds"]

df = pd.read_parquet(PARQUET)
df = df.sort_values(["temporada","rodada","atleta_id"]).reset_index(drop=True)
df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

man = pd.read_csv(MANIFEST)
folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]

rows = []
for t, R in folds:
    mask_t = df["temporada"] == t
    train = df[mask_t & (df["rodada"] < R) & df["elegivel"]]
    test = df[mask_t & (df["rodada"] == R) & df["elegivel"]]

    # prior por posicao no treino do fold
    prior_pos = train.groupby("posicao_id")["pontos_num"].mean()
    global_mean = float(train["pontos_num"].mean())

    # calcular hist_mean e hist_n por atleta no treino
    hist_stats = train.groupby("atleta_id")["pontos_num"].agg(
        hist_mean="mean", hist_n="count").reset_index()

    # merge com test
    test = test.merge(hist_stats, on="atleta_id", how="left")
    test["hist_n"] = test["hist_n"].fillna(0).astype(int)
    test["posicao_id_prior"] = test["posicao_id"].map(prior_pos)
    test["posicao_id_prior"] = test["posicao_id_prior"].fillna(global_mean)

    # hist_shrunk
    n = test["hist_n"].to_numpy(dtype=float)
    hm = test["hist_mean"].fillna(0).to_numpy(dtype=float)
    pr = test["posicao_id_prior"].to_numpy(dtype=float)
    test["hist_shrunk"] = np.where(
        n > 0,
        (n * hm + K_SHRINK * pr) / (n + K_SHRINK),
        np.nan
    )
    test["hist_shrinkage_delta"] = test["hist_shrunk"] - test["hist_mean"]

    for _, r in test.iterrows():
        rows.append({
            "temporada": t, "rodada": R,
            "atleta_id": int(r["atleta_id"]),
            "posicao_id": int(r["posicao_id"]),
            "hist_mean": r["hist_mean"],
            "hist_n": int(r["hist_n"]),
            "prior_pos": r["posicao_id_prior"],
            "hist_shrunk": r["hist_shrunk"],
            "hist_shrinkage_delta": r["hist_shrinkage_delta"],
        })

out = pd.DataFrame(rows)
out.to_parquet(OUT, index=False)
print(f"salvo: {OUT.relative_to(ROOT)}")
print(f"linhas: {len(out)}")
print(f"com hist_shrunk valido: {out['hist_shrunk'].notna().sum()}")
print(f"k usado: {K_SHRINK}")
