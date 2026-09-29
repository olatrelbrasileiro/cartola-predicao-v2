#!/usr/bin/env python3
"""Shrinkage de G+A por posicao, calculado dentro de cada fold.

Especificacao pre-registrada:
- k = 7 (fixo, nao otimizado)
- ga_rate = (G + A acumulados) / (jogos_num acumulados) ate a ultima obs pre-R
- prior_ga_pos = media de ga_rate por posicao no treino do fold
- ga_shrunk = (n * ga_rate + k * prior_ga_pos) / (n + k)
- n = numero de observacoes elegiveis anteriores
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
OUT = ROOT/"data"/"interim"/"features_ga_shrinkage_v1.parquet"

K_SHRINK = 7

cfg = yaml.safe_load(open(CONFIG))
mtr = cfg["validation"]["min_train_rounds"]

df = pd.read_parquet(PARQUET)
df = df.sort_values(["temporada","rodada","atleta_id"]).reset_index(drop=True)
df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

# Calcular diffs de scouts G e A
for s in ["G","A"]:
    col = f"scout_{s}"
    df[f"diff_{col}"] = df.groupby(["temporada","atleta_id"])[col].diff().fillna(df[col])

man = pd.read_csv(MANIFEST)
folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]

rows = []
for t, R in folds:
    mask_t = df["temporada"] == t
    train = df[mask_t & (df["rodada"] < R) & df["elegivel"]]
    test = df[mask_t & (df["rodada"] == R) & df["elegivel"]]

    # Para cada atleta no treino: ga_rate = (G+A acumulados) / jogos_num
    train_agg = train.groupby("atleta_id").agg(
        ga_total=("diff_scout_G", "sum"),
        jogos=("jogos_num", "last"),
        hist_n=("rodada", "count"),
        posicao_id=("posicao_id", "last"),
    ).reset_index()
    train_agg["ga_rate"] = train_agg["ga_total"] / train_agg["jogos"].clip(lower=1)
    # prior por posicao
    prior_ga = train_agg.groupby("posicao_id")["ga_rate"].mean()
    global_ga = float(train_agg["ga_rate"].mean())

    test = test.merge(train_agg[["atleta_id","ga_rate","hist_n","posicao_id"]],
                     on="atleta_id", how="left", suffixes=("","_tr"))
    test["posicao_id_prior_ga"] = test["posicao_id"].map(prior_ga)
    test["posicao_id_prior_ga"] = test["posicao_id_prior_ga"].fillna(global_ga)
    test["hist_n"] = test["hist_n"].fillna(0).astype(int)
    test["ga_rate"] = test["ga_rate"].fillna(0)

    n = test["hist_n"].to_numpy(dtype=float)
    gr = test["ga_rate"].to_numpy(dtype=float)
    pr = test["posicao_id_prior_ga"].to_numpy(dtype=float)
    test["ga_shrunk"] = np.where(
        n > 0,
        (n * gr + K_SHRINK * pr) / (n + K_SHRINK),
        np.nan
    )
    test["ga_shrinkage_delta"] = test["ga_shrunk"] - test["ga_rate"]

    for _, r in test.iterrows():
        rows.append({
            "temporada": t, "rodada": R,
            "atleta_id": int(r["atleta_id"]),
            "posicao_id": int(r["posicao_id"]),
            "ga_rate": r["ga_rate"],
            "hist_n": int(r["hist_n"]),
            "prior_ga_pos": r["posicao_id_prior_ga"],
            "ga_shrunk": r["ga_shrunk"],
            "ga_shrinkage_delta": r["ga_shrinkage_delta"],
        })

out = pd.DataFrame(rows)
out.to_parquet(OUT, index=False)
print(f"salvo: {OUT.relative_to(ROOT)}")
print(f"linhas: {len(out)}")
print(f"com ga_shrunk valido: {out['ga_shrunk'].notna().sum()}")
