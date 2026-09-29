#!/usr/bin/env python3
"""Aplica mascara de adiamento nas features de confronto.

Deriva diretamente do vinculo_atleta_partida_v1.parquet (auditavel).
Nao sobrescreve features_confronto_v1.parquet.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"

feats = pd.read_parquet(INTERIM/"features_confronto_v1.parquet")
vinculo = pd.read_parquet(INTERIM/"vinculo_atleta_partida_v1.parquet")

# O vinculo tem tudo o que precisamos
feats2 = feats.merge(
    vinculo[["temporada","rodada","atleta_id","tipo_match","odds_apos_cutoff"]],
    on=["temporada","rodada","atleta_id"], how="left"
)

mask = feats2["odds_apos_cutoff"].fillna(False)
print(f"linhas mascaradas: {int(mask.sum())}")

for c in ["p_home","p_draw","p_away","favoritismo_home_away"]:
    feats2.loc[mask, c] = np.nan

# mando preservado automaticamente
saida = INTERIM/"features_confronto_v1_masked.parquet"
feats2.to_parquet(saida, index=False)

print(f"salvo: {saida.relative_to(ROOT)}")
print(f"linhas: {len(feats2)}")
print(f"p_home NaN: {int(feats2['p_home'].isna().sum())}")
print(f"mando NaN: {int(feats2['mando'].isna().sum())}")
print(f"tipo_match NORMAL: {int((feats2['tipo_match']=='NORMAL').sum())}")
print(f"tipo_match ADIADA: {int((feats2['tipo_match']=='ADIADA').sum())}")
print(f"tipo_match SEM_MATCH: {int((feats2['tipo_match']=='SEM_MATCH').sum())}")
