#!/usr/bin/env python3
"""Template para auditoria de temporalidade do Confronto v1."""
from pathlib import Path
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

matches = pd.read_parquet(INTERIM/"matches_odds_cartola_v1.parquet")
matches["data"] = pd.to_datetime(matches["data"], errors="coerce")

g = matches.groupby(["temporada","rodada"]).agg(
    data_min=("data","min"),
    data_max=("data","max"),
    n_partidas=("bzzoiro_event_id","nunique"),
).reset_index()

g["data_min"] = g["data_min"].dt.strftime("%Y-%m-%d")
g["data_max"] = g["data_max"].dt.strftime("%Y-%m-%d")
g["dia_semana_primeiro_jogo"] = pd.to_datetime(g["data_min"]).dt.dayofweek
g["tipo_rodada"] = ""
g["categoria_evidencia"] = ""
g["cartola_cutoff"] = ""
g["fonte"] = ""
g["evidencia_textual"] = ""
g["confianca"] = ""

g = g[["temporada","rodada","tipo_rodada","data_min","data_max",
       "n_partidas","dia_semana_primeiro_jogo",
       "categoria_evidencia","cartola_cutoff","fonte","evidencia_textual","confianca"]]

out = REP/"auditoria_cutoff_confronto_v1.csv"
g.to_csv(out, index=False)
print(f"template salvo: {out.relative_to(ROOT)}")
print(f"rodadas: {len(g)}")
print(f"colunas: {list(g.columns)}")
