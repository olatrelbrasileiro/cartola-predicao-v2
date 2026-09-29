#!/usr/bin/env python3
"""Verificador v4 — regra de adiamento por delta interno da rodada.

Não depende de cutoff pesquisado nem de heurística de snapshot FD.
Classifica cada partida por quanto tempo rodou após data_min da rodada.
"""
from pathlib import Path
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

LIMITE_ADIAMENTO_DIAS = 7

matches = pd.read_parquet(INTERIM/"matches_odds_cartola_v1.parquet")
matches["data"] = pd.to_datetime(matches["data"], errors="coerce")

# data_min por rodada
data_min = matches.groupby(["temporada","rodada"])["data"].min().reset_index()
data_min = data_min.rename(columns={"data":"data_min_rodada"})

matches = matches.merge(data_min, on=["temporada","rodada"], how="left")
matches["delta_dias"] = (matches["data"] - matches["data_min_rodada"]).dt.days

def classifica(row):
    if pd.isna(row["delta_dias"]):
        return "C_sem_data"
    if row["delta_dias"] <= LIMITE_ADIAMENTO_DIAS:
        return "NORMAL"
    return "ADIADA"

matches["tipo_match"] = matches.apply(classifica, axis=1)

# agregar por rodada
agg = matches.groupby(["temporada","rodada"]).agg(
    n_matches=("bzzoiro_event_id","nunique"),
    n_normal=("tipo_match", lambda s: (s == "NORMAL").sum()),
    n_adiada=("tipo_match", lambda s: (s == "ADIADA").sum()),
    max_delta_dias=("delta_dias","max"),
).reset_index()

def status_rodada(row):
    if row["n_adiada"] == 0:
        return "TODAS_NORMAIS"
    if row["n_adiada"] == row["n_matches"]:
        return "TODAS_ADIADAS"
    return "PARCIALMENTE_ADIADA"

agg["status_rodada"] = agg.apply(status_rodada, axis=1)

matches[["temporada","rodada","data","home_team_cartola","away_team_cartola",
         "data_min_rodada","delta_dias","tipo_match"]].to_csv(
    REP/"auditoria_cutoff_confronto_matches.csv", index=False)
agg.to_csv(REP/"auditoria_cutoff_confronto_rodadas.csv", index=False)

print(f"partidas: {len(matches)}")
print()
print("== por tipo ==")
print(matches["tipo_match"].value_counts().to_string())
print()
print("== rodadas por status ==")
print(agg["status_rodada"].value_counts().to_string())
print()
print("== rodadas com adiamento ==")
adiadas = agg[agg["status_rodada"] != "TODAS_NORMAIS"]
if len(adiadas):
    print(adiadas.sort_values(["temporada","rodada"]).to_string(index=False))
else:
    print("  nenhuma")

# relatorio
L = []
L.append("# Auditoria de temporalidade — Confronto v1 (v4 delta-interno)\n\n")
L.append(f"- Limite de adiamento: > {LIMITE_ADIAMENTO_DIAS} dias após data_min da rodada.\n\n")
L.append("## Distribuicao por tipo\n\n")
for k, v in matches["tipo_match"].value_counts().items():
    L.append(f"- {k}: {v}\n")
L.append("\n## Rodadas por status\n\n")
for k, v in agg["status_rodada"].value_counts().items():
    L.append(f"- {k}: {v}\n")
L.append("\n## Rodadas com adiamento\n\n")
if len(adiadas):
    L.append(adiadas.sort_values(["temporada","rodada"]).to_string(index=False))
else:
    L.append("_Nenhuma._\n")

(REP/"auditoria_cutoff_confronto_v1.md").write_text("".join(L), encoding="utf-8")
print(f"\nrelatorio: reports/auditoria_cutoff_confronto_v1.md")
