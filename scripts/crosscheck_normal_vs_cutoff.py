#!/usr/bin/env python3
"""Cross-check conservador das partidas NORMAL contra o cutoff do Cartola.

Regra: snapshot_fd_estimado = data_partida - 1 dia (conservador).
Se snapshot_fd_estimado >= cutoff -> margem_curta (atencao).
"""
from pathlib import Path
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

matches = pd.read_parquet(INTERIM/"matches_odds_cartola_v1.parquet")
matches["data"] = pd.to_datetime(matches["data"], errors="coerce")

data_min = matches.groupby(["temporada","rodada"])["data"].min().reset_index()
data_min = data_min.rename(columns={"data":"data_min_rodada"})
matches = matches.merge(data_min, on=["temporada","rodada"], how="left")
matches["delta_dias"] = (matches["data"] - matches["data_min_rodada"]).dt.days
matches["tipo_match"] = matches["delta_dias"].apply(
    lambda d: "NORMAL" if pd.notna(d) and d <= 7 else "ADIADA")

cutoff = pd.read_csv(REP/"auditoria_cutoff_confronto_v1.csv")
cutoff["cartola_ts"] = pd.to_datetime(cutoff["cartola_cutoff"], errors="coerce")

normais = matches[matches["tipo_match"] == "NORMAL"].merge(
    cutoff[["temporada","rodada","cartola_ts"]],
    on=["temporada","rodada"], how="left")

normais["snapshot_fd_estimado"] = normais["data"] - pd.Timedelta(days=1)
normais["margem_horas_estimada"] = (
    (normais["cartola_ts"] - normais["snapshot_fd_estimado"]).dt.total_seconds() / 3600
)

def classifica(r):
    if pd.isna(r["cartola_ts"]):
        return "SEM_CUTOFF"
    if r["margem_horas_estimada"] > 12:
        return "OK"
    if r["margem_horas_estimada"] > 3:
        return "ATENCAO_MARGEM"
    return "RISCO"

normais["status_crosscheck"] = normais.apply(classifica, axis=1)

print(f"partidas NORMAL: {len(normais)}")
print()
print("== status ==")
print(normais["status_crosscheck"].value_counts().to_string())
print()
print("== partidas RISCO ==")
risco = normais[normais["status_crosscheck"] == "RISCO"]
if len(risco):
    print(risco[["temporada","rodada","data","home_team_cartola",
                 "away_team_cartola","cartola_ts","snapshot_fd_estimado",
                 "margem_horas_estimada"]].to_string(index=False))
else:
    print("  nenhuma")

normais[[
    "temporada","rodada","data","home_team_cartola","away_team_cartola",
    "cartola_ts","snapshot_fd_estimado","margem_horas_estimada","status_crosscheck"
]].to_csv(REP/"crosscheck_normal_vs_cutoff.csv", index=False)
print()
print(f"salvo: reports/crosscheck_normal_vs_cutoff.csv")
