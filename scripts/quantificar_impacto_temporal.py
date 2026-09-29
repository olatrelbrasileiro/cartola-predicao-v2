#!/usr/bin/env python3
"""Quantifica o impacto da politica temporal conservadora sobre o Confronto v1.

NAO gera artefato. Apenas mede.
"""
from datetime import timedelta
from pathlib import Path
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"
CSV_CUTOFF = REP/"auditoria_cutoff_confronto_v1.csv"


def fd_ceiling_utc(match_date):
    """Janela FD em UTC, versao ceiling (mais tardia possivel)."""
    if pd.isna(match_date):
        return pd.NaT
    wd = match_date.weekday()
    d = match_date.normalize()
    if wd in (5, 6):   # sab, dom -> sexta anterior, 16:00 UTC
        return (d - timedelta(days=wd - 4)).replace(hour=16, minute=0)
    if wd in (1, 2, 3):  # ter, qua, qui -> terca anterior, 12:00 UTC
        return (d - timedelta(days=wd - 1)).replace(hour=12, minute=0)
    return pd.NaT


def cutoff_brt_para_utc(ts_str):
    if pd.isna(ts_str) or ts_str == "":
        return pd.NaT
    try:
        t = pd.to_datetime(ts_str)
        return t + timedelta(hours=3)   # BRT -> UTC
    except Exception:
        return pd.NaT


# carregar
matches = pd.read_parquet(INTERIM/"matches_odds_cartola_v1.parquet")
matches["data"] = pd.to_datetime(matches["data"], errors="coerce")

dmin = matches.groupby(["temporada","rodada"])["data"].min().reset_index()
dmin = dmin.rename(columns={"data":"data_min_rodada"})
matches = matches.merge(dmin, on=["temporada","rodada"], how="left")
matches["delta_dias"] = (matches["data"] - matches["data_min_rodada"]).dt.days

cutoff = pd.read_csv(CSV_CUTOFF)
cutoff["cutoff_utc"] = cutoff["cartola_cutoff"].apply(cutoff_brt_para_utc)
matches = matches.merge(
    cutoff[["temporada","rodada","cutoff_utc"]],
    on=["temporada","rodada"], how="left")

matches["fd_ceiling_utc"] = matches["data"].apply(fd_ceiling_utc)


def classifica(r):
    if pd.notna(r["delta_dias"]) and r["delta_dias"] > 7:
        return "ADIADA"
    if pd.isna(r["cutoff_utc"]):
        return "SEM_CUTOFF"
    if pd.isna(r["fd_ceiling_utc"]):
        return "FD_INDETERMINADO"
    if r["fd_ceiling_utc"] >= r["cutoff_utc"]:
        return "RISCO_CUTOFF"
    return "COMPATIVEL_JANELA_FD"


matches["classificacao"] = matches.apply(classifica, axis=1)

# contagens
print(f"total partidas: {len(matches)}")
print()
print("== classificacao por partida ==")
print(matches["classificacao"].value_counts().to_string())
print()

# por temporada
print("== classificacao por temporada ==")
tab = matches.groupby(["temporada","classificacao"]).size().unstack(fill_value=0)
print(tab.to_string())
print()

# por rodada — quais rodadas tem pelo menos 1 RISCO_CUTOFF?
rod = matches.groupby(["temporada","rodada"]).agg(
    n=("bzzoiro_event_id","nunique"),
    n_risco=("classificacao", lambda s: (s == "RISCO_CUTOFF").sum()),
    n_adiada=("classificacao", lambda s: (s == "ADIADA").sum()),
    n_ok=("classificacao", lambda s: (s == "COMPATIVEL_JANELA_FD").sum()),
).reset_index()
rod["status"] = rod.apply(
    lambda r: "TODAS_OK" if r["n_ok"] == r["n"]
              else ("TODAS_PROBLEMA" if r["n_ok"] == 0 else "PARCIAL"), axis=1)
print("== rodadas por status ==")
print(rod["status"].value_counts().to_string())
print()
print("== rodadas com pelo menos 1 RISCO_CUTOFF ==")
com_risco = rod[rod["n_risco"] > 0]
print(com_risco.to_string(index=False))

# metricas de perda
n_total = len(matches)
n_perdidas_rigido = int((matches["classificacao"] != "COMPATIVEL_JANELA_FD").sum())
print()
print(f"== impacto da politica conservadora estrita ==")
print(f"partidas mantidas (COMPATIVEL_JANELA_FD): {n_total - n_perdidas_rigido}")
print(f"partidas mascaradas: {n_perdidas_rigido}")
print(f"cobertura: {100*(n_total - n_perdidas_rigido)/n_total:.2f}%")

# salvar
matches[["temporada","rodada","data","home_team_cartola","away_team_cartola",
         "cutoff_utc","fd_ceiling_utc","delta_dias","classificacao"]].to_csv(
    REP/"quantificacao_temporal_confronto.csv", index=False)
rod.to_csv(REP/"quantificacao_temporal_por_rodada.csv", index=False)
print()
print("salvos:")
print(f"  reports/quantificacao_temporal_confronto.csv")
print(f"  reports/quantificacao_temporal_por_rodada.csv")
