#!/usr/bin/env python3
"""Constroi o vinculo auditavel atleta x partida x tipo_match.

Para cada (temporada, rodada, atleta_id):
- clube_nome (original)
- clube_canon
- adversario_canon
- mando (1 = casa, 0 = fora)
- bzzoiro_event_id
- tipo_match (NORMAL / ADIADA / SEM_MATCH)
- odds_apos_cutoff (bool)

Nao aplica mascara ainda. Apenas materializa o vinculo para auditoria.
"""
from pathlib import Path
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

CARTOLA_TO_CANON = {
    "FLA":"Flamengo","BOT":"Botafogo","COR":"Corinthians","BAH":"Bahia",
    "FLU":"Fluminense","VAS":"Vasco","PAL":"Palmeiras","SAO":"São Paulo",
    "SAN":"Santos","RBB":"Bragantino","CAM":"Atlético-MG","CRU":"Cruzeiro",
    "GRE":"Grêmio","INT":"Internacional","JUV":"Juventude","VIT":"Vitória",
    "CEA":"Ceará","FOR":"Fortaleza","REM":"Remo","MIR":"Mirassol",
    "SPT":"Sport Recife","CAP":"Athletico-PR","CFC":"Coritiba",
    "CHA":"Chapecoense","CRI":"Criciúma",
    "Flamengo":"Flamengo","Botafogo":"Botafogo","Corinthians":"Corinthians",
    "Bahia":"Bahia","Fluminense":"Fluminense","Vasco":"Vasco",
    "Palmeiras":"Palmeiras","São Paulo":"São Paulo","Bragantino":"Bragantino",
    "Atlético-MG":"Atlético-MG","Cruzeiro":"Cruzeiro","Grêmio":"Grêmio",
    "Internacional":"Internacional","Juventude":"Juventude","Vitória":"Vitória",
    "Criciúma":"Criciúma","Athlético-PR":"Athletico-PR","Atlético-GO":"Atlético-GO",
    "Cuiabá":"Cuiabá","Fortaleza":"Fortaleza","Ceará":"Ceará",
    "Sport Recife":"Sport Recife","Coritiba":"Coritiba","Chapecoense":"Chapecoense",
    "Remo":"Remo","Mirassol":"Mirassol",
}

# --- matches com tipo_match (v4)
matches = pd.read_parquet(INTERIM/"matches_odds_cartola_v1.parquet")
matches["data"] = pd.to_datetime(matches["data"], errors="coerce")

# data_min por rodada
data_min = matches.groupby(["temporada","rodada"])["data"].min().reset_index()
data_min = data_min.rename(columns={"data":"data_min_rodada"})
matches = matches.merge(data_min, on=["temporada","rodada"], how="left")
matches["delta_dias"] = (matches["data"] - matches["data_min_rodada"]).dt.days
matches["tipo_match"] = matches["delta_dias"].apply(
    lambda d: "NORMAL" if pd.notna(d) and d <= 7 else ("ADIADA" if pd.notna(d) else "SEM_DATA"))
matches["odds_apos_cutoff"] = matches["tipo_match"] == "ADIADA"

# --- match_long: duas linhas por match (home e away)
rows = []
for _, m in matches.iterrows():
    common = {
        "temporada": int(m["temporada"]),
        "rodada": int(m["rodada"]),
        "bzzoiro_event_id": int(m["bzzoiro_event_id"]),
        "tipo_match": m["tipo_match"],
        "odds_apos_cutoff": bool(m["odds_apos_cutoff"]),
    }
    rows.append({**common,
                 "clube_canon": m["home_team_cartola"],
                 "adversario_canon": m["away_team_cartola"],
                 "mando": 1})
    rows.append({**common,
                 "clube_canon": m["away_team_cartola"],
                 "adversario_canon": m["home_team_cartola"],
                 "mando": 0})
match_long = pd.DataFrame(rows)

# checar unicidade (temporada, rodada, clube_canon)
dups = match_long.duplicated(subset=["temporada","rodada","clube_canon"]).sum()
print(f"matches longos: {len(match_long)}")
print(f"duplicatas (temporada, rodada, clube_canon): {int(dups)}")
if dups > 0:
    print("ATENCAO: ha clubes com mais de uma partida por rodada. Investigar.")
    dup_rows = match_long[match_long.duplicated(
        subset=["temporada","rodada","clube_canon"], keep=False)]
    print(dup_rows.head(20).to_string(index=False))

# --- cartola: atleta por rodada com clube
cart = pd.read_parquet(INTERIM/"cartola_jogador_rodada_v2.parquet")
cart_aux = cart[["temporada","rodada","atleta_id","clube_nome"]].copy()
cart_aux["clube_canon"] = cart_aux["clube_nome"].map(CARTOLA_TO_CANON)

sem_canon = cart_aux[cart_aux["clube_canon"].isna()]["clube_nome"].unique()
if len(sem_canon):
    print(f"AVISO: clubes sem canon no cartola: {list(sem_canon)}")

# --- join
vinculo = cart_aux.merge(match_long,
                          on=["temporada","rodada","clube_canon"],
                          how="left")

sem_match = vinculo["bzzoiro_event_id"].isna().sum()
vinculo["tipo_match"] = vinculo["tipo_match"].fillna("SEM_MATCH")
vinculo["odds_apos_cutoff"] = vinculo["odds_apos_cutoff"].fillna(False).astype(bool)
vinculo["mando"] = vinculo["mando"]  # mantem NaN quando SEM_MATCH

saida = INTERIM/"vinculo_atleta_partida_v1.parquet"
vinculo[["temporada","rodada","atleta_id","clube_nome","clube_canon",
         "adversario_canon","mando","bzzoiro_event_id",
         "tipo_match","odds_apos_cutoff"]].to_parquet(saida, index=False)

print()
print(f"salvo: {saida.relative_to(ROOT)}")
print(f"linhas: {len(vinculo)}")
print(f"sem match (SEM_MATCH): {int(sem_match)}")
print()
print("== tipo_match ==")
print(vinculo["tipo_match"].value_counts().to_string())
print()
print("== odds_apos_cutoff ==")
print(vinculo["odds_apos_cutoff"].value_counts().to_string())
print()
print("== atletas com mando ambiguo por (temporada, rodada) ==")
g = vinculo.groupby(["temporada","rodada","atleta_id"])["mando"].nunique()
print(f"atletas com >1 mando na mesma rodada: {int((g > 1).sum())}")
