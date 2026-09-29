#!/usr/bin/env python3
"""Matching Football-Data odds x Bzzoiro events x rodadas do Cartola."""
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
RAW_ODDS = ROOT/"data"/"raw"/"odds"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

# ---------------------------------------------- mapa explicito de nomes
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
    "Palmeiras":"Palmeiras","São Paulo":"São Paulo",
    "Bragantino":"Bragantino","Atlético-MG":"Atlético-MG","Cruzeiro":"Cruzeiro",
    "Grêmio":"Grêmio","Internacional":"Internacional","Juventude":"Juventude",
    "Vitória":"Vitória","Criciúma":"Criciúma","Athlético-PR":"Athletico-PR",
    "Atlético-GO":"Atlético-GO","Cuiabá":"Cuiabá",
}
FD_TO_CANON = {
    "Flamengo RJ":"Flamengo","Botafogo RJ":"Botafogo","Corinthians":"Corinthians",
    "Bahia":"Bahia","Fluminense":"Fluminense","Vasco":"Vasco",
    "Palmeiras":"Palmeiras","Sao Paulo":"São Paulo","Santos":"Santos",
    "Bragantino":"Bragantino","Atletico-MG":"Atlético-MG","Cruzeiro":"Cruzeiro",
    "Gremio":"Grêmio","Internacional":"Internacional","Juventude":"Juventude",
    "Vitoria":"Vitória","Criciuma":"Criciúma","Sport Recife":"Sport Recife",
    "Athletico-PR":"Athletico-PR","Chapecoense-SC":"Chapecoense",
    "Ceara":"Ceará","Fortaleza":"Fortaleza","Remo":"Remo",
    "Atletico GO":"Atlético-GO","Cuiaba":"Cuiabá","Mirassol":"Mirassol",
    "Coritiba":"Coritiba",
}
BZ_TO_CANON = {
    "Flamengo":"Flamengo","Botafogo":"Botafogo","Corinthians":"Corinthians",
    "Bahia":"Bahia","Fluminense":"Fluminense","Vasco da Gama":"Vasco",
    "Palmeiras":"Palmeiras","São Paulo":"São Paulo","Santos":"Santos",
    "Red Bull Bragantino":"Bragantino","Atlético Mineiro":"Atlético-MG",
    "Cruzeiro":"Cruzeiro","Grêmio":"Grêmio","Internacional":"Internacional",
    "Juventude":"Juventude","Vitória":"Vitória","Criciúma":"Criciúma",
    "Sport Recife":"Sport Recife","Athletico":"Athletico-PR",
    "Chapecoense":"Chapecoense","Ceará":"Ceará","Fortaleza":"Fortaleza",
    "Remo":"Remo","Atlético Goianiense":"Atlético-GO","Cuiabá":"Cuiabá",
    "Mirassol":"Mirassol","Coritiba":"Coritiba",
}

# ---------------------------------------------- carregar
print("== carregando ==")
fd = pd.read_csv(RAW_ODDS/"football_data_bra.csv", encoding="utf-8")
fd = fd[fd["Season"].isin([2024,2025,2026])].copy()
bz = pd.read_parquet(RAW_ODDS/"bzzoiro_events_2024_2026.parquet")
cart = pd.read_parquet(INTERIM/"cartola_jogador_rodada_v2.parquet")

print(f"FD: {len(fd)}  BZ: {len(bz)}")

fd["year"] = fd["Season"]
fd["home_canon"] = fd["Home"].map(FD_TO_CANON)
fd["away_canon"] = fd["Away"].map(FD_TO_CANON)
fd_nm = sorted(set(fd[fd["home_canon"].isna()]["Home"]) |
               set(fd[fd["away_canon"].isna()]["Away"]))
print(f"FD sem mapa: {fd_nm}")

bz["year"] = bz["ano_fd"]
bz["home_canon"] = bz["home_team"].map(BZ_TO_CANON)
bz["away_canon"] = bz["away_team"].map(BZ_TO_CANON)
bz_nm = sorted(set(bz[bz["home_canon"].isna()]["home_team"]) |
               set(bz[bz["away_canon"].isna()]["away_team"]))
print(f"BZ sem mapa: {bz_nm}")

# ---------------------------------------------- match
keys = ["year","home_canon","away_canon"]
print(f"\ndup FD nas chaves: {fd.duplicated(subset=keys).sum()}")
print(f"dup BZ nas chaves: {bz.duplicated(subset=keys).sum()}")

fd_k = fd[keys + ["Date","Time","Home","Away","AvgCH","AvgCD","AvgCA"]].copy()
bz_k = bz[keys + ["id","event_date","status","round_number"]].copy()

m = fd_k.merge(bz_k, on=keys, how="outer", indicator=True)
print("\nmerge:")
print(m["_merge"].value_counts().to_string())
so_fd = m[m["_merge"]=="left_only"]
so_bz = m[m["_merge"]=="right_only"]
ambos = m[m["_merge"]=="both"].copy()
print(f"SO_FD: {len(so_fd)}  SO_BZ: {len(so_bz)}  BOTH: {len(ambos)}")

if len(so_fd):
    print("\nExemplos SO_FD:")
    print(so_fd[keys + ["Date","Home","Away"]].head(8).to_string(index=False))
if len(so_bz):
    print("\nExemplos SO_BZ:")
    print(so_bz[keys + ["event_date","status","round_number","id"]].head(8).to_string(index=False))

# ---------------------------------------------- checagem com Cartola
cart_rounds = (cart.groupby(["temporada","rodada"])
                    .agg(n_clubes=("clube_id","nunique"),
                         clubes=("clube_nome", lambda x: set(x.unique())))
                    .reset_index())
cart_rounds["clubes_canon"] = cart_rounds["clubes"].apply(
    lambda s: frozenset(CARTOLA_TO_CANON.get(c, c) for c in s))
cart_lookup = cart_rounds.set_index(["temporada","rodada"])["clubes_canon"].to_dict()

ambos["temporada"] = ambos["year"]
ambos["rodada"] = ambos["round_number"]
ambos["data"] = pd.to_datetime(ambos["Date"], format="%d/%m/%Y", errors="coerce")
ambos["data_bz"] = pd.to_datetime(ambos["event_date"], errors="coerce", utc=True).dt.tz_convert(None)
ambos["date_diff_days"] = (ambos["data"].dt.normalize() - ambos["data_bz"].dt.normalize()).dt.days

def checa(row):
    clubes = cart_lookup.get((row["temporada"], row["rodada"]))
    if clubes is None: return "rodada_ausente_cartola"
    h = row["home_canon"] in clubes
    a = row["away_canon"] in clubes
    if h and a: return "ok"
    if h or a: return "parcial"
    return "fora"

ambos["checagem_cartola"] = ambos.apply(checa, axis=1)
print("\nchecagem_cartola:")
print(ambos["checagem_cartola"].value_counts().to_string())

print("\ndate_diff_days (FD - BZ):")
print(ambos["date_diff_days"].value_counts(dropna=False).sort_index().to_string())

# ---------------------------------------------- saida
out = ambos[[
    "temporada","rodada","data",
    "Home","Away","home_canon","away_canon",
    "AvgCH","AvgCD","AvgCA",
    "id","event_date","status",
    "checagem_cartola","date_diff_days"
]].rename(columns={
    "Home":"home_team_original","Away":"away_team_original",
    "home_canon":"home_team_cartola","away_canon":"away_team_cartola",
    "AvgCH":"odds_home","AvgCD":"odds_draw","AvgCA":"odds_away",
    "id":"bzzoiro_event_id","event_date":"bzzoiro_event_date",
    "status":"bzzoiro_status",
})
out["source"] = "football_data_bra.csv + bzzoiro"
out.to_parquet(INTERIM/"matches_odds_cartola_v1.parquet", index=False)
print(f"\nsalvo matches_odds_cartola_v1.parquet  shape={out.shape}")

# ---------------------------------------------- cobertura
todos = cart_rounds[["temporada","rodada","n_clubes"]].copy()
todos["n_partidas_cartola"] = (todos["n_clubes"] // 2).astype(int)
cob = out[out["checagem_cartola"]=="ok"].groupby(["temporada","rodada"]).size().reset_index(name="n_matching")
cob = todos.merge(cob, on=["temporada","rodada"], how="left").fillna(0)
cob["n_matching"] = cob["n_matching"].astype(int)
cob["cobertura"] = (cob["n_matching"] / cob["n_partidas_cartola"]).round(4)
cob.to_csv(REP/"cobertura_odds_por_rodada.csv", index=False)
print("\ncobertura por temporada:")
print(cob.groupby("temporada")["cobertura"].describe().to_string())

# ---------------------------------------------- relatorio
L = []
L.append("# Matching Football-Data x Bzzoiro x Cartola\n\n")
L.append(f"- Gerado em: {datetime.now().isoformat(timespec='seconds')}\n")
L.append(f"- FD partidas 2024-2026: {len(fd)}\n")
L.append(f"- BZ eventos 2024-2026: {len(bz)}\n")
L.append(f"- BOTH: {len(ambos)}\n")
L.append(f"- SO_FD: {len(so_fd)}\n")
L.append(f"- SO_BZ: {len(so_bz)}\n\n")
L.append("## checagem_cartola\n\n")
L.append(ambos["checagem_cartola"].value_counts().to_string() + "\n\n")
L.append("## date_diff_days\n\n")
L.append(ambos["date_diff_days"].value_counts(dropna=False).sort_index().to_string() + "\n\n")
L.append("## Clubes sem mapeamento\n\n")
L.append(f"- FD: {fd_nm}\n")
L.append(f"- BZ: {bz_nm}\n\n")
L.append("## Cobertura por temporada\n\n")
L.append(cob.groupby("temporada")["cobertura"].describe().to_string() + "\n")

(REP/"match_odds_cartola_auditoria.md").write_text("".join(L), encoding="utf-8")
print("salvo match_odds_cartola_auditoria.md")
