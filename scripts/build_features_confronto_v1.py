#!/usr/bin/env python3
"""Features de confronto v1 — p_home/p_draw/p_away + mando + favoritismo."""
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
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

print("== carregando ==")
matches = pd.read_parquet(INTERIM/"matches_odds_cartola_v1.parquet")
cart = pd.read_parquet(INTERIM/"cartola_jogador_rodada_v2.parquet")
print(f"matches: {len(matches)}  cartola: {len(cart)}")

# hash do parquet canonico
H_PARQUET_ANTES = hashlib.sha256(
    open(INTERIM/"cartola_jogador_rodada_v2.parquet","rb").read()).hexdigest()

# ---------------------------------------------- probabilidades
print("\n== calculando probabilidades ==")
m = matches.copy()
m["q_home"] = 1.0 / m["odds_home"]
m["q_draw"] = 1.0 / m["odds_draw"]
m["q_away"] = 1.0 / m["odds_away"]
m["overround"] = m["q_home"] + m["q_draw"] + m["q_away"]
m["overround_pct"] = (m["overround"] - 1) * 100
m["p_home"] = m["q_home"] / m["overround"]
m["p_draw"] = m["q_draw"] / m["overround"]
m["p_away"] = m["q_away"] / m["overround"]
m["favoritismo_home_away"] = np.log(m["p_home"] / m["p_away"])

# sanity
print(f"overround min={m['overround'].min():.4f}  max={m['overround'].max():.4f}")
print(f"overround medio={m['overround'].mean():.4f}")
print(f"p_home+p_draw+p_away max erro={abs(m['p_home']+m['p_draw']+m['p_away']-1).max():.2e}")

# ---------------------------------------------- lookup por clube
print("\n== construindo lookup por clube ==")
lookup = {}
conflitos = []
for _, r in m.iterrows():
    key_h = (int(r["temporada"]), int(r["rodada"]), r["home_team_cartola"])
    key_a = (int(r["temporada"]), int(r["rodada"]), r["away_team_cartola"])
    val_h = {"p_home": r["p_home"], "p_draw": r["p_draw"], "p_away": r["p_away"],
             "mando": 1, "favoritismo_home_away": r["favoritismo_home_away"]}
    val_a = {"p_home": r["p_home"], "p_draw": r["p_draw"], "p_away": r["p_away"],
             "mando": 0, "favoritismo_home_away": r["favoritismo_home_away"]}
    if key_h in lookup: conflitos.append(("home", key_h))
    if key_a in lookup: conflitos.append(("away", key_a))
    lookup[key_h] = val_h
    lookup[key_a] = val_a

print(f"chaves: {len(lookup)}  conflitos: {len(conflitos)}")
if conflitos:
    print("CONFLITOS:")
    for c in conflitos[:10]: print(f"  {c}")

# ---------------------------------------------- aplicar no Cartola
print("\n== aplicando no Cartola ==")
cart_feat = cart[["temporada","rodada","atleta_id","clube_nome"]].copy()
cart_feat["clube_canon"] = cart_feat["clube_nome"].map(CARTOLA_TO_CANON)
sem_map = cart_feat[cart_feat["clube_canon"].isna()]["clube_nome"].unique()
print(f"clubes sem mapa: {list(sem_map)}")

keys = list(zip(cart_feat["temporada"], cart_feat["rodada"], cart_feat["clube_canon"]))
cart_feat["p_home"] = [lookup.get(k, {}).get("p_home", np.nan) for k in keys]
cart_feat["p_draw"] = [lookup.get(k, {}).get("p_draw", np.nan) for k in keys]
cart_feat["p_away"] = [lookup.get(k, {}).get("p_away", np.nan) for k in keys]
cart_feat["mando"]  = [lookup.get(k, {}).get("mando", np.nan) for k in keys]
cart_feat["favoritismo_home_away"] = [lookup.get(k, {}).get("favoritismo_home_away", np.nan) for k in keys]

cart_feat["tem_odds"] = cart_feat["p_home"].notna().astype(int)
n_com = int(cart_feat["tem_odds"].sum())
n_sem = int(len(cart_feat) - n_com)
print(f"com odds: {n_com}  sem odds: {n_sem}  ({100*n_com/len(cart_feat):.2f}%)")

if n_sem > 0:
    faltando = cart_feat[cart_feat["tem_odds"]==0].groupby(["temporada","rodada"]).size()
    print("\nrodadas com jogadores sem odds (top 20):")
    print(faltando.head(20).to_string())

# ---------------------------------------------- saida
out = cart_feat[["temporada","rodada","atleta_id","clube_canon",
                 "p_home","p_draw","p_away","mando",
                 "favoritismo_home_away","tem_odds"]].copy()
out.to_parquet(INTERIM/"features_confronto_v1.parquet", index=False)
print(f"\nsalvo features_confronto_v1.parquet  shape={out.shape}")

# checagens de sanidade
print("\n== sanidade ==")
ok = out[out["tem_odds"]==1]
soma = ok["p_home"] + ok["p_draw"] + ok["p_away"]
print(f"soma p_home+p_draw+p_away: min={soma.min():.6f}  max={soma.max():.6f}")
print(f"mando unico por atleta+rodada: {out.groupby(['temporada','rodada','atleta_id'])['mando'].nunique().max()}")
print(f"mando min={out['mando'].min()}  max={out['mando'].max()}")
print(f"favoritismo min={out['favoritismo_home_away'].min():.4f}  max={out['favoritismo_home_away'].max():.4f}")

# hash canonico
H_PARQUET_DEPOIS = hashlib.sha256(
    open(INTERIM/"cartola_jogador_rodada_v2.parquet","rb").read()).hexdigest()
print(f"\nParquet canonico inalterado: {H_PARQUET_ANTES == H_PARQUET_DEPOIS}")

# ---------------------------------------------- relatorio
L = []
L.append("# Confronto v1 — features construídas\n\n")
L.append(f"- Gerado em: {datetime.now().isoformat(timespec='seconds')}\n")
L.append(f"- Fonte: Football-Data `AvgCH/AvgCD/AvgCA` (closing odds)\n")
L.append(f"- Fontes auxiliares: Bzzoiro (rodada), Cartola (clube_nome)\n\n")
L.append("## Cobertura\n\n")
L.append(f"- Cartola total: {len(cart_feat)}\n")
L.append(f"- Com odds: {n_com} ({100*n_com/len(cart_feat):.2f}%)\n")
L.append(f"- Sem odds: {n_sem}\n\n")
L.append("## Transformações\n\n")
L.append("- `q_x = 1/odds_x`\n")
L.append("- `p_x = q_x / (q_home+q_draw+q_away)`\n")
L.append("- `favoritismo_home_away = log(p_home/p_away)`\n")
L.append("- `mando = 1` se mandante, `0` se visitante\n\n")
L.append("## Overround\n\n")
L.append(f"- min={m['overround'].min():.4f}\n")
L.append(f"- mediana={m['overround'].median():.4f}\n")
L.append(f"- media={m['overround'].mean():.4f}\n")
L.append(f"- max={m['overround'].max():.4f}\n\n")
L.append("## Parquet canônico\n\n")
L.append(f"- SHA-256 inalterado: {H_PARQUET_ANTES == H_PARQUET_DEPOIS}\n")
L.append(f"- hash: {H_PARQUET_ANTES}\n")

(REP/"confronto_v1_summary.md").write_text("".join(L), encoding="utf-8")
print("salvo confronto_v1_summary.md")
