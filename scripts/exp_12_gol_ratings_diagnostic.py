#!/usr/bin/env python3
"""
Diagnóstico: Avaliando os Ratings contra os Scouts Intermediários
Versão Robustecida: Descobre automaticamente as colunas de rating.
"""
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path("/data/data/com.termux/files/home/cartola-predicao-v2")
RATINGS = ROOT / "data/interim/exp12/team_strength_ratings_v3.parquet"
GOL = ROOT / "data/interim/exp12/gol_pooling_2024_2026.parquet"

print("=" * 70)
print("DIAGNÓSTICO: Ratings vs Scouts Intermediários")
print("=" * 70)

# 1. Carregar
ratings = pd.read_parquet(RATINGS)
gol = pd.read_parquet(GOL)

print(f"\n[DEBUG] Colunas em ratings: {list(ratings.columns)}")
print(f"[DEBUG] Total de colunas em gol: {len(gol.columns)}")

# Filtrar elegíveis
if "elegivel_modelo" in gol.columns:
    gol = gol[gol["elegivel_modelo"] == True].copy()
elif "entrou_em_campo" in gol.columns:
    gol = gol[gol["entrou_em_campo"] == True].copy()

# 2. Descobrir automaticamente as colunas de Ataque e Defesa no ratings
att_col = next((c for c in ratings.columns if "attack" in c.lower() or "att" in c.lower()), None)
def_col = next((c for c in ratings.columns if "defense" in c.lower() or "def" in c.lower()), None)

if not att_col or not def_col:
    raise ValueError(f"Não encontrei colunas de ataque/defesa em ratings. Colunas: {list(ratings.columns)}")

print(f"[DEBUG] Coluna de Ataque detectada: {att_col}")
print(f"[DEBUG] Coluna de Defesa detectada: {def_col}")

# 3. Merge do Rating do Próprio Time (Defesa/Solidez)
gol_own = ratings[["temporada", "rodada", "clube_id", att_col, def_col]].copy()
gol_own = gol_own.rename(columns={
    att_col: "att_own",
    def_col: "def_own"
})
gol = gol.merge(gol_own, on=["temporada", "rodada", "clube_id"], how="left")

# 4. Merge do Rating do Adversário (Ataque/Volume)
gol_opp = ratings[["temporada", "rodada", "clube_id", att_col, def_col]].copy()
gol_opp = gol_opp.rename(columns={
    "clube_id": "adversario_id",
    att_col: "att_opp",
    def_col: "def_opp"
})
gol = gol.merge(gol_opp, on=["temporada", "rodada", "adversario_id"], how="left")

print(f"\n[DEBUG] Cobertura após merge:")
print(f"  att_own: {gol['att_own'].notna().mean():.1%}")
print(f"  def_own: {gol['def_own'].notna().mean():.1%}")
print(f"  att_opp: {gol['att_opp'].notna().mean():.1%}")
print(f"  def_opp: {gol['def_opp'].notna().mean():.1%}")

# 5. Descobrir alvos disponíveis (scouts brutos ou médias móveis)
keywords = ["SG", "DE", "GS", "FS", "FTT", "pontos", "goals_conceded"]
available_targets = [col for col in gol.columns if any(k.lower() in col.lower() for k in keywords)]
# Remover as próprias features de rating da lista de alvos
available_targets = [t for t in available_targets if "rating_" not in t and "att_" not in t and "def_" not in t]

print(f"\n[DEBUG] Alvos disponíveis encontrados: {available_targets}")

features = ["att_own", "def_own", "att_opp", "def_opp"]
results = []

for feat in features:
    for tgt in available_targets:
        sub = gol[[feat, tgt]].dropna()
        if len(sub) > 100:
            sp, _ = spearmanr(sub[feat], sub[tgt])
            results.append({
                "feature": feat, 
                "target": tgt, 
                "spearman": round(sp, 4), 
                "n": len(sub)
            })

df_res = pd.DataFrame(results)

if not df_res.empty:
    print("\n=== MATRIZ DE CORRELAÇÃO (Spearman) ===")
    pivot = df_res.pivot(index="target", columns="feature", values="spearman")
    # Ordenar por correlação com def_own (solidez)
    sort_col = "def_own" if "def_own" in pivot.columns else pivot.columns[0]
    pivot = pivot.sort_values(by=sort_col, ascending=False)
    print(pivot.to_string())
else:
    print("\n[ALERTA] Nenhuma correlação pôde ser calculada.")

print("\n" + "=" * 70)
print("INTERPRETAÇÃO:")
print("- Se 'def_own' correlaciona POSITIVAMENTE com SG (ou gk_sg_mean) e NEGATIVAMENTE com GS -> O modelo está PERFEITO fisicamente.")
print("- Se 'att_opp' correlaciona POSITIVAMENTE com FS/FTT/DE -> O modelo captura o volume de trabalho.")
print("- Se as correlações forem muito próximas de 0 -> O modelo Poisson é 'lento' demais e janelas curtas (momentum) são melhores.")
print("=" * 70)
