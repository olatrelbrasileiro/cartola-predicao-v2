#!/usr/bin/env python3
"""
Análise da Feature Importance do CatBoost v4
Foco: Verificar se os scouts de goleiro corrigidos ganharam relevância.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("=" * 85)
print("ANÁLISE DE FEATURE IMPORTANCE — CATBOOST v4")
print("=" * 85)

# Carregar importância
try:
    fi = pd.read_csv(REP / "catboost_v4_feature_importance.csv")
except FileNotFoundError:
    print("❌ Arquivo catboost_v4_feature_importance.csv não encontrado.")
    exit(1)

# Ordenar por importância
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
total_imp = fi["importance"].sum()

# 1. Top 15 Features Gerais
print("\n🏆 TOP 15 FEATURES MAIS IMPORTANTES (v4)")
print("-" * 85)
print(f"{'Rank':<4} | {'Importância':<12} | {'% do Total':<10} | {'Feature'}")
print("-" * 85)
for i, row in fi.head(15).iterrows():
    pct = (row["importance"] / total_imp) * 100
    print(f"{i+1:<4} | {row['importance']:<12.2f} | {pct:<10.2f}% | {row['feature']}")

# 2. Desempenho Específico dos Scouts de Goleiro
print("\n🧤 DESEMPENHO DOS SCOUTS DE GOLEIRO (DE, SG, DP)")
print("-" * 85)
goalie_scouts = fi[fi["feature"].str.contains(r"scout_(DE|SG|DP)_", regex=True)]

if len(goalie_scouts) > 0:
    goalie_scouts = goalie_scouts.sort_values("importance", ascending=False)
    total_goalie_imp = goalie_scouts["importance"].sum()
    pct_goalie = (total_goalie_imp / total_imp) * 100
    
    print(f"Importância TOTAL combinada de DE, SG, DP: {total_goalie_imp:.2f} ({pct_goalie:.2f}% do modelo)\n")
    print(f"{'Rank Geral':<10} | {'Importância':<12} | {'Feature'}")
    print("-" * 85)
    for _, row in goalie_scouts.iterrows():
        # Encontrar o rank geral desta feature
        rank = fi[fi["feature"] == row["feature"]].index[0] + 1
        print(f"#{rank:<8} | {row['importance']:<12.2f} | {row['feature']}")
else:
    print("❌ Nenhuma feature de goleiro (DE, SG, DP) encontrada na lista.")

# 3. Resumo por Grupo (para ver o panorama geral)
print("\n📊 RESUMO DE IMPORTÂNCIA POR GRUPO")
print("-" * 85)

def get_group(feat):
    if any(x in feat for x in ["hist_", "rec_", "momentum", "basic_", "ga_", "entrou_rate", "is_cold_start", "posicao_id"]):
        return "Base / Histórico"
    elif any(x in feat for x in ["p_home", "p_draw", "p_away", "mando", "favoritismo"]):
        return "Confronto (Odds)"
    elif any(x in feat for x in ["team_", "opp_", "diff_"]):
        return "Contexto Time/Adv"
    elif feat.startswith("scout_"):
        return "Scouts Granulares"
    else:
        return "Outros"

fi["grupo"] = fi["feature"].apply(get_group)
group_summary = fi.groupby("grupo")["importance"].sum().sort_values(ascending=False)

for grupo, imp in group_summary.items():
    pct = (imp / total_imp) * 100
    print(f"{grupo:<25} : {imp:>8.2f}  ({pct:>5.2f}%)")

print("\n" + "=" * 85)
print("CONCLUSÃO RÁPIDA:")
if pct_goalie < 2.0:
    print("⚠️  Os scouts de goleiro (DE, SG, DP) continuam sendo ignorados pelo modelo (< 2%).")
    print("   Isso valida que o v3 (sem esse ruído) pode ser uma escolha mais sólida, ou que")
    print("   precisamos de uma poda agressiva (Feature Selection) no v4.")
else:
    print(f"✅ Os scouts de goleiro ganharam relevância ({pct_goalie:.2f}%). Vale a pena investigar")
    print("   se uma versão podada do v4 (removendo apenas o lixo de baixa importância) supera o v3.")
print("=" * 85)
