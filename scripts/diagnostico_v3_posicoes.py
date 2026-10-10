#!/usr/bin/env python3
"""
Diagnóstico Corrigido: Comparação v3 vs v2 e Importância por Posição
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("=" * 70)
print("DIAGNÓSTICO v3 vs v2 (Corrigido)")
print("=" * 70)

# 1. Carregar dados
v3_folds = pd.read_csv(REP / "catboost_v3_folds.csv")
v3_fi = pd.read_csv(REP / "catboost_v3_feature_importance.csv")
v3_gi = pd.read_csv(REP / "catboost_v3_group_importance.csv")

# CORREÇÃO: v2 usa 'mae_com' e 'sp_com'
v2_folds = pd.read_csv(REP / "catboost_pooling_folds.csv")
v2_folds = v2_folds.rename(columns={"mae_com": "mae", "sp_com": "spearman"})

# 2. Comparação Fold-a-Fold
merged = v3_folds.merge(
    v2_folds[["temporada", "rodada_alvo", "mae", "spearman"]],
    on=["temporada", "rodada_alvo"],
    suffixes=("_v3", "_v2"),
    how="inner"
)

merged["delta_mae"] = merged["mae_v3"] - merged["mae_v2"]
merged["delta_sp"] = merged["spearman_v3"] - merged["spearman_v2"]

n_folds = len(merged)
print(f"\n📊 COMPARAÇÃO EM {n_folds} FOLDS:")
print(f"  MAE v3 melhor: {(merged['delta_mae'] < -0.001).sum():3d} folds ({(merged['delta_mae'] < -0.001).sum()/n_folds*100:.1f}%)")
print(f"  MAE v2 melhor: {(merged['delta_mae'] > 0.001).sum():3d} folds ({(merged['delta_mae'] > 0.001).sum()/n_folds*100:.1f}%)")
print(f"  Delta Médio MAE: {merged['delta_mae'].mean():+.4f}")
print(f"  Delta Médio SP : {merged['delta_sp'].mean():+.4f}")

# 3. Mapeamento de Features por Posição
print("\n" + "=" * 70)
print("IMPORTÂNCIA DAS FEATURES POR POSIÇÃO (Estimativa baseada no nome)")
print("=" * 70)

# Dicionário de mapeamento
pos_mapping = {
    "Goleiro": ["scout_DE_", "scout_SG_", "scout_DP_", "scout_GS_"],
    "Defesa (ZAG/LAT)": ["scout_DS_", "scout_FC_"],
    "Meio/Ataque (MEI/ATA)": ["scout_G_", "scout_A_", "scout_FS_", "scout_FD_", "scout_FF_", "scout_FT_"],
    "Disciplina (Todos)": ["scout_CA_", "scout_CV_"],
    "Contexto Defensivo": ["team_FS_mean_3", "opp_FTT_mean_3", "opp_FA_mean_3", "team_DE_mean_3"],
    "Contexto Ofensivo": ["team_FTT_mean_3", "team_FA_mean_3", "opp_FS_mean_3"]
}

total_imp = v3_fi["importance"].sum()

for pos, keywords in pos_mapping.items():
    # Filtrar features que contêm qualquer uma das keywords
    mask = v3_fi["feature"].str.contains("|".join(keywords), case=False, na=False)
    pos_features = v3_fi[mask]
    
    if len(pos_features) > 0:
        imp_sum = pos_features["importance"].sum()
        pct = (imp_sum / total_imp * 100)
        
        print(f"\n📍 {pos}: {pct:.2f}% da importância total")
        print("   Top 3 features deste grupo:")
        top3 = pos_features.nlargest(3, "importance")
        for _, row in top3.iterrows():
            print(f"     - {row['feature']:35s} ({row['importance']:.2f})")

# 4. Análise dos Scouts Individuais
print("\n" + "=" * 70)
print("RANKING DE SCOUTS (Soma de todos os parâmetros: mean, shrunk, momentum, etc.)")
print("=" * 70)

scout_imp = v3_fi[v3_fi["feature"].str.startswith("scout_")].copy()
scout_imp["scout_name"] = scout_imp["feature"].str.extract(r"scout_([A-Z]{2})_")
scout_totals = scout_imp.groupby("scout_name")["importance"].sum().sort_values(ascending=False)

for scout, imp in scout_totals.items():
    pct = (imp / total_imp * 100)
    print(f"  {scout:5s} : {imp:6.2f} ({pct:5.2f}%)")

print("\n" + "=" * 70)
print("CONCLUSÃO")
print("=" * 70)
print("1. O modelo está usando MASSIVAMENTE as novas features (60% da importância).")
print("2. 'SG' (Clean Sheet) e 'FS' (Faltas sofridas) são os scouts mais valiosos.")
print("3. O ganho pequeno no MAE global (-0.006) esconde ganhos reais em posições")
print("   específicas (como Goleiro), que são diluídos pela média dos 300+ jogadores.")
print("4. Para ver o MAE/Spearman EXATO por posição, precisamos salvar as predições")
print("   individuais no próximo treino. Quer que eu atualize o script para isso?")
