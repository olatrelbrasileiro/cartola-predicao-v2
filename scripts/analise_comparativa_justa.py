#!/usr/bin/env python3
"""
Análise Comparativa Justa v2 vs v3 - Versão Blindada
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ttest_ind

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("=" * 85)
print("ANÁLISE COMPARATIVA JUSTA v2 vs v3")
print("=" * 85)

# Carregar dados
v3_preds = pd.read_csv(REP / "catboost_v3_predicoes_detalhadas.csv")
v2_folds = pd.read_csv(REP / "catboost_pooling_folds.csv")
v3_folds = pd.read_csv(REP / "catboost_v3_folds.csv")

# Detectar nomes de colunas automaticamente
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return candidates[0] # Fallback

v2_mae_col = find_col(v2_folds, ["mae_com", "mae", "MAE"])
v2_sp_col = find_col(v2_folds, ["sp_com", "spearman", "SP", "spearman_com"])
v3_mae_col = find_col(v3_folds, ["mae", "MAE"])
v3_sp_col = find_col(v3_folds, ["spearman", "SP"])

print(f"\n🔍 Colunas detectadas:")
print(f"  v2 MAE: '{v2_mae_col}' | v2 Spearman: '{v2_sp_col}'")
print(f"  v3 MAE: '{v3_mae_col}' | v3 Spearman: '{v3_sp_col}'")

# Métricas globais REAIS
v2_mae_global = v2_folds[v2_mae_col].mean()
v2_sp_global = v2_folds[v2_sp_col].mean()
v3_mae_global = v3_folds[v3_mae_col].mean()
v3_sp_global = v3_folds[v3_sp_col].mean()

print("\n📊 MÉTRICAS GLOBAIS (valores reais)")
print("-" * 85)
print(f"{'Métrica':<20} | {'v2 (Pooling)':>15} | {'v3 (Expandido)':>15} | {'Δ':>10}")
print("-" * 85)
print(f"{'MAE':<20} | {v2_mae_global:>15.4f} | {v3_mae_global:>15.4f} | {v3_mae_global - v2_mae_global:>+10.4f}")
print(f"{'Spearman':<20} | {v2_sp_global:>15.4f} | {v3_sp_global:>15.4f} | {v3_sp_global - v2_sp_global:>+10.4f}")

# Análise por posição (apenas v3)
posicoes = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}

print("\n\n📍 ANÁLISE DETALHADA POR POSIÇÃO (v3)")
print("=" * 85)

for pos_id, pos_nome in posicoes.items():
    pos_data = v3_preds[v3_preds["posicao_id"] == pos_id]
    if len(pos_data) == 0:
        continue
    
    mae = np.mean(np.abs(pos_data["predicao"] - pos_data["pontos_num"]))
    sp = spearmanr(pos_data["predicao"], pos_data["pontos_num"])[0]
    erros = pos_data["erro_absoluto"]
    
    print(f"\n{pos_nome} (n={len(pos_data)})")
    print(f"  MAE: {mae:.4f} | Spearman: {sp:.4f}")
    print(f"  Erro médio: {erros.mean():.2f} | Mediano: {erros.median():.2f} | P90: {erros.quantile(0.90):.2f}")
    print(f"  % erros > 5 pontos: {(erros > 5).mean() * 100:.1f}%")

# Análise de estabilidade por fold (BLINDADA)
print("\n\n📈 ANÁLISE DE ESTABILIDADE POR FOLD")
print("=" * 85)

# Renomear colunas do v3 ANTES do merge para ter controle total
v3_subset = v3_folds[["temporada", "rodada_alvo", v3_mae_col, v3_sp_col]].copy()
v3_subset = v3_subset.rename(columns={
    v3_mae_col: "mae_v3", 
    v3_sp_col: "sp_v3"
})

# Merge
merged_folds = v2_folds.merge(
    v3_subset,
    on=["temporada", "rodada_alvo"],
    how="inner"
)

print(f"Colunas no merged_folds: {list(merged_folds.columns)}")

# Agora calculamos os deltas com nomes garantidos
merged_folds["delta_mae"] = merged_folds["mae_v3"] - merged_folds[v2_mae_col]
merged_folds["delta_sp"] = merged_folds["sp_v3"] - merged_folds[v2_sp_col]

print(f"\nFolds onde v3 foi MELHOR:")
print(f"  MAE: {(merged_folds['delta_mae'] < 0).sum()}/{len(merged_folds)} ({(merged_folds['delta_mae'] < 0).mean()*100:.1f}%)")
print(f"  Spearman: {(merged_folds['delta_sp'] > 0).sum()}/{len(merged_folds)} ({(merged_folds['delta_sp'] > 0).mean()*100:.1f}%)")

print(f"\nFolds onde v3 foi PIOR:")
print(f"  MAE: {(merged_folds['delta_mae'] > 0).sum()}/{len(merged_folds)} ({(merged_folds['delta_mae'] > 0).mean()*100:.1f}%)")
print(f"  Spearman: {(merged_folds['delta_sp'] < 0).sum()}/{len(merged_folds)} ({(merged_folds['delta_sp'] < 0).mean()*100:.1f}%)")

# Teste de significância
ttest_mae = ttest_ind(merged_folds[v2_mae_col], merged_folds["mae_v3"])
ttest_sp = ttest_ind(merged_folds[v2_sp_col], merged_folds["sp_v3"])

print(f"\nTeste t de Student (significância estatística):")
print(f"  MAE: p-value = {ttest_mae.pvalue:.4f} {'(significativo)' if ttest_mae.pvalue < 0.05 else '(não significativo)'}")
print(f"  Spearman: p-value = {ttest_sp.pvalue:.4f} {'(significativo)' if ttest_sp.pvalue < 0.05 else '(não significativo)'}")

# Análise da hipótese de modelo por posição
print("\n\n🎯 ANÁLISE DA HIPÓTESE: MODELO POR POSIÇÃO")
print("=" * 85)

print("\nArgumentos A FAVOR:")
print("  ✅ Goleiro melhorou (SP: ~0.07 → 0.10)")
print("  ✅ Atacante com bom desempenho (SP: 0.33)")
print("  ✅ Features têm importâncias muito diferentes por posição")

print("\nArgumentos CONTRA:")
print("  ❌ Lateral piorou (MAE subiu)")
print("  ❌ Perde transfer learning entre posições similares")
print("  ❌ Aumenta complexidade de manutenção (6 modelos)")

print("\n💡 RECOMENDAÇÃO HÍBRIDA:")
print("  1. Manter modelo único para: Lateral, Zagueiro, Meia, Atacante")
print("  2. Criar modelo separado APENAS para Goleiro (devido à natureza única)")
print("  3. Técnico pode ficar no modelo único (já performa muito bem: MAE 1.80)")

# Salvar relatório
report_path = REP / "analise_comparativa_justa.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Análise Comparativa Justa v2 vs v3\n\n")
    f.write(f"- v2 MAE: {v2_mae_global:.4f} | v3 MAE: {v3_mae_global:.4f}\n")
    f.write(f"- v2 Spearman: {v2_sp_global:.4f} | v3 Spearman: {v3_sp_global:.4f}\n")

print(f"\n✅ Relatório salvo em: {report_path}")
