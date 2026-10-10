#!/usr/bin/env python3
"""
AUDITORIA PROFUNDA DE LEAKAGE — CatBoost v5.1
Investiga se a melhora absurda é real ou se há data leakage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, shapiro
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
INTERIM = ROOT / "data" / "interim"

print("=" * 90)
print("🔍 AUDITORIA PROFUNDA DE LEAKAGE — CATBOOST v5.1")
print("=" * 90)

# 1. Carregar predições e folds
print("\n1. Carregando dados...")
preds = pd.read_csv(REP / "catboost_v5_final_predicoes.csv")
folds = pd.read_csv(REP / "catboost_v5_final_folds.csv")
print(f"   Predições: {len(preds)} linhas")
print(f"   Folds: {len(folds)} rodadas")

# 2. ANÁLISE 1: Variância das Predições vs Real
print("\n" + "=" * 90)
print("📊 ANÁLISE 1: VARIÂNCIA DAS PREDIÇÕES vs VALORES REAIS")
print("=" * 90)

var_pred = preds["predicao"].var()
var_real = preds["pontos_num"].var()
ratio = var_pred / var_real

print(f"   Variância das predições: {var_pred:.4f}")
print(f"   Variância dos valores reais: {var_real:.4f}")
print(f"   Razão (pred/real): {ratio:.4f}")

if ratio < 0.3:
    print("   ⚠️  ALERTA: Variância das predições muito baixa (modelo conservador)")
elif ratio > 1.5:
    print("   ⚠️  ALERTA: Variância das predições muito alta (possível overfitting)")
else:
    print("   ✅ Variância dentro do esperado (0.3 - 1.5)")

# 3. ANÁLISE 2: Distribuição dos Erros
print("\n" + "=" * 90)
print("📊 ANÁLISE 2: DISTRIBUIÇÃO DOS ERROS (MAE por fold)")
print("=" * 90)

erros_abs = np.abs(preds["predicao"] - preds["pontos_num"])
print(f"   MAE médio: {erros_abs.mean():.4f}")
print(f"   MAE mediano: {erros_abs.median():.4f}")
print(f"   MAE std: {erros_abs.std():.4f}")
print(f"   MAE min: {erros_abs.min():.4f}")
print(f"   MAE max: {erros_abs.max():.4f}")
print(f"   MAE P90: {erros_abs.quantile(0.90):.4f}")
print(f"   MAE P95: {erros_abs.quantile(0.95):.4f}")
print(f"   MAE P99: {erros_abs.quantile(0.99):.4f}")

# Teste de normalidade dos erros
stat, p_value = shapiro(erros_abs.sample(min(5000, len(erros_abs))))
print(f"\n   Teste de normalidade (Shapiro-Wilk): p-value = {p_value:.6f}")
if p_value < 0.05:
    print("   ⚠️  Erros NÃO seguem distribuição normal (comum em modelos reais)")
else:
    print("   ✅ Erros seguem distribuição normal")

# 4. ANÁLISE 3: Estabilidade Temporal (Leakage entre folds)
print("\n" + "=" * 90)
print("📊 ANÁLISE 3: ESTABILIDADE TEMPORAL (Leakage entre folds)")
print("=" * 90)

# Agrupar por fold e calcular métricas
fold_metrics = preds.groupby("fold").agg({
    "pontos_num": ["mean", "std", "count"],
    "predicao": ["mean", "std"],
    "erro_absoluto": ["mean", "std"]
}).reset_index()

fold_metrics.columns = ["fold", "real_mean", "real_std", "count", "pred_mean", "pred_std", "mae", "mae_std"]

print(f"   MAE médio por fold: {fold_metrics['mae'].mean():.4f} ± {fold_metrics['mae'].std():.4f}")
print(f"   MAE min (melhor fold): {fold_metrics['mae'].min():.4f}")
print(f"   MAE max (pior fold): {fold_metrics['mae'].max():.4f}")

# Verificar se há tendência temporal (leakage acumulado)
fold_metrics["fold_num"] = range(len(fold_metrics))
corr_temporal, p_temporal = pearsonr(fold_metrics["fold_num"], fold_metrics["mae"])
print(f"\n   Correlação temporal (fold_num vs MAE): {corr_temporal:.4f} (p={p_temporal:.4f})")

if abs(corr_temporal) > 0.3 and p_temporal < 0.05:
    print("   ⚠️  ALERTA: Tendência temporal significativa (possível leakage acumulado)")
else:
    print("   ✅ Sem tendência temporal significativa")

# 5. ANÁLISE 4: Correlação Feature-Target (Leakage direto)
print("\n" + "=" * 90)
print("📊 ANÁLISE 4: CORRELAÇÃO PREDIÇÃO vs TARGET (Leakage direto)")
print("=" * 90)

# Se houver leakage, a correlação seria próxima de 1.0
corr_pred_real, p_corr = pearsonr(preds["predicao"], preds["pontos_num"])
sp_pred_real, p_sp = spearmanr(preds["predicao"], preds["pontos_num"])

print(f"   Correlação de Pearson (pred vs real): {corr_pred_real:.4f} (p={p_corr:.6f})")
print(f"   Correlação de Spearman (pred vs real): {sp_pred_real:.4f} (p={p_sp:.6f})")

if corr_pred_real > 0.95:
    print("   ⚠️  ALERTA CRÍTICO: Correlação muito alta (possível leakage de target)")
elif corr_pred_real > 0.85:
    print("   ⚠️  Correlação alta (pode ser sinal forte ou leakage parcial)")
else:
    print("   ✅ Correlação dentro do esperado para modelo preditivo")

# 6. ANÁLISE 5: Análise por Posição (Leakage específico)
print("\n" + "=" * 90)
print("📊 ANÁLISE 5: ANÁLISE POR POSIÇÃO (Leakage específico)")
print("=" * 90)

posicoes = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}

for pos_id, pos_nome in posicoes.items():
    pos_data = preds[preds["posicao_id"] == pos_id]
    if len(pos_data) == 0:
        continue
    
    mae_pos = np.abs(pos_data["predicao"] - pos_data["pontos_num"]).mean()
    sp_pos = spearmanr(pos_data["predicao"], pos_data["pontos_num"])[0]
    var_ratio = pos_data["predicao"].var() / pos_data["pontos_num"].var()
    
    print(f"\n   {pos_nome}:")
    print(f"     MAE: {mae_pos:.4f} | Spearman: {sp_pos:.4f}")
    print(f"     Variância pred/real: {var_ratio:.4f}")
    
    # Verificar se há overfitting por posição
    if var_ratio > 1.5:
        print(f"     ⚠️  ALERTA: Variância alta (possível overfitting)")
    elif var_ratio < 0.3:
        print(f"     ⚠️  ALERTA: Variância baixa (modelo conservador)")

# 7. ANÁLISE 6: Comparação com Baseline v2
print("\n" + "=" * 90)
print("📊 ANÁLISE 6: COMPARAÇÃO COM BASELINE v2")
print("=" * 90)

try:
    v2_folds = pd.read_csv(REP / "catboost_pooling_folds.csv")
    v2_mae = v2_folds["mae_com"].mean()
    v2_sp = v2_folds["sp_com"].mean()
    
    v5_mae = folds["mae"].mean()
    v5_sp = folds["spearman"].mean()
    
    print(f"   v2 (Pooling): MAE = {v2_mae:.4f}, Spearman = {v2_sp:.4f}")
    print(f"   v5.1 (Poda):  MAE = {v5_mae:.4f}, Spearman = {v5_sp:.4f}")
    print(f"   Melhoria MAE: {(v2_mae - v5_mae) / v2_mae * 100:.2f}%")
    print(f"   Melhoria Spearman: {(v5_sp - v2_sp) / v2_sp * 100:.2f}%")
    
    if (v5_sp - v2_sp) / v2_sp > 0.5:
        print("   ⚠️  ALERTA: Melhoria > 50% (investigar causas)")
    else:
        print("   ✅ Melhoria dentro de limites razoáveis")
except:
    print("   ❌ Não foi possível carregar baseline v2")

# 8. ANÁLISE 7: Verificação de Features Suspeitas
print("\n" + "=" * 90)
print("📊 ANÁLISE 7: VERIFICAÇÃO DE FEATURES SUSPEITAS")
print("=" * 90)

# Carregar dataset para verificar features
try:
    df = pd.read_parquet(INTERIM / "cartola_jogador_rodada_v4.parquet")
    
    # Verificar se há features com correlação anômala com target
    target = "pontos_num"
    suspicious_features = []
    
    for col in df.columns:
        if col in ["temporada", "rodada", "atleta_id", "clube_id", "posicao_id", target]:
            continue
        
        try:
            valid_data = df[[col, target]].dropna()
            if len(valid_data) < 100:
                continue
            
            corr, _ = pearsonr(valid_data[col], valid_data[target])
            
            if abs(corr) > 0.8:
                suspicious_features.append((col, corr))
        except:
            continue
    
    if suspicious_features:
        print(f"   ⚠️  Encontradas {len(suspicious_features)} features com correlação > 0.8:")
        for feat, corr in sorted(suspicious_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"     - {feat}: {corr:.4f}")
    else:
        print("   ✅ Nenhuma feature com correlação anômala encontrada")
        
except Exception as e:
    print(f"   ❌ Erro ao carregar dataset: {e}")

# 9. CONCLUSÃO FINAL
print("\n" + "=" * 90)
print("🎯 CONCLUSÃO FINAL DA AUDITORIA")
print("=" * 90)

print("""
   Com base nas análises acima, avalie:
   
   ✅ INDICADORES DE MELHORA REAL:
   - Variância das predições dentro de 0.3-1.5x do real
   - Sem tendência temporal significativa nos erros
   - Correlação pred vs real entre 0.6-0.9
   - Melhoria consistente em todas as posições
   - Nenhuma feature com correlação > 0.8 com target
   
   ⚠️ INDICADORES DE POSSÍVEL LEAKAGE:
   - Variância das predições muito baixa (< 0.3) ou muito alta (> 1.5)
   - Tendência temporal nos erros (correlação > 0.3)
   - Correlação pred vs real > 0.95
   - Melhoria > 50% em relação ao baseline
   - Features com correlação > 0.8 com target
   - Goleiro com Spearman > 0.4 (suspeito)
""")

print("=" * 90)
