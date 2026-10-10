#!/usr/bin/env python3
"""
Análise detalhada de MAE e Spearman por posição — CatBoost v5.1
A PROVA DE FOGO para validar a melhora absurda.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("=" * 85)
print("🔍 PROVA DE FOGO: ANÁLISE POR POSIÇÃO — CATBOOST v5.1")
print("=" * 85)

# Carregar predições do v5.1
preds_file = REP / "catboost_v5_final_predicoes.csv"
if not preds_file.exists():
    print(f"❌ Arquivo não encontrado: {preds_file}")
    exit(1)

preds = pd.read_csv(preds_file)
print(f"\nTotal de predições carregadas: {len(preds)}")

posicoes = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}
resultados = []

for pos_id, pos_nome in posicoes.items():
    pos_data = preds[preds["posicao_id"] == pos_id]
    if len(pos_data) == 0:
        continue
    
    mae_pos = np.mean(np.abs(pos_data["predicao"] - pos_data["pontos_num"]))
    sp_pos = spearmanr(pos_data["predicao"], pos_data["pontos_num"])[0]
    
    resultados.append({
        "posicao": pos_nome,
        "n_amostras": len(pos_data),
        "mae": mae_pos,
        "spearman": sp_pos
    })

res_df = pd.DataFrame(resultados)

print("\n" + "=" * 85)
print(f"{'Posição':<12} | {'Amostras':>8} | {'MAE':>8} | {'Spearman':>10} | {'Status':>10}")
print("-" * 85)

for _, row in res_df.iterrows():
    # Critério simples: Spearman > 0.25 é considerado bom para essa posição
    status = "✅ REAL" if row["spearman"] > 0.20 else "⚠️ SUSPEITO"
    if row["posicao"] == "Goleiro":
        status = "🏆 MILAGRE" if row["spearman"] > 0.25 else ("✅ BOM" if row["spearman"] > 0.15 else "❌ FALHA")
        
    print(f"{row['posicao']:<12} | {row['n_amostras']:>8} | {row['mae']:>8.2f} | {row['spearman']:>10.4f} | {status:>10}")

print("=" * 85)
print("🎯 CONCLUSÃO:")
gk_sp = res_df[res_df["posicao"] == "Goleiro"]["spearman"].values[0] if "Goleiro" in res_df["posicao"].values else 0
if gk_sp > 0.20:
    print("✅ A MELHORA É REAL! O Goleiro saiu do buraco (Spearman > 0.20).")
    print("   As features de contexto e scouts granulares funcionaram como esperado.")
else:
    print("⚠️ ALERTA DE LEAKAGE OU BUG: O Goleiro continua ruim, mas o global subiu.")
    print("   Isso indica que a métrica global está sendo inflada por outra posição ou há erro no cálculo.")
print("=" * 85)
