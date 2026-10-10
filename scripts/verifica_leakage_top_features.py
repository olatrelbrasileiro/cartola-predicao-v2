#!/usr/bin/env python3
"""
Verifica se features de resultado de partida estão no topo da importância.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("=" * 80)
print("🔍 VERIFICAÇÃO DE LEAKAGE: TOP 15 FEATURES")
print("=" * 80)

fi = pd.read_csv(REP / "catboost_v5_final_feature_importance.csv")
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n{'Rank':<4} | {'Importância':<12} | {'Feature'}")
print("-" * 80)

leakage_suspects = ["GS", "SG", "DE", "_G_", "_A_"] # GS=Gols Sofridos, SG=Clean Sheet, DE=Defesas

for i, row in fi.head(15).iterrows():
    feat = row["feature"]
    imp = row["importance"]
    
    # Destacar em vermelho (texto) se for suspeito
    is_suspect = any(s in feat for s in leakage_suspects)
    marker = "⚠️ SUSPEITO DE LEAKAGE" if is_suspect else "✅ OK"
    
    print(f"#{i+1:<2} | {imp:<12.4f} | {feat:<40} {marker}")

print("\n" + "=" * 80)
print("🎯 CONCLUSÃO:")
top_5 = fi.head(5)["feature"].tolist()
if any("GS" in f or "SG" in f for f in top_5):
    print("❌ LEAKAGE CONFIRMADO: O modelo está usando o resultado da partida (Gols Sofridos / Clean Sheet) para prever a pontuação da mesma partida.")
    print("   Isso explica a melhora absurda e a baixa variância.")
else:
    print("✅ As top 5 features parecem seguras. O leakage pode estar em uma interação mais sutil.")
print("=" * 80)
