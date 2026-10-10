#!/usr/bin/env python3
"""
Raio-X Completo da Feature Importance (Todas as Features)
Mostra ranking, importância cumulativa e categorização para poda agressiva.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("=" * 90)
print("RAIO-X COMPLETO: IMPORTÂNCIA DE TODAS AS FEATURES (Base: v4)")
print("=" * 90)

# Carregar a importância completa do v4 (antes da poda)
fi_file = REP / "catboost_v4_feature_importance.csv"
try:
    fi = pd.read_csv(fi_file)
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {fi_file}")
    print("   Certifique-se de que o treino do v4 foi concluído.")
    exit(1)

# Ordenar e calcular cumulativo
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
total_imp = fi["importance"].sum()
fi["pct"] = (fi["importance"] / total_imp) * 100
fi["cumulative_pct"] = fi["pct"].cumsum()

# Função para categorizar
def categorizar(feat):
    if any(x in feat for x in ["scout_DE_", "scout_SG_", "scout_DP_", "scout_GS_"]):
        return "🧤 Goleiro"
    elif any(x in feat for x in ["team_", "opp_", "diff_"]):
        return "🌍 Contexto"
    elif any(x in feat for x in ["p_home", "p_draw", "p_away", "mando", "favoritismo"]):
        return "⚔️ Confronto"
    elif any(x in feat for x in ["hist_", "rec_", "momentum", "basic_", "ga_", "entrou_rate", "is_cold_start", "posicao_id"]):
        return "📊 Base"
    else:
        return "🎯 Scout Linha"

fi["categoria"] = fi["feature"].apply(categorizar)

# 1. Mostrar o Top 20 e a "Cauda" (Bottom 15)
print("\n🏆 TOP 20 FEATURES (Onde está o verdadeiro sinal)")
print("-" * 90)
print(f"{'Rank':<4} | {'Categoria':<12} | {'Imp':<6} | {'Cumulativo':<10} | Feature")
print("-" * 90)
for i, row in fi.head(20).iterrows():
    print(f"#{i+1:<2} | {row['categoria']:<12} | {row['pct']:<5.2f}% | {row['cumulative_pct']:<9.2f}% | {row['feature']}")

print("\n... (meio da lista omitido para brevidade) ...\n")

print("🗑️ CAUDA DA LISTA (As 15 features menos importantes - Candidatas a lixo)")
print("-" * 90)
for i, row in fi.tail(15).iterrows():
    print(f"#{i+1:<2} | {row['categoria']:<12} | {row['pct']:<5.2f}% | {row['cumulative_pct']:<9.2f}% | {row['feature']}")

# 2. Resumo por Categoria
print("\n" + "=" * 90)
print("📊 RESUMO DE IMPORTÂNCIA POR CATEGORIA")
print("=" * 90)
cat_summary = fi.groupby("categoria").agg(
    qtd_features=("feature", "count"),
    imp_total=("importance", "sum"),
    pct_total=("pct", "sum")
).sort_values("pct_total", ascending=False)

for cat, row in cat_summary.iterrows():
    print(f"{cat:<15} : {int(row['qtd_features']):>3} features | {row['pct_total']:>5.2f}% da importância total")

# 3. Sugestão de Corte Baseada em Cumulativo
print("\n" + "=" * 90)
print("💡 SUGESTÃO DE PODA AGRESSIVA")
print("=" * 90)
# Encontrar onde atinge 90% e 95% da importância
corte_90 = fi[fi["cumulative_pct"] <= 90.0]
corte_95 = fi[fi["cumulative_pct"] <= 95.0]

print(f"→ Para manter **90%** do poder do modelo, use apenas as Top {len(corte_90)} features.")
print(f"→ Para manter **95%** do poder do modelo, use apenas as Top {len(corte_95)} features.")
print(f"→ O modelo atual tem {len(fi)} features. Cortar abaixo de 95% elimina {len(fi) - len(corte_95)} features de ruído.")

# Salvar relatório completo em Markdown
md_path = REP / "feature_importance_completa.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Raio-X Completo da Feature Importance (v4)\n\n")
    f.write("## Ranking Completo\n\n")
    f.write(fi[["feature", "categoria", "pct", "cumulative_pct"]].to_markdown(index=False))
    f.write("\n\n## Resumo por Categoria\n\n")
    f.write(cat_summary.to_markdown())

print(f"\n✅ Relatório completo salvo em: {md_path}")
print("   (Abra este arquivo para ver a lista completa das 110 features)")
