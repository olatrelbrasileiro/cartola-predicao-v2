#!/usr/bin/env python3
"""
Diagnóstico do Cálculo de Diffs dos Scouts
Verifica como os scouts estão no dataset bruto e como o diff foi calculado.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON_V2 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"

print("=" * 80)
print("DIAGNÓSTICO DO CÁLCULO DE DIFFS DOS SCOUTS")
print("=" * 80)

# 1. Carregar dataset bruto v2
print("\n1. Carregando dataset canônico v2 (bruto)...")
df = pd.read_parquet(CANON_V2)
print(f"   Shape: {df.shape}")

# 2. Filtrar goleiros
goleiros = df[df["posicao_id"] == 1].copy()
print(f"   Total de linhas de goleiros: {len(goleiros)}")

# 3. Verificar scouts GS e DE no dataset bruto
print("\n2. Análise dos scouts GS e DE no dataset BRUTO (v2):")
print("-" * 80)

for scout in ["GS", "DE"]:
    col = f"scout_{scout}"
    if col in goleiros.columns:
        serie = goleiros[col]
        n_total = len(serie)
        n_nans = serie.isna().sum()
        n_zeros = (serie == 0).sum()
        media = serie.mean()
        std = serie.std()
        max_val = serie.max()
        min_val = serie.min()
        
        print(f"\n{col}:")
        print(f"  NaNs: {n_nans} ({n_nans/n_total*100:.1f}%)")
        print(f"  Zeros: {n_zeros} ({n_zeros/n_total*100:.1f}%)")
        print(f"  Média: {media:.2f} | Std: {std:.2f}")
        print(f"  [Min: {min_val}, Max: {max_val}]")
        
        # Verificar se é acumulado ou por rodada
        # Se for acumulado, os valores devem aumentar ao longo da temporada
        amostra = goleiros[goleiros["atleta_id"] == goleiros["atleta_id"].mode()[0]].sort_values("rodada").head(10)
        print(f"\n  Amostra do goleiro {goleiros['atleta_id'].mode()[0]} (primeiras 10 rodadas):")
        print(f"  {'Rodada':<8} | {col:<12}")
        print(f"  {'-'*25}")
        for _, row in amostra.iterrows():
            print(f"  {int(row['rodada']):<8} | {row[col]:<12}")
    else:
        print(f"\n❌ Coluna '{col}' NÃO ENCONTRADA!")

# 4. Verificar se o diff foi calculado no v3
print("\n3. Verificando se o diff foi calculado no dataset v3:")
print("-" * 80)

CANON_V3 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v3.parquet"
df_v3 = pd.read_parquet(CANON_V3)

for scout in ["GS", "DE"]:
    col_diff = f"diff_scout_{scout}"
    if col_diff in df_v3.columns:
        print(f"✅ '{col_diff}' existe no v3")
        serie = df_v3[df_v3["posicao_id"] == 1][col_diff]
        print(f"   NaNs: {serie.isna().sum()} | Zeros: {(serie == 0).sum()} | Média: {serie.mean():.2f}")
    else:
        print(f"❌ '{col_diff}' NÃO existe no v3")

# 5. Testar o cálculo do diff manualmente
print("\n4. Testando o cálculo do diff manualmente para GS:")
print("-" * 80)

# Pegar um goleiro específico
atleta_id = goleiros["atleta_id"].mode()[0]
gk_data = goleiros[goleiros["atleta_id"] == atleta_id].sort_values("rodada").copy()

print(f"\nGoleiro ID: {atleta_id}")
print(f"{'Rodada':<8} | {'scout_GS':<10} | {'diff_calculado':<15}")
print("-" * 40)

for i, row in gk_data.iterrows():
    gs = row["scout_GS"]
    if pd.isna(gs):
        diff = np.nan
    else:
        # Tentar calcular o diff
        if i == gk_data.index[0]:
            diff = gs  # Primeira rodada, assume que é o valor acumulado
        else:
            prev_gs = gk_data.loc[gk_data.index[gk_data.index.get_loc(i) - 1], "scout_GS"]
            if pd.notna(prev_gs):
                diff = gs - prev_gs
            else:
                diff = gs
    
    print(f"{int(row['rodada']):<8} | {str(gs):<10} | {str(diff):<15}")

# 6. Verificar a lógica do script build_scouts_shrunk_v1.py
print("\n5. Verificando a lógica do script build_scouts_shrunk_v1.py:")
print("-" * 80)

SCRIPT_PATH = ROOT / "scripts" / "build_scouts_shrunk_v1.py"
if SCRIPT_PATH.exists():
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Procurar a linha do diff
    if "diff()" in content or ".diff(" in content:
        print("✅ O script usa .diff() para calcular diferenças")
        
        # Extrair o contexto do diff
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if ".diff(" in line:
                print(f"\n   Linha {i+1}: {line.strip()}")
                # Mostrar contexto (3 linhas antes e depois)
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                print(f"\n   Contexto:")
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"   {marker} {j+1:4d}: {lines[j]}")
    else:
        print("❌ O script NÃO usa .diff()")
else:
    print(f"❌ Script não encontrado: {SCRIPT_PATH}")

print("\n" + "=" * 80)
print("DIAGNÓSTICO CONCLUÍDO")
print("=" * 80)
