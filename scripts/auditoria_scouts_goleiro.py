#!/usr/bin/env python3
"""
Auditoria de Saúde dos Scouts de Goleiro (GS e DE)
Verifica se os dados foram calculados corretamente ou se estão corrompidos/zerados.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCOUTS_FILE = ROOT / "data" / "interim" / "features_scouts_shrunk_v1.parquet"
CANON_V3 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v3.parquet"

print("=" * 80)
print("AUDITORIA DE SAÚDE DOS SCOUTS DE GOLEIRO (GS e DE)")
print("=" * 80)

# 1. Carregar dados
print("\n1. Carregando datasets...")
df_scouts = pd.read_parquet(SCOUTS_FILE)
df_v3 = pd.read_parquet(CANON_V3)

# Filtrar apenas goleiros (posicao_id == 1) que participaram (para ter dados reais)
# Vamos usar o df_v3 que tem a flag entrou_em_campo
goleiros = df_v3[df_v3["posicao_id"] == 1].copy()
print(f"   Total de linhas de goleiros no v3: {len(goleiros)}")

# 2. Verificar colunas de GS e DE
scouts_alvo = ["GS", "DE"]
parametros = ["mean_hist", "mean_3", "mean_5", "shrunk", "momentum", "std_5"]

print("\n2. Análise detalhada por Scout e Parâmetro (Apenas Goleiros):")
print("-" * 80)

for scout in scouts_alvo:
    print(f"\n🔍 SCOUT: {scout}")
    for param in parametros:
        col_name = f"scout_{scout}_{param}"
        
        if col_name not in goleiros.columns:
            print(f"   ❌ Coluna '{col_name}' NÃO ENCONTRADA!")
            continue
            
        serie = goleiros[col_name]
        n_total = len(serie)
        n_nans = serie.isna().sum()
        n_zeros = (serie == 0).sum()
        media = serie.mean()
        std = serie.std()
        max_val = serie.max()
        min_val = serie.min()
        
        pct_nans = (n_nans / n_total) * 100
        pct_zeros = (n_zeros / n_total) * 100
        
        print(f"   {param:12s} | NaNs: {n_nans:4d} ({pct_nans:5.1f}%) | Zeros: {n_zeros:4d} ({pct_zeros:5.1f}%) | Média: {media:6.2f} | Std: {std:6.2f} | [Min: {min_val:5.1f}, Max: {max_val:5.1f}]")

# 3. Verificar o cálculo do DIFF no dataset original (antes do shrinkage)
print("\n3. Verificando a fonte do cálculo (diff_scout no dataset bruto processado):")
# Precisamos ver se o diff foi calculado corretamente
# Vamos pegar uma amostra de goleiros que jogaram
amostra = goleiros[goleiros["entrou_em_campo"] == True].head(10)

print("\n   Amostra de 10 goleiros que entraram em campo:")
print(f"   {'atleta_id':<10} | {'temp':<5} | {'rod':<4} | {'scout_GS_raw':<12} | {'diff_GS':<10} | {'scout_DE_raw':<12} | {'diff_DE':<10}")
print("   " + "-" * 76)

for _, row in amostra.iterrows():
    # Tenta pegar o scout bruto e o diff
    gs_raw = row.get("scout_GS", np.nan)
    diff_gs = row.get("diff_scout_GS", np.nan)
    de_raw = row.get("scout_DE", np.nan)
    diff_de = row.get("diff_scout_DE", np.nan)
    
    print(f"   {int(row['atleta_id']):<10} | {int(row['temporada']):<5} | {int(row['rodada']):<4} | "
          f"{str(gs_raw):<12} | {str(diff_gs):<10} | {str(de_raw):<12} | {str(diff_de):<10}")

# 4. Diagnóstico Final
print("\n" + "=" * 80)
print("DIAGNÓSTICO:")
print("=" * 80)

# Verificar se a coluna diff_scout_GS existe e tem variância
if "diff_scout_GS" in df_v3.columns:
    var_diff_gs = df_v3[df_v3["posicao_id"] == 1]["diff_scout_GS"].var()
    if var_diff_gs < 1e-6:
        print("⚠️  ALERTA CRÍTICO: A variância de 'diff_scout_GS' para goleiros é praticamente ZERO.")
        print("   Isso explica a importância 0.00% no CatBoost. O cálculo do diff falhou ou os dados brutos estão incorretos.")
    else:
        print("✅ 'diff_scout_GS' possui variância. O problema pode estar na etapa de shrinkage ou no merge.")
else:
    print("❌ 'diff_scout_GS' não existe no dataset v3. O cálculo do diff não foi executado corretamente.")

if "diff_scout_DE" in df_v3.columns:
    var_diff_de = df_v3[df_v3["posicao_id"] == 1]["diff_scout_DE"].var()
    if var_diff_de < 1e-6:
        print("⚠️  ALERTA CRÍTICO: A variância de 'diff_scout_DE' para goleiros é muito baixa.")
    else:
        print("✅ 'diff_scout_DE' possui variância razoável.")
else:
    print("❌ 'diff_scout_DE' não existe no dataset v3.")

print("\n💡 PRÓXIMO PASSO:")
print("Se os diffs estiverem zerados, o problema está no script 'build_scouts_shrunk_v1.py'")
print("na hora de calcular o .diff() dos scouts acumulados da API.")
