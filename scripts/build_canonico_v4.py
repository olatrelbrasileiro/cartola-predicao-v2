#!/usr/bin/env python3
"""
Build Dataset Canônico v4 (Versão Definitiva)
1. Base: cartola_jogador_rodada_v2.parquet
2. + Contexto Time/Adversário (features_team_context_v2.parquet)
3. + Scouts por Rodada CORRIGIDOS (features_scouts_rodada_v1.parquet)
"""
import pandas as pd
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON_V2 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
SCOUTS_NOVOS = ROOT / "data" / "interim" / "features_scouts_rodada_v1.parquet"
CONTEXT = ROOT / "data" / "interim" / "features_team_context_v2.parquet"
OUT = ROOT / "data" / "interim" / "cartola_jogador_rodada_v4.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def main():
    print("1. Carregando dataset base (v2)...")
    df = pd.read_parquet(CANON_V2)
    print(f"   Shape: {df.shape}")

    print("\n2. Removendo colunas de scouts antigas (acumuladas/breakadas)...")
    # Remove qualquer coluna que comece com scout_ que já exista no v2 (se houver)
    old_scout_cols = [c for c in df.columns if c.startswith("scout_") and c not in ["scout_G", "scout_A", "scout_DE", "scout_GS", "scout_SG", "scout_DS", "scout_FC", "scout_DP", "scout_CA", "scout_CV", "scout_FD", "scout_FF", "scout_FT", "scout_FS", "scout_I", "scout_PC", "scout_PS", "scout_PP"]]
    if old_scout_cols:
        df = df.drop(columns=old_scout_cols)
        print(f"   Removidas: {old_scout_cols}")

    print("\n3. Merge com novos scouts (por rodada)...")
    df_scouts = pd.read_parquet(SCOUTS_NOVOS)
    print(f"   Scouts carregados: {df_scouts.shape}")
    
    # Pegar apenas as features calculadas (exclui as chaves de merge)
    scout_cols = [c for c in df_scouts.columns if c not in ["temporada", "rodada", "atleta_id", "posicao_id"]]
    df = df.merge(df_scouts[["temporada", "rodada", "atleta_id"] + scout_cols],
                  on=["temporada", "rodada", "atleta_id"], how="left")

    print("\n4. Merge com contexto time/adversário...")
    df_ctx = pd.read_parquet(CONTEXT)
    ctx_cols = [c for c in df_ctx.columns if c not in ["temporada", "rodada", "atleta_id", "clube_id"]]
    df = df.merge(df_ctx[["temporada", "rodada", "atleta_id"] + ctx_cols],
                  on=["temporada", "rodada", "atleta_id"], how="left")

    print("\n5. Preenchendo NaNs (0 para scouts e contexto)...")
    new_cols = scout_cols + ctx_cols
    df[new_cols] = df[new_cols].fillna(0)

    print("\n6. Validando integridade...")
    original_len = len(pd.read_parquet(CANON_V2))
    assert len(df) == original_len, f"❌ Linhas alteradas! {len(df)} vs {original_len}"
    
    print(f"   ✅ Total de linhas: {len(df)} (inalterado)")
    print(f"   ✅ Total de colunas: {df.shape[1]}")
    
    print("\n7. Salvando canônico v4...")
    df.to_parquet(OUT, index=False)
    print(f"   ✅ Salvo: {OUT}")

if __name__ == "__main__":
    main()
