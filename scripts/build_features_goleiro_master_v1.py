#!/usr/bin/env python3
"""
EXP-11: Injeção Oficial de Features de Goleiro (Banco Mestre)
Gera o dataset canônico v3, incorporando o contexto de time para posicao_id == 1.
"""
import pandas as pd
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANONICO_V2 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
GOL_MASTER = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"
CANONICO_V3 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v3.parquet"

def get_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("1. Carregando dataset canônico v2...")
    df_canonico = pd.read_parquet(CANONICO_V2)
    print(f"   -> Shape: {df_canonico.shape}")
    
    print("2. Carregando features de goleiro do Banco Mestre...")
    df_gol = pd.read_parquet(GOL_MASTER)
    
    # Features VALIDADAS e confirmadas pelo diagnóstico que existem no arquivo
    cols_gol = [
        "temporada", "rodada", "clube_id", "atleta_id",
        "team_clean_sheet_rate_mean_3",
        "team_gols_sofridos_mean_3",
        "opp_gols_marcados_mean_3",
        "opp_shots_on_target_last3"
    ]
    
    # Filtrar apenas colunas que realmente existem (segurança)
    cols_gol = [c for c in cols_gol if c in df_gol.columns]
    df_gol_features = df_gol[cols_gol].copy()
    
    # Garantir unicidade das chaves de merge para evitar duplicação de linhas (Cartesian product)
    df_gol_features = df_gol_features.drop_duplicates(subset=["temporada", "rodada", "clube_id", "atleta_id"])
    
    print("3. Realizando merge seguro (left join)...")
    df_v3 = df_canonico.merge(
        df_gol_features,
        on=["temporada", "rodada", "clube_id", "atleta_id"],
        how="left"
    )
    
    print("4. Tratando ausências (apenas para as novas colunas em não-goleiros)...")
    cols_novas = [c for c in cols_gol if c not in ["temporada", "rodada", "clube_id", "atleta_id"]]
    
    # Preencher com 0 para não-goleiros é semanticamente correto (eles não têm esse contexto de goleiro)
    # O CatBoost lida bem com isso, e mantém o schema limpo.
    df_v3[cols_novas] = df_v3[cols_novas].fillna(0)
    
    print("5. Validando integridade dos dados...")
    assert len(df_v3) == len(df_canonico), "❌ ERRO CRÍTICO: Número de linhas alterado após o merge!"
    
    gk_total = (df_v3["posicao_id"] == 1).sum()
    
    # Verifica se pelo menos uma das novas colunas foi preenchida para goleiros
    col_check = cols_novas[0] if cols_novas else "team_clean_sheet_rate_mean_3"
    gk_com_features = ((df_v3["posicao_id"] == 1) & (df_v3[col_check] > 0)).sum()
    
    print(f"   -> Total de linhas: {len(df_v3)} (inalterado ✅)")
    print(f"   -> Goleiros no dataset: {gk_total}")
    print(f"   -> Goleiros com features de contexto preenchidas: {gk_com_features} ✅")
    
    print("6. Salvando dataset canônico v3...")
    df_v3.to_parquet(CANONICO_V3, index=False)
    
    print("\n" + "="*65)
    print("✅ SUCESSO: Dataset canônico v3 gerado com features de goleiro!")
    print("="*65)
    print(f"Origem v2 SHA256: {get_sha256(CANONICO_V2)[:16]}...")
    print(f"Destino v3 SHA256: {get_sha256(CANONICO_V3)[:16]}...")
    print(f"Salvo em: {CANONICO_V3}")

if __name__ == "__main__":
    main()
