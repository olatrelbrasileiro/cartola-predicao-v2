#!/usr/bin/env python3
"""
Build Dataset Canônico v5 (Versão Final - 92 features)
- Base: cartola_jogador_rodada_v2.parquet
- + Scouts granulares corrigidos (features_scouts_rodada_v1.parquet)
- + Contexto time/adversário v3 (features_team_context_v3.parquet)
- Remove features indesejadas (basic_*, ga_*, diff_*, is_cold_start, scouts de PP/DP/PS/CV/PC/CA/I)
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON_V2 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
SCOUTS = ROOT / "data" / "interim" / "features_scouts_rodada_v1.parquet"
CONTEXT = ROOT / "data" / "interim" / "features_team_context_v3.parquet"
OUT = ROOT / "data" / "interim" / "cartola_jogador_rodada_v5.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Lista das 92 features finais
FEATURES_FINAIS = [
    # Bloco 1: Pontos Gerais (8)
    "hist_mean", "hist_std", "hist_n", "hist_shrunk_k7",
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    
    # Bloco 2: Oportunidade/Posição (2)
    "entrou_rate", "posicao_id",
    
    # Bloco 3: Confronto (5)
    "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away",
    
    # Bloco 4: Contexto Time (9)
    "team_FTT_mean_3", "team_FA_mean_3", "team_DS_mean_3", "team_FC_mean_3",
    "team_clean_sheet_rate_3", "team_G_mean_3", "team_FS_mean_3", "team_DE_mean_3", "team_GS_mean_3",
    
    # Bloco 4: Contexto Adversário (9)
    "opp_FTT_mean_3", "opp_FA_mean_3", "opp_DS_mean_3", "opp_FC_mean_3",
    "opp_clean_sheet_rate_3", "opp_G_mean_3", "opp_FS_mean_3", "opp_DE_mean_3", "opp_GS_mean_3",
    
    # Bloco 5: Scouts Ofensivos (6 scouts × 5 params = 30)
    "scout_G_mean_hist", "scout_G_mean_3", "scout_G_mean_5", "scout_G_shrunk", "scout_G_momentum",
    "scout_A_mean_hist", "scout_A_mean_3", "scout_A_mean_5", "scout_A_shrunk", "scout_A_momentum",
    "scout_FD_mean_hist", "scout_FD_mean_3", "scout_FD_mean_5", "scout_FD_shrunk", "scout_FD_momentum",
    "scout_FF_mean_hist", "scout_FF_mean_3", "scout_FF_mean_5", "scout_FF_shrunk", "scout_FF_momentum",
    "scout_FT_mean_hist", "scout_FT_mean_3", "scout_FT_mean_5", "scout_FT_shrunk", "scout_FT_momentum",
    "scout_FS_mean_hist", "scout_FS_mean_3", "scout_FS_mean_5", "scout_FS_shrunk", "scout_FS_momentum",
    
    # Bloco 5: Scouts Defensivos (5 scouts × 5 params = 25)
    "scout_DE_mean_hist", "scout_DE_mean_3", "scout_DE_mean_5", "scout_DE_shrunk", "scout_DE_momentum",
    "scout_SG_mean_hist", "scout_SG_mean_3", "scout_SG_mean_5", "scout_SG_shrunk", "scout_SG_momentum",
    "scout_DS_mean_hist", "scout_DS_mean_3", "scout_DS_mean_5", "scout_DS_shrunk", "scout_DS_momentum",
    "scout_FC_mean_hist", "scout_FC_mean_3", "scout_FC_mean_5", "scout_FC_shrunk", "scout_FC_momentum",
    "scout_GS_mean_hist", "scout_GS_mean_3", "scout_GS_mean_5", "scout_GS_shrunk", "scout_GS_momentum",
    
    # Bloco 5: Std de Consistência (4)
    "scout_G_std_5", "scout_A_std_5", "scout_DE_std_5", "scout_SG_std_5",
]

def main():
    print("=" * 70)
    print("BUILD CANÔNICO v5 — 92 FEATURES FINAIS")
    print("=" * 70)
    
    print("\n1. Carregando dataset base (v2)...")
    df = pd.read_parquet(CANON_V2)
    print(f"   Shape: {df.shape}")

    print("\n2. Merge com scouts granulares corrigidos...")
    df_scouts = pd.read_parquet(SCOUTS)
    print(f"   Scouts carregados: {df_scouts.shape}")
    
    scout_cols = [c for c in df_scouts.columns if c not in ["temporada", "rodada", "atleta_id", "posicao_id"]]
    df = df.merge(df_scouts[["temporada", "rodada", "atleta_id"] + scout_cols],
                  on=["temporada", "rodada", "atleta_id"], how="left")

    print("\n3. Merge com contexto time/adversário v3...")
    df_ctx = pd.read_parquet(CONTEXT)
    ctx_cols = [c for c in df_ctx.columns if c not in ["temporada", "rodada", "atleta_id", "clube_id"]]
    df = df.merge(df_ctx[["temporada", "rodada", "atleta_id"] + ctx_cols],
                  on=["temporada", "rodada", "atleta_id"], how="left")

    print("\n4. Preenchendo NaNs (0 para scouts e contexto)...")
    new_cols = scout_cols + ctx_cols
    df[new_cols] = df[new_cols].fillna(0)

    print("\n5. Validando integridade...")
    original_len = len(pd.read_parquet(CANON_V2))
    assert len(df) == original_len, f"❌ Linhas alteradas! {len(df)} vs {original_len}"
    
    print(f"   ✅ Total de linhas: {len(df)} (inalterado)")
    print(f"   ✅ Total de colunas: {df.shape[1]}")
    
    # Verificar se todas as 92 features estão presentes
    missing = [f for f in FEATURES_FINAIS if f not in df.columns]
    if missing:
        print(f"\n⚠️  {len(missing)} features finais não encontradas:")
        for m in missing[:10]:
            print(f"   - {m}")
        if len(missing) > 10:
            print(f"   ... e mais {len(missing) - 10}")
    else:
        print(f"   ✅ Todas as {len(FEATURES_FINAIS)} features finais presentes")
    
    print("\n6. Salvando canônico v5...")
    df.to_parquet(OUT, index=False)
    print(f"   ✅ Salvo: {OUT}")
    
    # Relatório
    report_path = REPORTS / "build_canonico_v5_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Build Canônico v5 — Relatório\n\n")
        f.write(f"- Shape final: {df.shape}\n")
        f.write(f"- Features finais planejadas: {len(FEATURES_FINAIS)}\n")
        f.write(f"- Features presentes: {len(FEATURES_FINAIS) - len(missing)}\n")
        f.write(f"- Features ausentes: {len(missing)}\n")
        if missing:
            f.write(f"\n## Features Ausentes\n\n")
            for m in missing:
                f.write(f"- {m}\n")
    
    print(f"\n✅ Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()
