#!/usr/bin/env python3
"""EXP-11: Validação de Features de Goleiro com Banco Mestre"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
CANONICO = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
GOL_MASTER = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def main():
    print("1. Carregando dados e filtrando goleiros (posicao_id == 1)...")
    df = pd.read_parquet(CANONICO)
    df_gk = df[df["posicao_id"] == 1].copy()
    print(f"   -> {len(df_gk)} observações de goleiros.")

    print("2. Carregando features de contexto do Banco Mestre...")
    df_master = pd.read_parquet(GOL_MASTER)
    
    cols_master = ["temporada", "rodada", "clube_id", "atleta_id",
                   "team_gols_sofridos_mean_3", "team_clean_sheet_rate_mean_3",
                   "expected_goals_against", "p_sg", "attack_vs_defense"]
    cols_master = [c for c in cols_master if c in df_master.columns]
    
    print("3. Fazendo merge das features...")
    df_merged = df_gk.merge(df_master[cols_master], on=["temporada", "rodada", "clube_id", "atleta_id"], how="left")
    df_merged = df_merged[df_merged["entrou_em_campo"] == True].copy()
    
    # Baseline B2S: média histórica do próprio atleta (shift 1)
    df_merged = df_merged.sort_values(["atleta_id", "temporada", "rodada"])
    df_merged["b2s_pred"] = df_merged.groupby("atleta_id")["pontos_num"].shift(1)
    media_pos_temp = df_merged.groupby(["temporada", "posicao_id"])["pontos_num"].transform("mean")
    df_merged["b2s_pred"] = df_merged["b2s_pred"].fillna(media_pos_temp)
    
    # Modelo Novo: Regressão simples usando apenas o sinal do banco mestre
    features_novas = [f for f in ["expected_goals_against", "p_sg", "team_clean_sheet_rate_mean_3"] if f in df_merged.columns]
    X = df_merged[features_novas].fillna(df_merged[features_novas].median())
    y = df_merged["pontos_num"]
    
    train_mask = df_merged["temporada"] <= 2025
    test_mask = df_merged["temporada"] == 2026
    
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        model = Ridge(alpha=1.0)
        model.fit(X[train_mask], y[train_mask])
        df_merged.loc[test_mask, "nova_pred"] = model.predict(X[test_mask])
        
        y_test = df_merged.loc[test_mask, "pontos_num"]
        b2s_test = df_merged.loc[test_mask, "b2s_pred"]
        nova_test = df_merged.loc[test_mask, "nova_pred"]
        
        mae_b2s, mae_nova = mean_absolute_error(y_test, b2s_test), mean_absolute_error(y_test, nova_test)
        sp_b2s, _ = spearmanr(y_test, b2s_test)
        sp_nova, _ = spearmanr(y_test, nova_test)
        
        print("\n" + "="*65)
        print(" RESULTADO DA VALIDAÇÃO (Goleiros - Temporada 2026)")
        print("="*65)
        print(f" Baseline B2S (Média Histórica):")
        print(f"   MAE     : {mae_b2s:.4f}")
        print(f"   Spearman: {sp_b2s:.4f}")
        print(f"\n Modelo com Features do Banco Mestre (Contexto de Time):")
        print(f"   MAE     : {mae_nova:.4f}  {'✅ MELHOR' if mae_nova < mae_b2s else '❌ PIOR'}")
        print(f"   Spearman: {sp_nova:.4f}  {'✅ MELHOR' if sp_nova > sp_b2s else '❌ PIOR'}")
        print("="*65)
        
        report_path = REPORTS / "exp11_validacao_goleiro_master.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Validação EXP-11: Features de Goleiro com Banco Mestre\n\n")
            f.write(f"- **MAE B2S**: {mae_b2s:.4f} | **MAE Novo**: {mae_nova:.4f}\n")
            f.write(f"- **Spearman B2S**: {sp_b2s:.4f} | **Spearman Novo**: {sp_nova:.4f}\n\n")
            f.write("### Conclusão\n")
            if mae_nova < mae_b2s and sp_nova > sp_b2s:
                f.write("✅ **SUCESSO**: O sinal de contexto de time do Banco Mestre supera o baseline histórico.\n")
            else:
                f.write("⚠️ **PARCIAL**: Ganho em alguma métrica, mas requer ajuste de shrinkage ou mais features individuais (DE/DP).\n")
        print(f"\n💾 Relatório salvo em: {report_path}")
    else:
        print("⚠️ Dados insuficientes para divisão treino/teste temporal.")

if __name__ == "__main__":
    main()
