#!/usr/bin/env python3
"""
Análise detalhada de MAE e Spearman por posição usando predições salvas
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"

print("=" * 70)
print("ANÁLISE DETALHADA POR POSIÇÃO")
print("=" * 70)

# Carregar predições
preds = pd.read_csv(REP / "catboost_v3_predicoes_detalhadas.csv")
print(f"\nTotal de predições carregadas: {len(preds)}")

# Mapeamento de posições
posicoes = {
    1: "Goleiro",
    2: "Lateral", 
    3: "Zagueiro",
    4: "Meia",
    5: "Atacante",
    6: "Técnico"
}

resultados = []

for pos_id, pos_nome in posicoes.items():
    pos_data = preds[preds["posicao_id"] == pos_id]
    
    if len(pos_data) == 0:
        continue
    
    # Métricas globais para esta posição
    mae_pos = np.mean(np.abs(pos_data["predicao"] - pos_data["pontos_num"]))
    rmse_pos = np.sqrt(np.mean((pos_data["predicao"] - pos_data["pontos_num"]) ** 2))
    spearman_pos = spearmanr(pos_data["predicao"], pos_data["pontos_num"])[0]
    
    # Métricas por fold (rodada)
    fold_metrics = []
    for fold in pos_data["fold"].unique():
        fold_data = pos_data[pos_data["fold"] == fold]
        if len(fold_data) < 3:  # Mínimo para Spearman
            continue
        
        fold_mae = np.mean(np.abs(fold_data["predicao"] - fold_data["pontos_num"]))
        fold_sp = spearmanr(fold_data["predicao"], fold_data["pontos_num"])[0]
        fold_metrics.append({"mae": fold_mae, "spearman": fold_sp})
    
    fold_df = pd.DataFrame(fold_metrics)
    mae_medio_folds = fold_df["mae"].mean() if len(fold_df) > 0 else np.nan
    sp_medio_folds = fold_df["spearman"].mean() if len(fold_df) > 0 else np.nan
    
    resultados.append({
        "posicao": pos_nome,
        "posicao_id": pos_id,
        "n_amostras": len(pos_data),
        "n_folds": len(fold_df),
        "mae_global": mae_pos,
        "rmse_global": rmse_pos,
        "spearman_global": spearman_pos,
        "mae_medio_folds": mae_medio_folds,
        "spearman_medio_folds": sp_medio_folds
    })

res_df = pd.DataFrame(resultados)

print("\n" + "=" * 85)
print(f"{'Posição':<12} | {'Amostras':>8} | {'Folds':>5} | {'MAE Global':>10} | {'MAE Folds':>10} | {'SP Global':>10} | {'SP Folds':>10}")
print("-" * 85)

for _, row in res_df.iterrows():
    print(f"{row['posicao']:<12} | {row['n_amostras']:>8} | {row['n_folds']:>5} | "
          f"{row['mae_global']:>10.4f} | {row['mae_medio_folds']:>10.4f} | "
          f"{row['spearman_global']:>10.4f} | {row['spearman_medio_folds']:>10.4f}")

# Comparação com baseline v2 (valores conhecidos da auditoria anterior)
print("\n" + "=" * 85)
print("COMPARAÇÃO COM BASELINE v2 (Pooling)")
print("=" * 85)

# Valores aproximados do v2 para comparação
v2_metrics = {
    "Goleiro": {"mae": 3.25, "spearman": 0.07},
    "Lateral": {"mae": 2.80, "spearman": 0.25},
    "Zagueiro": {"mae": 2.70, "spearman": 0.20},
    "Meia": {"mae": 2.80, "spearman": 0.31},
    "Atacante": {"mae": 2.90, "spearman": 0.25},
    "Técnico": {"mae": 3.00, "spearman": 0.40}
}

print(f"\n{'Posição':<12} | {'MAE v2':>8} | {'MAE v3':>8} | {'Δ MAE':>8} | {'SP v2':>8} | {'SP v3':>8} | {'Δ SP':>8}")
print("-" * 85)

for _, row in res_df.iterrows():
    pos = row["posicao"]
    if pos in v2_metrics:
        v2 = v2_metrics[pos]
        delta_mae = row["mae_global"] - v2["mae"]
        delta_sp = row["spearman_global"] - v2["spearman"]
        
        mae_ind = "✅" if delta_mae < 0 else "❌"
        sp_ind = "✅" if delta_sp > 0 else "❌"
        
        print(f"{pos:<12} | {v2['mae']:>8.2f} | {row['mae_global']:>8.2f} | {delta_mae:>+8.2f}{mae_ind} | "
              f"{v2['spearman']:>8.2f} | {row['spearman_global']:>8.2f} | {delta_sp:>+8.2f}{sp_ind}")

# Salvar relatório
report_path = REP / "analise_por_posicao_v3.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Análise Detalhada por Posição — CatBoost v3\n\n")
    f.write("## Métricas por Posição\n\n")
    f.write(res_df.to_markdown(index=False))
    f.write(f"\n\n- Total de predições analisadas: {len(preds)}\n")

print(f"\n✅ Relatório salvo em: {report_path}")
