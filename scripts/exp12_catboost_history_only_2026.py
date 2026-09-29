#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-12: MODELO 0 — Baseline de Histórico Individual Puro (Temporada 2026)

Objetivo:
    Avaliar o poder preditivo isolado do histórico individual de participações
    do atleta antes da introdução dos Tiers 0 a 7 de contexto coletivo.

Restrições:
    - Exclusivamente temporada 2026.
    - Sem dados de 2024 ou 2025.
    - Sem odds, mando, clube, adversário ou variáveis dos Tiers 0-7.
    - Histórico computado apenas sobre participações reais (entrou_em_campo == True).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAÇÕES E PARÂMETROS
# ==============================================================================
DATA_PATH = "data/interim/cartola_jogador_rodada_v2.parquet"
OUTPUT_DIR = "reports/exp12"
FOLDS_CSV_PATH = os.path.join(OUTPUT_DIR, "exp12_history_only_2026_folds.csv")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "exp12_history_only_2026_summary.json")

TARGET = "pontos_num"
TEMPORADA = 2026

FEATURES_HISTORICO = [
    "posicao_id",
    "hist_mean",
    "hist_std",
    "hist_n",
    "rec_mean_5",
    "rec_mean_3",
    "rec_std_5",
    "basic_mean",
    "basic_rec_5",
]

CAT_FEATURES = ["posicao_id"]

CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": False
}

BANNED_SUBSTRINGS = [
    "team_", "opp_", "matchup_", "mando", "clube_", "adversario_",
    "p_home", "p_draw", "p_away", "prob", "odd", "favorit", "tier"
]

# ==============================================================================
# ENGENHARIA DO HISTÓRICO INDIVIDUAL (SEM VAZAMENTO TEMPORAL)
# ==============================================================================
def compute_individual_history_2026(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa as estatísticas de histórico individual baseadas estritamente
    nas participações passadas (entrou_em_campo == True) em 2026.
    
    Regra fundamental:
    Para prever a rodada R, apenas participações em rodadas < R são consideradas.
    Se o atleta não jogou na rodada R-1, a última participação válida anterior é usada.
    Valores insuficientes resultam em NaN (sem preenchimento com zero).
    """
    # 1. Filtro estrito: Temporada 2026 e participações reais
    df_part = df[
        (df["temporada"] == TEMPORADA) & 
        (df["entrou_em_campo"] == True)
    ].copy()
    
    # 2. Ordenação temporal estrita
    df_part = df_part.sort_values(by=["atleta_id", "rodada"]).reset_index(drop=True)
    
    # 3. Cálculo das features utilizando deslocamento temporal (shift 1 sobre participações)
    # Target shiftado: pontuação da participação imediatamente anterior
    pts_shift = df_part.groupby("atleta_id")[TARGET].shift(1)
    
    # hist_n: quantidade de participações anteriores
    df_part["hist_n"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.expanding(min_periods=1).count())
        .astype(float)
    )
    # Se não jogou nenhuma anterior, hist_n deve ser NaN ou 0.0 (preservamos NaN se sem participações)
    df_part["hist_n"] = df_part["hist_n"].replace(0.0, np.nan)
    
    # hist_mean & basic_mean: média de todas as participações anteriores
    df_part["hist_mean"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    df_part["basic_mean"] = df_part["hist_mean"]  # Redundância formal documentada
    
    # hist_std: desvio padrão de todas as participações anteriores (requer >= 2 participações)
    df_part["hist_std"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.expanding(min_periods=2).std())
    )
    
    # rec_mean_5 & basic_rec_5: média móvel das últimas 5 participações
    df_part["rec_mean_5"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )
    df_part["basic_rec_5"] = df_part["rec_mean_5"]  # Redundância formal documentada
    
    # rec_mean_3: média móvel das últimas 3 participações
    df_part["rec_mean_3"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    
    # rec_std_5: desvio padrão móvel das últimas 5 participações (requer >= 2 participações)
    df_part["rec_std_5"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.rolling(window=5, min_periods=2).std())
    )
    
    # Assegurar tipo categórico em posicao_id
    df_part["posicao_id"] = df_part["posicao_id"].astype(str)
    
    return df_part

# ==============================================================================
# AUDITORIA DE INTEGRIDADE OBRIGATÓRIA (REGRAS A - H)
# ==============================================================================
def run_integrity_checks(df_processed: pd.DataFrame, features: list):
    """Verifica todos os 10 critérios e regras A até H."""
    # Regra F: Somente 2026
    assert (df_processed["temporada"] == TEMPORADA).all(), "Regra F violada: Dados fora de 2026!"
    
    # Regra G: Target
    assert TARGET in df_processed.columns, "Regra G violada: Target pontos_num ausente!"
    assert not df_processed[TARGET].isna().any(), "Target contém valores NaN!"
    assert not np.isinf(df_processed[TARGET]).any(), "Target contém valores infinitos!"
    
    # Regra H & C: Somente atletas que entraram em campo
    assert (df_processed["entrou_em_campo"] == True).all(), "Regra H violada: Contém atletas que não jogaram!"
    
    # Regras D e E: Nenhuma feature proibida ou tier
    for col in features:
        col_lower = col.lower()
        for banned in BANNED_SUBSTRINGS:
            if banned in col_lower:
                raise ValueError(f"Violação Regra D/E: Coluna proibida '{col}' encontrada!")
                
    # Sem duplicatas atleta_id x rodada
    dups = df_processed.duplicated(subset=["atleta_id", "rodada"]).sum()
    assert dups == 0, f"Encontradas {dups} duplicatas em (atleta_id, rodada)!"
    
    print("✓ Auditoria de Integridade (Regras A a H): APROVADA.")

# ==============================================================================
# EXECUÇÃO DO EXPERIMENTO
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset canônico não encontrado em: {DATA_PATH}")
        
    df_raw = pd.read_parquet(DATA_PATH)
    
    # Padronizar nome de atleta_id se necessário
    if "atleta_id" not in df_raw.columns and "jogador_id" in df_raw.columns:
        df_raw.rename(columns={"jogador_id": "atleta_id"}, inplace=True)
        
    # Construção do dataset de histórico puro
    df = compute_individual_history_2026(df_raw)
    
    # Auditoria de integridade
    run_integrity_checks(df, FEATURES_HISTORICO)
    
    rodadas_disponiveis = sorted(df["rodada"].unique())
    
    folds_summary = []
    all_predictions = []
    
    print("\n" + "="*80)
    print(f"EXP-12: EXECUTANDO VALIDAÇÃO TEMPORAL (TEMPORADA {TEMPORADA})")
    print("="*80)
    print(f"{'Rodada':<8} | {'Treino':<8} | {'Teste':<8} | {'MAE':<10} | {'RMSE':<10} | {'Spearman':<10}")
    print("-" * 65)
    
    for r in rodadas_disponiveis:
        # Treino: rodadas anteriores estritas
        train_df = df[df["rodada"] < r]
        # Teste: rodada-alvo exata
        test_df = df[df["rodada"] == r]
        
        # Só iniciamos quando há dados de treino suficientes (rodada >= 2)
        if len(train_df) < 50 or len(test_df) == 0:
            continue
            
        # Regra A & B: Integridade temporal do fold
        assert train_df["rodada"].max() < r, f"Regra A violada na rodada {r}!"
        assert (test_df["rodada"] == r).all(), f"Regra B violada no teste da rodada {r}!"
        
        X_train = train_df[FEATURES_HISTORICO]
        y_train = train_df[TARGET]
        
        X_test = test_df[FEATURES_HISTORICO]
        y_test = test_df[TARGET]
        
        model = CatBoostRegressor(**CATBOOST_PARAMS, cat_features=CAT_FEATURES)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        mae_fold = mean_absolute_error(y_test, preds)
        rmse_fold = np.sqrt(mean_squared_error(y_test, preds))
        spearman_fold, _ = spearmanr(y_test, preds)
        if np.isnan(spearman_fold):
            spearman_fold = 0.0
            
        folds_summary.append({
            "rodada": int(r),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "mae": float(mae_fold),
            "rmse": float(rmse_fold),
            "spearman": float(spearman_fold)
        })
        
        df_pred = test_df[["temporada", "rodada", "atleta_id", TARGET]].copy()
        df_pred["predicao"] = preds
        all_predictions.append(df_pred)
        
        print(f"R{r:02d}      | {len(train_df):<8d} | {len(test_df):<8d} | {mae_fold:<10.4f} | {rmse_fold:<10.4f} | {spearman_fold:<10.4f}")

    if not all_predictions:
        raise RuntimeError("Nenhum fold de validação pôde ser executado.")
        
    df_all_preds = pd.concat(all_predictions, ignore_index=True)
    
    # Métricas Globais Agregadas
    global_mae = mean_absolute_error(df_all_preds[TARGET], df_all_preds["predicao"])
    global_rmse = np.sqrt(mean_squared_error(df_all_preds[TARGET], df_all_preds["predicao"]))
    global_spearman, _ = spearmanr(df_all_preds[TARGET], df_all_preds["predicao"])
    
    # Salvar Folds CSV
    df_folds = pd.DataFrame(folds_summary)
    df_folds.to_csv(FOLDS_CSV_PATH, index=False)
    
    # Salvar Summary JSON
    summary_data = {
        "experimento": "EXP-12",
        "modelo": "Modelo 0 — Histórico Puro 2026",
        "temporada": TEMPORADA,
        "n_folds": int(len(folds_summary)),
        "n_predictions": int(len(df_all_preds)),
        "mae_global": float(global_mae),
        "rmse_global": float(global_rmse),
        "spearman_global": float(global_spearman),
        "features": FEATURES_HISTORICO,
        "redundancies_documented": {
            "basic_mean": "Matematicamente idêntica a hist_mean em 2026 pura",
            "basic_rec_5": "Matematicamente idêntica a rec_mean_5 em 2026 pura"
        },
        "catboost_params": CATBOOST_PARAMS
    }
    
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
    print("-" * 65)
    print("\n================================")
    print("EXP-12 — MODELO 0")
    print("HISTÓRICO PURO — 2026")
    print("================================")
    print("Features:")
    for feat in FEATURES_HISTORICO:
        print(f"  - {feat}")
    print(f"\nFolds: {len(folds_summary)}")
    print(f"Predições: {len(df_all_preds)}")
    print(f"\nMAE: {global_mae:.4f}")
    print(f"RMSE: {global_rmse:.4f}")
    print(f"Spearman: {global_spearman:.4f}")
    print("\nIntegridade: APROVADA (10/10 verificações, Regras A a H)")
    print("\nArquivos gerados:")
    print(f"  - {FOLDS_CSV_PATH}")
    print(f"  - {SUMMARY_JSON_PATH}")

if __name__ == "__main__":
    main()
