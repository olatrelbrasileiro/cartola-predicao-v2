#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-12: CatBoost Progressive Model Runner (Temporada 2026)
Execução controlada via CLI para avaliação cumulativa e isolada dos Tiers 0 a 7.

Uso:
    python scripts/exp12_catboost_progressive_2026.py --mode cumulative --tier N
    python scripts/exp12_catboost_progressive_2026.py --mode single --tier N
"""

import os
import sys
import json
import hashlib
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAÇÕES E DEFINIÇÃO CANÔNICA DOS TIERS
# ==============================================================================
TARGET = "pontos_num"
TEMPORADA = 2026
EXPECTED_FOLDS = list(range(2, 27))  # R02 até R26 (24 rodadas)

DATA_PLAYER_PATH = "data/interim/cartola_jogador_rodada_v2.parquet"
DATA_CONTEXT_PATH = "data/interim/exp12/team_context_features_v1.parquet"
DATA_SCOUT_PATH = "data/interim/exp12/team_scout_history.parquet"

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

TIERS_CATALOG = {
    0: [
        "mando",
        "rodada",
        "team_games_played_season",
        "opp_games_played_season",
    ],
    1: [
        "team_FTT_last5_mean",
        "team_FA_last5_mean",
        "team_G_last5_mean",
    ],
    2: [
        "team_gols_sofridos_last5_mean",
        "team_SG_last5_rate",
        "opp_gols_sofridos_last5_mean",
        "opp_SG_last5_rate",
    ],
    3: [
        "opp_FTT_last5_mean",
        "opp_FA_last5_mean",
        "opp_G_last5_mean",
    ],
    4: [
        "team_DS_last5_mean",
        "opp_DS_last5_mean",
        "team_DE_last5_mean",
        "opp_DE_last5_mean",
        "team_FS_last5_mean",
    ],
    5: [
        "team_shot_accuracy_last5",
        "team_conversion_rate_last5",
    ],
    6: [
        "team_SG_expanding_rate",
        "team_G_expanding_mean",
    ],
    7: [
        "matchup_G_diff_last5",
    ],
}

TIER_NAMES = {
    0: "Tier 0 — calendário/suporte",
    1: "Tier 1 — ofensivo próprio",
    2: "Tier 2 — vulnerabilidade defensiva",
    3: "Tier 3 — pressão do adversário",
    4: "Tier 4 — duelo/estilo",
    5: "Tier 5 — eficiência",
    6: "Tier 6 — expansão",
    7: "Tier 7 — matchup",
}

CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": False,
}

# ==============================================================================
# FUNÇÕES DE HISTÓRICO, INTEGRIDADE E JOIN EXPLÍCITO DO MANDO
# ==============================================================================
def get_file_sha256(filepath: str) -> str:
    """Calcula o hash SHA-256 de um arquivo caso exista."""
    if not os.path.exists(filepath):
        return "N/A"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_individual_history_2026(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa o histórico individual exatamente alinhado ao Modelo 0.
    Somente participações reais (entrou_em_campo == True) em 2026.
    """
    df_part = df[(df["temporada"] == TEMPORADA) & (df["entrou_em_campo"] == True)].copy()

    if "atleta_id" not in df_part.columns and "jogador_id" in df_part.columns:
        df_part.rename(columns={"jogador_id": "atleta_id"}, inplace=True)

    df_part = df_part.sort_values(by=["atleta_id", "rodada"]).reset_index(drop=True)

    pts_shift = df_part.groupby("atleta_id")[TARGET].shift(1)

    df_part["hist_n"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.expanding(min_periods=1).count())
        .astype(float)
        .replace(0.0, np.nan)
    )

    df_part["hist_mean"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    df_part["basic_mean"] = df_part["hist_mean"]

    df_part["hist_std"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.expanding(min_periods=2).std())
    )

    df_part["rec_mean_5"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )
    df_part["basic_rec_5"] = df_part["rec_mean_5"]

    df_part["rec_mean_3"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    df_part["rec_std_5"] = (
        pts_shift.groupby(df_part["atleta_id"])
        .transform(lambda s: s.rolling(window=5, min_periods=2).std())
    )

    df_part["posicao_id"] = df_part["posicao_id"].astype(str)
    return df_part


def merge_team_context_explicit(df_player: pd.DataFrame, df_context: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o join explícito das features de contexto, validando unicidade da chave
    e garantindo que a coluna 'mando' utilizada venha expressamente do contexto.
    """
    initial_rows = len(df_player)

    context_team_col = "clube_id" if "clube_id" in df_context.columns else "team_id"
    player_team_col = "clube_id" if "clube_id" in df_player.columns else "team_id"

    if player_team_col not in df_player.columns:
        raise RuntimeError("FALHA: Coluna do clube/equipe ausente no dataset do jogador.")

    df_ctx = df_context.copy()
    if context_team_col != player_team_col:
        df_ctx.rename(columns={context_team_col: player_team_col}, inplace=True)

    df_player["temporada"] = df_player["temporada"].astype(int)
    df_player["rodada"] = df_player["rodada"].astype(int)
    df_ctx["temporada"] = df_ctx["temporada"].astype(int)
    df_ctx["rodada"] = df_ctx["rodada"].astype(int)

    join_keys = ["temporada", "rodada", player_team_col]

    # Validação de unicidade no dataset de contexto antes do join
    ctx_duplicates = df_ctx.duplicated(subset=join_keys).sum()
    if ctx_duplicates > 0:
        raise RuntimeError(
            f"FALHA DE INTEGRIDADE: Contexto possui {ctx_duplicates} duplicatas na chave {join_keys}!"
        )

    # Tratamento explícito de 'mando'
    df_p = df_player.copy()
    if "mando" in df_p.columns:
        df_p.rename(columns={"mando": "mando_player"}, inplace=True)

    if "mando" in df_ctx.columns:
        df_ctx.rename(columns={"mando": "mando_context"}, inplace=True)

    cols_to_use = [c for c in df_ctx.columns if c not in df_p.columns or c in join_keys]
    df_merged = pd.merge(df_p, df_ctx[cols_to_use], on=join_keys, how="left")

    # Validação pós-join de contagem de linhas
    if len(df_merged) != initial_rows:
        raise RuntimeError(
            f"FALHA DE INTEGRIDADE: Linhas multiplicadas de {initial_rows} para {len(df_merged)}!"
        )

    # Definição explícita da coluna final 'mando' a partir do contexto
    if "mando_context" in df_merged.columns:
        df_merged["mando"] = df_merged["mando_context"]
        df_merged.drop(columns=["mando_context"], inplace=True)

    if "mando_player" in df_merged.columns:
        df_merged.drop(columns=["mando_player"], inplace=True)

    return df_merged


def verify_10_rules(df: pd.DataFrame, expected_features: list, actual_features: list):
    """Executa a verificação programática estrita das regras essenciais de integridade."""
    # Temporada exclusivamente 2026
    if not (df["temporada"] == TEMPORADA).all():
        raise RuntimeError("FALHA DE INTEGRIDADE: Temporadas fora de 2026 encontradas!")

    # Teste somente participantes
    if not (df["entrou_em_campo"] == True).all():
        raise RuntimeError("FALHA DE INTEGRIDADE: Observações onde entrou_em_campo != True presentes!")

    # Target válido
    if df[TARGET].isna().any() or np.isinf(df[TARGET]).any():
        raise RuntimeError("FALHA DE INTEGRIDADE: Target contém valores NaN ou Infinitos!")

    # Unicidade jogador x rodada
    dups = df.duplicated(subset=["temporada", "atleta_id", "rodada"]).sum()
    if dups > 0:
        raise RuntimeError(f"FALHA DE INTEGRIDADE: {dups} duplicatas em (temporada, atleta_id, rodada)!")

    # Features históricas presentes
    for feat in FEATURES_HISTORICO:
        if feat not in df.columns:
            raise RuntimeError(f"FALHA DE INTEGRIDADE: Feature histórica '{feat}' ausente!")

    # Conjunto final de features exato
    if sorted(expected_features) != sorted(actual_features):
        missing = set(expected_features) - set(actual_features)
        extra = set(actual_features) - set(expected_features)
        raise RuntimeError(f"FALHA DE INTEGRIDADE: Features não correspondem ao modo solicitado! Ausentes: {missing} | Extras: {extra}")

# ==============================================================================
# FLUXO PRINCIPAL DO EXPERIMENTO
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP-12 Progressive CatBoost Runner (2026)")
    parser.add_argument("--mode", type=str, required=True, choices=["cumulative", "single"],
                        help="Modo: 'cumulative' ou 'single'")
    parser.add_argument("--tier", type=int, required=True, choices=range(0, 8),
                        help="Índice do Tier (0 a 7)")
    args = parser.parse_args()

    mode = args.mode
    target_tier = args.tier

    # 1. Definição dos Tiers e Features Ativas
    if mode == "cumulative":
        active_tiers = list(range(0, target_tier + 1))
        model_name = f"EXP12_M{target_tier+1}_Cumulative_T0_to_T{target_tier}"
        out_subdir = "reports/exp12/progressive"
        file_prefix = f"exp12_cumulative_tier{target_tier}"
    else:
        active_tiers = [target_tier]
        model_name = f"EXP12_Single_Tier{target_tier}"
        out_subdir = "reports/exp12/single_tiers"
        file_prefix = f"exp12_single_tier{target_tier}"

    context_features_used = []
    for t in active_tiers:
        context_features_used.extend(TIERS_CATALOG[t])

    # Preservar ordem sem duplicatas
    seen = set()
    context_features_used = [x for x in context_features_used if not (x in seen or seen.add(x))]
    expected_feature_list = FEATURES_HISTORICO + context_features_used

    os.makedirs(out_subdir, exist_ok=True)
    folds_csv_path = os.path.join(out_subdir, f"{file_prefix}_folds.csv")
    summary_json_path = os.path.join(out_subdir, f"{file_prefix}_summary.json")
    importance_csv_path = os.path.join(out_subdir, f"{file_prefix}_feature_importance.csv")
    pos_metrics_csv_path = os.path.join(out_subdir, f"{file_prefix}_position_metrics.csv")
    predictions_csv_path = os.path.join(out_subdir, f"{file_prefix}_predictions.csv")

    # 2. Carregamento dos dados
    if not os.path.exists(DATA_PLAYER_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PLAYER_PATH}")
    if not os.path.exists(DATA_CONTEXT_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_CONTEXT_PATH}")

    df_raw = pd.read_parquet(DATA_PLAYER_PATH)
    df_ctx_raw = pd.read_parquet(DATA_CONTEXT_PATH)

    # 3. Engenharia de Histórico e Join de Contexto
    df = compute_individual_history_2026(df_raw)
    df = merge_team_context_explicit(df, df_ctx_raw)

    actual_feature_list = [f for f in expected_feature_list if f in df.columns]

    # 4. Auditoria das Regras Globais
    verify_10_rules(df, expected_feature_list, actual_feature_list)

    # 5. Validação dos Folds Obrigatórios (R02 a R27)
    available_rounds = sorted(df["rodada"].unique())
    for exp_r in EXPECTED_FOLDS:
        if exp_r not in available_rounds:
            raise RuntimeError(f"FALHA DE INTEGRIDADE: Rodada R{exp_r:02d} ausente no dataset!")

    print("=" * 60)
    print("EXP-12 — MODELO PROGRESSIVO 2026")
    print("=" * 60)
    print(f"Modo: {mode}")
    print(f"Tier(s): {active_tiers} ({', '.join(['T'+str(t) for t in active_tiers])})")
    print(f"Nome do Modelo: {model_name}\n")
    print(f"Features históricas: {len(FEATURES_HISTORICO)}")
    for t in active_tiers:
        print(f"Features {TIER_NAMES[t]}: {len(TIERS_CATALOG[t])}")
    print(f"\nTotal de features: {len(actual_feature_list)}")
    print("=" * 60)

    # ==============================================================================
    # 6. VALIDAÇÃO TEMPORAL WALK-FORWARD (R02 A R27)
    # ==============================================================================
    folds_results = []
    all_predictions = []
    fold_feature_importances = []

    print(f"{'Rodada':<7} | {'Treino':<7} | {'Teste':<7} | {'MAE':<8} | {'RMSE':<8} | {'Spearman':<8}")
    print("-" * 56)

    for r in EXPECTED_FOLDS:
        train_df = df[df["rodada"] < r]
        test_df = df[df["rodada"] == r]

        if len(train_df) == 0:
            raise RuntimeError(f"FALHA DE INTEGRIDADE: Treino vazio para rodada R{r:02d}!")
        if len(test_df) == 0:
            raise RuntimeError(f"FALHA DE INTEGRIDADE: Teste vazio para rodada R{r:02d}!")

        # Temporalidade estrita
        if train_df["rodada"].max() >= r:
            raise RuntimeError(f"FALHA DE INTEGRIDADE: Vazamento temporal no treino de R{r:02d}!")
        if not (test_df["rodada"] == r).all():
            raise RuntimeError(f"FALHA DE INTEGRIDADE: Dados de rodadas externas no teste de R{r:02d}!")

        X_train = train_df[actual_feature_list]
        y_train = train_df[TARGET]

        X_test = test_df[actual_feature_list]
        y_test = test_df[TARGET]
        cat_feature_indices = [
            actual_feature_list.index("posicao_id"),
        ]
        if "mando" in actual_feature_list:
            cat_feature_indices.append(actual_feature_list.index("mando"))

        model = CatBoostRegressor(**CATBOOST_PARAMS)

        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
        )

        preds = model.predict(X_test)

        # Importância das features no fold (PredictionValuesChange normalizada)
        raw_imp = model.get_feature_importance(type="PredictionValuesChange")
        sum_imp = np.sum(raw_imp)
        norm_imp = (raw_imp / sum_imp * 100.0) if sum_imp > 0 else np.zeros_like(raw_imp)
        fold_feature_importances.append(norm_imp)

        mae_fold = mean_absolute_error(y_test, preds)
        rmse_fold = np.sqrt(mean_squared_error(y_test, preds))
        spearman_fold, _ = spearmanr(y_test, preds)
        if np.isnan(spearman_fold):
            spearman_fold = 0.0

        folds_results.append({
            "rodada": int(r),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "mae": float(mae_fold),
            "rmse": float(rmse_fold),
            "spearman": float(spearman_fold)
        })

        # Predições individuais com identificadores para auditoria independente
        pred_cols = ["temporada", "rodada", "atleta_id", "posicao_id", TARGET]
        if "clube_id" in test_df.columns:
            pred_cols.insert(3, "clube_id")
        
        df_p = test_df[pred_cols].copy()
        df_p["predicao"] = preds
        all_predictions.append(df_p)

        print(f"R{r:02d}     | {len(train_df):<7d} | {len(test_df):<7d} | {mae_fold:<8.4f} | {rmse_fold:<8.4f} | {spearman_fold:<8.4f}")

    df_all_preds = pd.concat(all_predictions, ignore_index=True)

    # 7. Salvar Predições Individuais (Artefato para Auditoria Independente)
    df_all_preds.to_csv(predictions_csv_path, index=False)

    # 8. Métricas Agregadas Globais
    global_mae = mean_absolute_error(df_all_preds[TARGET], df_all_preds["predicao"])
    global_rmse = np.sqrt(mean_squared_error(df_all_preds[TARGET], df_all_preds["predicao"]))
    global_spearman, _ = spearmanr(df_all_preds[TARGET], df_all_preds["predicao"])

    # 9. Métricas Agregadas por Posição
    pos_metrics_dict = {}
    pos_metrics_list = []
    for pos in sorted(df_all_preds["posicao_id"].unique()):
        df_pos = df_all_preds[df_all_preds["posicao_id"] == pos]
        p_mae = mean_absolute_error(df_pos[TARGET], df_pos["predicao"])
        p_rmse = np.sqrt(mean_squared_error(df_pos[TARGET], df_pos["predicao"]))
        p_spearman, _ = spearmanr(df_pos[TARGET], df_pos["predicao"])
        if np.isnan(p_spearman):
            p_spearman = 0.0

        pos_info = {
            "posicao_id": str(pos),
            "n_predictions": int(len(df_pos)),
            "mae": float(p_mae),
            "rmse": float(p_rmse),
            "spearman": float(p_spearman)
        }
        pos_metrics_dict[str(pos)] = pos_info
        pos_metrics_list.append(pos_info)

    df_pos_metrics = pd.DataFrame(pos_metrics_list)
    df_pos_metrics.to_csv(pos_metrics_csv_path, index=False)

    # 10. Importância Média das Features e Agregação por Grupo
    mean_importances = np.mean(fold_feature_importances, axis=0)
    total_mean_imp = np.sum(mean_importances)
    if total_mean_imp > 0:
        mean_importances = (mean_importances / total_mean_imp) * 100.0

    imp_records = []
    imp_dict = {}
    for feat, imp in zip(actual_feature_list, mean_importances):
        imp_dict[feat] = float(imp)
        imp_records.append({"feature": feat, "importancia_pct": float(imp)})

    df_imp = pd.DataFrame(imp_records).sort_values(by="importancia_pct", ascending=False).reset_index(drop=True)
    df_imp.to_csv(importance_csv_path, index=False)

    group_imp = {"historico": 0.0}
    for t in active_tiers:
        group_imp[f"tier_{t}"] = 0.0

    for feat, imp in zip(actual_feature_list, mean_importances):
        if feat in FEATURES_HISTORICO:
            group_imp["historico"] += imp
        else:
            for t in active_tiers:
                if feat in TIERS_CATALOG[t]:
                    group_imp[f"tier_{t}"] += imp
                    break

    group_imp = {k: float(v) for k, v in group_imp.items()}

    # 11. Salvar Folds CSV e Summary JSON
    df_folds = pd.DataFrame(folds_results)
    df_folds.to_csv(folds_csv_path, index=False)

    summary_data = {
        "experimento": "EXP-12",
        "modo": mode,
        "target_tier": target_tier,
        "tiers_ativos": active_tiers,
        "nome_modelo": model_name,
        "temporada": TEMPORADA,
        "n_folds": int(len(folds_results)),
        "n_predictions": int(len(df_all_preds)),
        "mae_global": float(global_mae),
        "rmse_global": float(global_rmse),
        "spearman_global": float(global_spearman),
        "metricas_por_posicao": pos_metrics_dict,
        "features_total": len(actual_feature_list),
        "features_historico": FEATURES_HISTORICO,
        "features_contexto": context_features_used,
        "importancia_features": imp_dict,
        "importancia_por_grupo": group_imp,
        "catboost_params": CATBOOST_PARAMS,
        "hashes_insumos": {
            "cartola_jogador_rodada_v2": get_file_sha256(DATA_PLAYER_PATH),
            "team_context_features_v1": get_file_sha256(DATA_CONTEXT_PATH),
            "team_scout_history": get_file_sha256(DATA_SCOUT_PATH)
        },
        "integridade": "APROVADA (10/10 verificações)",
        "folds_detalhe": folds_results
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    # ==============================================================================
    # 12. IMPRESSÃO DA SAÍDA OFICIAL NO TERMINAL
    # ==============================================================================
    print("=" * 60)
    print("RESULTADO AGREGADO")
    print("=" * 60)
    print(f"MAE: {global_mae:.4f}")
    print(f"RMSE: {global_rmse:.4f}")
    print(f"Spearman: {global_spearman:.4f}")
    print(f"Predições: {len(df_all_preds)}")

    print("\n" + "-" * 60)
    print("MÉTRICAS POR POSIÇÃO")
    print("-" * 60)
    print(f"{'Posição':<8} | {'N':<6} | {'MAE':<8} | {'RMSE':<8} | {'Spearman':<8}")
    print("-" * 46)
    for p_info in pos_metrics_list:
        print(f"{p_info['posicao_id']:<8} | {p_info['n_predictions']:<6d} | {p_info['mae']:<8.4f} | {p_info['rmse']:<8.4f} | {p_info['spearman']:<8.4f}")

    print("\n" + "-" * 60)
    print("IMPORTÂNCIA DAS FEATURES (Top 10)")
    print("-" * 60)
    print(f"{'Feature':<30} | {'Importância %':<12}")
    print("-" * 46)
    for _, row in df_imp.head(10).iterrows():
        print(f"{row['feature']:<30} | {row['importancia_pct']:<12.2f}")

    print("\n" + "-" * 60)
    print("IMPORTÂNCIA POR GRUPO")
    print("-" * 60)
    for grp, imp_val in group_imp.items():
        print(f"{grp:<15} | {imp_val:.2f}%")

    print("\n" + "-" * 60)
    print("INTEGRIDADE")
    print("-" * 60)
    print("APROVADA (10/10)")
    print("=" * 60)
    print("Arquivos salvos:")
    print(f"  - {folds_csv_path}")
    print(f"  - {summary_json_path}")
    print(f"  - {importance_csv_path}")
    print(f"  - {pos_metrics_csv_path}")
    print(f"  - {predictions_csv_path}\n")


if __name__ == "__main__":
    main()
