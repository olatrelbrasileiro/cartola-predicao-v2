#!/usr/bin/env python3
"""
Predict Rodada Otimizado (v6.2 - 92 Features)
Salva os modelos treinados para reutilização.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
import argparse
import pickle

def detect_root():
    for c in [Path.home() / "cartola-predicao-v2", Path("/data/data/com.termux/files/home/cartola-predicao-v2"), Path(__file__).resolve().parent.parent]:
        if (c / "reports" / "harness_folds_v2.csv").exists(): return c
    raise RuntimeError("Raiz não encontrada")

ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
CONFIG = ROOT / "config"
PRED_DB = ROOT / "data" / "predictions"
MODEL_DB = ROOT / "data" / "models"  # Novo: pasta para salvar modelos
PRED_DB.mkdir(parents=True, exist_ok=True)
MODEL_DB.mkdir(parents=True, exist_ok=True)

CANON_V4 = INTERIM / "cartola_jogador_rodada_v4.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"

FEATURES_92 = [
    "hist_mean", "hist_std", "hist_n", "hist_shrunk_k7",
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    "entrou_rate", "posicao_id",
    "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away",
    "team_FTT_mean_3", "team_FA_mean_3", "team_DS_mean_3", "team_FC_mean_3",
    "team_clean_sheet_rate_3", "team_G_mean_3", "team_FS_mean_3", "team_DE_mean_3", "team_GS_mean_3",
    "opp_FTT_mean_3", "opp_FA_mean_3", "opp_DS_mean_3", "opp_FC_mean_3",
    "opp_clean_sheet_rate_3", "opp_G_mean_3", "opp_FS_mean_3", "opp_DE_mean_3", "opp_GS_mean_3",
    "scout_G_mean_hist", "scout_G_mean_3", "scout_G_mean_5", "scout_G_shrunk", "scout_G_momentum",
    "scout_A_mean_hist", "scout_A_mean_3", "scout_A_mean_5", "scout_A_shrunk", "scout_A_momentum",
    "scout_FD_mean_hist", "scout_FD_mean_3", "scout_FD_mean_5", "scout_FD_shrunk", "scout_FD_momentum",
    "scout_FF_mean_hist", "scout_FF_mean_3", "scout_FF_mean_5", "scout_FF_shrunk", "scout_FF_momentum",
    "scout_FT_mean_hist", "scout_FT_mean_3", "scout_FT_mean_5", "scout_FT_shrunk", "scout_FT_momentum",
    "scout_FS_mean_hist", "scout_FS_mean_3", "scout_FS_mean_5", "scout_FS_shrunk", "scout_FS_momentum",
    "scout_DE_mean_hist", "scout_DE_mean_3", "scout_DE_mean_5", "scout_DE_shrunk", "scout_DE_momentum",
    "scout_SG_mean_hist", "scout_SG_mean_3", "scout_SG_mean_5", "scout_SG_shrunk", "scout_SG_momentum",
    "scout_DS_mean_hist", "scout_DS_mean_3", "scout_DS_mean_5", "scout_DS_shrunk", "scout_DS_momentum",
    "scout_FC_mean_hist", "scout_FC_mean_3", "scout_FC_mean_5", "scout_FC_shrunk", "scout_FC_momentum",
    "scout_GS_mean_hist", "scout_GS_mean_3", "scout_GS_mean_5", "scout_GS_shrunk", "scout_GS_momentum",
    "scout_G_std_5", "scout_A_std_5", "scout_DE_std_5", "scout_SG_std_5",
]

CATBOOST_PARAMS = dict(
    loss_function="MAE", iterations=500, depth=6, learning_rate=0.05,
    l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False
)

def _load_ridge_v2():
    spec = importlib.util.spec_from_file_location("ridge_v2_module", str(ROOT / "scripts" / "ridge_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
rv2 = _load_ridge_v2()

def main():
    parser = argparse.ArgumentParser(description="Gerar predições para uma rodada específica (otimizado)")
    parser.add_argument("--temporada", type=int, required=True)
    parser.add_argument("--rodada", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Forçar retreino do modelo")
    args = parser.parse_args()
    
    t, R = args.temporada, args.rodada
    
    print("=" * 70)
    print(f"🔮 PREDICT RODADA OTIMIZADO")
    print(f"   Temporada: {t} | Rodada: {R}")
    print("=" * 70)
    
    # Caminhos dos arquivos
    pred_file = PRED_DB / f"predictions_{t}_R{R}.parquet"
    model_file = MODEL_DB / f"model_{t}_R{R}.pkl"
    priors_file = MODEL_DB / f"priors_{t}_R{R}.pkl"
    
    # Verificar se predições já existem
    if pred_file.exists() and not args.force:
        print(f"\n✅ Predições já existem: {pred_file}")
        return
    
    # Verificar se modelo já existe
    if model_file.exists() and not args.force:
        print(f"\n📦 Modelo já treinado: {model_file}")
        print("   Carregando modelo...")
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(priors_file, "rb") as f:
            prior_pos, prior_ga, b0_const = pickle.load(f)
        print("   ✅ Modelo carregado")
    else:
        print("\n1. Treinando modelo (dados até R-1)...")
        df = pd.read_parquet(CANON_V4)
        
        # Filtrar dados até R-1
        mask_train = (df["temporada"] < t) | ((df["temporada"] == t) & (df["rodada"] < R))
        df_hist = df[mask_train].copy()
        
        # Processar features
        df_hist["elegivel"] = ~((df_hist["temporada"] == 2025) & (df_hist["rodada"] == 1))
        audit = rv2.load_yaml(AUDIT_YAML)
        scout_cols_basicos = [f"scout_{s}" for s in audit["scouts_basicos"]]
        df_hist = rv2.apply_cumulative_diff(df_hist, scout_cols_basicos + ["scout_G", "scout_A"])
        df_hist["basic_score"] = rv2.compute_basic_score(df_hist, audit)
        
        feats_hist = rv2.compute_participation_features(df_hist)
        
        # Merge com odds e contexto
        ts = pd.read_parquet(TEMPORAL_SAFE)
        ts_cols = ["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
        feats_hist = feats_hist.merge(ts[ts_cols], on=["temporada", "rodada", "atleta_id"], how="left")
        
        new_cols = [c for c in df.columns if c.startswith("scout_") or c.startswith("team_") or c.startswith("opp_")]
        df_new = df[["temporada", "rodada", "atleta_id"] + new_cols].drop_duplicates()
        feats_hist = feats_hist.merge(df_new, on=["temporada", "rodada", "atleta_id"], how="left")
        feats_hist[new_cols] = feats_hist[new_cols].fillna(0)
        
        # Remover scouts crus
        scouts_crus = [c for c in feats_hist.columns if c.startswith("scout_") and not any(x in c for x in ["_mean_", "_shrunk", "_momentum", "_std_"])]
        feats_hist = feats_hist.drop(columns=scouts_crus, errors="ignore")
        
        # Garantir 92 features
        missing_features = {feat: 0.0 for feat in FEATURES_92 if feat not in feats_hist.columns}
        if missing_features:
            feats_hist = feats_hist.assign(**missing_features)
        feats_hist[FEATURES_92] = feats_hist[FEATURES_92].fillna(0.0)
        
        # Filtrar quem entrou em campo
        feats_train = feats_hist[feats_hist["entrou_em_campo"].fillna(False)].copy()
        
        # Calcular priors
        prior_pos, prior_ga, b0_const = rv2.compute_priors(feats_train)
        feats_train = rv2.add_shrunk_features(feats_train, prior_pos, prior_ga, b0_const)
        
        # Treinar modelo
        available = [f for f in FEATURES_92 if f in feats_train.columns]
        cat_feats = ["posicao_id"] if "posicao_id" in available else []
        
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(feats_train[available], feats_train["pontos_num"], cat_features=cat_feats)
        print(f"   ✅ Modelo treinado com {len(feats_train)} amostras")
        
        # Salvar modelo
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        with open(priors_file, "wb") as f:
            pickle.dump((prior_pos, prior_ga, b0_const), f)
        print(f"   💾 Modelo salvo: {model_file}")
    
    # Gerar predições
    print("\n2. Gerando predições para rodada alvo...")
    df = pd.read_parquet(CANON_V4)
    
    # Combinar histórico + rodada alvo
    df_test = df[(df["temporada"] == t) & (df["rodada"] == R)].copy()
    df_combined = pd.concat([df[df["temporada"] < t], df_test], ignore_index=True)
    
    # Processar features
    df_combined["elegivel"] = ~((df_combined["temporada"] == 2025) & (df_combined["rodada"] == 1))
    audit = rv2.load_yaml(AUDIT_YAML)
    scout_cols_basicos = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df_combined = rv2.apply_cumulative_diff(df_combined, scout_cols_basicos + ["scout_G", "scout_A"])
    df_combined["basic_score"] = rv2.compute_basic_score(df_combined, audit)
    
    feats_combined = rv2.compute_participation_features(df_combined)
    
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    feats_combined = feats_combined.merge(ts[ts_cols], on=["temporada", "rodada", "atleta_id"], how="left")
    
    new_cols = [c for c in df.columns if c.startswith("scout_") or c.startswith("team_") or c.startswith("opp_")]
    df_new = df[["temporada", "rodada", "atleta_id"] + new_cols].drop_duplicates()
    feats_combined = feats_combined.merge(df_new, on=["temporada", "rodada", "atleta_id"], how="left")
    feats_combined[new_cols] = feats_combined[new_cols].fillna(0)
    
    scouts_crus = [c for c in feats_combined.columns if c.startswith("scout_") and not any(x in c for x in ["_mean_", "_shrunk", "_momentum", "_std_"])]
    feats_combined = feats_combined.drop(columns=scouts_crus, errors="ignore")
    
    missing_features = {feat: 0.0 for feat in FEATURES_92 if feat not in feats_combined.columns}
    if missing_features:
        feats_combined = feats_combined.assign(**missing_features)
    feats_combined[FEATURES_92] = feats_combined[FEATURES_92].fillna(0.0)
    
    # Filtrar rodada alvo
    test_feats = feats_combined[
        (feats_combined["temporada"] == t) & 
        (feats_combined["rodada"] == R) &
        (feats_combined["entrou_em_campo"].fillna(False))
    ].copy()
    
    if len(test_feats) == 0:
        print(f"\n❌ Nenhum jogador encontrado na rodada {R}")
        return
    
    # Aplicar shrinkage
    test_feats = rv2.add_shrunk_features(test_feats, prior_pos, prior_ga, b0_const)
    available_test = [f for f in FEATURES_92 if f in test_feats.columns]
    
    # Predizer
    pred = model.predict(test_feats[available_test])
    
    # Montar output
    out_df = test_feats[["temporada", "rodada", "atleta_id", "posicao_id", "pontos_num"]].copy()
    out_df["predicao"] = pred
    
    # Rankings
    out_df["rank_geral_pred"] = out_df["predicao"].rank(ascending=False, method="min").astype(int)
    out_df["rank_geral_real"] = out_df["pontos_num"].rank(ascending=False, method="min").astype(int)
    out_df["rank_pos_pred"] = out_df.groupby("posicao_id")["predicao"].rank(ascending=False, method="min").astype(int)
    out_df["rank_pos_real"] = out_df.groupby("posicao_id")["pontos_num"].rank(ascending=False, method="min").astype(int)
    
    # Métricas
    from scipy.stats import spearmanr
    mae = np.mean(np.abs(out_df["predicao"] - out_df["pontos_num"]))
    sp = spearmanr(out_df["predicao"], out_df["pontos_num"])[0]
    
    print("\n" + "=" * 70)
    print("📊 RESULTADOS")
    print("=" * 70)
    print(f"MAE: {mae:.4f}")
    print(f"Spearman: {sp:.4f}")
    print(f"Jogadores: {len(out_df)}")
    
    # Salvar predições
    out_df.to_parquet(pred_file, index=False)
    print(f"\n✅ Predições salvas: {pred_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
