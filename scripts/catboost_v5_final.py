#!/usr/bin/env python3
"""
CatBoost v5.1 — Modelo Final com Poda Inteligente
Usa o dataset v4 (que tem os scouts corrigidos) e remove apenas o ruído comprovado.
"""
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

def detect_root():
    for c in [Path.home() / "cartola-predicao-v2", Path("/data/data/com.termux/files/home/cartola-predicao-v2"), Path(__file__).resolve().parent.parent]:
        if (c / "reports" / "harness_folds_v2.csv").exists(): return c
    raise RuntimeError("Raiz não encontrada")

ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
CONFIG = ROOT / "config"

# Usamos o v4 que já tem os scouts corrigidos (DE, SG, G, A, etc.)
CANON_V4 = INTERIM / "cartola_jogador_rodada_v4.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"

OUT_FOLDS = REP / "catboost_v5_final_folds.csv"
OUT_SUMMARY = REP / "catboost_v5_final_summary.md"
OUT_FI = REP / "catboost_v5_final_feature_importance.csv"
OUT_GI = REP / "catboost_v5_final_group_importance.csv"
OUT_PREDS = REP / "catboost_v5_final_predicoes.csv"

TARGET = "pontos_num"
CATBOOST_PARAMS = dict(loss_function="MAE", iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False)

def _load_ridge_v2():
    spec = importlib.util.spec_from_file_location("ridge_v2_module", str(ROOT / "scripts" / "ridge_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
rv2 = _load_ridge_v2()

def mae(p, y): return float(np.mean(np.abs(p - y))) if len(y) else np.nan
def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan
def spearman(p, y):
    if len(p) < 2 or np.all(p == p[0]) or np.all(y == y[0]): return np.nan
    rx, ry = pd.Series(p).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def main():
    inicio = datetime.now()
    print("=" * 70)
    print("CATBOOST v5.1 — MODELO FINAL COM PODA INTELIGENTE")
    print("=" * 70)
    
    # 1. Carregar dados
    print("\n1. Carregando dataset v4 (com scouts corrigidos)...")
    df = pd.read_parquet(CANON_V4)
    print(f"   Shape original: {df.shape}")
    
    # 2. Definir lista de features a REMOVER (o ruído que concordamos em cortar)
    features_to_remove = [
        "is_cold_start", "basic_mean", "basic_rec_5", "ga_per_game", "ga_rec_5", "ga_shrunk_k7",
        "diff_FTT", "diff_FA", "diff_DS",
        # Scouts de ruído comprovado
        "scout_PP_mean_hist", "scout_PP_mean_3", "scout_PP_mean_5", "scout_PP_shrunk", "scout_PP_momentum",
        "scout_DP_mean_hist", "scout_DP_mean_3", "scout_DP_mean_5", "scout_DP_shrunk", "scout_DP_momentum", "scout_DP_std_5",
        "scout_PS_mean_hist", "scout_PS_mean_3", "scout_PS_mean_5", "scout_PS_shrunk", "scout_PS_momentum",
        "scout_CV_mean_hist", "scout_CV_mean_3", "scout_CV_mean_5", "scout_CV_shrunk", "scout_CV_momentum",
        "scout_PC_mean_hist", "scout_PC_mean_3", "scout_PC_mean_5", "scout_PC_shrunk", "scout_PC_momentum",
        "scout_CA_mean_hist", "scout_CA_mean_3", "scout_CA_mean_5", "scout_CA_shrunk", "scout_CA_momentum",
        "scout_I_mean_hist", "scout_I_mean_3", "scout_I_mean_5", "scout_I_shrunk", "scout_I_momentum",
    ]
    
    # 3. Preparar dataframe
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    audit = rv2.load_yaml(AUDIT_YAML)
    scout_cols_basicos = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols_basicos + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)
    
    feats = rv2.compute_participation_features(df)
    ts = pd.read_parquet(TEMPORAL_SAFE)
    feats = feats.merge(ts[["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]], on=["temporada", "rodada", "atleta_id"], how="left")
    
    # Merge com scouts e contexto do v4
    new_cols = [c for c in df.columns if c.startswith("scout_") or c.startswith("team_") or c.startswith("opp_")]
    df_new = df[["temporada", "rodada", "atleta_id"] + new_cols].drop_duplicates()
    feats = feats.merge(df_new, on=["temporada", "rodada", "atleta_id"], how="left")
    feats[new_cols] = feats[new_cols].fillna(0)
    
    # 4. Definir features finais (Todas as colunas, menos as removidas e metadados)
    exclude_cols = ["temporada", "rodada", "atleta_id", "clube_id", "entrou_em_campo", "elegivel", "pontos_num", "basic_score", "_is_part", "n_prev", "sum_prev", "ga_rate_last"]
    exclude_cols.extend(features_to_remove)
    
    # Filtrar apenas colunas que realmente existem no dataframe
    final_features = [c for c in feats.columns if c not in exclude_cols and not c.startswith("diff_scout_")]
    
    print(f"\n2. Configuração de Features:")
    print(f"   Total de colunas no df: {len(feats.columns)}")
    print(f"   Features a remover (ruído): {len(features_to_remove)}")
    print(f"   ✅ FEATURES FINAIS UTILIZADAS: {len(final_features)}")

    # 5. Loop de treino
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"\n3. Iniciando treino ({len(folds)} folds)...")
    
    rows_fold, all_predictions, fi_dict = [], [], {}
    
    for idx, (t, R) in enumerate(folds):
        mask_train = (((df["temporada"] < t) | ((df["temporada"] == t) & (df["rodada"] < R))) & feats["elegivel"])
        tr = feats[mask_train][feats[mask_train]["entrou_em_campo"].fillna(False)].copy()
        
        mask_test = ((df["temporada"] == t) & (df["rodada"] == R) & feats["elegivel"])
        te = feats[mask_test][feats[mask_test]["entrou_em_campo"].fillna(False)].copy()
        
        if len(tr) == 0 or len(te) == 0: continue
        
        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te = rv2.add_shrunk_features(te, prior_pos, prior_ga, b0_const)
        
        # Garantir que usamos apenas features que estão no treino
        available = [f for f in final_features if f in tr.columns]
        cat_feats = ["posicao_id"] if "posicao_id" in available else []
        
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(tr[available], tr[TARGET], cat_features=cat_feats)
        pred = m.predict(te[available])
        y_te = te[TARGET].to_numpy(dtype=float)
        
        # Acumular FI
        for feat, imp in zip(available, m.get_feature_importance()):
            fi_dict.setdefault(feat, []).append(imp)
            
        rows_fold.append({"temporada": t, "rodada_alvo": R, "n_train": len(tr), "n_test": len(te), 
                          "mae": mae(pred, y_te), "rmse": rmse(pred, y_te), "spearman": spearman(pred, y_te)})
        
        te_res = te[["temporada", "rodada", "atleta_id", "posicao_id", TARGET]].copy()
        te_res["predicao"], te_res["erro_absoluto"], te_res["fold"] = pred, np.abs(pred - y_te), f"{t}_R{R}"
        all_predictions.append(te_res)
        
        if (idx + 1) % 15 == 0:
            print(f"   Fold {idx+1}/{len(folds)} concluído. MAE={mae(pred, y_te):.4f}")

    # 6. Resultados
    dfr = pd.DataFrame(rows_fold)
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"] ** 2 * dfr["n_test"]).sum() / total_n))
    sp_global = float(dfr["spearman"].mean())
    
    print("\n" + "=" * 70)
    print("RESULTADO FINAL v5.1 (PODA INTELIGENTE)")
    print("=" * 70)
    print(f"MAE Global:     {mae_global:.4f}")
    print(f"RMSE Global:    {rmse_global:.4f}")
    print(f"Spearman Médio: {sp_global:.4f}")
    print(f"Features Usadas: {len(available)}")
    print("=" * 70)
    
    dfr.to_csv(OUT_FOLDS, index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_PREDS, index=False)
    
    fi_df = pd.DataFrame([{"feature": f, "importance": np.mean(imp)} for f, imp in fi_dict.items()])
    fi_df.sort_values("importance", ascending=False).to_csv(OUT_FI, index=False)
    
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"# CatBoost v5.1 (Poda Inteligente)\n- Features: {len(available)}\n- MAE: {mae_global:.4f}\n- Spearman: {sp_global:.4f}\n")
    
    print(f"✅ Predições e métricas salvas. Rode a análise por posição agora.")

if __name__ == "__main__":
    main()
