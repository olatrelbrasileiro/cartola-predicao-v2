#!/usr/bin/env python3
"""
CatBoost v7 — Poda Inteligente Sem Leakage
Replica EXATAMENTE a estratégia do v5.1 (82 features, MAE 2.30, SP 0.54)
mas remove os scouts crus da rodada atual para eliminar o leakage.
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

CANON_V4 = INTERIM / "cartola_jogador_rodada_v4.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"

OUT_FOLDS = REP / "catboost_v7_folds.csv"
OUT_SUMMARY = REP / "catboost_v7_summary.md"
OUT_FI = REP / "catboost_v7_feature_importance.csv"
OUT_GI = REP / "catboost_v7_group_importance.csv"
OUT_PREDS = REP / "catboost_v7_predicoes.csv"

TARGET = "pontos_num"
CATBOOST_PARAMS = dict(loss_function="MAE", iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False)

# MESMA LISTA DE REMOÇÃO DO v5.1 (features de ruído)
FEATURES_RUIDO = [
    "is_cold_start", "basic_mean", "basic_rec_5", "ga_per_game", "ga_rec_5", "ga_shrunk_k7",
    "diff_FTT", "diff_FA", "diff_DS",
    "scout_PP_mean_hist", "scout_PP_mean_3", "scout_PP_mean_5", "scout_PP_shrunk", "scout_PP_momentum",
    "scout_DP_mean_hist", "scout_DP_mean_3", "scout_DP_mean_5", "scout_DP_shrunk", "scout_DP_momentum", "scout_DP_std_5",
    "scout_PS_mean_hist", "scout_PS_mean_3", "scout_PS_mean_5", "scout_PS_shrunk", "scout_PS_momentum",
    "scout_CV_mean_hist", "scout_CV_mean_3", "scout_CV_mean_5", "scout_CV_shrunk", "scout_CV_momentum",
    "scout_PC_mean_hist", "scout_PC_mean_3", "scout_PC_mean_5", "scout_PC_shrunk", "scout_PC_momentum",
    "scout_CA_mean_hist", "scout_CA_mean_3", "scout_CA_mean_5", "scout_CA_shrunk", "scout_CA_momentum",
    "scout_I_mean_hist", "scout_I_mean_3", "scout_I_mean_5", "scout_I_shrunk", "scout_I_momentum",
]

# NOVO: Scouts crus da rodada atual (LEAKAGE)
# Estes são os valores da partida atual, não os calculados até R-1
SCOUTS_CRUS = [
    "scout_G", "scout_A", "scout_FD", "scout_FF", "scout_FT", "scout_FS",
    "scout_DE", "scout_SG", "scout_DS", "scout_FC", "scout_DP", "scout_GS",
    "scout_CA", "scout_CV", "scout_I", "scout_PP", "scout_PS", "scout_PC",
]

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
    print("CATBOOST v7 — PODA INTELIGENTE SEM LEAKAGE")
    print("Replicando estratégia v5.1 + removendo scouts crus")
    print("=" * 70)
    
    print("\n1. Carregando dataset v4...")
    df = pd.read_parquet(CANON_V4)
    print(f"   Shape: {df.shape}")
    
    print("\n2. Processando base (diffs, scores, participação)...")
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    audit = rv2.load_yaml(AUDIT_YAML)
    scout_cols_basicos = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols_basicos + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)
    
    feats = rv2.compute_participation_features(df)
    
    print("\n3. Merge com confronto...")
    ts = pd.read_parquet(TEMPORAL_SAFE)
    feats = feats.merge(ts[["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]], on=["temporada", "rodada", "atleta_id"], how="left")
    
    print("\n4. Merge com scouts e contexto do v4...")
    new_cols = [c for c in df.columns if c.startswith("scout_") or c.startswith("team_") or c.startswith("opp_")]
    df_new = df[["temporada", "rodada", "atleta_id"] + new_cols].drop_duplicates()
    feats = feats.merge(df_new, on=["temporada", "rodada", "atleta_id"], how="left")
    feats[new_cols] = feats[new_cols].fillna(0)
    
    print("\n5. REMOVENDO scouts crus da rodada atual (LEAKAGE)...")
    scouts_crus_encontrados = [c for c in SCOUTS_CRUS if c in feats.columns]
    print(f"   ❌ Removendo {len(scouts_crus_encontrados)} scouts crus: {scouts_crus_encontrados}")
    feats = feats.drop(columns=scouts_crus_encontrados, errors="ignore")
    
    print("\n6. REMOVENDO features de ruído (mesma poda do v5.1)...")
    ruido_encontrado = [c for c in FEATURES_RUIDO if c in feats.columns]
    print(f"   ❌ Removendo {len(ruido_encontrado)} features de ruído")
    feats = feats.drop(columns=ruido_encontrado, errors="ignore")
    
    # Define features finais
    exclude_cols = ["temporada", "rodada", "atleta_id", "clube_id", "entrou_em_campo", "elegivel", "pontos_num", "basic_score", "_is_part", "n_prev", "sum_prev", "ga_rate_last"]
    # Também remove diff_scout_* que são intermediários
    diff_cols = [c for c in feats.columns if c.startswith("diff_scout_")]
    exclude_cols.extend(diff_cols)
    
    final_features = [c for c in feats.columns if c not in exclude_cols]
    
    print(f"\n7. Configuração final:")
    print(f"   ✅ Features utilizadas: {len(final_features)}")
    print(f"   - Históricas: {len([f for f in final_features if f.startswith('hist_') or f.startswith('rec_') or f in ['momentum', 'entrou_rate', 'posicao_id']])}")
    print(f"   - Scouts calculados: {len([f for f in final_features if f.startswith('scout_')])}")
    print(f"   - Contexto: {len([f for f in final_features if f.startswith('team_') or f.startswith('opp_')])}")
    print(f"   - Confronto: {len([f for f in final_features if f in ['p_home', 'p_draw', 'p_away', 'mando', 'favoritismo_home_away']])}")

    print("\n8. Iniciando treino walk-forward...")
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    
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
        
        available = [f for f in final_features if f in tr.columns]
        cat_feats = ["posicao_id"] if "posicao_id" in available else []
        
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(tr[available], tr[TARGET], cat_features=cat_feats)
        pred = m.predict(te[available])
        y_te = te[TARGET].to_numpy(dtype=float)
        
        for feat, imp in zip(available, m.get_feature_importance()):
            fi_dict.setdefault(feat, []).append(imp)
            
        rows_fold.append({"temporada": t, "rodada_alvo": R, "n_train": len(tr), "n_test": len(te), 
                          "mae": mae(pred, y_te), "rmse": rmse(pred, y_te), "spearman": spearman(pred, y_te)})
        
        te_res = te[["temporada", "rodada", "atleta_id", "posicao_id", TARGET]].copy()
        te_res["predicao"], te_res["erro_absoluto"], te_res["fold"] = pred, np.abs(pred - y_te), f"{t}_R{R}"
        all_predictions.append(te_res)
        
        if (idx + 1) % 15 == 0:
            print(f"   Fold {idx+1}/{len(folds)} concluído. MAE={mae(pred, y_te):.4f}, SP={spearman(pred, y_te):.4f}")

    dfr = pd.DataFrame(rows_fold)
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"] ** 2 * dfr["n_test"]).sum() / total_n))
    sp_global = float(dfr["spearman"].mean())
    
    print("\n" + "=" * 70)
    print("RESULTADO FINAL v7 (PODA INTELIGENTE SEM LEAKAGE)")
    print("=" * 70)
    print(f"MAE Global:     {mae_global:.4f}")
    print(f"RMSE Global:    {rmse_global:.4f}")
    print(f"Spearman Médio: {sp_global:.4f}")
    print(f"Features Usadas: {len(available)}")
    print("=" * 70)
    print("Comparação:")
    print(f"  v2 (Pooling):     MAE=2.6322  SP=0.3416  (baseline)")
    print(f"  v5.1 (c/leakage): MAE=2.3009  SP=0.5434  (teto irreal)")
    print(f"  v7 (sem leakage): MAE={mae_global:.4f}  SP={sp_global:.4f}  (real)")
    print("=" * 70)
    
    dfr.to_csv(OUT_FOLDS, index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_PREDS, index=False)
    
    fi_df = pd.DataFrame([{"feature": f, "importance": np.mean(imp)} for f, imp in fi_dict.items()])
    fi_df.sort_values("importance", ascending=False).to_csv(OUT_FI, index=False)
    
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"# CatBoost v7 (Poda Sem Leakage)\n- Features: {len(available)}\n- MAE: {mae_global:.4f}\n- Spearman: {sp_global:.4f}\n")
    
    print(f"✅ Artefatos salvos em {REP}")

if __name__ == "__main__":
    main()
