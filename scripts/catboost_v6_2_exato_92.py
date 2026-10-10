#!/usr/bin/env python3
"""
CatBoost v6.2 — EXATAMENTE 92 Features (Sem Leakage)
Usa APENAS as 92 features definidas, removendo scouts crus da rodada atual.
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

CANON_V2 = INTERIM / "cartola_jogador_rodada_v2.parquet"
SCOUTS_CALCULADOS = INTERIM / "features_scouts_rodada_v1.parquet"
CONTEXT_V3 = INTERIM / "features_team_context_v3.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"

OUT_FOLDS = REP / "catboost_v6_2_folds.csv"
OUT_SUMMARY = REP / "catboost_v6_2_summary.md"
OUT_FI = REP / "catboost_v6_2_feature_importance.csv"
OUT_GI = REP / "catboost_v6_2_group_importance.csv"
OUT_PREDS = REP / "catboost_v6_2_predicoes.csv"

TARGET = "pontos_num"
CATBOOST_PARAMS = dict(loss_function="MAE", iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False)

# LISTA EXATA DAS 92 FEATURES DEFINIDAS
FEATURES_92 = [
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
    print("CATBOOST v6.2 — EXATAMENTE 92 FEATURES (SEM LEAKAGE)")
    print("=" * 70)
    
    print("\n1. Carregando dataset base...")
    df = pd.read_parquet(CANON_V2)
    print(f"   Shape: {df.shape}")
    
    print("\n2. Processando features históricas (até R-1)...")
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    audit = rv2.load_yaml(AUDIT_YAML)
    scout_cols_basicos = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols_basicos + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)
    
    feats = rv2.compute_participation_features(df)
    
    print("\n3. Merge com Odds do jogo...")
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    feats = feats.merge(ts[ts_cols], on=["temporada", "rodada", "atleta_id"], how="left")
    
    print("\n4. Merge com Scouts Granulares CALCULADOS (até R-1)...")
    df_scouts = pd.read_parquet(SCOUTS_CALCULADOS)
    scout_cols = [c for c in df_scouts.columns if c not in ["temporada", "rodada", "atleta_id", "posicao_id"]]
    feats = feats.merge(df_scouts[["temporada", "rodada", "atleta_id"] + scout_cols], on=["temporada", "rodada", "atleta_id"], how="left")
    feats[scout_cols] = feats[scout_cols].fillna(0)
    
    print("\n5. Merge com Contexto Time/Adversário (até R-1)...")
    df_ctx = pd.read_parquet(CONTEXT_V3)
    ctx_cols = [c for c in df_ctx.columns if c not in ["temporada", "rodada", "atleta_id", "clube_id"]]
    feats = feats.merge(df_ctx[["temporada", "rodada", "atleta_id"] + ctx_cols], on=["temporada", "rodada", "atleta_id"], how="left")
    feats[ctx_cols] = feats[ctx_cols].fillna(0)
    
    print("\n6. REMOVENDO scouts crus da rodada atual (LEAKAGE)...")
    scouts_crus = [c for c in feats.columns if c.startswith("scout_") and not any(x in c for x in ["_mean_", "_shrunk", "_momentum", "_std_"])]
    print(f"   ❌ Removendo {len(scouts_crus)} scouts crus")
    feats = feats.drop(columns=scouts_crus, errors="ignore")
    
    print("\n7. GARANTINDO AS 92 FEATURES EXATAS...")
    missing_count = 0
    for feat in FEATURES_92:
        if feat not in feats.columns:
            feats[feat] = 0.0
            missing_count += 1
    
    if missing_count > 0:
        print(f"   ⚠️  {missing_count} features não estavam no dataset e foram criadas com valor 0.0")
    else:
        print("   ✅ Todas as 92 features já estavam presentes!")
    
    feats[FEATURES_92] = feats[FEATURES_92].fillna(0.0)
    print(f"   ✅ Features finais: {len([c for c in feats.columns if c in FEATURES_92])} (deveria ser 92)")

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
        
        # Usa EXATAMENTE as 92 features
        available = [f for f in FEATURES_92 if f in tr.columns]
        cat_feats = ["posicao_id"] if "posicao_id" in available else []
        
        if idx == 0:
            print(f"   [CONFIRMAÇÃO] Features utilizadas no Fold 1: {len(available)} (Deveria ser 92)")
        
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

    # 9. Resultados
    dfr = pd.DataFrame(rows_fold)
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"] ** 2 * dfr["n_test"]).sum() / total_n))
    sp_global = float(dfr["spearman"].mean())
    
    print("\n" + "=" * 70)
    print("RESULTADO FINAL v6.2 (92 FEATURES EXATAS, SEM LEAKAGE)")
    print("=" * 70)
    print(f"MAE Global:     {mae_global:.4f}")
    print(f"RMSE Global:    {rmse_global:.4f}")
    print(f"Spearman Médio: {sp_global:.4f}")
    print(f"Features Usadas: {len(available)}")
    print("=" * 70)
    print("✅ Este modelo usa EXATAMENTE as 92 features definidas.")
    print("   Scouts calculados até R-1 (sem leakage).")
    print("=" * 70)
    
    dfr.to_csv(OUT_FOLDS, index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_PREDS, index=False)
    
    fi_df = pd.DataFrame([{"feature": f, "importance": np.mean(imp)} for f, imp in fi_dict.items()])
    fi_df.sort_values("importance", ascending=False).to_csv(OUT_FI, index=False)
    
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"# CatBoost v6.2 (92 Features Exatas)\n- Features: {len(available)}\n- MAE: {mae_global:.4f}\n- Spearman: {sp_global:.4f}\n")
    
    print(f"✅ Artefatos salvos em {REP}")

if __name__ == "__main__":
    main()
