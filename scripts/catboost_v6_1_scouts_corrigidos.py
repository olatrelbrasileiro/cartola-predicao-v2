#!/usr/bin/env python3
"""
CatBoost v6.1 — Scouts Corrigidos (Sem Leakage)
Usa scouts granulares calculados até R-1 + contexto + odds.
Remove APENAS scouts crus da rodada atual.
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
SCOUTS_CALCULADOS = INTERIM / "features_scouts_rodada_v1.parquet"  # Scouts calculados até R-1
CONTEXT_V3 = INTERIM / "features_team_context_v3.parquet"  # Contexto calculado até R-1
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"

OUT_FOLDS = REP / "catboost_v6_1_folds.csv"
OUT_SUMMARY = REP / "catboost_v6_1_summary.md"
OUT_FI = REP / "catboost_v6_1_feature_importance.csv"
OUT_GI = REP / "catboost_v6_1_group_importance.csv"
OUT_PREDS = REP / "catboost_v6_1_predicoes.csv"

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
    print("CATBOOST v6.1 — SCOUTS CORRIGIDOS (SEM LEAKAGE)")
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
    print(f"   ✅ Scouts calculados mergeados: {len(scout_cols)} features")
    
    print("\n5. Merge com Contexto Time/Adversário (até R-1)...")
    df_ctx = pd.read_parquet(CONTEXT_V3)
    ctx_cols = [c for c in df_ctx.columns if c not in ["temporada", "rodada", "atleta_id", "clube_id"]]
    feats = feats.merge(df_ctx[["temporada", "rodada", "atleta_id"] + ctx_cols], on=["temporada", "rodada", "atleta_id"], how="left")
    feats[ctx_cols] = feats[ctx_cols].fillna(0)
    print(f"   ✅ Contexto mergeado: {len(ctx_cols)} features")
    
    print("\n6. REMOVENDO scouts crus da rodada atual (LEAKAGE)...")
    # Remove APENAS os scouts crus (scout_G, scout_A, etc.) que vêm do dataset v2
    # Mantém os scouts calculados (scout_G_mean_hist, scout_G_shrunk, etc.)
    scouts_crus = [c for c in feats.columns if c.startswith("scout_") and "_" not in c.replace("scout_", "")]
    # Isso remove: scout_G, scout_A, scout_DE, etc. (valores da rodada atual)
    # Mas mantém: scout_G_mean_hist, scout_G_shrunk, etc. (calculados até R-1)
    
    # Na verdade, o padrão é: scout_X é cru, scout_X_* é calculado
    scouts_crus = [c for c in feats.columns if c.startswith("scout_") and not any(x in c for x in ["_mean_", "_shrunk", "_momentum", "_std_"])]
    
    print(f"   ❌ Removendo {len(scouts_crus)} scouts crus: {scouts_crus[:10]}...")
    feats = feats.drop(columns=scouts_crus, errors="ignore")
    
    # Define features finais (tudo que sobrou, exceto metadados)
    exclude_cols = ["temporada", "rodada", "atleta_id", "clube_id", "entrou_em_campo", "elegivel", "pontos_num", "basic_score", "_is_part", "n_prev", "sum_prev", "ga_rate_last", "diff_scout_G", "diff_scout_A"]
    final_features = [c for c in feats.columns if c not in exclude_cols]
    
    print(f"\n   ✅ Features finais: {len(final_features)}")
    print(f"   - Históricas: {len([f for f in final_features if f.startswith('hist_') or f.startswith('rec_') or f in ['momentum', 'entrou_rate', 'is_cold_start', 'posicao_id']])}")
    print(f"   - Scouts calculados: {len([f for f in final_features if f.startswith('scout_')])}")
    print(f"   - Contexto: {len([f for f in final_features if f.startswith('team_') or f.startswith('opp_')])}")
    print(f"   - Confronto: {len([f for f in final_features if f in ['p_home', 'p_draw', 'p_away', 'mando', 'favoritismo_home_away']])}")

    print("\n7. Iniciando treino walk-forward...")
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
        
        final_features_fold = [f for f in final_features if f in tr.columns]
        cat_feats = ["posicao_id"] if "posicao_id" in final_features_fold else []
        
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(tr[final_features_fold], tr[TARGET], cat_features=cat_feats)
        pred = m.predict(te[final_features_fold])
        y_te = te[TARGET].to_numpy(dtype=float)
        
        for feat, imp in zip(final_features_fold, m.get_feature_importance()):
            fi_dict.setdefault(feat, []).append(imp)
            
        rows_fold.append({"temporada": t, "rodada_alvo": R, "n_train": len(tr), "n_test": len(te), 
                          "mae": mae(pred, y_te), "rmse": rmse(pred, y_te), "spearman": spearman(pred, y_te)})
        
        te_res = te[["temporada", "rodada", "atleta_id", "posicao_id", TARGET]].copy()
        te_res["predicao"], te_res["erro_absoluto"], te_res["fold"] = pred, np.abs(pred - y_te), f"{t}_R{R}"
        all_predictions.append(te_res)
        
        if (idx + 1) % 15 == 0:
            print(f"   Fold {idx+1}/{len(folds)} concluído. MAE={mae(pred, y_te):.4f}, SP={spearman(pred, y_te):.4f}")

    # 8. Resultados
    dfr = pd.DataFrame(rows_fold)
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"] ** 2 * dfr["n_test"]).sum() / total_n))
    sp_global = float(dfr["spearman"].mean())
    
    print("\n" + "=" * 70)
    print("RESULTADO FINAL v6.1 (SCOUTS CORRIGIDOS)")
    print("=" * 70)
    print(f"MAE Global:     {mae_global:.4f}")
    print(f"RMSE Global:    {rmse_global:.4f}")
    print(f"Spearman Médio: {sp_global:.4f}")
    print(f"Features Usadas: {len(final_features_fold)}")
    print("=" * 70)
    print("✅ Este modelo usa scouts granulares calculados até R-1.")
    print("   É 100% seguro para uso prático (escala na sexta-feira).")
    print("=" * 70)
    
    dfr.to_csv(OUT_FOLDS, index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_PREDS, index=False)
    
    fi_df = pd.DataFrame([{"feature": f, "importance": np.mean(imp)} for f, imp in fi_dict.items()])
    fi_df.sort_values("importance", ascending=False).to_csv(OUT_FI, index=False)
    
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"# CatBoost v6.1 (Scouts Corrigidos)\n- Features: {len(final_features_fold)}\n- MAE: {mae_global:.4f}\n- Spearman: {sp_global:.4f}\n")
    
    print(f"✅ Artefatos salvos em {REP}")

if __name__ == "__main__":
    main()
