#!/usr/bin/env python3
"""
CatBoost v5 — Modelo Final (92 features) - VERSÃO CORRIGIDA
Correção: Merge adequado de TODAS as features (incluindo scouts) no dataframe de treino.
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
    for c in [Path.home() / "cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path("/data/data/com.termux/files/home/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c / "reports" / "harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz não encontrada")

ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
CONFIG = ROOT / "config"

CANON_V5 = INTERIM / "cartola_jogador_rodada_v5.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"

OUT_FOLDS = REP / "catboost_v5_folds.csv"
OUT_SUMMARY = REP / "catboost_v5_summary.md"
OUT_FI = REP / "catboost_v5_feature_importance.csv"
OUT_GI = REP / "catboost_v5_group_importance.csv"
OUT_PREDS = REP / "catboost_v5_predicoes_detalhadas.csv"

TARGET = "pontos_num"

CATBOOST_PARAMS = dict(
    loss_function="MAE", iterations=500, depth=6, learning_rate=0.05,
    l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False,
)

# Lista das 92 features finais
FEATURES_COM = [
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

CAT_FEATURES = ["posicao_id"]

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

def fit_predict_catboost(X_tr, y_tr, X_te, cat_features):
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(X_tr, y_tr, cat_features=cat_features)
    return m.predict(X_te), m

def main():
    inicio = datetime.now()
    print("=" * 70)
    print("CATBOOST v5 — MODELO FINAL (92 FEATURES) - CORRIGIDO")
    print("=" * 70)
    print(f"Início: {inicio.isoformat(timespec='seconds')}")

    print("\n1. Carregando dataset canônico v5...")
    df = pd.read_parquet(CANON_V5)
    print(f"   Shape: {df.shape}")

    print("\n2. Criando coluna elegivel...")
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("\n3. Processando diffs e scores básicos...")
    audit = rv2.load_yaml(AUDIT_YAML)
    scout_cols_basicos = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols_basicos + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    print("\n4. Computando features de participação (Base)...")
    feats = rv2.compute_participation_features(df)

    print("\n5. Merge com features de confronto...")
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    feats = feats.merge(ts[ts_cols], on=["temporada", "rodada", "atleta_id"], how="left")

    print("\n6. Merge com Scouts Granulares e Contexto (CORREÇÃO AQUI)...")
    # Pegar TODAS as features novas do df (scouts + contexto)
    new_cols = [c for c in df.columns if c.startswith("scout_") or c.startswith("team_") or c.startswith("opp_")]
    df_new = df[["temporada", "rodada", "atleta_id"] + new_cols].drop_duplicates()
    
    feats = feats.merge(df_new, on=["temporada", "rodada", "atleta_id"], how="left")
    feats[new_cols] = feats[new_cols].fillna(0)
    
    print(f"   Total de colunas no dataframe de treino: {len(feats.columns)}")

    print("\n7. Carregar harness...")
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"   Folds: {len(folds)}")

    print("\n8. Iniciando treino walk-forward...")
    rows_fold, all_predictions, fi_dict = [], [], {}
    n_folds_fi = 0

    for idx, (t, R) in enumerate(folds):
        mask_train = (((df["temporada"] < t) | ((df["temporada"] == t) & (df["rodada"] < R))) & feats["elegivel"])
        tr = feats[mask_train][feats[mask_train]["entrou_em_campo"].fillna(False)].copy()
        
        mask_test = ((df["temporada"] == t) & (df["rodada"] == R) & feats["elegivel"])
        te_metric = feats[mask_test][feats[mask_test]["entrou_em_campo"].fillna(False)].copy()
        
        if len(tr) == 0 or len(te_metric) == 0: continue
        
        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te_metric = rv2.add_shrunk_features(te_metric, prior_pos, prior_ga, b0_const)
        
        # Filtrar APENAS as 92 features definidas que existem no dataframe
        available_features = [f for f in FEATURES_COM if f in tr.columns]
        cat_features = ["posicao_id"] if "posicao_id" in available_features else []
        
        print(f"   [DEBUG Fold {idx+1}] Features disponíveis: {len(available_features)}") # Monitoramento
        
        X_tr, y_tr = tr[available_features], tr[TARGET].to_numpy(dtype=float)
        X_te, y_te = te_metric[available_features], te_metric[TARGET].to_numpy(dtype=float)

        pred, model = fit_predict_catboost(X_tr, y_tr, X_te, cat_features)
        
        fi_values = model.get_feature_importance()
        for feat_name, imp_value in zip(available_features, fi_values):
            fi_dict.setdefault(feat_name, []).append(imp_value)
        n_folds_fi += 1

        rows_fold.append({"temporada": t, "rodada_alvo": R, "n_train": len(tr), "n_test": len(te_metric), 
                          "mae": mae(pred, y_te), "rmse": rmse(pred, y_te), "spearman": spearman(pred, y_te)})
        
        te_res = te_metric[["temporada", "rodada", "atleta_id", "posicao_id", TARGET]].copy()
        te_res["predicao"], te_res["erro_absoluto"], te_res["fold"] = pred, np.abs(pred - y_te), f"{t}_R{R}"
        all_predictions.append(te_res)
        
        if (idx + 1) % 10 == 0:
            print(f"   fold {idx+1}/{len(folds)}: {t}/R{R} n_tr={len(tr)} n_te={len(te_metric)} MAE={mae(pred, y_te):.4f} SP={spearman(pred, y_te):.4f}")

    dfr = pd.DataFrame(rows_fold)
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"] ** 2 * dfr["n_test"]).sum() / total_n))
    sp_global = float(dfr["spearman"].mean())

    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS — CatBoost v5 (CORRIGIDO)")
    print("=" * 70)
    print(f"MAE global:     {mae_global:.4f}")
    print(f"RMSE global:    {rmse_global:.4f}")
    print(f"Spearman médio: {sp_global:.4f}")
    print(f"Tempo total:    {datetime.now() - inicio}")
    print(f"Features utilizadas: {len(available_features)} (Deveria ser 92)")
    print("=" * 70)

    dfr.to_csv(OUT_FOLDS, index=False)
    
    fi_df = pd.DataFrame([{"feature": feat, "importance": np.mean(values)} for feat, values in fi_dict.items()])
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(OUT_FI, index=False)
    
    GRUPO_FEAT = {
        "Pontos_Gerais": ["hist_mean", "hist_std", "hist_n", "hist_shrunk_k7", "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum"],
        "Oportunidade_Posicao": ["entrou_rate", "posicao_id"],
        "Confronto": ["p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"],
        "Contexto_Time": [f for f in available_features if f.startswith("team_")],
        "Contexto_Adversario": [f for f in available_features if f.startswith("opp_")],
        "Scouts_Granulares": [f for f in available_features if f.startswith("scout_")],
    }
    
    grp_rows = [{"grupo": g, "importance_total": float(fi_df[fi_df["feature"].isin(fgs)]["importance"].sum())} for g, fgs in GRUPO_FEAT.items()]
    pd.DataFrame(grp_rows).sort_values("importance_total", ascending=False).to_csv(OUT_GI, index=False)

    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_PREDS, index=False)
    
    fim = datetime.now()
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"# CatBoost v5 (Corrigido)\n- Features: {len(available_features)}\n- MAE: {mae_global:.4f}\n- Spearman: {sp_global:.4f}\n")
    
    print(f"\n✅ Artefatos salvos em {REP}")

if __name__ == "__main__":
    main()
