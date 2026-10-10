#!/usr/bin/env python3
"""
CatBoost v4.1 (Poda) — Removendo ruído de baixa importância
Mantém apenas features com importância >= 0.25 (aprox. 0.25% do total)
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

OUT_FOLDS = REP / "catboost_v4_poda_folds.csv"
OUT_SUMMARY = REP / "catboost_v4_poda_summary.md"
OUT_PREDS = REP / "catboost_v4_poda_predicoes.csv"

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
    print("== CatBoost v4.1 (Poda de Features) ==")
    
    # 1. Ler importância e definir corte
    fi = pd.read_csv(REP / "catboost_v4_feature_importance.csv")
    fi = fi.sort_values("importance", ascending=False)
    
    # Corte: manter features com importância >= 0.25
    MIN_IMP = 0.25
    features_keep = fi[fi["importance"] >= MIN_IMP]["feature"].tolist()
    print(f"\n🔪 PODA: Mantendo {len(features_keep)} features (Importância >= {MIN_IMP})")
    print(f"   Excluídas: {len(fi) - len(features_keep)} features de ruído (ex: DP, DE fraco)")

    # 2. Carregar dados
    df = pd.read_parquet(CANON_V4)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    
    audit = rv2.load_yaml(AUDIT_YAML)
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)
    
    feats = rv2.compute_participation_features(df)
    ts = pd.read_parquet(TEMPORAL_SAFE)
    feats = feats.merge(ts[["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]], on=["temporada", "rodada", "atleta_id"], how="left")

    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    
    rows_fold, all_predictions = [], []

    print("\n🚀 Iniciando treino podado...")
    for idx, (t, R) in enumerate(folds):
        mask_train = (((df["temporada"] < t) | ((df["temporada"] == t) & (df["rodada"] < R))) & feats["elegivel"])
        tr = feats[mask_train][feats[mask_train]["entrou_em_campo"].fillna(False)].copy()
        
        mask_test = ((df["temporada"] == t) & (df["rodada"] == R) & feats["elegivel"])
        te = feats[mask_test][feats[mask_test]["entrou_em_campo"].fillna(False)].copy()
        
        if len(tr) == 0 or len(te) == 0: continue
        
        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te = rv2.add_shrunk_features(te, prior_pos, prior_ga, b0_const)
        
        # Filtrar APENAS as features mantidas
        available = [f for f in features_keep if f in tr.columns]
        cat_feats = ["posicao_id"] if "posicao_id" in available else []
        
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(tr[available], tr[TARGET], cat_features=cat_feats)
        pred = m.predict(te[available])
        y_te = te[TARGET].to_numpy(dtype=float)
        
        rows_fold.append({"temporada": t, "rodada_alvo": R, "n_train": len(tr), "n_test": len(te), "mae": mae(pred, y_te), "spearman": spearman(pred, y_te)})
        
        te_res = te[["temporada", "rodada", "atleta_id", "posicao_id", TARGET]].copy()
        te_res["predicao"] = pred
        te_res["erro_absoluto"] = np.abs(pred - y_te)
        te_res["fold"] = f"{t}_R{R}"
        all_predictions.append(te_res)
        
        if (idx + 1) % 15 == 0: print(f"   Fold {idx+1}/{len(folds)} concluído")

    # Resultados
    dfr = pd.DataFrame(rows_fold)
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    sp_global = float(dfr["spearman"].mean())
    
    print("\n" + "="*65)
    print(f"RESULTADO FINAL v4.1 (PODA)")
    print(f"MAE Global:     {mae_global:.4f}")
    print(f"Spearman Médio: {sp_global:.4f}")
    print(f"Features Usadas: {len(available)}")
    print("="*65)
    
    dfr.to_csv(OUT_FOLDS, index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_PREDS, index=False)
    
    # Summary
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"# CatBoost v4.1 (Poda)\n- MAE: {mae_global:.4f}\n- Spearman: {sp_global:.4f}\n- Features: {len(available)}\n")
    print(f"✅ Predições salvas em: {OUT_PREDS}")

if __name__ == "__main__":
    main()
