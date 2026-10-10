#!/usr/bin/env python3
"""
CatBoost v3 — Modelo oficial com features expandidas
- Dataset: cartola_jogador_rodada_v3.parquet
- Pipeline: Reutiliza a lógica validada de P0-correction e shrinkage do ridge_v2
- Novas features: Scouts granulares + Contexto Time/Adversário (pré-calculadas no v3)
"""
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

# ------------------------------------------------------------- paths
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

CANON_V3 = INTERIM / "cartola_jogador_rodada_v3.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"
P0_FOLDS = REP / "baselines_p0_corrected_folds.csv"

OUT_FOLDS = REP / "catboost_v3_folds.csv"
OUT_SUMMARY = REP / "catboost_v3_summary.md"
OUT_FI = REP / "catboost_v3_feature_importance.csv"
OUT_GI = REP / "catboost_v3_group_importance.csv"
OUT_VALID = REP / "catboost_v3_validation.json"

TARGET = "pontos_num"

CATBOOST_PARAMS = dict(
    loss_function="MAE",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    thread_count=-1,
    verbose=0,
    allow_writing_files=False,
)

# ------------------------------------------------------------- Importar lógica validada do ridge_v2
def _load_ridge_v2():
    spec = importlib.util.spec_from_file_location("ridge_v2_module", str(ROOT / "scripts" / "ridge_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

rv2 = _load_ridge_v2()

# ------------------------------------------------------------- Definir Features
SCOUTS = ["G", "A", "FD", "FF", "FT", "FS", "DE", "SG", "DS", "FC", "DP", "GS", "CA", "CV"]
SCOUTS_STD = ["G", "A", "DE", "DP", "SG"]
PARAMS = ["mean_hist", "mean_3", "mean_5", "shrunk", "momentum"]

scout_features = []
for s in SCOUTS:
    for p in PARAMS:
        scout_features.append(f"scout_{s}_{p}")
    if s in SCOUTS_STD:
        scout_features.append(f"scout_{s}_std_5")

context_features = [
    "team_FTT_mean_3", "team_FA_mean_3", "team_DS_mean_3", "team_FC_mean_3",
    "team_G_mean_3", "team_FS_mean_3", "team_DE_mean_3", "team_clean_sheet_rate_3",
    "opp_FTT_mean_3", "opp_FA_mean_3", "opp_DS_mean_3", "opp_FC_mean_3",
    "opp_G_mean_3", "opp_FS_mean_3", "opp_DE_mean_3", "opp_clean_sheet_rate_3",
    "diff_FTT", "diff_FA", "diff_DS"
]

# Features base (serão geradas por rv2.compute_participation_features)
base_features = [
    "hist_mean", "hist_std", "hist_n", "hist_shrunk_k7",
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    "basic_mean", "basic_rec_5",
    "ga_per_game", "ga_rec_5", "ga_shrunk_k7",
    "entrou_rate", "is_cold_start",
    "posicao_id",
    "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away",
]

FEATURES_COM = base_features + scout_features + context_features
CAT_FEATURES = ["posicao_id"]

print(f"Total de features alvo: {len(FEATURES_COM)}")
print(f"  Base (geradas on-the-fly): {len(base_features)}")
print(f"  Scouts (pré-calculadas): {len(scout_features)}")
print(f"  Contexto (pré-calculadas): {len(context_features)}")

# ------------------------------------------------------------- utils
def mae(p, y):
    return float(np.mean(np.abs(p - y))) if len(y) else np.nan

def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan

def spearman(p, y):
    if len(p) < 2:
        return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]):
        return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def fit_predict_catboost(X_tr, y_tr, X_te):
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(X_tr, y_tr, cat_features=CAT_FEATURES)
    return m.predict(X_te), m

# ------------------------------------------------------------- main
def main():
    inicio = datetime.now()
    print("== CatBoost v3 — Features Expandidas ==")
    print(f"Início: {inicio.isoformat(timespec='seconds')}")

    # 1. Carregar dataset v3
    print("\n1. Carregando dataset canônico v3...")
    df = pd.read_parquet(CANON_V3)
    print(f"   Shape original: {df.shape}")

    # Extrair colunas de scouts e contexto para merge posterior
    new_features_cols = [c for c in df.columns if c.startswith("scout_") or c.startswith("team_") or c.startswith("opp_") or c.startswith("diff_")]
    print(f"   Novas features encontradas no v3: {len(new_features_cols)}")

    # 2. Criar coluna elegivel (CRÍTICO - usada por ridge_v2)
    print("\n2. Criando coluna elegivel...")
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    print(f"   ✅ Coluna elegivel criada")

    # 3. Processamento básico (reutilizando lógica validada do v2)
    print("\n3. Processando diffs e scores básicos...")
    audit = rv2.load_yaml(AUDIT_YAML)
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    print("4. Computando features de participação (hist, rec, etc.)...")
    feats = rv2.compute_participation_features(df)

    print("5. Merge com features de confronto...")
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id", "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    feats = feats.merge(ts[ts_cols], on=["temporada", "rodada", "atleta_id"], how="left")

    print("6. Merge com novas features de scouts e contexto...")
    # Pegar apenas as colunas necessárias do df original para evitar duplicação
    df_new = df[["temporada", "rodada", "atleta_id"] + new_features_cols].drop_duplicates()
    feats = feats.merge(df_new, on=["temporada", "rodada", "atleta_id"], how="left")
    
    # Preencher NaNs das novas features com 0
    feats[new_features_cols] = feats[new_features_cols].fillna(0)

    print(f"   Shape final de features: {feats.shape}")

    # 7. Carregar harness
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"\n7. Folds: {len(folds)}")

    # 8. Loop de treino
    print("\n8. Iniciando treino walk-forward...")
    rows_fold = []
    fi_acc = np.zeros(len(FEATURES_COM))
    n_folds_fi = 0

    for idx, (t, R) in enumerate(folds):
        # Treino pooled: temporadas < t + temporada t com rodada < R
        mask_train = (
            ((df["temporada"] < t) | ((df["temporada"] == t) & (df["rodada"] < R)))
            & feats["elegivel"]
        )
        tr_all = feats[mask_train].copy()
        tr = tr_all[tr_all["entrou_em_campo"].fillna(False)].copy()

        # Teste: temporada t, rodada R
        mask_test = (
            (df["temporada"] == t)
            & (df["rodada"] == R)
            & feats["elegivel"]
        )
        te_all = feats[mask_test].copy()
        te_metric = te_all[te_all["entrou_em_campo"].fillna(False)].copy()

        if len(tr) == 0 or len(te_metric) == 0:
            continue

        # Calcular priors e aplicar shrinkage (lógica validada do v2)
        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te_metric = rv2.add_shrunk_features(te_metric, prior_pos, prior_ga, b0_const)

        # Filtrar apenas features disponíveis
        available_features = [f for f in FEATURES_COM if f in tr.columns]
        
        X_tr = tr[available_features]
        y_tr = tr[TARGET].to_numpy(dtype=float)
        X_te = te_metric[available_features]
        y_te = te_metric[TARGET].to_numpy(dtype=float)

        # Treinar
        pred, model = fit_predict_catboost(X_tr, y_tr, X_te)
        
        # Acumular feature importance
        fi_acc += np.array(model.get_feature_importance())
        n_folds_fi += 1

        # Métricas
        mae_fold = mae(pred, y_te)
        rmse_fold = rmse(pred, y_te)
        sp_fold = spearman(pred, y_te)

        rows_fold.append({
            "temporada": t,
            "rodada_alvo": R,
            "n_train": int(len(tr)),
            "n_test": int(len(te_metric)),
            "mae": mae_fold,
            "rmse": rmse_fold,
            "spearman": sp_fold,
        })

        if (idx + 1) % 10 == 0:
            print(f"   fold {idx+1}/{len(folds)}: {t}/R{R} n_tr={len(tr)} n_te={len(te_metric)} MAE={mae_fold:.4f} SP={sp_fold:.4f}")

    # 9. Resultados agregados
    dfr = pd.DataFrame(rows_fold)
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"] ** 2 * dfr["n_test"]).sum() / total_n))
    sp_global = float(dfr["spearman"].mean())

    print("\n" + "=" * 65)
    print("RESULTADOS FINAIS — CatBoost v3")
    print("=" * 65)
    print(f"MAE global:     {mae_global:.4f}")
    print(f"RMSE global:    {rmse_global:.4f}")
    print(f"Spearman médio: {sp_global:.4f}")
    print(f"Tempo total:    {datetime.now() - inicio}")
    print("=" * 65)

    # 10. Salvar artefatos
    dfr.to_csv(OUT_FOLDS, index=False)
    
    # Feature Importance
    fi_df = pd.DataFrame({"feature": available_features, "importance": fi_acc / max(n_folds_fi, 1)})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(OUT_FI, index=False)
    
    # Group Importance
    GRUPO_FEAT = {
        "Historico": ["hist_mean", "hist_std", "hist_n", "hist_shrunk_k7"],
        "Recente": ["rec_mean_5", "rec_mean_3", "rec_std_5", "momentum"],
        "Basico": ["basic_mean", "basic_rec_5"],
        "GA": ["ga_per_game", "ga_rec_5", "ga_shrunk_k7"],
        "Oportunidade": ["entrou_rate", "is_cold_start"],
        "Posicao": ["posicao_id"],
        "Confronto": ["p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"],
        "Scouts_Granulares": [f for f in available_features if f.startswith("scout_")],
        "Contexto_Time_Adv": [f for f in available_features if f.startswith("team_") or f.startswith("opp_") or f.startswith("diff_")]
    }
    
    grp_rows = []
    for grupo, fgs in GRUPO_FEAT.items():
        imp = float(fi_df[fi_df["feature"].isin(fgs)]["importance"].sum())
        grp_rows.append({"grupo": grupo, "importance_total": imp})
    pd.DataFrame(grp_rows).sort_values("importance_total", ascending=False).to_csv(OUT_GI, index=False)

    print(f"\nArtefatos salvos:")
    print(f"  {OUT_FOLDS}")
    print(f"  {OUT_FI}")
    print(f"  {OUT_GI}")

    # Summary
    fim = datetime.now()
    L = []
    L.append("# CatBoost v3 — Summary\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Duração: {(fim - inicio).total_seconds():.1f}s\n")
    L.append(f"- Features utilizadas: {len(available_features)}\n")
    L.append(f"- Folds: {len(dfr)}\n\n")
    L.append("## Métricas Globais\n\n")
    L.append(f"- **MAE ponderado: {mae_global:.4f}**\n")
    L.append(f"- RMSE ponderado: {rmse_global:.4f}\n")
    L.append(f"- Spearman médio: {sp_global:.4f}\n\n")
    L.append("## Comparação com v2 (Pooling)\n\n")
    L.append(f"- CatBoost v2 MAE: 2.6322\n")
    L.append(f"- CatBoost v3 MAE: {mae_global:.4f}\n")
    L.append(f"- Delta: {mae_global - 2.6322:+.4f}\n\n")
    L.append("## Importância por Grupo\n\n")
    grp_df = pd.DataFrame(grp_rows).sort_values("importance_total", ascending=False)
    L.append(grp_df.to_markdown(index=False))

    OUT_SUMMARY.write_text("".join(L), encoding="utf-8")
    print(f"  {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
