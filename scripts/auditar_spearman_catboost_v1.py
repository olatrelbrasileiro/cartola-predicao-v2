#!/usr/bin/env python3
"""Auditoria diagnóstica fold-a-fold do CatBoost v1 nos 5 folds
com maior perda de Spearman em relação ao B2S.

NÃO altera modelos nem artefatos. Apenas gera diagnóstico.
"""
import hashlib, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def _detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz nao encontrada")


ROOT = _detect_root()
INTERIM = ROOT/"data"/"interim"
CACHE = INTERIM/"cache"
REP = ROOT/"reports"

CANON = INTERIM/"cartola_jogador_rodada_v2.parquet"
MANIFEST = REP/"harness_folds_v2.csv"
B2S_FOLDS = REP/"baseline_b2s_folds.csv"
CB_FOLDS = REP/"catboost_v1_folds.csv"
CACHE_N1 = CACHE/"catboost_v1_prepared.parquet"

OUT_MD       = REP/"catboost_v1_spearman_audit.md"
OUT_PLAYERS  = REP/"catboost_v1_spearman_audit_players.csv"
OUT_BY_POS   = REP/"catboost_v1_spearman_audit_by_position.csv"
OUT_BY_RANGE = REP/"catboost_v1_spearman_audit_by_range.csv"

TARGET = "pontos_num"
K_SHRINK = 7
SEED = 42

FEATURES_COM = [
    "hist_mean","hist_std","hist_n","hist_shrunk",
    "rec_mean_5","rec_mean_3","rec_std_5","momentum",
    "basic_mean","basic_rec_5",
    "ga_per_game","ga_rec_5","ga_shrunk",
    "entrou_rate","is_cold_start",
    "posicao_id",
    "p_home","p_draw","p_away","mando","favoritismo_home_away",
]
CAT_FEATURES = ["posicao_id"]
CATBOOST_PARAMS = dict(
    loss_function="MAE", iterations=500, depth=6, learning_rate=0.05,
    l2_leaf_reg=3.0, random_seed=SEED, thread_count=-1, verbose=0,
    allow_writing_files=False,
)

FOLDS_ALVO = [(2024,35),(2025,38),(2024,18),(2026,20),(2025,11)]

POS_NAMES = {1:"GOL", 2:"LAT", 3:"ZAG", 4:"MEI", 5:"ATA", 6:"TEC"}


# ------------------------------------------------------------------ utils
def mae(p, y): return float(np.mean(np.abs(p - y)))
def rmse(p, y): return float(np.sqrt(np.mean((p - y)**2)))
def pearson(p, y):
    if len(p) < 2: return np.nan
    if np.std(p) == 0 or np.std(y) == 0: return np.nan
    return float(np.corrcoef(p, y)[0,1])
def spearman(p, y):
    if len(p) < 2: return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])
def slope_regression(p, y):
    """Slope de p ~ y. < 1 = compressao; = 1 = calibrado."""
    if len(p) < 2 or np.var(y) == 0: return np.nan
    return float(np.cov(p, y)[0,1] / np.var(y))


# ------------------------------------------------------------------ load
def main():
    inicio = datetime.now()
    print("== Auditoria Spearman — CatBoost v1 ==")

    if not CACHE_N1.exists():
        print(f"ERRO: cache {CACHE_N1} nao encontrado.")
        print("Rode scripts/catboost_v1.py primeiro para popular o cache.")
        return

    print(f"[load N1] {CACHE_N1.name}")
    df = pd.read_parquet(CACHE_N1)
    print(f"  shape: {df.shape}")

    b2s = pd.read_csv(B2S_FOLDS).set_index(["temporada","rodada_alvo"])

    # acumuladores
    all_players = []
    all_by_pos = []
    all_by_range = []
    L_folds = []

    for (t, R) in FOLDS_ALVO:
        print(f"\n--- fold {t}/R{R} ---")
        mask_t = df["temporada"] == t
        train = df[mask_t & (df["rodada"] < R) & df["elegivel"]].copy()
        test_all = df[mask_t & (df["rodada"] == R) & df["elegivel"]].copy()

        # prior + shrunk (mesma logica do catboost_v1.py)
        prior_pos = train.groupby("posicao_id")[TARGET].mean()
        last_obs = (train.sort_values("rodada")
                    .groupby("atleta_id").tail(1)[["atleta_id","posicao_id","ga_rate_last"]]
                    .dropna(subset=["ga_rate_last"]))
        prior_ga = last_obs.groupby("posicao_id")["ga_rate_last"].mean()
        global_mean = float(train[TARGET].mean())
        global_ga = float(last_obs["ga_rate_last"].mean()) if len(last_obs) else 0.0

        def add_shrunk(s):
            s = s.copy()
            n = s["hist_n"].fillna(0).to_numpy(dtype=float)
            hm = s["hist_mean"].fillna(0).to_numpy(dtype=float)
            pp = s["posicao_id"].map(prior_pos).fillna(global_mean).to_numpy(dtype=float)
            s["hist_shrunk"] = np.where(n > 0, (n*hm + K_SHRINK*pp)/(n+K_SHRINK), np.nan)
            gr = s["ga_rate_last"].fillna(0).to_numpy(dtype=float)
            pg = s["posicao_id"].map(prior_ga).fillna(global_ga).to_numpy(dtype=float)
            s["ga_shrunk"] = np.where(n > 0, (n*gr + K_SHRINK*pg)/(n+K_SHRINK), np.nan)
            return s

        train = add_shrunk(train)
        test_all = add_shrunk(test_all)
        test = test_all[~test_all["flag_ambigua_B"]].copy()

        y = test[TARGET].to_numpy(dtype=float)
        n = len(y)
        print(f"  train={len(train)}  test_raw={len(test_all)}  test_metric={n}")

        # --- CatBoost
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(train[FEATURES_COM], train[TARGET], cat_features=CAT_FEATURES)
        pred_cb = m.predict(test[FEATURES_COM])

        # --- B2S (recomputado a partir dos artefatos oficiais)
        alpha_star = float(b2s.loc[(t,R), "alpha_star"])
        anchor_median = float(b2s.loc[(t,R), "anchor_median"])

        train_sorted = train.sort_values(["atleta_id","rodada"], kind="mergesort")
        recent5 = (train_sorted.groupby("atleta_id").tail(5)
                   .groupby("atleta_id")[TARGET].mean())
        pred_b2 = test["atleta_id"].map(recent5).to_numpy(dtype=float)
        has_hist = ~np.isnan(pred_b2)
        train_mean_global = float(train[TARGET].mean())
        pred_b2_filled = np.where(has_hist, pred_b2, train_mean_global)
        pred_b2s = alpha_star * pred_b2_filled + (1 - alpha_star) * anchor_median

        # --- metricas
        mae_cb = mae(pred_cb, y)
        mae_b2s = mae(pred_b2s, y)
        sp_cb = spearman(pred_cb, y)
        sp_b2s = spearman(pred_b2s, y)
        pe_cb = pearson(pred_cb, y)
        pe_b2s = pearson(pred_b2s, y)

        std_real = float(np.std(y))
        std_cb = float(np.std(pred_cb))
        std_b2s = float(np.std(pred_b2s))
        slope_cb = slope_regression(pred_cb, y)
        slope_b2s = slope_regression(pred_b2s, y)

        # tabela player x fold
        tp = pd.DataFrame({
            "temporada": t, "rodada_alvo": R,
            "atleta_id": test["atleta_id"].to_numpy(),
            "posicao_id": test["posicao_id"].to_numpy(),
            "posicao_nome": test["posicao_id"].map(POS_NAMES).to_numpy(),
            "pontos_real": y,
            "pred_cb": pred_cb,
            "pred_b2s": pred_b2s,
            "err_cb": pred_cb - y,
            "err_b2s": pred_b2s - y,
            "abs_err_cb": np.abs(pred_cb - y),
            "abs_err_b2s": np.abs(pred_b2s - y),
        })
        tp["rank_real"] = tp["pontos_real"].rank(ascending=False, method="min").astype(int)
        tp["rank_cb"]   = tp["pred_cb"].rank(ascending=False, method="min").astype(int)
        tp["rank_b2s"]  = tp["pred_b2s"].rank(ascending=False, method="min").astype(int)
        tp["rank_err_cb"]  = (tp["rank_cb"] - tp["rank_real"]).abs()
        tp["rank_err_b2s"] = (tp["rank_b2s"] - tp["rank_real"]).abs()
        all_players.append(tp)

        # breakdown por posicao
        for pos, sub in tp.groupby("posicao_id"):
            sp_cb_p = spearman(sub["pred_cb"].to_numpy(), sub["pontos_real"].to_numpy())
            sp_b2s_p = spearman(sub["pred_b2s"].to_numpy(), sub["pontos_real"].to_numpy())
            all_by_pos.append({
                "temporada": t, "rodada_alvo": R, "posicao_id": int(pos),
                "posicao_nome": POS_NAMES.get(int(pos), str(pos)),
                "n": int(len(sub)),
                "mae_cb": mae(sub["pred_cb"].to_numpy(), sub["pontos_real"].to_numpy()),
                "mae_b2s": mae(sub["pred_b2s"].to_numpy(), sub["pontos_real"].to_numpy()),
                "sp_cb": sp_cb_p, "sp_b2s": sp_b2s_p,
                "delta_sp": sp_cb_p - sp_b2s_p if not (np.isnan(sp_cb_p) or np.isnan(sp_b2s_p)) else np.nan,
                "std_cb": float(sub["pred_cb"].std()),
                "std_b2s": float(sub["pred_b2s"].std()),
                "std_real": float(sub["pontos_real"].std()),
            })

        # breakdown por faixa de pontos
        def bucket(x):
            if x < 0: return "lt_0"
            if x == 0: return "eq_0"
            if x <= 3: return "0_3"
            if x <= 6: return "3_6"
            if x <= 10: return "6_10"
            return "gt_10"
        tp["bucket"] = tp["pontos_real"].apply(bucket)
        for b, sub in tp.groupby("bucket"):
            all_by_range.append({
                "temporada": t, "rodada_alvo": R, "bucket": b,
                "n": int(len(sub)),
                "mae_cb": mae(sub["pred_cb"].to_numpy(), sub["pontos_real"].to_numpy()),
                "mae_b2s": mae(sub["pred_b2s"].to_numpy(), sub["pontos_real"].to_numpy()),
                "mean_pred_cb": float(sub["pred_cb"].mean()),
                "mean_pred_b2s": float(sub["pred_b2s"].mean()),
                "mean_real": float(sub["pontos_real"].mean()),
            })

        # guarda info do fold para o relatorio
        L_folds.append({
            "fold": f"{t}/R{R}",
            "n": n,
            "mae_cb": mae_cb, "mae_b2s": mae_b2s,
            "sp_cb": sp_cb, "sp_b2s": sp_b2s, "delta_sp": sp_cb - sp_b2s,
            "pe_cb": pe_cb, "pe_b2s": pe_b2s,
            "std_real": std_real, "std_cb": std_cb, "std_b2s": std_b2s,
            "slope_cb": slope_cb, "slope_b2s": slope_b2s,
            "top10_real": tp.nlargest(10, "pontos_real")[[
                "atleta_id","posicao_nome","pontos_real",
                "pred_cb","pred_b2s","rank_cb","rank_b2s"]].to_dict("records"),
            "top10_cb": tp.nlargest(10, "pred_cb")[[
                "atleta_id","posicao_nome","pontos_real",
                "pred_cb","pred_b2s","rank_real"]].to_dict("records"),
            "top10_b2s": tp.nlargest(10, "pred_b2s")[[
                "atleta_id","posicao_nome","pontos_real",
                "pred_cb","pred_b2s","rank_real"]].to_dict("records"),
        })

    df_players = pd.concat(all_players, ignore_index=True)
    df_by_pos = pd.DataFrame(all_by_pos)
    df_by_range = pd.DataFrame(all_by_range)
    df_players.to_csv(OUT_PLAYERS, index=False)
    df_by_pos.to_csv(OUT_BY_POS, index=False)
    df_by_range.to_csv(OUT_BY_RANGE, index=False)
    print(f"\n[write] {OUT_PLAYERS.name}")
    print(f"[write] {OUT_BY_POS.name}")
    print(f"[write] {OUT_BY_RANGE.name}")

    # ---- relatorio markdown
    L = []
    L.append("# CatBoost v1 — Auditoria dos 5 folds com maior perda de Spearman\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Folds: {len(L_folds)}\n\n")

    L.append("## Visao geral por fold\n\n")
    L.append("| fold | n | MAE_CB | MAE_B2S | Δ_MAE | SP_CB | SP_B2S | Δ_SP | Pearson_CB | Pearson_B2S |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in L_folds:
        L.append(f"| {r['fold']} | {r['n']} | "
                 f"{r['mae_cb']:.4f} | {r['mae_b2s']:.4f} | {r['mae_cb']-r['mae_b2s']:+.4f} | "
                 f"{r['sp_cb']:.4f} | {r['sp_b2s']:.4f} | {r['delta_sp']:+.4f} | "
                 f"{r['pe_cb']:.4f} | {r['pe_b2s']:.4f} |\n")
    L.append("\n")

    L.append("## Compressao (std e slope)\n\n")
    L.append("| fold | std_real | std_CB | std_B2S | slope_CB | slope_B2S |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in L_folds:
        L.append(f"| {r['fold']} | {r['std_real']:.3f} | {r['std_cb']:.3f} | {r['std_b2s']:.3f} | "
                 f"{r['slope_cb']:.3f} | {r['slope_b2s']:.3f} |\n")
    L.append("\n")
    L.append("slope < 1 indica compressao (previsao subestima amplitude real).\n\n")

    for r in L_folds:
        L.append(f"## Fold {r['fold']}\n\n")
        L.append("### Top 10 reais\n\n")
        L.append(pd.DataFrame(r["top10_real"]).to_string(index=False))
        L.append("\n\n### Top 10 previstos pelo CatBoost\n\n")
        L.append(pd.DataFrame(r["top10_cb"]).to_string(index=False))
        L.append("\n\n### Top 10 previstos pelo B2S\n\n")
        L.append(pd.DataFrame(r["top10_b2s"]).to_string(index=False))
        L.append("\n\n")

    # breakdown por posicao — agregado
    L.append("## Breakdown por posicao (agregado nos 5 folds)\n\n")
    agg_pos = df_by_pos.groupby(["posicao_id","posicao_nome"]).agg(
        n=("n","sum"),
        mae_cb=("mae_cb","mean"),
        mae_b2s=("mae_b2s","mean"),
        delta_mae=("mae_cb", lambda s: 0),  # placeholder
        sp_cb=("sp_cb","mean"),
        sp_b2s=("sp_b2s","mean"),
        delta_sp=("delta_sp","mean"),
    ).reset_index()
    agg_pos["delta_mae"] = agg_pos["mae_cb"] - agg_pos["mae_b2s"]
    L.append(agg_pos.to_string(index=False))
    L.append("\n\n")

    # breakdown por faixa de pontos — agregado
    L.append("## Breakdown por faixa de pontos (agregado nos 5 folds)\n\n")
    agg_rng = df_by_range.groupby("bucket").agg(
        n=("n","sum"),
        mae_cb=("mae_cb","mean"),
        mae_b2s=("mae_b2s","mean"),
        mean_pred_cb=("mean_pred_cb","mean"),
        mean_pred_b2s=("mean_pred_b2s","mean"),
        mean_real=("mean_real","mean"),
    ).reset_index()
    agg_rng["delta_mae"] = agg_rng["mae_cb"] - agg_rng["mae_b2s"]
    L.append(agg_rng.to_string(index=False))
    L.append("\n\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")
    print(f"[write] {OUT_MD.name}")

    # resumo no console
    print("\n=== RESUMO ===")
    for r in L_folds:
        print(f"  {r['fold']}: Δ_SP={r['delta_sp']:+.4f}  Δ_MAE={r['mae_cb']-r['mae_b2s']:+.4f}  "
              f"slope_CB={r['slope_cb']:.3f}  slope_B2S={r['slope_b2s']:.3f}")


if __name__ == "__main__":
    main()
