#!/usr/bin/env python3
"""
Auditoria B2 x B3 — analise comparativa.
Nao cria previsoes novas, nao combina modelos, nao treina nada.
"""
import hashlib, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path.home()/"cartola-predicao-v2"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
CONFIG = ROOT/"config.yaml"
AUDIT_YAML = ROOT/"config"/"pontuacao_scouts_auditoria.yaml"
MANIFEST = ROOT/"reports"/"harness_folds_v2.csv"
B0_FOLDS = ROOT/"reports"/"baseline_b0_folds.csv"
B1_FOLDS = ROOT/"reports"/"baseline_b1_folds.csv"
B2_FOLDS = ROOT/"reports"/"baseline_b2_folds.csv"
B3_FOLDS = ROOT/"reports"/"baseline_b3_folds.csv"
REP = ROOT/"reports"

CSV_FOLDS = REP/"auditoria_b2_b3_folds.csv"
CSV_ERROS = REP/"auditoria_b2_b3_erros.csv"
CSV_TARGET = REP/"auditoria_b2_b3_por_target.csv"
CSV_TEMP = REP/"auditoria_b2_b3_por_temporada.csv"
CSV_HIST = REP/"auditoria_b2_b3_por_historico.csv"
SUMMARY = REP/"auditoria_b2_b3_summary.md"

TARGET_COL = "pontos_num"
RECENT_WINDOW = 5

HASHES_ANTES = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def load_config() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_audit() -> dict:
    with open(AUDIT_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cumulative_diff(df: pd.DataFrame, scout_cols: list) -> pd.DataFrame:
    out = df.sort_values(["temporada","atleta_id","rodada"],
                         kind="mergesort").reset_index(drop=True)
    for col in scout_cols:
        ff = out.groupby(["temporada","atleta_id"], sort=False)[col].ffill()
        chave = out[["temporada","atleta_id"]].copy()
        chave["_ff"] = ff.values
        prev = chave.groupby(["temporada","atleta_id"], sort=False)["_ff"].shift(1)
        out[f"diff_{col}"] = ff.fillna(0).to_numpy() - prev.fillna(0).to_numpy()
    return out


def compute_basic_score(df: pd.DataFrame, audit: dict) -> pd.Series:
    pesos = audit["pesos_por_temporada"]
    total = pd.Series(0.0, index=df.index)
    for s in audit["scouts_basicos"]:
        col = f"diff_scout_{s}"
        peso_linha = df["temporada"].map({int(t): pesos[t][s] for t in pesos})
        total = total + df[col].to_numpy() * peso_linha.to_numpy()
    return total.astype(float)


def bucket_target(x: float) -> str:
    if x < 0: return "pontos_lt_0"
    if x == 0: return "pontos_eq_0"
    if x <= 3: return "pontos_0_3"
    if x <= 6: return "pontos_3_6"
    if x <= 10: return "pontos_6_10"
    return "pontos_gt_10"


def bucket_historico(n: int) -> str:
    if n == 0: return "n_0"
    if n <= 2: return "n_1_2"
    if n <= 5: return "n_3_5"
    if n <= 10: return "n_6_10"
    if n <= 20: return "n_11_20"
    return "n_gt_20"


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2: return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    inicio = datetime.now()
    print("== AUDITORIA B2 x B3 ==")

    # imutabilidade
    for k, p in [("parquet", PARQUET), ("b0", B0_FOLDS), ("b1", B1_FOLDS),
                 ("b2", B2_FOLDS), ("b3", B3_FOLDS),
                 ("b2_sum", REP/"baseline_b2_summary.md"),
                 ("b2_spot", REP/"baseline_b2_spotcheck.csv"),
                 ("b3_sum", REP/"baseline_b3_summary.md"),
                 ("b3_spot", REP/"baseline_b3_spotcheck.csv")]:
        HASHES_ANTES[k] = sha256(p)

    # dados
    cfg = load_config()
    mtr = cfg["validation"]["min_train_rounds"]
    audit = load_audit()

    df = pd.read_parquet(PARQUET)
    df = df.sort_values(["temporada","rodada","atleta_id"], kind="mergesort").reset_index(drop=True)
    df["flag_ambigua_B"] = (df["entrou_em_campo"] == False) & (df[TARGET_COL] != 0)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = apply_cumulative_diff(df, scout_cols)
    df["basic_score"] = compute_basic_score(df, audit)
    df = df.sort_values(["temporada","rodada","atleta_id"], kind="mergesort").reset_index(drop=True)

    # folds
    man = pd.read_csv(MANIFEST)
    folds = [{"temporada": int(r["temporada"]), "rodada_alvo": int(r["rodada_alvo"])}
             for _, r in man.iterrows()]

    # ---- per-fold e per-observation ----
    fold_rows = []
    obs_rows = []

    for f in folds:
        t = f["temporada"]; r = f["rodada_alvo"]
        mask_t = df["temporada"] == t
        train = df[mask_t & (df["rodada"] < r) & df["elegivel"]]
        test_raw = df[mask_t & (df["rodada"] == r) & df["elegivel"]]
        test_metric = test_raw[~test_raw["flag_ambigua_B"]].copy()

        train_mean_global = float(train[TARGET_COL].mean())

        # B2: media das ultimas 5 obs do jogador no treino
        train_sorted = train.sort_values(["atleta_id","rodada"], kind="mergesort")
        recent_means = (train_sorted.groupby("atleta_id").tail(RECENT_WINDOW)
                        .groupby("atleta_id")[TARGET_COL].mean())
        # B3: media do basic_score
        basic_means = train.groupby("atleta_id")["basic_score"].mean()
        # historico_n
        n_obs = train.groupby("atleta_id").size()

        test_metric["_b2"] = test_metric["atleta_id"].map(recent_means)
        test_metric["_b3"] = test_metric["atleta_id"].map(basic_means)
        test_metric["_hist_n"] = test_metric["atleta_id"].map(n_obs).fillna(0).astype(int)
        test_metric["fallback_b2"] = test_metric["_b2"].isna()
        test_metric["fallback_b3"] = test_metric["_b3"].isna()

        test_metric["pred_b2"] = test_metric["_b2"].fillna(train_mean_global)
        test_metric["pred_b3"] = test_metric["_b3"].fillna(train_mean_global)

        y = test_metric[TARGET_COL].to_numpy(dtype=float)
        p2 = test_metric["pred_b2"].to_numpy(dtype=float)
        p3 = test_metric["pred_b3"].to_numpy(dtype=float)

        e2 = p2 - y; e3 = p3 - y
        ae2 = np.abs(e2); ae3 = np.abs(e3)

        mae2 = float(ae2.mean()); mae3 = float(ae3.mean())
        rmse2 = float(np.sqrt((e2**2).mean()))
        rmse3 = float(np.sqrt((e3**2).mean()))
        sp2 = spearman_safe(p2, y)
        sp3 = spearman_safe(p3, y)

        def iqr(a): return float(np.percentile(a,75) - np.percentile(a,25))

        fold_rows.append({
            "temporada": t, "rodada": r,
            "n_test": len(y),
            "b2_mae": mae2, "b3_mae": mae3, "delta_mae": mae3 - mae2,
            "b2_rmse": rmse2, "b3_rmse": rmse3,
            "b2_spearman": sp2, "b3_spearman": sp3,
            "delta_spearman": (sp3 - sp2) if (not np.isnan(sp2) and not np.isnan(sp3)) else np.nan,
            "std_prediction_b2": float(np.std(p2)),
            "std_prediction_b3": float(np.std(p3)),
            "iqr_prediction_b2": iqr(p2),
            "iqr_prediction_b3": iqr(p3),
            "range_prediction_b2": float(p2.max() - p2.min()),
            "range_prediction_b3": float(p3.max() - p3.min()),
            "n_unique_prediction_b2": int(pd.Series(p2).nunique()),
            "n_unique_prediction_b3": int(pd.Series(p3).nunique()),
            "n_fallback_b2": int(test_metric["fallback_b2"].sum()),
            "n_fallback_b3": int(test_metric["fallback_b3"].sum()),
            "pct_target_zero": float((y == 0).mean()),
            "pct_target_negative": float((y < 0).mean()),
            "pct_target_gt6": float((y > 6).mean()),
            "pct_target_gt10": float((y > 10).mean()),
        })

        # obs-level
        for i in range(len(y)):
            obs_rows.append({
                "temporada": t, "rodada": r,
                "atleta_id": int(test_metric["atleta_id"].iloc[i]),
                "posicao_id": int(test_metric["posicao_id"].iloc[i]),
                "pontos_real": float(y[i]),
                "prediction_b2": float(p2[i]),
                "prediction_b3": float(p3[i]),
                "erro_b2": float(e2[i]),
                "erro_b3": float(e3[i]),
                "abs_erro_b2": float(ae2[i]),
                "abs_erro_b3": float(ae3[i]),
                "delta_abs_erro": float(ae3[i] - ae2[i]),
                "melhor_modelo_por_mae": "B3" if ae3[i] < ae2[i] else ("B2" if ae2[i] < ae3[i] else "EMPATE"),
                "historico_n": int(test_metric["_hist_n"].iloc[i]),
                "fallback_b2": bool(test_metric["fallback_b2"].iloc[i]),
                "fallback_b3": bool(test_metric["fallback_b3"].iloc[i]),
            })

    df_folds = pd.DataFrame(fold_rows)
    df_obs = pd.DataFrame(obs_rows)

    df_folds.to_csv(CSV_FOLDS, index=False)
    df_obs.to_csv(CSV_ERROS, index=False)

    # ---- por target ----
    df_obs["bucket_target"] = df_obs["pontos_real"].apply(bucket_target)
    tgt_rows = []
    for b, sub in df_obs.groupby("bucket_target"):
        tgt_rows.append({
            "bucket_target": b,
            "n": int(len(sub)),
            "mae_b2": float(sub["abs_erro_b2"].mean()),
            "mae_b3": float(sub["abs_erro_b3"].mean()),
            "delta_mae_b3_b2": float(sub["abs_erro_b3"].mean() - sub["abs_erro_b2"].mean()),
            "mean_pred_b2": float(sub["prediction_b2"].mean()),
            "mean_pred_b3": float(sub["prediction_b3"].mean()),
            "mean_y_true": float(sub["pontos_real"].mean()),
            "folds_b3_melhor": int((sub["delta_abs_erro"] < 0).sum()),
            "folds_b2_melhor": int((sub["delta_abs_erro"] > 0).sum()),
            "empates": int((sub["delta_abs_erro"] == 0).sum()),
        })
    df_target = pd.DataFrame(tgt_rows)
    df_target.to_csv(CSV_TARGET, index=False)

    # ---- por temporada ----
    temp_rows = []
    for t, sub in df_folds.groupby("temporada"):
        # weighted by n_test
        n = sub["n_test"].sum()
        b2w = (sub["b2_mae"] * sub["n_test"]).sum() / n
        b3w = (sub["b3_mae"] * sub["n_test"]).sum() / n
        temp_rows.append({
            "temporada": int(t),
            "n_folds": int(len(sub)),
            "b2_mae_ponderado": float(b2w),
            "b3_mae_ponderado": float(b3w),
            "delta_mae": float(b3w - b2w),
            "melhoria_pct": float((b2w - b3w) / b2w * 100),
            "folds_b3_melhor": int((sub["delta_mae"] < 0).sum()),
            "folds_b2_melhor": int((sub["delta_mae"] > 0).sum()),
            "media_delta": float(sub["delta_mae"].mean()),
            "mediana_delta": float(sub["delta_mae"].median()),
            "b2_spearman_medio": float(sub["b2_spearman"].mean()),
            "b3_spearman_medio": float(sub["b3_spearman"].mean()),
        })
    df_temp = pd.DataFrame(temp_rows)
    df_temp.to_csv(CSV_TEMP, index=False)

    # ---- por historico ----
    df_obs["bucket_historico"] = df_obs["historico_n"].apply(bucket_historico)
    hist_rows = []
    for b in ["n_0","n_1_2","n_3_5","n_6_10","n_11_20","n_gt_20"]:
        sub = df_obs[df_obs["bucket_historico"] == b]
        if len(sub) == 0:
            continue
        hist_rows.append({
            "bucket_historico": b,
            "n": int(len(sub)),
            "mae_b2": float(sub["abs_erro_b2"].mean()),
            "mae_b3": float(sub["abs_erro_b3"].mean()),
            "delta_mae_b3_b2": float(sub["abs_erro_b3"].mean() - sub["abs_erro_b2"].mean()),
            "mean_pred_b2": float(sub["prediction_b2"].mean()),
            "mean_pred_b3": float(sub["prediction_b3"].mean()),
            "mean_y_true": float(sub["pontos_real"].mean()),
        })
    df_hist = pd.DataFrame(hist_rows)
    df_hist.to_csv(CSV_HIST, index=False)

    # ---- correlações ----
    corr_erro = float(np.corrcoef(df_obs["erro_b2"], df_obs["erro_b3"])[0,1])
    corr_abs  = float(np.corrcoef(df_obs["abs_erro_b2"], df_obs["abs_erro_b3"])[0,1])

    # ---- complementaridade ----
    combined = np.concatenate([df_obs["abs_erro_b2"].to_numpy(), df_obs["abs_erro_b3"].to_numpy()])
    thr_p75 = float(np.percentile(combined, 75))
    df_obs["erro_alto_b2"] = df_obs["abs_erro_b2"] > thr_p75
    df_obs["erro_alto_b3"] = df_obs["abs_erro_b3"] > thr_p75
    df_obs["ambos_erram_alto"] = df_obs["erro_alto_b2"] & df_obs["erro_alto_b3"]

    pct_b3_melhor = float((df_obs["delta_abs_erro"] < 0).mean())
    pct_b2_melhor = float((df_obs["delta_abs_erro"] > 0).mean())
    pct_empate    = float((df_obs["delta_abs_erro"] == 0).mean())
    pct_ambos_alto = float(df_obs["ambos_erram_alto"].mean())

    # ---- globais ----
    total_n = int(df_folds["n_test"].sum())
    b2_mae_pond = float((df_folds["b2_mae"] * df_folds["n_test"]).sum() / total_n)
    b3_mae_pond = float((df_folds["b3_mae"] * df_folds["n_test"]).sum() / total_n)
    b2_rmse_pond = float(np.sqrt((df_folds["b2_rmse"]**2 * df_folds["n_test"]).sum() / total_n))
    b3_rmse_pond = float(np.sqrt((df_folds["b3_rmse"]**2 * df_folds["n_test"]).sum() / total_n))
    b2_sp_medio = float(df_folds["b2_spearman"].mean())
    b3_sp_medio = float(df_folds["b3_spearman"].mean())
    n_b3_melhor = int((df_folds["delta_mae"] < 0).sum())
    n_b2_melhor = int((df_folds["delta_mae"] > 0).sum())
    n_empate    = int((df_folds["delta_mae"] == 0).sum())
    n_b3_sp_melhor = int((df_folds["delta_spearman"] > 0).sum())
    n_b2_sp_melhor = int((df_folds["delta_spearman"] < 0).sum())

    # ---- TOPs ----
    d = df_folds.sort_values("delta_mae")
    top_b3 = d.head(10)[["temporada","rodada","b2_mae","b3_mae","delta_mae"]]
    top_b2 = d.tail(10)[["temporada","rodada","b2_mae","b3_mae","delta_mae"]]

    ds = df_folds.sort_values("delta_spearman")
    top_b3_sp = ds.head(10)[["temporada","rodada","b2_spearman","b3_spearman","delta_spearman"]]
    top_b2_sp = ds.tail(10)[["temporada","rodada","b2_spearman","b3_spearman","delta_spearman"]]

    # ---- variância das previsões ----
    mean_std_b2 = float(df_folds["std_prediction_b2"].mean())
    mean_std_b3 = float(df_folds["std_prediction_b3"].mean())
    mean_iqr_b2 = float(df_folds["iqr_prediction_b2"].mean())
    mean_iqr_b3 = float(df_folds["iqr_prediction_b3"].mean())
    mean_rng_b2 = float(df_folds["range_prediction_b2"].mean())
    mean_rng_b3 = float(df_folds["range_prediction_b3"].mean())
    mean_uni_b2 = float(df_folds["n_unique_prediction_b2"].mean())
    mean_uni_b3 = float(df_folds["n_unique_prediction_b3"].mean())

    # ---- por posição (usa df_obs) ----
    pos_rows = []
    for p, sub in df_obs.groupby("posicao_id"):
        pos_rows.append({
            "posicao_id": int(p),
            "n": int(len(sub)),
            "mae_b2": float(sub["abs_erro_b2"].mean()),
            "mae_b3": float(sub["abs_erro_b3"].mean()),
            "delta_mae": float(sub["abs_erro_b3"].mean() - sub["abs_erro_b2"].mean()),
        })
    df_pos = pd.DataFrame(pos_rows)

    # ---- fallback ----
    n_fb_b2 = int(df_obs["fallback_b2"].sum())
    n_fb_b3 = int(df_obs["fallback_b3"].sum())
    mesmo_fb = bool((df_obs["fallback_b2"] == df_obs["fallback_b3"]).all())
    sub_fb = df_obs[df_obs["fallback_b2"] | df_obs["fallback_b3"]]
    fb_mae_b2 = float(sub_fb["abs_erro_b2"].mean()) if len(sub_fb) else np.nan
    fb_mae_b3 = float(sub_fb["abs_erro_b3"].mean()) if len(sub_fb) else np.nan

    # ---- 53 / 23 folds ----
    win53 = df_folds[df_folds["delta_mae"] < 0].copy()
    lose23 = df_folds[df_folds["delta_mae"] > 0].copy()

    def stats_grupo(g):
        if len(g) == 0: return {}
        return {
            "n_folds": int(len(g)),
            "mean_pct_zero": float(g["pct_target_zero"].mean()),
            "mean_pct_neg": float(g["pct_target_negative"].mean()),
            "mean_pct_gt6": float(g["pct_target_gt6"].mean()),
            "mean_pct_gt10": float(g["pct_target_gt10"].mean()),
            "mean_std_b2": float(g["std_prediction_b2"].mean()),
            "mean_std_b3": float(g["std_prediction_b3"].mean()),
            "mean_b2_mae": float(g["b2_mae"].mean()),
            "mean_b3_mae": float(g["b3_mae"].mean()),
        }

    s_win = stats_grupo(win53)
    s_lose = stats_grupo(lose23)

    # ---- imutabilidade ----
    pos_inalt = {}
    for k, p in [("parquet", PARQUET), ("b0", B0_FOLDS), ("b1", B1_FOLDS),
                 ("b2", B2_FOLDS), ("b3", B3_FOLDS),
                 ("b2_sum", REP/"baseline_b2_summary.md"),
                 ("b2_spot", REP/"baseline_b2_spotcheck.csv"),
                 ("b3_sum", REP/"baseline_b3_summary.md"),
                 ("b3_spot", REP/"baseline_b3_spotcheck.csv")]:
        pos_inalt[k] = HASHES_ANTES[k] == sha256(p)

    # ---- SUMMARY ----
    fim = datetime.now()
    L = []
    L.append("# Auditoria B2 x B3\n\n")
    L.append(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.2f}s\n")
    L.append(f"- Observacoes analisadas: {len(df_obs)}\n")
    L.append(f"- Folds: {len(df_folds)}\n\n")

    L.append("## Factos observados\n\n")
    L.append(f"- B2 MAE ponderado: {b2_mae_pond:.4f}\n")
    L.append(f"- B3 MAE ponderado: {b3_mae_pond:.4f}\n")
    L.append(f"- Delta MAE (B3-B2): {b3_mae_pond-b2_mae_pond:+.4f}\n")
    L.append(f"- B2 RMSE ponderado: {b2_rmse_pond:.4f}\n")
    L.append(f"- B3 RMSE ponderado: {b3_rmse_pond:.4f}\n")
    L.append(f"- B2 Spearman medio: {b2_sp_medio:.4f}\n")
    L.append(f"- B3 Spearman medio: {b3_sp_medio:.4f}\n")
    L.append(f"- Folds B3 melhor (MAE): {n_b3_melhor}/{len(df_folds)}\n")
    L.append(f"- Folds B2 melhor (MAE): {n_b2_melhor}/{len(df_folds)}\n")
    L.append(f"- Empates (MAE): {n_empate}\n")
    L.append(f"- Folds B3 melhor (Spearman): {n_b3_sp_melhor}/{len(df_folds)}\n")
    L.append(f"- Folds B2 melhor (Spearman): {n_b2_sp_melhor}/{len(df_folds)}\n\n")

    L.append("## Por temporada\n\n")
    L.append(df_temp.to_string(index=False))
    L.append("\n\n")

    L.append("## Variancia das previsoes (media por fold)\n\n")
    L.append(f"- std B2 = {mean_std_b2:.4f}   std B3 = {mean_std_b3:.4f}\n")
    L.append(f"- IQR B2 = {mean_iqr_b2:.4f}   IQR B3 = {mean_iqr_b3:.4f}\n")
    L.append(f"- range B2 = {mean_rng_b2:.4f}  range B3 = {mean_rng_b3:.4f}\n")
    L.append(f"- n_unique B2 = {mean_uni_b2:.1f}  n_unique B3 = {mean_uni_b3:.1f}\n\n")

    L.append("## Por faixa do target\n\n")
    L.append(df_target.to_string(index=False))
    L.append("\n\n")

    L.append("## Por historico_n\n\n")
    L.append(df_hist.to_string(index=False))
    L.append("\n\n")

    L.append("## Por posicao\n\n")
    L.append(df_pos.to_string(index=False))
    L.append("\n\n")

    L.append("## Correlações de erro\n\n")
    L.append(f"- corr(erro_b2, erro_b3) = {corr_erro:.4f}\n")
    L.append(f"- corr(abs_erro_b2, abs_erro_b3) = {corr_abs:.4f}\n\n")

    L.append("## Complementaridade\n\n")
    L.append(f"- Limiar erro alto (p75 global combinado) = {thr_p75:.4f}\n")
    L.append(f"- % obs B3 melhor que B2: {pct_b3_melhor*100:.2f}%\n")
    L.append(f"- % obs B2 melhor que B3: {pct_b2_melhor*100:.2f}%\n")
    L.append(f"- % empates: {pct_empate*100:.2f}%\n")
    L.append(f"- % ambos erram alto: {pct_ambos_alto*100:.2f}%\n\n")

    L.append("## TOP 10 folds onde B3 mais melhora (MAE)\n\n")
    L.append(top_b3.to_string(index=False))
    L.append("\n\n## TOP 10 folds onde B3 mais piora (MAE)\n\n")
    L.append(top_b2.to_string(index=False))
    L.append("\n\n## TOP 10 quedas de Spearman B3 vs B2\n\n")
    L.append(top_b3_sp.to_string(index=False))
    L.append("\n\n## TOP 10 ganhos de Spearman B3 vs B2\n\n")
    L.append(top_b2_sp.to_string(index=False))
    L.append("\n\n")

    L.append("## Perfil dos 53 folds em que B3 ganha\n\n")
    for k, v in s_win.items():
        L.append(f"- {k}: {v}\n")
    L.append("\n## Perfil dos 23 folds em que B3 perde\n\n")
    for k, v in s_lose.items():
        L.append(f"- {k}: {v}\n")

    L.append(f"\n## Fallback\n\n")
    L.append(f"- n_fallback B2: {n_fb_b2}\n")
    L.append(f"- n_fallback B3: {n_fb_b3}\n")
    L.append(f"- Mesmo conjunto: {mesmo_fb}\n")
    L.append(f"- MAE fallback B2: {fb_mae_b2:.4f}\n")
    L.append(f"- MAE fallback B3: {fb_mae_b3:.4f}\n\n")

    L.append("## Imutabilidade\n\n")
    for k, v in pos_inalt.items():
        L.append(f"- {k}: {'OK' if v else 'MUDOU'}\n")

    SUMMARY.write_text("".join(L), encoding="utf-8")

    print("Artefatos:")
    for p in [CSV_FOLDS, CSV_ERROS, CSV_TARGET, CSV_TEMP, CSV_HIST, SUMMARY]:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")
    print()
    print(f"MAE B2: {b2_mae_pond:.4f}  MAE B3: {b3_mae_pond:.4f}")
    print(f"Spearman B2: {b2_sp_medio:.4f}  Spearman B3: {b3_sp_medio:.4f}")
    print(f"Folds B3 melhor MAE: {n_b3_melhor}/{len(df_folds)}")
    print(f"corr(erro) = {corr_erro:.4f}  corr(|erro|) = {corr_abs:.4f}")
    print(f"Imutabilidade: {pos_inalt}")


if __name__ == "__main__":
    main()
