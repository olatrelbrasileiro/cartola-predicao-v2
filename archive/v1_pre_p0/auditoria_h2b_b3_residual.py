#!/usr/bin/env python3
"""
H2B — basic_score tem sinal ortogonal ao B2?

Usa auditoria_b2_b3_erros.csv (y, pred_b2, pred_b3).
Calcula:
- corr(pred_b2 - y, pred_b3 - pred_b2) global e por fold
- corr(pred_b2 - y, pred_b3)
- bucket analysis por quintil de inovacao
- regressao simples (diagnostica, nao e modelo)
"""
import hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
REP = ROOT/"reports"
ERROS = REP/"auditoria_b2_b3_erros.csv"
CSV_FOLDS = REP/"auditoria_h2b_folds.csv"
CSV_BUCKETS = REP/"auditoria_h2b_buckets.csv"
SUMMARY = REP/"auditoria_h2b_summary.md"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    inicio = datetime.now()
    print("== H2B — residual B2 x sinal do B3 ==")
    hash_antes = sha256(ERROS)

    obs = pd.read_csv(ERROS)
    print(f"obs: {len(obs)}")

    obs["resid_b2"] = obs["prediction_b2"] - obs["pontos_real"]
    obs["innovation_b3"] = obs["prediction_b3"] - obs["prediction_b2"]

    # global
    corr_signed = float(np.corrcoef(obs["resid_b2"], obs["innovation_b3"])[0,1])
    corr_with_b3 = float(np.corrcoef(obs["resid_b2"], obs["prediction_b3"])[0,1])
    corr_abs = float(np.corrcoef(obs["abs_erro_b2"], obs["abs_erro_b3"])[0,1])

    # regressao diagnostica (nao e modelo, so mede slope)
    x = obs["innovation_b3"].to_numpy()
    y = obs["resid_b2"].to_numpy()
    beta = float(np.cov(x, y)[0,1] / np.var(x)) if np.var(x) > 0 else np.nan

    # por fold
    fold_rows = []
    for (t, r), sub in obs.groupby(["temporada","rodada"]):
        if len(sub) < 20: continue
        if np.std(sub["innovation_b3"]) == 0: continue
        r1 = float(np.corrcoef(sub["resid_b2"], sub["innovation_b3"])[0,1])
        r2 = float(np.corrcoef(sub["resid_b2"], sub["prediction_b3"])[0,1])
        fold_rows.append({"temporada": int(t), "rodada": int(r),
                          "n": int(len(sub)),
                          "corr_resid_innov": r1,
                          "corr_resid_b3": r2})
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(CSV_FOLDS, index=False)

    mean_corr_fold = float(df_folds["corr_resid_innov"].mean()) if len(df_folds) else np.nan
    median_corr_fold = float(df_folds["corr_resid_innov"].median()) if len(df_folds) else np.nan
    n_pos = int((df_folds["corr_resid_innov"] > 0).sum())
    n_neg = int((df_folds["corr_resid_innov"] < 0).sum())

    # buckets de inovacao
    obs["q_innov"] = pd.qcut(obs["innovation_b3"], 5, labels=False, duplicates="drop")
    buckets = obs.groupby("q_innov").agg(
        n=("resid_b2","size"),
        innov_min=("innovation_b3","min"),
        innov_max=("innovation_b3","max"),
        innov_mean=("innovation_b3","mean"),
        resid_b2_mean=("resid_b2","mean"),
        resid_b2_std=("resid_b2","std"),
        y_mean=("pontos_real","mean"),
        pred_b2_mean=("prediction_b2","mean"),
        pred_b3_mean=("prediction_b3","mean"),
        abs_erro_b2_mean=("abs_erro_b2","mean"),
        abs_erro_b3_mean=("abs_erro_b3","mean"),
    ).reset_index()
    buckets.to_csv(CSV_BUCKETS, index=False)

    # bucket analysis: monotonicidade de resid_b2_mean
    means = buckets["resid_b2_mean"].to_numpy()
    diffs = np.diff(means)
    monotonic_up = bool(np.all(diffs > 0))
    monotonic_down = bool(np.all(diffs < 0))
    range_resid = float(means.max() - means.min())

    hash_depois = sha256(ERROS)
    imut = (hash_antes == hash_depois)

    fim = datetime.now()
    L = []
    L.append("# H2B — sinal ortogonal do B3 vs B2\n\n")
    L.append(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- Obs: {len(obs)}\n- Folds: {len(df_folds)}\n\n")

    L.append("## Correlacoes globais\n\n")
    L.append(f"- corr(resid_b2, innovation_b3) = {corr_signed:.4f}\n")
    L.append(f"- corr(resid_b2, prediction_b3) = {corr_with_b3:.4f}\n")
    L.append(f"- corr(|erro_b2|, |erro_b3|) = {corr_abs:.4f}\n")
    L.append(f"- beta (resid_b2 ~ innovation_b3) = {beta:.4f}\n\n")

    L.append("## Por fold\n\n")
    L.append(f"- corr media: {mean_corr_fold:.4f}\n")
    L.append(f"- corr mediana: {median_corr_fold:.4f}\n")
    L.append(f"- folds com corr > 0: {n_pos}/{len(df_folds)}\n")
    L.append(f"- folds com corr < 0: {n_neg}/{len(df_folds)}\n\n")

    L.append("## Buckets de inovacao\n\n")
    L.append(buckets.to_string(index=False))
    L.append("\n\n")
    L.append(f"- resid_b2_mean monotonico crescente: {monotonic_up}\n")
    L.append(f"- resid_b2_mean monotonico decrescente: {monotonic_down}\n")
    L.append(f"- range de resid_b2_mean entre buckets: {range_resid:.4f}\n\n")

    L.append("## Imutabilidade\n\n")
    L.append(f"- erros CSV SHA: {'OK' if imut else 'MUDOU'}\n")

    SUMMARY.write_text("".join(L), encoding="utf-8")

    print()
    print(f"corr(resid_b2, innovation_b3): {corr_signed:.4f}")
    print(f"corr(resid_b2, pred_b3):       {corr_with_b3:.4f}")
    print(f"corr(|erro|) globais:          {corr_abs:.4f}")
    print(f"beta:                          {beta:.4f}")
    print(f"corr media por fold:           {mean_corr_fold:.4f}")
    print(f"range resid_b2 entre buckets:  {range_resid:.4f}")
    print(f"imutavel: {imut}")


if __name__ == "__main__":
    main()
