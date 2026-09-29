#!/usr/bin/env python3
"""
H2A — alpha walk-forward vs alpha global.

Para cada fold (t, R), escolhe alpha* usando SOMENTE rounds < R
(inner walk-forward dentro do proprio treino), aplica no round R.
Compara com alpha=0.4 fixo (contaminado) e com B2 puro.
"""
import hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
REP = ROOT/"reports"
CSV_RESULTS = REP/"auditoria_h2a_folds.csv"
CSV_ALPHAS = REP/"auditoria_h2a_alphas.csv"
SUMMARY = REP/"auditoria_h2a_summary.md"

ALPHAS = [round(a, 2) for a in np.arange(0, 1.01, 0.05)]
MIN_INNER = 3
WINDOW = 5


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def build_cache(df):
    cache = {}
    for t in sorted(df["temporada"].unique()):
        sub_t = df[df["temporada"] == t]
        for r in sorted(sub_t["rodada"].unique()):
            train = sub_t[(sub_t["rodada"] < r) & sub_t["elegivel"]]
            if len(train) == 0:
                continue
            recent = (train.sort_values(["atleta_id","rodada"])
                      .groupby("atleta_id").tail(WINDOW)
                      .groupby("atleta_id")["pontos_num"].mean())
            mean_g = float(train["pontos_num"].mean())
            median_g = float(train["pontos_num"].median())
            test = sub_t[(sub_t["rodada"] == r) & sub_t["elegivel"]].copy()
            test = test[~((test["entrou_em_campo"] == False) & (test["pontos_num"] != 0))]
            if len(test) == 0:
                continue
            pred_b2 = test["atleta_id"].map(recent).fillna(mean_g).to_numpy()
            cache[(int(t), int(r))] = {
                "y": test["pontos_num"].to_numpy(dtype=float),
                "pred_b2": pred_b2,
                "mean_g": mean_g,
                "median_g": median_g,
                "n": int(len(test)),
            }
    return cache


def mae_at_alpha(cache, t, r, alpha):
    c = cache[(t, r)]
    p2c = alpha * c["pred_b2"] + (1 - alpha) * c["median_g"]
    return float(np.mean(np.abs(c["y"] - p2c)))


def spearman_of_b2(cache, t, r):
    c = cache[(t, r)]
    x = c["pred_b2"]; y = c["y"]
    if len(x) < 2: return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx)==0 or np.std(ry)==0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])


def main():
    inicio = datetime.now()
    print("== H2A — alpha walk-forward ==")
    hash_antes = sha256(PARQUET)

    df = pd.read_parquet(PARQUET)
    df = df.sort_values(["temporada","rodada","atleta_id"]).reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("Pre-computando cache...")
    cache = build_cache(df)
    print(f"cache: {len(cache)} (t, r)")

    folds = [(t, r) for (t, r) in sorted(cache.keys()) if r > 9]
    print(f"Folds: {len(folds)}")

    rows = []
    alpha_rows = []
    for (t, R) in folds:
        # outer MAE para cada alpha
        outer_maes = {a: mae_at_alpha(cache, t, R, a) for a in ALPHAS}
        for a in ALPHAS:
            alpha_rows.append({"temporada": t, "rodada_alvo": R, "alpha": a,
                               "mae": outer_maes[a]})

        # inner walk-forward: media de MAE sobre r_i em [MIN_INNER, R-1]
        inner_rounds = [r for (tt, r) in cache.keys() if tt == t and MIN_INNER <= r <= R-1]
        inner_means = {}
        for a in ALPHAS:
            mas = [mae_at_alpha(cache, t, ri, a) for ri in inner_rounds]
            inner_means[a] = float(np.mean(mas)) if mas else np.nan
        alpha_star = min(inner_means, key=lambda a: inner_means[a]
                         if not np.isnan(inner_means[a]) else np.inf)

        rows.append({
            "temporada": t, "rodada_alvo": R,
            "alpha_star": float(alpha_star),
            "mae_b2": outer_maes[1.0],
            "mae_alpha_star": outer_maes[alpha_star],
            "mae_alpha_04_fixo": outer_maes.get(0.4, np.nan),
            "spearman_b2": spearman_of_b2(cache, t, R),
            "n_inner": len(inner_rounds),
            "n_test": cache[(t, R)]["n"],
            "inner_mean_mae_star": inner_means[alpha_star],
            "inner_mean_mae_04": inner_means.get(0.4, np.nan),
        })

    dfr = pd.DataFrame(rows)
    dfr.to_csv(CSV_RESULTS, index=False)
    pd.DataFrame(alpha_rows).to_csv(CSV_ALPHAS, index=False)

    n = int(dfr["n_test"].sum())
    b2_w = float((dfr["mae_b2"]*dfr["n_test"]).sum()/n)
    star_w = float((dfr["mae_alpha_star"]*dfr["n_test"]).sum()/n)
    fixo_w = float((dfr["mae_alpha_04_fixo"]*dfr["n_test"]).sum()/n)

    alpha_counts = dfr["alpha_star"].value_counts().sort_index()
    alpha_mean = float(dfr["alpha_star"].mean())
    alpha_median = float(dfr["alpha_star"].median())
    n_comp = int((dfr["alpha_star"] < 1.0).sum())
    n_full = int((dfr["alpha_star"] == 1.0).sum())

    por_temp = dfr.groupby("temporada").apply(lambda g: pd.Series({
        "n_folds": len(g),
        "b2_mae": float((g["mae_b2"]*g["n_test"]).sum()/g["n_test"].sum()),
        "wf_mae": float((g["mae_alpha_star"]*g["n_test"]).sum()/g["n_test"].sum()),
        "fixo_04_mae": float((g["mae_alpha_04_fixo"]*g["n_test"]).sum()/g["n_test"].sum()),
        "alpha_medio": float(g["alpha_star"].mean()),
    })).reset_index()

    # folds B2 melhor / wf melhor
    n_wf_better = int((dfr["mae_alpha_star"] < dfr["mae_alpha_04_fixo"]).sum())
    n_fixo_better = int((dfr["mae_alpha_04_fixo"] < dfr["mae_alpha_star"]).sum())
    n_tie = int((dfr["mae_alpha_04_fixo"] == dfr["mae_alpha_star"]).sum())

    hash_depois = sha256(PARQUET)
    imut = (hash_antes == hash_depois)

    fim = datetime.now()
    L = []
    L.append("# H2A — alpha walk-forward vs alpha global\n\n")
    L.append(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- Folds: {len(dfr)}\n- Obs: {n}\n")
    L.append(f"- Grid: {len(ALPHAS)} alphas (passo 0.05)\n")
    L.append(f"- min_inner_rounds: {MIN_INNER}\n\n")

    L.append("## Resultado global\n\n")
    L.append(f"- B2 puro (α=1):       MAE = {b2_w:.4f}\n")
    L.append(f"- B2 α=0.4 fixo:       MAE = {fixo_w:.4f}  (contaminado)\n")
    L.append(f"- B2 α walk-forward:   MAE = {star_w:.4f}\n\n")

    L.append("## Distribuicao de α*\n\n")
    L.append(f"- media: {alpha_mean:.3f}\n- mediana: {alpha_median:.3f}\n")
    L.append(f"- folds com α* < 1: {n_comp}/{len(dfr)}\n")
    L.append(f"- folds com α* = 1: {n_full}/{len(dfr)}\n")
    L.append(f"- folds wf melhor que fixo(0.4): {n_wf_better}\n")
    L.append(f"- folds fixo(0.4) melhor que wf: {n_fixo_better}\n")
    L.append(f"- empates: {n_tie}\n\n")
    L.append("Contagem de α* por fold:\n\n")
    for a, c in alpha_counts.items():
        L.append(f"- {a}: {c}\n")
    L.append("\n")

    L.append("## Por temporada\n\n")
    L.append(por_temp.to_string(index=False))
    L.append("\n\n")

    L.append("## Imutabilidade\n\n")
    L.append(f"- Parquet SHA: {'OK' if imut else 'MUDOU'}\n")

    SUMMARY.write_text("".join(L), encoding="utf-8")

    print()
    print(f"B2 puro:          {b2_w:.4f}")
    print(f"B2 α=0.4 fixo:    {fixo_w:.4f}")
    print(f"B2 α walk-fwd:    {star_w:.4f}")
    print(f"α* media: {alpha_mean:.3f}  mediana: {alpha_median:.3f}")
    print(f"folds compressao: {n_comp}/{len(dfr)}")
    print(f"imutavel: {imut}")


if __name__ == "__main__":
    main()
