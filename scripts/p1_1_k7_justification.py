#!/usr/bin/env python3
"""
P1.1 — Justificativa quantitativa do k=7.

Modelo hierarquico:
   y_ij ~ N(mu_i, sigma^2)   (obs do jogador i)
   mu_i ~ N(mu_0, tau^2)     (media do jogador)
Empirical-Bayes: k = sigma^2 / tau^2.

Estimadores:
  sigma^2 : variancia within-player pooled (somente n_i >= 2)
  tau^2   : Var(mean_i) - sigma^2 * E[1/n_i]  (todos com n_i >= 1)
            Se tau^2 corrigido <= 0 -> DEGENERADO (k = NaN).

NAO altera k=7. Sem info futura.
"""
import hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("raiz nao encontrada")


ROOT = detect_root()
CANON = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
MANIFEST = ROOT/"reports"/"harness_folds_v2.csv"
OUT_CSV = ROOT/"reports"/"p1_1_k7_estimates.csv"
OUT_MD = ROOT/"reports"/"p1_1_k7_summary.md"

TARGET = "pontos_num"
HASH_ESPERADO = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
TAU2_EPS = 1e-9


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def estimate_k(train_df):
    grp = train_df.groupby("atleta_id")[TARGET]
    stats = grp.agg(mean="mean", var="var", count="count")
    n_all = int(len(stats))
    n_var = int((stats["count"] >= 2).sum())

    if n_var == 0:
        # sem players com n>=2 -> sigma^2 indefinido
        return {
            "sigma2": np.nan, "tau2_raw": float(stats["mean"].var(ddof=1)) if n_all>=2 else np.nan,
            "tau2_corrected": np.nan, "k_estimated": np.nan,
            "n_players_all": n_all, "n_players_var": n_var,
            "mean_n": float(stats["count"].mean()), "degenerate": True,
            "nota": "n_var=0",
        }

    valid_var = stats[stats["count"] >= 2]
    w = valid_var["count"] - 1
    sigma2 = float((valid_var["var"] * w).sum() / w.sum())

    all_means = stats["mean"]
    all_ns = stats["count"]
    tau2_raw = float(all_means.var(ddof=1)) if n_all >= 2 else np.nan
    corr = sigma2 * float((1.0 / all_ns).mean())
    tau2_corr = tau2_raw - corr if not np.isnan(tau2_raw) else np.nan
    degenerate = (not np.isnan(tau2_corr) and tau2_corr <= TAU2_EPS) or np.isnan(tau2_corr)
    k = sigma2 / tau2_corr if not degenerate else np.nan

    return {
        "sigma2": sigma2,
        "tau2_raw": tau2_raw,
        "tau2_corrected": tau2_corr,
        "k_estimated": k,
        "n_players_all": n_all,
        "n_players_var": n_var,
        "mean_n": float(all_ns.mean()),
        "degenerate": bool(degenerate),
        "nota": "ok" if not degenerate else "tau2_corr<=0",
    }


def main():
    inicio = datetime.now()
    h_before = sha256(CANON)
    if h_before != HASH_ESPERADO:
        print("ERRO: hash do parquet nao bate", file=sys.stderr); sys.exit(2)

    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)

    man = pd.read_csv(MANIFEST)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]

    rows = []
    for (t, R) in folds:
        mask = df["temporada"] == t
        train = df[mask & (df["rodada"] < R) & df["elegivel"] & df["entrou_bool"]]
        if len(train) < 50:
            continue
        est = estimate_k(train)
        est.update({"temporada": t, "rodada_alvo": R, "n_train": int(len(train))})
        rows.append(est)

    dfr = pd.DataFrame(rows)
    dfr.to_csv(OUT_CSV, index=False)

    valid = dfr[dfr["k_estimated"].notna()]
    deg = dfr[dfr["degenerate"] == True]

    L = []
    L.append("# P1.1 — Justificativa quantitativa do k=7\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Folds avaliados: {len(dfr)}\n")
    L.append(f"- Modelo: `y_ij ~ N(mu_i, sigma^2)`, `mu_i ~ N(mu_0, tau^2)`\n")
    L.append(f"- Estimador: `k = sigma^2 / tau^2`\n")
    L.append(f"- k pre-registrado: 7.0 (NAO alterado)\n\n")

    L.append("## Contagens por fold\n\n")
    L.append(f"- folds com estimativa valida: {len(valid)}/{len(dfr)}\n")
    L.append(f"- folds degenerados: {len(deg)}\n\n")

    L.append("### n_players_var (players com n_i >= 2)\n")
    L.append(f"- min    : {int(dfr['n_players_var'].min())}\n")
    L.append(f"- mediana: {int(dfr['n_players_var'].median())}\n")
    L.append(f"- max    : {int(dfr['n_players_var'].max())}\n\n")
    L.append("### n_players_all (players com n_i >= 1)\n")
    L.append(f"- min    : {int(dfr['n_players_all'].min())}\n")
    L.append(f"- mediana: {int(dfr['n_players_all'].median())}\n")
    L.append(f"- max    : {int(dfr['n_players_all'].max())}\n\n")

    if len(valid) > 0:
        L.append("## Distribuicao de k (folds validos)\n\n")
        for c in ["sigma2", "tau2_corrected", "k_estimated"]:
            s = valid[c].dropna()
            if len(s) == 0: continue
            L.append(f"### {c}\n")
            L.append(f"- n      : {len(s)}\n")
            L.append(f"- min    : {s.min():.4f}\n")
            L.append(f"- p25    : {s.quantile(.25):.4f}\n")
            L.append(f"- mediana: {s.median():.4f}\n")
            L.append(f"- p75    : {s.quantile(.75):.4f}\n")
            L.append(f"- max    : {s.max():.4f}\n\n")

    L.append("## Por temporada (k mediano)\n\n")
    for t in sorted(dfr["temporada"].unique()):
        sub = dfr[(dfr["temporada"] == t) & dfr["k_estimated"].notna()]
        if len(sub) == 0:
            L.append(f"- {t}: sem folds validos\n")
            continue
        L.append(f"- {t}: n={len(sub)}  "
                 f"k_mediana={sub['k_estimated'].median():.3f}  "
                 f"k_media={sub['k_estimated'].mean():.3f}  "
                 f"k_min={sub['k_estimated'].min():.3f}  "
                 f"k_max={sub['k_estimated'].max():.3f}\n")
    L.append("\n")

    L.append("## Conclusao\n\n")
    if len(valid) > 0:
        med = float(valid["k_estimated"].median())
        L.append(f"- mediana de k entre folds validos: **{med:.3f}**\n")
        L.append(f"- k pre-registrado: 7.0\n\n")
        if 3.0 <= med <= 15.0:
            L.append("**FATO:** mediana de k na mesma ordem de magnitude de 7.\n")
            L.append("**INTERPRETACAO:** k=7 e compativel com a estrutura de variancia "
                     "dos dados de treino.\n")
        else:
            L.append("**FATO:** mediana de k difere substancialmente de 7.\n")
            L.append("**RECOMENDACAO:** registrar a discrepancia. Nao alterar k agora.\n")
    L.append(f"\n- Folds degenerados foram EXCLUIDOS da mediana, nao silenciados.\n")
    L.append(f"- Nenhum tuning. k=7 mantido.\n")

    h_after = sha256(CANON)
    L.append(f"\n## Integridade\n\n- Parquet inalterado: {h_before == h_after}\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    print(f"folds validos: {len(valid)}/{len(dfr)}   degenerados: {len(deg)}")
    if len(valid):
        print(f"k mediano: {valid['k_estimated'].median():.3f}")
    print("artefatos: reports/p1_1_k7_estimates.csv, reports/p1_1_k7_summary.md")


if __name__ == "__main__":
    main()
