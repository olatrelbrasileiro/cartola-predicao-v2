#!/usr/bin/env python3
"""
Auditoria H1 — Separar ganho de informacao de ganho por compressao no B3.

3 experimentos deterministicos:
  H1.1: comprimir B2 em direcao a train_median (alpha sweep)
  H1.2: permutar prediction_b3 dentro de cada fold (controle)
  H1.3: rescalar B2 para bater mean/std do B3

NAO cria previsao nova, NAO combina B2+B3, NAO treina modelo.
"""
import hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
ERROS = ROOT/"reports"/"auditoria_b2_b3_erros.csv"
B2_FOLDS = ROOT/"reports"/"baseline_b2_folds.csv"
B3_FOLDS = ROOT/"reports"/"baseline_b3_folds.csv"
REP = ROOT/"reports"

CSV_H11 = REP/"auditoria_h1_b2_compressao.csv"
CSV_H12 = REP/"auditoria_h1_b3_permutado.csv"
CSV_H13 = REP/"auditoria_h1_b2_stdmatch.csv"
SUMMARY = REP/"auditoria_h1_summary.md"

ALPHAS = [round(a, 2) for a in np.arange(0, 1.01, 0.1)]
SEEDS = [42, 123, 777, 2024, 2025]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    inicio = datetime.now()
    print("== AUDITORIA H1 ==")

    hashes_antes = {
        "parquet": sha256(PARQUET),
        "b2_folds": sha256(B2_FOLDS),
        "b3_folds": sha256(B3_FOLDS),
        "erros": sha256(ERROS),
    }

    obs = pd.read_csv(ERROS)
    print(f"Obs: {len(obs)}")

    # train stats por fold
    dfp = pd.read_parquet(PARQUET)
    dfp["elegivel"] = ~((dfp["temporada"] == 2025) & (dfp["rodada"] == 1))
    folds = sorted(obs[["temporada","rodada"]].drop_duplicates().itertuples(index=False, name=None))
    print(f"Folds: {len(folds)}")

    train_stats = {}
    for (t, r) in folds:
        train = dfp[(dfp["temporada"]==t) & (dfp["rodada"] < r) & dfp["elegivel"]]
        train_stats[(int(t), int(r))] = {
            "median": float(train["pontos_num"].median()),
            "mean": float(train["pontos_num"].mean()),
            "std": float(train["pontos_num"].std()),
        }

    obs_by_fold = {k: v for k, v in obs.groupby(["temporada","rodada"])}

    # ---- H1.1 ----
    h11 = []
    for (t, r), sub in obs_by_fold.items():
        y = sub["pontos_real"].to_numpy()
        p2 = sub["prediction_b2"].to_numpy()
        anchor = train_stats[(int(t), int(r))]["median"]
        for a in ALPHAS:
            p2c = a * p2 + (1 - a) * anchor
            h11.append({
                "temporada": int(t), "rodada": int(r), "alpha": float(a),
                "anchor": "train_median",
                "mae": float(np.mean(np.abs(y - p2c))),
                "rmse": float(np.sqrt(np.mean((y - p2c)**2))),
                "n": int(len(y)),
            })
    df_h11 = pd.DataFrame(h11)
    df_h11.to_csv(CSV_H11, index=False)

    # ---- H1.2 ----
    h12 = []
    for (t, r), sub in obs_by_fold.items():
        y = sub["pontos_real"].to_numpy()
        p3 = sub["prediction_b3"].to_numpy()
        for s in SEEDS:
            rng = np.random.default_rng(s)
            p3p = rng.permutation(p3)
            h12.append({
                "temporada": int(t), "rodada": int(r), "seed": int(s),
                "mae": float(np.mean(np.abs(y - p3p))),
                "rmse": float(np.sqrt(np.mean((y - p3p)**2))),
                "n": int(len(y)),
            })
    df_h12 = pd.DataFrame(h12)
    df_h12.to_csv(CSV_H12, index=False)

    # ---- H1.3 ----
    h13 = []
    for (t, r), sub in obs_by_fold.items():
        y = sub["pontos_real"].to_numpy()
        p2 = sub["prediction_b2"].to_numpy()
        p3 = sub["prediction_b3"].to_numpy()
        m2, s2 = float(p2.mean()), float(p2.std())
        m3, s3 = float(p3.mean()), float(p3.std())
        if s2 == 0:
            p2c = np.full_like(p2, m3)
        else:
            p2c = m3 + (p2 - m2) * (s3 / s2)
        h13.append({
            "temporada": int(t), "rodada": int(r),
            "mae": float(np.mean(np.abs(y - p2c))),
            "rmse": float(np.sqrt(np.mean((y - p2c)**2))),
            "n": int(len(y)),
            "mean_b2": m2, "std_b2": s2,
            "mean_b3": m3, "std_b3": s3,
        })
    df_h13 = pd.DataFrame(h13)
    df_h13.to_csv(CSV_H13, index=False)

    # ---- Agregacao ----
    b2_mae_global = float(obs["abs_erro_b2"].mean())
    b3_mae_global = float(obs["abs_erro_b3"].mean())

    g11 = df_h11.groupby("alpha").apply(
        lambda d: pd.Series({
            "mae_global": float((d["mae"]*d["n"]).sum() / d["n"].sum()),
            "rmse_global": float(np.sqrt((d["rmse"]**2 * d["n"]).sum() / d["n"].sum())),
        })
    ).reset_index()

    best_alpha = float(g11.loc[g11["mae_global"].idxmin(), "alpha"])
    best_mae_alpha = float(g11["mae_global"].min())

    h12_agg = df_h12.groupby("seed").apply(
        lambda d: pd.Series({"mae_global": float((d["mae"]*d["n"]).sum()/d["n"].sum())})
    ).reset_index()
    perm_mae_mean = float(h12_agg["mae_global"].mean())
    perm_mae_min = float(h12_agg["mae_global"].min())
    perm_mae_max = float(h12_agg["mae_global"].max())

    stdmatch_mae = float((df_h13["mae"]*df_h13["n"]).sum()/df_h13["n"].sum())

    b2_sp = float(pd.read_csv(B2_FOLDS)["spearman"].mean())
    b3_sp = float(pd.read_csv(B3_FOLDS)["spearman"].mean())

    # ---- summary ----
    fim = datetime.now()
    L = []
    L.append("# Auditoria H1 — B3: compressão vs informação\n\n")
    L.append(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.2f}s\n")
    L.append(f"- Observações: {len(obs)}\n")
    L.append(f"- Folds: {len(folds)}\n\n")

    L.append("## Referências\n\n")
    L.append(f"- B2 MAE: {b2_mae_global:.4f}  Spearman: {b2_sp:.4f}\n")
    L.append(f"- B3 MAE: {b3_mae_global:.4f}  Spearman: {b3_sp:.4f}\n\n")

    L.append("## H1.1 — B2 comprimido em direção a train_median\n\n")
    L.append("`B2c(α) = α·B2 + (1−α)·train_median`  (Spearman preservado p/ α>0)\n\n")
    L.append(g11.to_string(index=False))
    L.append(f"\n\n- Melhor α (menor MAE): {best_alpha}\n")
    L.append(f"- MAE em α*: {best_mae_alpha:.4f}\n\n")

    L.append("## H1.2 — B3 permutado dentro do fold\n\n")
    L.append(h12_agg.to_string(index=False))
    L.append(f"\n\n- MAE permutado (média): {perm_mae_mean:.4f}\n")
    L.append(f"- MAE permutado (min/max): {perm_mae_min:.4f} / {perm_mae_max:.4f}\n\n")

    L.append("## H1.3 — B2 com mean/std igualados ao B3\n\n")
    L.append(f"- MAE B2c_std: {stdmatch_mae:.4f}\n")
    L.append(f"- Spearman preservado (linear monotônica): {b2_sp:.4f}\n\n")

    L.append("## Comparações diretas\n\n")
    L.append(f"- B3 vs B2c(α*): MAE {b3_mae_global:.4f} vs {best_mae_alpha:.4f}  "
             f"delta={b3_mae_global - best_mae_alpha:+.4f}\n")
    L.append(f"- B3 vs B3_perm: MAE {b3_mae_global:.4f} vs {perm_mae_mean:.4f}  "
             f"delta={b3_mae_global - perm_mae_mean:+.4f}\n")
    L.append(f"- B3 vs B2c_std: MAE {b3_mae_global:.4f} vs {stdmatch_mae:.4f}  "
             f"delta={b3_mae_global - stdmatch_mae:+.4f}\n\n")

    L.append("## Imutabilidade\n\n")
    for k, p in [("parquet", PARQUET), ("b2_folds", B2_FOLDS),
                 ("b3_folds", B3_FOLDS), ("erros", ERROS)]:
        L.append(f"- {k}: {'OK' if hashes_antes[k] == sha256(p) else 'MUDOU'}\n")

    SUMMARY.write_text("".join(L), encoding="utf-8")

    print("\nArtefatos:")
    for p in [CSV_H11, CSV_H12, CSV_H13, SUMMARY]:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")
    print()
    print(f"B2 MAE: {b2_mae_global:.4f}")
    print(f"B3 MAE: {b3_mae_global:.4f}")
    print(f"Best alpha: {best_alpha}  MAE: {best_mae_alpha:.4f}")
    print(f"B3_perm MAE (media): {perm_mae_mean:.4f}")
    print(f"B2c_std MAE: {stdmatch_mae:.4f}")
    print(f"B2 Spearman: {b2_sp:.4f}   B3 Spearman: {b3_sp:.4f}")


if __name__ == "__main__":
    main()
