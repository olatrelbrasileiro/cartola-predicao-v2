#!/usr/bin/env python3
"""
ETAPA 5 — Baseline B2.

Formula:
  H(i,r) = media de TODAS as observacoes elegiveis do jogador i
           em rodadas < r, mesma temporada
  R(i,r) = media das ULTIMAS 5 observacoes elegiveis do jogador i
           em rodadas < r, mesma temporada
  innovation = R - H
  prediction = H + innovation = R      (numericamente usado R)

Fallback: sem historico -> train_mean_global do fold.
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path.home() / "cartola-predicao-v2"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
CONFIG = ROOT / "config.yaml"
MANIFEST = ROOT / "reports" / "harness_folds_v2.csv"
B0_CSV = ROOT / "reports" / "baseline_b0_folds.csv"
B1_CSV = ROOT / "reports" / "baseline_b1_folds.csv"
B0_SUM = ROOT / "reports" / "baseline_b0_summary.md"
B1_SUM = ROOT / "reports" / "baseline_b1_summary.md"
REP = ROOT / "reports"

CSV_FOLDS = REP / "baseline_b2_folds.csv"
CSV_SPOT = REP / "baseline_b2_spotcheck.csv"
SUMMARY = REP / "baseline_b2_summary.md"

TARGET_COL = "pontos_num"
KEY_COLS = ["temporada", "rodada", "atleta_id"]
RECENT_WINDOW = 5


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def load_config() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_canonical() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    return df.sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)


def derive_runtime_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flag_ambigua_B"] = (
        (out["entrou_em_campo"] == False) & (out[TARGET_COL] != 0)
    )
    out["elegivel"] = ~((out["temporada"] == 2025) & (out["rodada"] == 1))
    return out


def build_folds(df: pd.DataFrame, min_train_rounds: int) -> list:
    folds = []
    for temporada in sorted(df["temporada"].unique()):
        sub = df[df["temporada"] == temporada]
        for r in sorted(sub["rodada"].unique()):
            if r <= min_train_rounds:
                continue
            folds.append({"temporada": int(temporada), "rodada_alvo": int(r)})
    return folds


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def run_fold_b2(df: pd.DataFrame, fold: dict, b0_row: dict, b1_row: dict) -> dict:
    t = fold["temporada"]
    r = fold["rodada_alvo"]

    mask_t = df["temporada"] == t
    train = df[mask_t & (df["rodada"] < r) & df["elegivel"]]
    test_raw = df[mask_t & (df["rodada"] == r) & df["elegivel"]]
    test_metric = test_raw[~test_raw["flag_ambigua_B"]].copy()

    train_mean_global = float(train[TARGET_COL].mean())

    # H: media por jogador sobre TODAS as observacoes do treino
    hist_means = train.groupby("atleta_id")[TARGET_COL].mean()

    # R: media das ultimas 5 observacoes por jogador (ordem por rodada)
    train_sorted = train.sort_values(["atleta_id", "rodada"], kind="mergesort")
    recent_means = (train_sorted.groupby("atleta_id")
                    .tail(RECENT_WINDOW)
                    .groupby("atleta_id")[TARGET_COL].mean())

    test_metric["historico_mean"] = test_metric["atleta_id"].map(hist_means)
    test_metric["recente_mean"] = test_metric["atleta_id"].map(recent_means)

    has_hist = test_metric["historico_mean"].notna()
    test_metric["innovation_recent"] = (
        test_metric["recente_mean"] - test_metric["historico_mean"]
    )
    # prediction = H + innovation = R  (usar R direto para exatidao)
    test_metric["prediction"] = test_metric["recente_mean"]
    test_metric.loc[~has_hist, "prediction"] = train_mean_global
    test_metric["used_fallback_b0"] = ~has_hist

    y_true = test_metric[TARGET_COL].to_numpy(dtype=float)
    preds = test_metric["prediction"].to_numpy(dtype=float)
    n = len(y_true)

    mae = float(np.mean(np.abs(y_true - preds))) if n else np.nan
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2))) if n else np.nan
    sp = spearman_safe(preds, y_true)

    n_com_hist = int(has_hist.sum())
    n_fallback = int(n - n_com_hist)
    coverage = float(n_com_hist / n) if n else np.nan

    mean_h = float(test_metric.loc[has_hist, "historico_mean"].mean()) if n_com_hist else np.nan
    mean_r = float(test_metric.loc[has_hist, "recente_mean"].mean()) if n_com_hist else np.nan
    mean_inn = float(test_metric.loc[has_hist, "innovation_recent"].mean()) if n_com_hist else np.nan

    b0_mae = float(b0_row["mae"])
    b1_mae = float(b1_row["mae"])
    delta_b1 = mae - b1_mae
    melhoria_b1 = b1_mae - mae

    return {
        "temporada": int(t),
        "rodada_alvo": int(r),
        "n_train": int(len(train)),
        "n_test_metric": int(n),
        "n_test_ambiguous_B": int(len(test_raw) - n),
        "train_mean_global": train_mean_global,
        "n_test_com_historico": n_com_hist,
        "n_test_fallback_b0": n_fallback,
        "coverage_historico": coverage,
        "mean_historico": mean_h,
        "mean_recente": mean_r,
        "mean_innovation": mean_inn,
        "mae": mae,
        "rmse": rmse,
        "spearman": sp,
        "b0_mae": b0_mae,
        "b1_mae": b1_mae,
        "delta_mae_vs_b1": delta_b1,
        "melhoria_vs_b1": melhoria_b1,
        "n_b2_predictions_from_recent": n_com_hist,
        "train_max_round": int(train["rodada"].max()),
        "test_round": int(test_raw["rodada"].min()),
        "_test_metric_df": test_metric,   # guardado temporariamente p/ spotcheck
    }


def build_spotcheck(df: pd.DataFrame, folds: list, b0: pd.DataFrame, b1: pd.DataFrame) -> pd.DataFrame:
    """Amostra auditavel: 20+ linhas cobrindo categorias."""
    rng = np.random.default_rng(20240920)
    rows = []
    picks = []
    # 3 folds por ano
    for t in [2024, 2025, 2026]:
        idxs = [i for i, f in enumerate(folds) if f["temporada"] == t]
        escolhidos = list(rng.choice(idxs, size=min(4, len(idxs)), replace=False))
        picks.extend(escolhidos)
    picks = sorted(set(picks))

    for i in picks:
        f = folds[i]
        b0_row = b0.set_index(["temporada","rodada_alvo"]).loc[(f["temporada"], f["rodada_alvo"])]
        b1_row = b1.set_index(["temporada","rodada_alvo"]).loc[(f["temporada"], f["rodada_alvo"])]
        res = run_fold_b2(df, f, b0_row, b1_row)
        tm = res["_test_metric_df"]
        # pega 3 jogadores: com historico longo, com poucos, fallback
        com_hist = tm[tm["historico_mean"].notna()]
        fallback = tm[tm["used_fallback_b0"]]
        if len(com_hist):
            for _, row in com_hist.sample(n=min(2, len(com_hist)), random_state=1).iterrows():
                rows.append({
                    "temporada": row["temporada"],
                    "rodada_alvo": row["rodada"],
                    "atleta_id": row["atleta_id"],
                    "historico_mean": row["historico_mean"],
                    "recente_mean": row["recente_mean"],
                    "innovation_recent": row["innovation_recent"],
                    "prediction": row["prediction"],
                    "train_mean_global": res["train_mean_global"],
                    "used_fallback_b0": bool(row["used_fallback_b0"]),
                    "y_true": row[TARGET_COL],
                })
        if len(fallback):
            row = fallback.iloc[0]
            rows.append({
                "temporada": row["temporada"],
                "rodada_alvo": row["rodada"],
                "atleta_id": row["atleta_id"],
                "historico_mean": row["historico_mean"],
                "recente_mean": row["recente_mean"],
                "innovation_recent": row["innovation_recent"],
                "prediction": row["prediction"],
                "train_mean_global": res["train_mean_global"],
                "used_fallback_b0": bool(row["used_fallback_b0"]),
                "y_true": row[TARGET_COL],
            })
    return pd.DataFrame(rows)


def main():
    inicio = datetime.now()
    print("== ETAPA 5 — B2 ==")

    hash_antes = sha256(PARQUET)
    n_antes = len(pd.read_parquet(PARQUET))
    h_b0f = sha256(B0_CSV); h_b0s = sha256(B0_SUM)
    h_b1f = sha256(B1_CSV); h_b1s = sha256(B1_SUM)
    print(f"Parquet SHA-256 (antes): {hash_antes[:16]}...")

    cfg = load_config()
    mtr = cfg.get("validation", {}).get("min_train_rounds")
    if not isinstance(mtr, int) or mtr <= 0:
        print(f"ERRO: config.yaml sem min_train_rounds: {mtr!r}", file=sys.stderr)
        sys.exit(3)

    df = load_canonical()
    df = derive_runtime_flags(df)
    print(f"Linhas: {len(df)}")

    b0 = pd.read_csv(B0_CSV)
    b1 = pd.read_csv(B1_CSV)
    b0_idx = b0.set_index(["temporada","rodada_alvo"])
    b1_idx = b1.set_index(["temporada","rodada_alvo"])

    b0_global = float((b0["mae"] * b0["n_test_metric"]).sum() / b0["n_test_metric"].sum())
    b1_global = float((b1["mae"] * b1["n_test_metric"]).sum() / b1["n_test_metric"].sum())
    print(f"B0 MAE global: {b0_global:.4f}")
    print(f"B1 MAE global: {b1_global:.4f}")

    folds = build_folds(df, mtr)
    print(f"Folds: {len(folds)}")

    man = pd.read_csv(MANIFEST)
    a = set(map(tuple, man[["temporada","rodada_alvo"]].values.tolist()))
    b = set((f["temporada"], f["rodada_alvo"]) for f in folds)
    if a != b:
        print("ERRO: folds B2 != harness", file=sys.stderr)
        sys.exit(4)
    print("Consistencia com harness: OK")

    resultados = []
    for f in folds:
        b0_row = b0_idx.loc[(f["temporada"], f["rodada_alvo"])]
        b1_row = b1_idx.loc[(f["temporada"], f["rodada_alvo"])]
        res = run_fold_b2(df, f, b0_row, b1_row)
        resultados.append(res)

    # spotcheck antes de dropar coluna temporaria
    spot = build_spotcheck(df, folds, b0, b1)
    spot.to_csv(CSV_SPOT, index=False)

    # dropar coluna interna
    for res in resultados:
        res.pop("_test_metric_df", None)

    dfr = pd.DataFrame(resultados)

    # auditorias internas
    assert len(dfr) == 76
    assert (dfr["train_max_round"] < dfr["rodada_alvo"]).all()
    assert (dfr["test_round"] == dfr["rodada_alvo"]).all()
    assert (dfr["n_test_com_historico"] + dfr["n_test_fallback_b0"] == dfr["n_test_metric"]).all()
    assert ((dfr["coverage_historico"] >= 0) & (dfr["coverage_historico"] <= 1)).all()
    assert (dfr["mae"] >= 0).all()
    assert (dfr["rmse"] >= 0).all()
    assert np.isfinite(dfr["train_mean_global"]).all()

    dfr.to_csv(CSV_FOLDS, index=False)

    total_n = int(dfr["n_test_metric"].sum())
    mae_medio = float(dfr["mae"].mean())
    mae_pond = float((dfr["mae"] * dfr["n_test_metric"]).sum() / total_n)
    rmse_pond = float(np.sqrt((dfr["rmse"]**2 * dfr["n_test_metric"]).sum() / total_n))

    sp_val = dfr["spearman"].dropna()
    sp_medio = float(sp_val.mean()) if len(sp_val) else np.nan
    n_sp_val = int(len(sp_val))
    n_sp_nan = int(dfr["spearman"].isna().sum())

    n_com_hist_tot = int(dfr["n_test_com_historico"].sum())
    n_fallback_tot = int(dfr["n_test_fallback_b0"].sum())
    cov_global = n_com_hist_tot / total_n if total_n else np.nan

    melhor = dfr["delta_mae_vs_b1"] < 0
    pior = dfr["delta_mae_vs_b1"] > 0
    empate = dfr["delta_mae_vs_b1"] == 0
    n_melhor = int(melhor.sum())
    n_pior = int(pior.sum())
    n_empate = int(empate.sum())
    mediana_delta = float(dfr["delta_mae_vs_b1"].median())
    media_delta = float(dfr["delta_mae_vs_b1"].mean())

    melhoria_abs = b1_global - mae_pond
    melhoria_pct = (b1_global - mae_pond) / b1_global * 100

    hash_depois = sha256(PARQUET)
    n_depois = len(pd.read_parquet(PARQUET))
    imutavel = (hash_antes == hash_depois and n_antes == n_depois)
    b0_inalt = (h_b0f == sha256(B0_CSV)) and (h_b0s == sha256(B0_SUM))
    b1_inalt = (h_b1f == sha256(B1_CSV)) and (h_b1s == sha256(B1_SUM))

    fim = datetime.now()
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# ETAPA 5 — B2 (historico expansivo + inovacao recente)\n\n")
        w(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
        w(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
        w(f"- Duracao: {(fim-inicio).total_seconds():.2f}s\n\n")

        w("## Definicao\n\n")
        w("H = media historica expansiva do jogador (mesma temp., rodada < alvo)\n")
        w("R = media das ultimas 5 observacoes anteriores do jogador\n")
        w("innovation = R - H\n")
        w("prediction = H + innovation = R\n\n")
        w("Fallback: sem historico -> train_mean_global do fold.\n\n")

        w("## Folds\n\n")
        w(f"- Total: {len(dfr)}\n")
        for t in sorted(dfr["temporada"].unique()):
            w(f"- {int(t)}: {int((dfr['temporada']==t).sum())}\n")
        w("\n")

        w("## Metricas\n\n")
        w(f"- MAE medio simples: {mae_medio:.4f}\n")
        w(f"- **MAE global ponderado: {mae_pond:.4f}**  (principal)\n")
        w(f"- RMSE agregado: {rmse_pond:.4f}\n")
        w(f"- Spearman medio: {sp_medio:.4f}\n")
        w(f"- Folds com Spearman valido: {n_sp_val}\n")
        w(f"- Folds com Spearman indefinido: {n_sp_nan}\n")
        w(f"- Observacoes avaliadas: {total_n}\n\n")

        w("## Coverage\n\n")
        w(f"- Com historico: {n_com_hist_tot}\n")
        w(f"- Fallback B0: {n_fallback_tot}\n")
        w(f"- Coverage global: {cov_global:.4f}\n\n")

        w("## Comparacao com B1\n\n")
        w(f"- B1 MAE: {b1_global:.4f}\n")
        w(f"- B2 MAE: {mae_pond:.4f}\n")
        w(f"- Melhoria absoluta: {melhoria_abs:+.4f}\n")
        w(f"- Melhoria percentual: {melhoria_pct:+.2f}%\n")
        w(f"- Folds B2 < B1: {n_melhor}\n")
        w(f"- Folds B2 > B1: {n_pior}\n")
        w(f"- Empates: {n_empate}\n")
        w(f"- Mediana delta (B2 - B1): {mediana_delta:+.4f}\n")
        w(f"- Media delta (B2 - B1): {media_delta:+.4f}\n\n")

        w("## Notas\n\n")
        w("- Nenhuma feature externa utilizada.\n")
        w("- Unica mudanca em relacao ao B1: substituir H pela forma recente R.\n")
        w("- Janela recente fixa em 5 observacoes. Sem tuning.\n")
        w("- Historico intra-temporada; janela expansiva; 2025/R1 excluida.\n")
        w("- B (ambiguas) permanece no treino; removida so das metricas.\n\n")

        w("## Imutabilidade\n\n")
        w(f"- Parquet SHA-256 antes: {hash_antes}\n")
        w(f"- Parquet SHA-256 depois: {hash_depois}\n")
        w(f"- Parquet imutavel: {imutavel}\n")
        w(f"- B0 inalterado: {b0_inalt}\n")
        w(f"- B1 inalterado: {b1_inalt}\n")

    print()
    print("=== RESUMO B2 ===")
    print(f"folds: {len(dfr)}")
    print(f"MAE medio simples: {mae_medio:.4f}")
    print(f"MAE global ponderado: {mae_pond:.4f}")
    print(f"RMSE agregado: {rmse_pond:.4f}")
    print(f"Spearman medio: {sp_medio:.4f} (validos={n_sp_val}, indefinidos={n_sp_nan})")
    print(f"Coverage global: {cov_global:.4f}  fallback: {n_fallback_tot}")
    print(f"B1 MAE: {b1_global:.4f}")
    print(f"B2 MAE: {mae_pond:.4f}")
    print(f"Melhoria vs B1: {melhoria_abs:+.4f} ({melhoria_pct:+.2f}%)")
    print(f"Folds B2<B1: {n_melhor}  B2>B1: {n_pior}  empates: {n_empate}")
    print(f"Mediana delta: {mediana_delta:+.4f}")
    print(f"Parquet imutavel: {imutavel}")
    print(f"B0 inalterado: {b0_inalt}")
    print(f"B1 inalterado: {b1_inalt}")
    print(f"CSV: {CSV_FOLDS.relative_to(ROOT)}")
    print(f"Spotcheck: {CSV_SPOT.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY.relative_to(ROOT)}")

    if not imutavel or not b0_inalt or not b1_inalt:
        print("ERRO: arquivo alterado indevidamente.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
