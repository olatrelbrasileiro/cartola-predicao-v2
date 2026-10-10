#!/usr/bin/env python3
"""
ETAPA 5 — Baseline B3 (componente basico, corrigido).

Correcao critica: scouts no canonico sao CUMULATIVOS (ver Etapa 3.c).
Este baseline converte cumulativo -> contribuicao por rodada via diff.

Para cada fold (temporada, rodada_alvo):
  diff_scout_R = ffilled_cumul(R) - ffilled_cumul(R-1)
  basic_score_R = sum(diff_scout_R * peso_scout(temporada))
  prediction_i  = mean(basic_score historico do jogador) ou train_mean_global
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
AUDIT = ROOT / "config" / "pontuacao_scouts_auditoria.yaml"
MANIFEST = ROOT / "reports" / "harness_folds_v2.csv"
B0_CSV = ROOT / "reports" / "baseline_b0_folds.csv"
B1_CSV = ROOT / "reports" / "baseline_b1_folds.csv"
B2_CSV = ROOT / "reports" / "baseline_b2_folds.csv"
REP = ROOT / "reports"

CSV_FOLDS = REP / "baseline_b3_folds.csv"
CSV_SPOT = REP / "baseline_b3_spotcheck.csv"
SUMMARY = REP / "baseline_b3_summary.md"

TARGET_COL = "pontos_num"
KEY_COLS = ["temporada", "rodada", "atleta_id"]


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
    with open(AUDIT, "r", encoding="utf-8") as f:
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


def apply_cumulative_diff(df: pd.DataFrame, scout_cols: list) -> pd.DataFrame:
    """
    Converte scouts cumulativos em contribuicao por rodada.
    Retorna df com colunas diff_scout_X.
    """
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
    """basic_score = sum(diff_scout * peso(temporada)) sobre scouts basicos."""
    pesos = audit["pesos_por_temporada"]
    total = pd.Series(0.0, index=df.index)
    for s in audit["scouts_basicos"]:
        col = f"diff_scout_{s}"
        peso_linha = df["temporada"].map({int(t): pesos[t][s] for t in pesos})
        total = total + df[col].to_numpy() * peso_linha.to_numpy()
    return total.astype(float)


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


def run_fold(df: pd.DataFrame, fold: dict, b0_row, b1_row, b2_row) -> dict:
    t = fold["temporada"]; r = fold["rodada_alvo"]
    mask_t = df["temporada"] == t
    train = df[mask_t & (df["rodada"] < r) & df["elegivel"]]
    test_raw = df[mask_t & (df["rodada"] == r) & df["elegivel"]]
    test_metric = test_raw[~test_raw["flag_ambigua_B"]].copy()

    train_mean_global = float(train[TARGET_COL].mean())
    player_basic_mean = train.groupby("atleta_id")["basic_score"].mean()

    test_metric["basic_mean"] = test_metric["atleta_id"].map(player_basic_mean)
    has_hist = test_metric["basic_mean"].notna()
    test_metric["prediction"] = test_metric["basic_mean"]
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
    basic_hist_mean = (float(test_metric.loc[has_hist, "basic_mean"].mean())
                       if n_com_hist else np.nan)

    b0_mae = float(b0_row["mae"]); b1_mae = float(b1_row["mae"]); b2_mae = float(b2_row["mae"])
    delta_b2 = mae - b2_mae
    melhoria_b2 = b2_mae - mae

    return {
        "temporada": int(t), "rodada_alvo": int(r),
        "n_train": int(len(train)),
        "n_test_metric": int(n),
        "n_test_ambiguous_B": int(len(test_raw) - n),
        "train_mean_global": train_mean_global,
        "n_test_com_basic_history": n_com_hist,
        "n_test_fallback_b0": n_fallback,
        "coverage_basic_history": coverage,
        "basic_history_mean": basic_hist_mean,
        "mae": mae, "rmse": rmse, "spearman": sp,
        "b0_mae": b0_mae, "b1_mae": b1_mae, "b2_mae": b2_mae,
        "delta_mae_vs_b2": delta_b2, "melhoria_vs_b2": melhoria_b2,
        "fold_b3_better_than_b2": bool(delta_b2 < 0),
        "train_max_round": int(train["rodada"].max()),
        "test_round": int(test_raw["rodada"].min()),
        "_tm": test_metric,
    }


def build_spotcheck(df, folds, b0, b1, b2, audit) -> pd.DataFrame:
    rng = np.random.default_rng(20240920)
    rows = []; picks = []
    for t in [2024, 2025, 2026]:
        idxs = [i for i, f in enumerate(folds) if f["temporada"] == t]
        picks.extend(list(rng.choice(idxs, size=min(5, len(idxs)), replace=False)))
    picks = sorted(set(picks))
    scouts = audit["scouts_basicos"]
    scout_cols = [f"diff_scout_{s}" for s in scouts]  # diff, nao cumulativo

    for i in picks:
        f = folds[i]
        b0_r = b0.set_index(["temporada","rodada_alvo"]).loc[(f["temporada"], f["rodada_alvo"])]
        b1_r = b1.set_index(["temporada","rodada_alvo"]).loc[(f["temporada"], f["rodada_alvo"])]
        b2_r = b2.set_index(["temporada","rodada_alvo"]).loc[(f["temporada"], f["rodada_alvo"])]
        res = run_fold(df, f, b0_r, b1_r, b2_r)
        tm = res["_tm"]
        com_hist = tm[tm["basic_mean"].notna()]
        fallback = tm[tm["used_fallback_b0"]]
        t, r = f["temporada"], f["rodada_alvo"]
        train = df[(df["temporada"]==t) & (df["rodada"]<r) & df["elegivel"]]
        n_obs = train.groupby("atleta_id").size()

        for pool, n_take in [(com_hist, 2), (fallback, 1)]:
            if len(pool) == 0: continue
            sample = pool.sample(n=min(n_take, len(pool)), random_state=1)
            for _, row in sample.iterrows():
                out = {
                    "temporada": int(row["temporada"]),
                    "rodada_alvo": int(row["rodada"]),
                    "atleta_id": int(row["atleta_id"]),
                    "n_observacoes_historicas": int(n_obs.get(row["atleta_id"], 0)),
                    "basic_mean": row["basic_mean"],
                    "prediction": row["prediction"],
                    "train_mean_global": res["train_mean_global"],
                    "used_fallback_b0": bool(row["used_fallback_b0"]),
                    "y_true": row[TARGET_COL],
                }
                for s, c in zip(scouts, scout_cols):
                    out[f"diff_{s}"] = row[c]
                rows.append(out)
    return pd.DataFrame(rows)


def main():
    inicio = datetime.now()
    print("== ETAPA 5 — B3 (corrigido) ==")

    hash_antes = sha256(PARQUET)
    n_antes = len(pd.read_parquet(PARQUET))
    hashes_prev = {
        "b0_folds": sha256(B0_CSV),
        "b1_folds": sha256(B1_CSV),
        "b2_folds": sha256(B2_CSV),
    }

    cfg = load_config()
    mtr = cfg.get("validation", {}).get("min_train_rounds")
    if not isinstance(mtr, int) or mtr <= 0:
        print(f"ERRO: config.yaml sem min_train_rounds valido: {mtr!r}", file=sys.stderr)
        sys.exit(3)

    audit = load_audit()
    df = load_canonical()
    df = derive_runtime_flags(df)

    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = apply_cumulative_diff(df, scout_cols)
    df["basic_score"] = compute_basic_score(df, audit)

    print(f"Linhas: {len(df)}")
    print(f"min_train_rounds: {mtr}")
    print(f"Scouts basicos: {audit['scouts_basicos']}")
    print(f"Diffs negativos (deve ser 0): "
          f"{int(sum((df[f'diff_{c}'] < 0).sum() for c in scout_cols))}")

    b0 = pd.read_csv(B0_CSV); b1 = pd.read_csv(B1_CSV); b2 = pd.read_csv(B2_CSV)
    b0_idx = b0.set_index(["temporada","rodada_alvo"])
    b1_idx = b1.set_index(["temporada","rodada_alvo"])
    b2_idx = b2.set_index(["temporada","rodada_alvo"])
    b2_global = float((b2["mae"] * b2["n_test_metric"]).sum() / b2["n_test_metric"].sum())
    print(f"B2 MAE global: {b2_global:.4f}")

    folds = build_folds(df, mtr)
    print(f"Folds: {len(folds)}")

    man = pd.read_csv(MANIFEST)
    a = set(map(tuple, man[["temporada","rodada_alvo"]].values.tolist()))
    b = set((f["temporada"], f["rodada_alvo"]) for f in folds)
    if a != b:
        print("ERRO: folds B3 != harness", file=sys.stderr); sys.exit(4)
    print("Consistencia com harness: OK")

    resultados = []
    for f in folds:
        b0_r = b0_idx.loc[(f["temporada"], f["rodada_alvo"])]
        b1_r = b1_idx.loc[(f["temporada"], f["rodada_alvo"])]
        b2_r = b2_idx.loc[(f["temporada"], f["rodada_alvo"])]
        resultados.append(run_fold(df, f, b0_r, b1_r, b2_r))

    spot = build_spotcheck(df, folds, b0, b1, b2, audit)
    spot.to_csv(CSV_SPOT, index=False)

    for res in resultados:
        res.pop("_tm", None)
    dfr = pd.DataFrame(resultados)

    assert len(dfr) == 76
    assert (dfr["train_max_round"] < dfr["rodada_alvo"]).all()
    assert (dfr["test_round"] == dfr["rodada_alvo"]).all()
    assert (dfr["n_test_com_basic_history"] + dfr["n_test_fallback_b0"] == dfr["n_test_metric"]).all()
    assert ((dfr["coverage_basic_history"] >= 0) & (dfr["coverage_basic_history"] <= 1)).all()
    assert (dfr["mae"] >= 0).all()
    assert (dfr["rmse"] >= 0).all()

    dfr.to_csv(CSV_FOLDS, index=False)

    total_n = int(dfr["n_test_metric"].sum())
    mae_medio = float(dfr["mae"].mean())
    mae_pond = float((dfr["mae"] * dfr["n_test_metric"]).sum() / total_n)
    rmse_pond = float(np.sqrt((dfr["rmse"]**2 * dfr["n_test_metric"]).sum() / total_n))

    sp_val = dfr["spearman"].dropna()
    sp_medio = float(sp_val.mean()) if len(sp_val) else np.nan
    n_sp_val = int(len(sp_val)); n_sp_nan = int(dfr["spearman"].isna().sum())

    n_com_hist_tot = int(dfr["n_test_com_basic_history"].sum())
    n_fallback_tot = int(dfr["n_test_fallback_b0"].sum())
    cov_global = n_com_hist_tot / total_n if total_n else np.nan

    melhor = dfr["delta_mae_vs_b2"] < 0
    pior = dfr["delta_mae_vs_b2"] > 0
    empate = dfr["delta_mae_vs_b2"] == 0
    n_melhor = int(melhor.sum()); n_pior = int(pior.sum()); n_empate = int(empate.sum())
    mediana_delta = float(dfr["delta_mae_vs_b2"].median())
    media_delta = float(dfr["delta_mae_vs_b2"].mean())

    melhoria_abs = b2_global - mae_pond
    melhoria_pct = (b2_global - mae_pond) / b2_global * 100

    hash_depois = sha256(PARQUET)
    n_depois = len(pd.read_parquet(PARQUET))
    imutavel = (hash_antes == hash_depois and n_antes == n_depois)
    prev_inalt = {
        "b0_folds": hashes_prev["b0_folds"] == sha256(B0_CSV),
        "b1_folds": hashes_prev["b1_folds"] == sha256(B1_CSV),
        "b2_folds": hashes_prev["b2_folds"] == sha256(B2_CSV),
    }

    fim = datetime.now()
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# ETAPA 5 — B3 (componente basico, corrigido)\n\n")
        w(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
        w(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
        w(f"- Duracao: {(fim-inicio).total_seconds():.2f}s\n\n")

        w("## Definicao\n\n")
        w("`diff_scout_R = ffilled_cumul(R) - ffilled_cumul(R-1)`\n")
        w("`basic_score_R = sum(diff_scout_R * peso_scout(temporada))`\n")
        w("`prediction_i = mean(basic_score historico do jogador)` ou fallback B0\n\n")
        w("Scouts no canonico sao CUMULATIVOS (Etapa 3.c). Diff aplicado.\n")
        w("Pesos por temporada: 2024 DE=1.0, DS=1.2; 2025-2026 DE=1.3, DS=1.5.\n\n")

        w("## Folds\n\n")
        w(f"- Total: {len(dfr)}\n")
        for t in sorted(dfr["temporada"].unique()):
            w(f"- {int(t)}: {int((dfr['temporada']==t).sum())}\n")
        w("\n")

        w("## Metricas\n\n")
        w(f"- MAE medio simples: {mae_medio:.4f}\n")
        w(f"- **MAE global ponderado: {mae_pond:.4f}**\n")
        w(f"- RMSE agregado: {rmse_pond:.4f}\n")
        w(f"- Spearman medio: {sp_medio:.4f}\n")
        w(f"- Folds com Spearman valido: {n_sp_val}\n")
        w(f"- Folds com Spearman indefinido: {n_sp_nan}\n")
        w(f"- Observacoes avaliadas: {total_n}\n\n")

        w("## Coverage\n\n")
        w(f"- Com historico basico: {n_com_hist_tot}\n")
        w(f"- Fallback B0: {n_fallback_tot}\n")
        w(f"- Coverage: {cov_global:.4f}\n\n")

        w("## Comparacao com B2\n\n")
        w(f"- B2 MAE: {b2_global:.4f}\n")
        w(f"- B3 MAE: {mae_pond:.4f}\n")
        w(f"- Melhoria absoluta: {melhoria_abs:+.4f}\n")
        w(f"- Melhoria percentual: {melhoria_pct:+.2f}%\n")
        w(f"- Folds B3 < B2: {n_melhor}\n")
        w(f"- Folds B3 > B2: {n_pior}\n")
        w(f"- Empates: {n_empate}\n")
        w(f"- Mediana delta: {mediana_delta:+.4f}\n")
        w(f"- Media delta: {media_delta:+.4f}\n\n")

        w("## Definicao congelada dos pesos (por temporada)\n\n")
        w("| scout | peso_2024 | peso_2025 | peso_2026 | incluido_no_basico |\n")
        w("|---|---|---|---|---|\n")
        p = audit["pesos_por_temporada"]
        for s in audit["scouts_basicos"]:
            w(f"| {s} | {p[2024][s]} | {p[2025][s]} | {p[2026][s]} | SIM |\n")
        for s in ["G","A","SG","PS","PP","DP","GC","GS","PC","V"]:
            w(f"| {s} | — | — | — | NAO |\n")
        w("\n")

        w("## Imutabilidade\n\n")
        w(f"- Parquet antes: {hash_antes}\n")
        w(f"- Parquet depois: {hash_depois}\n")
        w(f"- Imutavel: {imutavel}\n")
        w(f"- B0/B1/B2 inalterados: {prev_inalt}\n")

    print()
    print("=== RESUMO B3 (corrigido) ===")
    print(f"folds: {len(dfr)}")
    print(f"MAE medio simples: {mae_medio:.4f}")
    print(f"MAE global ponderado: {mae_pond:.4f}")
    print(f"RMSE agregado: {rmse_pond:.4f}")
    print(f"Spearman medio: {sp_medio:.4f} (validos={n_sp_val}, indefinidos={n_sp_nan})")
    print(f"Coverage: {cov_global:.4f}  fallback: {n_fallback_tot}")
    print(f"B2 MAE: {b2_global:.4f}")
    print(f"B3 MAE: {mae_pond:.4f}")
    print(f"Melhoria vs B2: {melhoria_abs:+.4f} ({melhoria_pct:+.2f}%)")
    print(f"Folds B3<B2: {n_melhor}  B3>B2: {n_pior}  empates: {n_empate}")
    print(f"Mediana delta: {mediana_delta:+.4f}")
    print(f"Parquet imutavel: {imutavel}")
    print(f"B0/B1/B2 inalterados: {prev_inalt}")

    if not imutavel or not all(prev_inalt.values()):
        print("ERRO: arquivo alterado indevidamente.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
