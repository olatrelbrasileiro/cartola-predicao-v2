#!/usr/bin/env python3
"""
ETAPA 5 — Baseline B1 (media historica expansiva do proprio jogador).

Para cada fold (temporada, rodada_alvo):
  train       = elegivel=True, temporada=t, rodada < alvo
  test_raw    = elegivel=True, temporada=t, rodada == alvo
  test_metric = test_raw sem flag_ambigua_B

  prediction_i = mean(pontos_num do jogador i em train)  se houver historico
                 train_mean_global                        caso contrario

Metricas sobre test_metric. Spearman pode ser NaN.
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
B0_SUM = ROOT / "reports" / "baseline_b0_summary.md"
REP = ROOT / "reports"

CSV_FOLDS = REP / "baseline_b1_folds.csv"
SUMMARY = REP / "baseline_b1_summary.md"

TARGET_COL = "pontos_num"
KEY_COLS = ["temporada", "rodada", "atleta_id"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
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
    """Spearman via Pearson sobre ranks. Retorna NaN se indefinido."""
    if len(x) < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def run_fold_b1(df: pd.DataFrame, fold: dict, b0_row: dict) -> dict:
    t = fold["temporada"]
    r = fold["rodada_alvo"]

    mask_t = df["temporada"] == t
    train = df[mask_t & (df["rodada"] < r) & df["elegivel"]]
    test_raw = df[mask_t & (df["rodada"] == r) & df["elegivel"]]
    test_metric = test_raw[~test_raw["flag_ambigua_B"]].copy()

    train_mean_global = float(train[TARGET_COL].mean())

    # media por jogador usando SOMENTE atleta_id + pontos_num do train
    player_means = train.groupby("atleta_id")[TARGET_COL].mean()

    preds_series = test_metric["atleta_id"].map(player_means)
    has_history = preds_series.notna()
    preds = preds_series.fillna(train_mean_global).to_numpy(dtype=float)

    y_true = test_metric[TARGET_COL].to_numpy(dtype=float)
    n = len(y_true)

    mae = float(np.mean(np.abs(y_true - preds))) if n else np.nan
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2))) if n else np.nan
    sp = spearman_safe(preds, y_true)

    n_com_hist = int(has_history.sum())
    n_fallback = int(n - n_com_hist)
    coverage = float(n_com_hist / n) if n else np.nan

    b0_mae = float(b0_row["mae"])
    delta = mae - b0_mae
    melhoria_abs = b0_mae - mae

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
        "mae": mae,
        "rmse": rmse,
        "spearman": sp,
        "b0_mae": b0_mae,
        "delta_mae": delta,
        "melhoria_absoluta": melhoria_abs,
        "train_max_round": int(train["rodada"].max()),
        "test_round": int(test_raw["rodada"].min()),
    }


def main():
    inicio = datetime.now()
    print("== ETAPA 5 — B1 ==")

    # imutabilidade antes
    hash_antes = sha256(PARQUET)
    n_antes = len(pd.read_parquet(PARQUET))
    hash_b0_antes = sha256(B0_CSV)
    hash_b0sum_antes = sha256(B0_SUM)
    print(f"Parquet SHA-256 (antes): {hash_antes[:16]}...")
    print(f"B0 folds SHA-256 (antes): {hash_b0_antes[:16]}...")

    # config
    cfg = load_config()
    mtr = cfg.get("validation", {}).get("min_train_rounds")
    if not isinstance(mtr, int) or mtr <= 0:
        print(f"ERRO: config.yaml sem min_train_rounds valido: {mtr!r}",
              file=sys.stderr)
        sys.exit(3)

    # dados
    df = load_canonical()
    df = derive_runtime_flags(df)
    print(f"Linhas: {len(df)}")

    # B0 como referencia
    b0 = pd.read_csv(B0_CSV)
    b0_idx = b0.set_index(["temporada", "rodada_alvo"])
    b0_global = float((b0["mae"] * b0["n_test_metric"]).sum()
                      / b0["n_test_metric"].sum())
    print(f"B0 MAE global ponderado: {b0_global:.4f}")

    # folds
    folds = build_folds(df, mtr)
    print(f"Folds: {len(folds)}")

    # consistencia com manifesto
    man = pd.read_csv(MANIFEST)
    a = set(map(tuple, man[["temporada","rodada_alvo"]].values.tolist()))
    b = set((f["temporada"], f["rodada_alvo"]) for f in folds)
    if a != b:
        print("ERRO: folds B1 != folds harness", file=sys.stderr)
        sys.exit(4)
    print("Consistencia com harness: OK")

    # executa
    resultados = []
    for f in folds:
        b0_row = b0_idx.loc[(f["temporada"], f["rodada_alvo"])]
        resultados.append(run_fold_b1(df, f, b0_row))

    dfr = pd.DataFrame(resultados)

    # auditorias internas
    assert len(dfr) == 76
    assert (dfr["train_max_round"] < dfr["rodada_alvo"]).all()
    assert (dfr["test_round"] == dfr["rodada_alvo"]).all()
    assert (dfr["n_test_com_historico"] + dfr["n_test_fallback_b0"]
            == dfr["n_test_metric"]).all()
    assert ((dfr["coverage_historico"] >= 0) & (dfr["coverage_historico"] <= 1)).all()
    assert (dfr["mae"] >= 0).all()
    assert (dfr["rmse"] >= 0).all()
    assert np.isfinite(dfr["train_mean_global"]).all()

    dfr.to_csv(CSV_FOLDS, index=False)

    # agregacao
    total_n = int(dfr["n_test_metric"].sum())
    mae_medio = float(dfr["mae"].mean())
    mae_pond = float((dfr["mae"] * dfr["n_test_metric"]).sum() / total_n)
    mse_pond = float((dfr["rmse"]**2 * dfr["n_test_metric"]).sum() / total_n)
    rmse_pond = float(np.sqrt(mse_pond))

    sp_val = dfr["spearman"].dropna()
    sp_medio = float(sp_val.mean()) if len(sp_val) else np.nan
    n_sp_val = int(len(sp_val))
    n_sp_nan = int(dfr["spearman"].isna().sum())

    n_com_hist_tot = int(dfr["n_test_com_historico"].sum())
    n_fallback_tot = int(dfr["n_test_fallback_b0"].sum())
    cov_global = n_com_hist_tot / total_n if total_n else np.nan
    cov_medio_fold = float(dfr["coverage_historico"].mean())

    melhor = dfr["delta_mae"] < 0
    pior = dfr["delta_mae"] > 0
    empate = dfr["delta_mae"] == 0
    n_melhor = int(melhor.sum())
    n_pior = int(pior.sum())
    n_empate = int(empate.sum())
    mediana_delta = float(dfr["delta_mae"].median())

    melhoria_abs = b0_global - mae_pond
    melhoria_pct = (b0_global - mae_pond) / b0_global * 100

    # imutabilidade depois
    hash_depois = sha256(PARQUET)
    n_depois = len(pd.read_parquet(PARQUET))
    hash_b0_depois = sha256(B0_CSV)
    hash_b0sum_depois = sha256(B0_SUM)
    imutavel = (hash_antes == hash_depois and n_antes == n_depois)
    b0_inalterado = (hash_b0_antes == hash_b0_depois
                     and hash_b0sum_antes == hash_b0sum_depois)

    # summary
    fim = datetime.now()
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# ETAPA 5 — B1 (media historica expansiva)\n\n")
        w(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
        w(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
        w(f"- Duracao: {(fim-inicio).total_seconds():.2f}s\n\n")

        w("## Definicao\n\n")
        w("`prediction_i = mean(pontos_num historico do jogador i)`\n")
        w("Historico: mesmo atleta_id, mesma temporada, rodada < alvo, elegivel.\n")
        w("Fallback: se jogador sem historico, usa `train_mean_global` do fold.\n")
        w("Nenhuma feature alem de (temporada, rodada, atleta_id, pontos_num).\n\n")

        w("## Folds\n\n")
        w(f"- Total: {len(dfr)}\n")
        for t in sorted(dfr["temporada"].unique()):
            w(f"- {int(t)}: {int((dfr['temporada']==t).sum())}\n")
        w("\n")

        w("## Metricas\n\n")
        w(f"- MAE medio simples: {mae_medio:.4f}\n")
        w(f"- **MAE global ponderado: {mae_pond:.4f}**  (principal oficial)\n")
        w(f"- RMSE agregado: {rmse_pond:.4f}\n")
        w(f"- Spearman medio (folds validos): {sp_medio:.4f}\n")
        w(f"- Folds com Spearman valido: {n_sp_val}\n")
        w(f"- Folds com Spearman indefinido: {n_sp_nan}\n")
        w(f"- Observacoes avaliadas (soma): {total_n}\n\n")

        w("## Coverage historico\n\n")
        w(f"- Soma com historico: {n_com_hist_tot}\n")
        w(f"- Soma fallback B0: {n_fallback_tot}\n")
        w(f"- Coverage global (ponderado): {cov_global:.4f}\n")
        w(f"- Coverage medio por fold: {cov_medio_fold:.4f}\n\n")

        w("## Comparacao com B0\n\n")
        w(f"- B0 MAE: {b0_global:.4f}\n")
        w(f"- B1 MAE: {mae_pond:.4f}\n")
        w(f"- Melhoria absoluta: {melhoria_abs:+.4f}\n")
        w(f"- Melhoria percentual: {melhoria_pct:+.2f}%\n")
        w(f"- Folds B1 < B0: {n_melhor}\n")
        w(f"- Folds B1 > B0: {n_pior}\n")
        w(f"- Empates: {n_empate}\n")
        w(f"- Mediana delta MAE (B1 - B0): {mediana_delta:+.4f}\n\n")

        w("## Notas metodologicas\n\n")
        w("- Historico intra-temporada; nao ha cruzamento entre temporadas.\n")
        w("- Janela expansiva; nao ha rolling window.\n")
        w("- 2025/R1 nao entra em treino nem teste; historico de 2025 comeca em R2.\n")
        w("- B (108 ambiguas) permanece no TREINO; removida somente das METRICAS.\n")
        w("- Fallback B0 aplicado a jogadores sem historico elegivel.\n")
        w("- Nenhuma informacao futura utilizada.\n\n")

        w("## Imutabilidade\n\n")
        w(f"- Parquet SHA-256 antes: {hash_antes}\n")
        w(f"- Parquet SHA-256 depois: {hash_depois}\n")
        w(f"- Parquet imutavel: {imutavel}\n")
        w(f"- B0 files inalterados: {b0_inalterado}\n")

    print()
    print("=== RESUMO B1 ===")
    print(f"folds: {len(dfr)}")
    print(f"MAE medio simples: {mae_medio:.4f}")
    print(f"MAE global ponderado: {mae_pond:.4f}")
    print(f"RMSE agregado: {rmse_pond:.4f}")
    print(f"Spearman medio: {sp_medio:.4f} (validos={n_sp_val}, indefinidos={n_sp_nan})")
    print(f"Coverage global: {cov_global:.4f}  fallback total: {n_fallback_tot}")
    print(f"B0 MAE: {b0_global:.4f}")
    print(f"B1 MAE: {mae_pond:.4f}")
    print(f"Melhoria absoluta: {melhoria_abs:+.4f}")
    print(f"Melhoria percentual: {melhoria_pct:+.2f}%")
    print(f"Folds B1<B0: {n_melhor}  B1>B0: {n_pior}  empates: {n_empate}")
    print(f"Mediana delta MAE: {mediana_delta:+.4f}")
    print(f"Parquet imutavel: {imutavel}")
    print(f"B0 inalterado: {b0_inalterado}")
    print(f"CSV: {CSV_FOLDS.relative_to(ROOT)}")
    print(f"SUMMARY: {SUMMARY.relative_to(ROOT)}")

    if not imutavel or not b0_inalterado:
        print("ERRO: algum arquivo foi alterado indevidamente.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
