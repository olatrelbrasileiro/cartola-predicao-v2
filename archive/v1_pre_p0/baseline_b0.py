#!/usr/bin/env python3
"""
ETAPA 5 — Baseline B0 (previsao constante pela media do treino).

Definicao:
  Para cada fold (temporada, rodada_alvo):
    train = linhas elegiveis com rodada < rodada_alvo, mesma temporada
    y_pred = mean(pontos_num) do train
    test_metric = linhas elegiveis com rodada == rodada_alvo, mesma temporada,
                  sem flag_ambigua_B=True
    MAE / RMSE sobre test_metric.
    Spearman = indefinido (predicao constante).

Nao usa features. Nao usa dados do teste. Nao usa media global.
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
POLICY = ROOT / "config" / "elegibilidade_v2.yaml"
MANIFEST = ROOT / "reports" / "harness_folds_v2.csv"
REP = ROOT / "reports"

CSV_FOLDS = REP / "baseline_b0_folds.csv"
SUMMARY = REP / "baseline_b0_summary.md"

TARGET_COL = "pontos_num"
KEY_COLS = ["temporada", "rodada", "atleta_id"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
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
        rodadas = sorted(sub["rodada"].unique())
        for r in rodadas:
            if r <= min_train_rounds:
                continue
            folds.append({"temporada": int(temporada), "rodada_alvo": int(r)})
    return folds


def run_fold_b0(df: pd.DataFrame, fold: dict) -> dict:
    t = fold["temporada"]
    r = fold["rodada_alvo"]

    mask_t = df["temporada"] == t
    train = df[mask_t & (df["rodada"] < r) & df["elegivel"]]
    test_raw = df[mask_t & (df["rodada"] == r) & df["elegivel"]]
    test_metric = test_raw[~test_raw["flag_ambigua_B"]].copy()

    # previsao: media do treino (usa o alvo bruto, incluindo B ambigua no treino
    # conforme decisao oficial da Etapa 3)
    train_mean = float(train[TARGET_COL].mean())

    y_true = test_metric[TARGET_COL].to_numpy(dtype=float)
    n = len(y_true)

    if n == 0:
        mae = np.nan
        rmse = np.nan
    else:
        y_pred = np.full(n, train_mean, dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "temporada": int(t),
        "rodada_alvo": int(r),
        "n_train": int(len(train)),
        "n_test_raw": int(len(test_raw)),
        "n_test_metric": int(n),
        "n_test_ambiguous_B": int(len(test_raw) - n),
        "train_mean": train_mean,
        "mae": mae,
        "rmse": rmse,
        "spearman": np.nan,   # indefinido por construcao
        "train_min_round": int(train["rodada"].min()) if len(train) else None,
        "train_max_round": int(train["rodada"].max()) if len(train) else None,
        "test_round": int(test_raw["rodada"].min()) if len(test_raw) else None,
    }


def main():
    inicio = datetime.now()
    print("== ETAPA 5 — B0 BASELINE ==")

    # imutabilidade antes
    hash_antes = sha256(PARQUET)
    n_linhas_antes = len(pd.read_parquet(PARQUET))
    print(f"Parquet SHA-256 (antes): {hash_antes[:16]}...")

    # config
    cfg = load_config()
    mtr = cfg.get("validation", {}).get("min_train_rounds")
    if not isinstance(mtr, int) or mtr <= 0:
        print(f"ERRO: config.yaml sem validation.min_train_rounds valido: {mtr!r}",
              file=sys.stderr)
        sys.exit(3)

    # dados
    df = load_canonical()
    df = derive_runtime_flags(df)
    print(f"Linhas: {len(df)}")
    print(f"min_train_rounds (config.yaml): {mtr}")

    # folds (mesma regra do harness)
    folds = build_folds(df, mtr)
    print(f"Folds construidos: {len(folds)}")

    # consistencia com o manifesto do harness (se existir)
    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST)
        if len(man) != len(folds):
            print(f"ERRO: manifesto tem {len(man)} folds, B0 construiu {len(folds)}",
                  file=sys.stderr)
            sys.exit(4)
        # confere pares (temporada, rodada_alvo)
        a = set(map(tuple, man[["temporada","rodada_alvo"]].values.tolist()))
        b = set((f["temporada"], f["rodada_alvo"]) for f in folds)
        if a != b:
            print("ERRO: conjunto de folds do B0 difere do manifesto.",
                  file=sys.stderr)
            sys.exit(4)
        print("Consistencia com manifesto: OK")

    # executa
    resultados = [run_fold_b0(df, f) for f in folds]
    dfr = pd.DataFrame(resultados)

    # auditorias embutidas
    assert len(dfr) == 76, f"folds={len(dfr)}"
    assert (dfr["train_max_round"] < dfr["rodada_alvo"]).all(), "leakage temporal"
    assert (dfr["test_round"] == dfr["rodada_alvo"]).all(), "teste fora do alvo"
    assert (dfr["n_test_metric"] + dfr["n_test_ambiguous_B"] == dfr["n_test_raw"]).all()
    assert (dfr["mae"] >= 0).all() or dfr["mae"].isna().any()
    assert (dfr["rmse"] >= 0).all() or dfr["rmse"].isna().any()
    assert np.isfinite(dfr["train_mean"]).all(), "train_mean nao finito"

    dfr.to_csv(CSV_FOLDS, index=False)

    # metricas agregadas
    total_n = int(dfr["n_test_metric"].sum())
    mae_medio = float(dfr["mae"].mean())
    mae_pond = float((dfr["mae"] * dfr["n_test_metric"]).sum() / total_n)
    dfr["_mse"] = dfr["rmse"] ** 2
    rmse_pond = float(np.sqrt((dfr["_mse"] * dfr["n_test_metric"]).sum() / total_n))
    dfr = dfr.drop(columns=["_mse"])

    tm = dfr["train_mean"]

    # imutabilidade depois
    hash_depois = sha256(PARQUET)
    n_linhas_depois = len(pd.read_parquet(PARQUET))
    imutavel = (hash_antes == hash_depois and n_linhas_antes == n_linhas_depois)

    # relatorio
    fim = datetime.now()
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# Baseline B0 — resumo\n\n")
        w(f"- **Inicio:** {inicio.isoformat(timespec='seconds')}\n")
        w(f"- **Fim:** {fim.isoformat(timespec='seconds')}\n")
        w(f"- **Duracao:** {(fim-inicio).total_seconds():.2f}s\n\n")

        w("## Definicao\n\n")
        w("`y_pred = mean(pontos_num)` do treino elegivel do fold. ")
        w("Aplicada como constante a todos os jogadores do teste.\n")
        w("Nenhuma feature usada. Nenhuma media global. Nenhum dado do teste.\n\n")

        w("## Folds\n\n")
        w(f"- Total: {len(dfr)}\n")
        for t in sorted(dfr["temporada"].unique()):
            w(f"- {int(t)}: {int((dfr['temporada']==t).sum())}\n")
        w("\n")

        w("## Metricas\n\n")
        w(f"- **MAE medio simples (por fold):** {mae_medio:.4f}\n")
        w(f"- **MAE global ponderado (por n):** {mae_pond:.4f}  ← **principal oficial**\n")
        w(f"- **RMSE agregado (por n):** {rmse_pond:.4f}\n")
        w(f"- **Spearman:** INDEFINIDO (previsao constante dentro de cada fold; "
          "nao ha variacao entre jogadores para gerar ranking)\n")
        w(f"- Observacoes avaliadas (soma): {total_n}\n\n")

        w("## Estatisticas de `train_mean`\n\n")
        w(f"- min: {tm.min():.4f}\n")
        w(f"- max: {tm.max():.4f}\n")
        w(f"- mean: {tm.mean():.4f}\n")
        w(f"- median: {tm.median():.4f}\n")
        w(f"- std: {tm.std():.4f}\n\n")

        w("## Decisao B (108 linhas ambiguas)\n\n")
        w("- Permanecem no TREINO (politica atual)\n")
        w("- Removidas SOMENTE da metrica principal do TESTE\n")
        w(f"- Removidas no total dos testes: {int(dfr['n_test_ambiguous_B'].sum())}\n\n")

        w("## 2025/R1\n\n")
        w("- Excluida globalmente pelo filtro `elegivel`\n")
        w("- Nenhum fold usa R1 de 2025 em treino nem teste\n\n")

        w("## Imutabilidade do Parquet\n\n")
        w(f"- SHA-256 antes: `{hash_antes}`\n")
        w(f"- SHA-256 depois: `{hash_depois}`\n")
        w(f"- Linhas antes/depois: {n_linhas_antes}/{n_linhas_depois}\n")
        w(f"- **Imutavel: {imutavel}**\n")

    print()
    print("=== RESUMO B0 ===")
    print(f"folds: {len(dfr)}")
    print(f"MAE medio simples: {mae_medio:.4f}")
    print(f"MAE global ponderado: {mae_pond:.4f}")
    print(f"RMSE agregado: {rmse_pond:.4f}")
    print(f"train_mean: min={tm.min():.4f} max={tm.max():.4f} mean={tm.mean():.4f}")
    print(f"Parquet imutavel: {imutavel}")
    print(f"CSV: {CSV_FOLDS.relative_to(ROOT)}")
    print(f"SUMMARY: {SUMMARY.relative_to(ROOT)}")

    if not imutavel:
        print("ERRO: parquet alterado.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
