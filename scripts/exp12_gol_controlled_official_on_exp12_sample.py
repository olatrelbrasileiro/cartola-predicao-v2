#!/usr/bin/env python3
"""
exp12_gol_controlled_official_on_exp12_sample.py

TESTE CONTROLADO 1 — Isolar efeito da amostra

Roda o CatBoost OFICIAL (features + hiperparâmetros do catboost_v1)
SOMENTE na amostra elegível da EXP-12 GOL (elegivel_modelo=True).

Compara diretamente com os resultados da EXP-12 (braço D) nas mesmas linhas/folds.

Uso:
    python scripts/exp12_gol_controlled_official_on_exp12_sample.py

Saídas:
    reports/exp12/exp12_gol_controlled_official_on_sample_summary.csv
    reports/exp12/exp12_gol_controlled_official_on_sample_by_fold.csv
    reports/exp12/exp12_gol_controlled_official_on_sample_report.md
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent
GOL_POOL = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"
PREPARED = ROOT / "data" / "interim" / "cache" / "catboost_v1_prepared.parquet"

REPORTS_DIR = ROOT / "reports" / "exp12"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = REPORTS_DIR / "exp12_gol_controlled_official_on_sample_summary.csv"
OUT_BY_FOLD = REPORTS_DIR / "exp12_gol_controlled_official_on_sample_by_fold.csv"
OUT_REPORT  = REPORTS_DIR / "exp12_gol_controlled_official_on_sample_report.md"

# ------------------------------------------------------------------ constantes oficiais
TARGET = "pontos_num"
K_SHRINK = 7
SEED = 42

FEATURES_OFFICIAL = [
    "hist_mean", "hist_std", "hist_n", "hist_shrunk",
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    "basic_mean", "basic_rec_5",
    "ga_per_game", "ga_rec_5", "ga_shrunk",
    "entrou_rate", "is_cold_start",
    "posicao_id",
    "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away",
]

CAT_FEATURES = ["posicao_id"]

CATBOOST_PARAMS = dict(
    loss_function="MAE",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    thread_count=-1,
    verbose=0,
    allow_writing_files=False,
)

MIN_TRAIN_ROWS = 50
MIN_TEST_ROWS = 3


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if np.isfinite(corr) else np.nan


def add_shrunk_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replica exatamente a lógica do catboost_v1.py:
    hist_shrunk e ga_shrunk são calculados com prior do conjunto de TREINO do fold.
    """
    train = train_df.copy()
    test = test_df.copy()

    # priors do treino
    prior_pos = train.groupby("posicao_id")[TARGET].mean()
    global_mean = float(train[TARGET].mean())

    last_obs = (
        train.sort_values("rodada")
        .groupby("atleta_id")
        .tail(1)[["atleta_id", "posicao_id", "ga_rate_last"]]
        .dropna(subset=["ga_rate_last"])
    )
    prior_ga = last_obs.groupby("posicao_id")["ga_rate_last"].mean()
    global_ga = float(last_obs["ga_rate_last"].mean()) if len(last_obs) else 0.0

    def _apply(df):
        df = df.copy()
        n = df["hist_n"].fillna(0).to_numpy(dtype=float)
        hm = df["hist_mean"].fillna(0).to_numpy(dtype=float)
        pp = df["posicao_id"].map(prior_pos).fillna(global_mean).to_numpy(dtype=float)
        df["hist_shrunk"] = np.where(n > 0, (n * hm + K_SHRINK * pp) / (n + K_SHRINK), np.nan)

        gr = df["ga_rate_last"].fillna(0).to_numpy(dtype=float)
        pg = df["posicao_id"].map(prior_ga).fillna(global_ga).to_numpy(dtype=float)
        df["ga_shrunk"] = np.where(n > 0, (n * gr + K_SHRINK * pg) / (n + K_SHRINK), np.nan)
        return df

    return _apply(train), _apply(test)


def main():
    print("=" * 70)
    print("TESTE CONTROLADO 1 — CatBoost Oficial na amostra EXP-12 GOL")
    print("=" * 70)

    # ------------------------------------------------------------------ 1. Carregar dados
    print("\n[1/5] Carregando datasets...")
    pool = pd.read_parquet(GOL_POOL)
    prep = pd.read_parquet(PREPARED)

    # Filtra apenas elegíveis da EXP-12
    eleg = pool[pool["elegivel_modelo"] == True].copy()
    print(f"     gol_pooling elegíveis: {len(eleg)}")

    # Chave de junção
    key = ["temporada", "atleta_id", "rodada"]

    # Colunas oficiais que vamos trazer do prepared
    official_cols = [
        "hist_mean", "hist_std", "hist_n",
        "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
        "basic_mean", "basic_rec_5",
        "ga_per_game", "ga_rec_5", "ga_rate_last",
        "entrou_rate", "is_cold_start",
        "posicao_id",
        "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away",
        "pontos_num", "entrou_em_campo",
    ]

    # Garante que todas existem
    missing = [c for c in official_cols if c not in prep.columns]
    if missing:
        raise RuntimeError(f"Colunas oficiais faltando no prepared: {missing}")

    prep_sub = prep[key + official_cols].copy()

    # Join
    df = eleg[key + ["elegivel_modelo"]].merge(
        prep_sub,
        on=key,
        how="inner",
        validate="one_to_one"
    )

    print(f"     Após join com prepared: {len(df)} linhas")
    print(f"     Cobertura do join: {len(df) / len(eleg):.1%}")

    if len(df) < 1000:
        raise RuntimeError("Join resultou em poucas linhas. Verifique chaves.")

    # ------------------------------------------------------------------ 2. Definir folds (mesmo protocolo EXP-12)
    print("\n[2/5] Gerando folds walk-forward expansivo...")

    folds = []
    for t in sorted(df["temporada"].unique()):
        rodadas = sorted(df.loc[df["temporada"] == t, "rodada"].unique())
        for R in rodadas:
            # treino: todas as temporadas anteriores + mesma temporada com rodada < R
            n_train = len(df[
                ((df["temporada"] < t) | ((df["temporada"] == t) & (df["rodada"] < R)))
            ])
            n_test = len(df[(df["temporada"] == t) & (df["rodada"] == R)])

            if n_train >= MIN_TRAIN_ROWS and n_test >= MIN_TEST_ROWS:
                folds.append((int(t), int(R)))

    print(f"     Folds válidos: {len(folds)}")

    # ------------------------------------------------------------------ 3. Walk-forward
    print("\n[3/5] Executando walk-forward com features oficiais...")

    results = []
    all_preds = []

    for i, (t, R) in enumerate(folds, 1):
        train_mask = (df["temporada"] < t) | ((df["temporada"] == t) & (df["rodada"] < R))
        test_mask = (df["temporada"] == t) & (df["rodada"] == R)

        train_raw = df[train_mask].copy()
        test_raw = df[test_mask].copy()

        # Calcula hist_shrunk e ga_shrunk com prior do treino
        train, test = add_shrunk_features(train_raw, test_raw)

        # Features disponíveis
        feat_cols = [c for c in FEATURES_OFFICIAL if c in train.columns]
        missing_feats = set(FEATURES_OFFICIAL) - set(feat_cols)
        if missing_feats:
            print(f"     [AVISO] Fold {t}/R{R}: features faltando: {missing_feats}")

        X_train = train[feat_cols]
        y_train = train[TARGET].to_numpy(dtype=float)
        X_test = test[feat_cols]
        y_test = test[TARGET].to_numpy(dtype=float)

        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, cat_features=[c for c in CAT_FEATURES if c in feat_cols])

        pred = model.predict(X_test)

        fold_mae = mae(y_test, pred)
        fold_rmse = rmse(y_test, pred)
        fold_sp = spearman(y_test, pred)

        results.append({
            "temporada": t,
            "rodada_alvo": R,
            "n_train": len(train),
            "n_test": len(test),
            "mae": fold_mae,
            "rmse": fold_rmse,
            "spearman": fold_sp,
        })

        # Guarda previsões para métricas pooled
        tmp = test[key + [TARGET]].copy()
        tmp["y_pred"] = pred
        all_preds.append(tmp)

        if i % 10 == 0 or i == len(folds):
            print(f"     [{i:3d}/{len(folds)}] {t}/R{R:2d}  n_test={len(test):3d}  MAE={fold_mae:.4f}  SP={fold_sp:.4f}")

    # ------------------------------------------------------------------ 4. Agregações
    print("\n[4/5] Agregando resultados...")

    res_df = pd.DataFrame(results)
    preds_df = pd.concat(all_preds, ignore_index=True)

    # Métricas pooled (todas as previsões juntas)
    y_true = preds_df[TARGET].to_numpy()
    y_pred = preds_df["y_pred"].to_numpy()

    pooled_mae = mae(y_true, y_pred)
    pooled_rmse = rmse(y_true, y_pred)
    pooled_sp = spearman(y_true, y_pred)

    summary = pd.DataFrame([{
        "arm": "Official_on_EXP12_sample",
        "n_folds": len(res_df),
        "n_pooled": len(preds_df),
        "MAE_mean_folds": res_df["mae"].mean(),
        "RMSE_mean_folds": res_df["rmse"].mean(),
        "Spearman_mean_folds": res_df["spearman"].mean(),
        "MAE_pooled": pooled_mae,
        "RMSE_pooled": pooled_rmse,
        "Spearman_pooled": pooled_sp,
    }])

    # ------------------------------------------------------------------ 5. Salvar
    print("\n[5/5] Salvando resultados...")

    res_df.to_csv(OUT_BY_FOLD, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    report = f"""# Teste Controlado 1 — CatBoost Oficial na amostra EXP-12 GOL

- Gerado em: {datetime.now().isoformat(timespec="seconds")}
- Amostra: apenas `elegivel_modelo=True` do gol_pooling (participantes)
- Features: exatamente as do catboost_v1 (FEATURES_COM)
- Hiperparâmetros: loss=MAE, depth=6, lr=0.05, iters=500, l2=3, seed=42
- Protocolo: walk-forward expansivo (mesmo da EXP-12)

## Resultado Pooled

| Métrica              | Valor    |
|----------------------|----------|
| n_folds              | {len(res_df)} |
| n_pooled             | {len(preds_df)} |
| MAE pooled           | {pooled_mae:.4f} |
| RMSE pooled          | {pooled_rmse:.4f} |
| Spearman pooled      | {pooled_sp:.4f} |

## Comparação rápida com EXP-12 (braço D)

| Pipeline                        | MAE pooled | Spearman pooled |
|---------------------------------|------------|-----------------|
| EXP-12 braço D (sua última run) | 3.3378     | 0.0258          |
| **Official features nesta amostra** | **{pooled_mae:.4f}** | **{pooled_sp:.4f}** |

## Interpretação

- Se o Official nesta amostra ficar **muito melhor** que o braço D → o problema principal são as **features** da EXP-12.
- Se ficar **parecido** com o braço D → o problema principal é a **dificuldade intrínseca** de prever apenas participantes (e o MAE baixo do oficial original vinha dos zeros).
"""

    OUT_REPORT.write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    print("RESULTADO FINAL (pooled)")
    print(f"  MAE      = {pooled_mae:.4f}")
    print(f"  RMSE     = {pooled_rmse:.4f}")
    print(f"  Spearman = {pooled_sp:.4f}")
    print("=" * 70)
    print(f"\nArquivos salvos em:\n  {OUT_SUMMARY}\n  {OUT_BY_FOLD}\n  {OUT_REPORT}")


if __name__ == "__main__":
    main()
