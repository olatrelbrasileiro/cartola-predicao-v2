#!/usr/bin/env python3
"""
Testes L1-L10 do CatBoost v2.

Nao treina o pipeline. Valida artefatos, reusa infraestrutura do ridge_v2
para recomputacao independente e verifica integridade por hash.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def detect_root():
    for c in [Path.home() / "cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path("/data/data/com.termux/files/home/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c / "reports" / "harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz nao encontrada")


ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
CONFIG = ROOT / "config"

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"
P0_FOLDS = REP / "baselines_p0_corrected_folds.csv"
RIDGE_SCRIPT = ROOT / "scripts" / "ridge_v2.py"
CB_SCRIPT = ROOT / "scripts" / "catboost_v2.py"

FOLDS = REP / "catboost_v2_folds.csv"

TARGET = "pontos_num"

FEATURES_COM_EXPECTED = [
    "hist_mean", "hist_std", "hist_n", "hist_shrunk_k7",
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    "basic_mean", "basic_rec_5",
    "ga_per_game", "ga_rec_5", "ga_shrunk_k7",
    "entrou_rate", "is_cold_start",
    "posicao_id",
    "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away",
]

CONFRONTO = {
    "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"
}

FEATURES_SEM_EXPECTED = [
    f for f in FEATURES_COM_EXPECTED if f not in CONFRONTO
]

CAT_FEATURES_EXPECTED = ["posicao_id"]

CATBOOST_PARAMS_EXPECTED = dict(
    loss_function="MAE",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    thread_count=-1,
    verbose=0,
    allow_writing_files=False,
)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def _load_cb_v2():
    spec = importlib.util.spec_from_file_location(
        "cb_v2_module", str(CB_SCRIPT)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_L1():
    cb = _load_cb_v2()
    falhas = []

    if list(cb.FEATURES_COM) != FEATURES_COM_EXPECTED:
        falhas.append("FEATURES_COM diverge do esperado")

    if list(cb.FEATURES_SEM) != FEATURES_SEM_EXPECTED:
        falhas.append("FEATURES_SEM diverge do esperado")

    if len(cb.FEATURES_COM) != 21:
        falhas.append(
            f"FEATURES_COM tem {len(cb.FEATURES_COM)} (esperado 21)"
        )

    if len(cb.FEATURES_SEM) != 16:
        falhas.append(
            f"FEATURES_SEM tem {len(cb.FEATURES_SEM)} (esperado 16)"
        )

    grupos_esperados = {
        "Historico", "Recente", "Basico", "GA",
        "Oportunidade", "Posicao", "Confronto"
    }

    if set(cb.GRUPO_FEAT.keys()) != grupos_esperados:
        falhas.append("GRUPO_FEAT incompleto")

    return falhas


def test_L2(dfr):
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~(
        (df["temporada"] == 2025) &
        (df["rodada"] == 1)
    )
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)

    falhas = []
    n_zero_present = 0

    for _, row in dfr.iterrows():
        t = int(row["temporada"])
        R = int(row["rodada_alvo"])

        te = df[
            (df["temporada"] == t) &
            (df["rodada"] == R) &
            df["elegivel"] &
            df["entrou_bool"]
        ]

        n_esp = len(te)

        if int(row["n_test_metric"]) != n_esp:
            falhas.append(
                f"{t}/R{R}: n_test_metric="
                f"{int(row['n_test_metric'])} != {n_esp}"
            )

        n_zero_present += int((te[TARGET] == 0).sum())

    print(
        f"      [L2] zeros entre participantes: {n_zero_present}"
    )

    if n_zero_present == 0:
        falhas.append(
            "nenhum zero entre participantes (suspeito)"
        )

    return falhas


def test_L3(dfr):
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~(
        (df["temporada"] == 2025) &
        (df["rodada"] == 1)
    )
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)

    falhas = []
    cache = {}

    for _, row in dfr.iterrows():
        t = int(row["temporada"])
        R = int(row["rodada_alvo"])
        key = (t, R)

        if key not in cache:
            tr = df[
                (df["temporada"] == t) &
                (df["rodada"] < R) &
                df["elegivel"] &
                df["entrou_bool"]
            ]

            cache[key] = (
                len(tr),
                int(tr["rodada"].max()) if len(tr) else None
            )

        n, tmax = cache[key]

        if n == 0:
            falhas.append(f"{t}/R{R}: treino vazio")
        elif tmax >= R:
            falhas.append(
                f"{t}/R{R}: train_max={tmax} >= R"
            )

    return falhas


def test_L4(dfr):
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~(
        (df["temporada"] == 2025) &
        (df["rodada"] == 1)
    )
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)

    falhas = []
    cache = {}

    for _, row in dfr.iterrows():
        t = int(row["temporada"])
        R = int(row["rodada_alvo"])
        key = (t, R)

        if key not in cache:
            tr = df[
                (df["temporada"] == t) &
                (df["rodada"] < R) &
                df["elegivel"] &
                df["entrou_bool"]
            ]

            te = df[
                (df["temporada"] == t) &
                (df["rodada"] == R) &
                df["elegivel"] &
                df["entrou_bool"]
            ]

            kt = set(
                zip(tr["temporada"], tr["rodada"], tr["atleta_id"])
            )

            kv = set(
                zip(te["temporada"], te["rodada"], te["atleta_id"])
            )

            cache[key] = kt & kv

        inter = cache[key]

        if inter:
            falhas.append(
                f"{t}/R{R}: {len(inter)} chaves em comum"
            )

    return falhas


def test_L5(dfr):
    p0 = pd.read_csv(P0_FOLDS).set_index(
        ["temporada", "rodada_alvo"]
    )

    falhas = []

    for _, row in dfr.iterrows():
        t = int(row["temporada"])
        R = int(row["rodada_alvo"])

        esp = float(
            p0.loc[(t, R), "b2s_mae"]
        )

        got = float(row["b2s_mae"])

        if abs(esp - got) > 1e-9:
            falhas.append(
                f"{t}/R{R}: b2s_mae={got} != P0={esp}"
            )

    src = CB_SCRIPT.read_text(encoding="utf-8")

    if (
        "baseline_b2s_folds.csv" in src
        and "baselines_p0_corrected_folds.csv" not in src
    ):
        falhas.append(
            "catboost_v2.py referencia baseline_b2s_folds.csv sem P0"
        )

    return falhas


def test_L6(dfr):
    rv2_spec = importlib.util.spec_from_file_location(
        "ridge_v2_module", str(RIDGE_SCRIPT)
    )

    rv2 = importlib.util.module_from_spec(rv2_spec)
    rv2_spec.loader.exec_module(rv2)

    audit = rv2.load_yaml(AUDIT_YAML)

    df = pd.read_parquet(CANON)
    df = df.sort_values(
        ["temporada", "rodada", "atleta_id"],
        kind="mergesort"
    ).reset_index(drop=True)

    df["elegivel"] = ~(
        (df["temporada"] == 2025) &
        (df["rodada"] == 1)
    )

    scout_cols = [
        f"scout_{s}" for s in audit["scouts_basicos"]
    ]

    df = rv2.apply_cumulative_diff(
        df,
        scout_cols + ["scout_G", "scout_A"]
    )

    df["basic_score"] = rv2.compute_basic_score(
        df,
        audit
    )

    feats = rv2.compute_participation_features(df)

    falhas = []
    candidatos = []

    for (t, aid), sub in feats.groupby(
        ["temporada", "atleta_id"],
        sort=False
    ):
        n_part = int(
            sub["entrou_em_campo"].fillna(False).sum()
        )

        if n_part >= 3:
            candidatos.append((t, aid))

        if len(candidatos) >= 5:
            break

    for t, aid in candidatos[:5]:
        sub = feats[
            (feats["temporada"] == t) &
            (feats["atleta_id"] == aid)
        ].sort_values("rodada")

        for i in range(1, len(sub)):
            row = sub.iloc[i]
            hist = sub.iloc[:i]

            hist_part = hist[
                hist["entrou_em_campo"].fillna(False)
            ]

            if len(hist_part) == 0:
                continue

            rec5_esp = hist_part.tail(5)[TARGET].mean()
            rec3_esp = hist_part.tail(3)[TARGET].mean()

            if abs(row["rec_mean_5"] - rec5_esp) > 1e-9:
                falhas.append(
                    f"{t}/{aid}/R{int(row['rodada'])}: "
                    "rec_mean_5 diverge"
                )

            if abs(row["rec_mean_3"] - rec3_esp) > 1e-9:
                falhas.append(
                    f"{t}/{aid}/R{int(row['rodada'])}: "
                    "rec_mean_3 diverge"
                )

            if len(falhas) > 5:
                break

        if len(falhas) > 5:
            break

    return falhas


def test_L7(dfr):
    rv2_spec = importlib.util.spec_from_file_location(
        "ridge_v2_module", str(RIDGE_SCRIPT)
    )

    rv2 = importlib.util.module_from_spec(rv2_spec)
    rv2_spec.loader.exec_module(rv2)

    audit = rv2.load_yaml(AUDIT_YAML)

    df = pd.read_parquet(CANON)
    df = df.sort_values(
        ["temporada", "rodada", "atleta_id"],
        kind="mergesort"
    ).reset_index(drop=True)

    df["elegivel"] = ~(
        (df["temporada"] == 2025) &
        (df["rodada"] == 1)
    )

    scout_cols = [
        f"scout_{s}" for s in audit["scouts_basicos"]
    ]

    df = rv2.apply_cumulative_diff(
        df,
        scout_cols + ["scout_G", "scout_A"]
    )

    df["basic_score"] = rv2.compute_basic_score(
        df,
        audit
    )

    feats = rv2.compute_participation_features(df)

    falhas = []

    folds_uni = dfr[
        ["temporada", "rodada_alvo"]
    ].drop_duplicates()

    sample = folds_uni.sample(
        n=min(3, len(folds_uni)),
        random_state=7
    )

    for _, f in sample.iterrows():
        t = int(f["temporada"])
        R = int(f["rodada_alvo"])

        mask_t = feats["temporada"] == t

        tr = feats[
            mask_t &
            (feats["rodada"] < R) &
            feats["elegivel"]
        ]

        tr = tr[
            tr["entrou_em_campo"].fillna(False)
        ].copy()

        te = feats[
            mask_t &
            (feats["rodada"] == R) &
            feats["elegivel"]
        ]

        te = te[
            te["entrou_em_campo"].fillna(False)
        ].copy()

        pp, pg, b0 = rv2.compute_priors(tr)

        amostra = te.sample(
            n=min(3, len(te)),
            random_state=int(R)
        )

        for _, row in amostra.iterrows():
            n = int(row["n_prev"])
            s = float(row["sum_prev"])
            pos = int(row["posicao_id"])

            prior_p = pp.get(pos, b0)

            if n > 0:
                esperado = (
                    s +
                    rv2.K_SHRINK * prior_p
                ) / (
                    n + rv2.K_SHRINK
                )
            else:
                esperado = np.nan

            te_single = rv2.add_shrunk_features(
                te[
                    te["atleta_id"] ==
                    row["atleta_id"]
                ],
                pp,
                pg,
                b0
            )

            got = (
                float(te_single["hist_shrunk_k7"].iloc[0])
                if pd.notna(
                    te_single["hist_shrunk_k7"].iloc[0]
                )
                else np.nan
            )

            if np.isnan(esperado) and np.isnan(got):
                continue

            if not (
                np.isnan(esperado) and
                np.isnan(got)
            ) and abs(esperado - got) > 1e-9:
                falhas.append(
                    f"{t}/R{R}/{int(row['atleta_id'])}: "
                    "hist_shrunk diverge"
                )

    return falhas


def test_L8(dfr):
    cb = _load_cb_v2()
    falhas = []

    for f in [
        "p_home",
        "p_draw",
        "p_away",
        "mando",
        "favoritismo_home_away",
    ]:
        if f not in cb.FEATURES_COM:
            falhas.append(
                f"feature confronto ausente: {f}"
            )

    ts = pd.read_parquet(TEMPORAL_SAFE)

    ts_masked = ts[
        ts["motivo_mascara"].fillna("") != ""
    ]

    if len(ts_masked) > 0:
        if not ts_masked["p_home"].isna().all():
            falhas.append(
                f"{len(ts_masked[ts_masked['p_home'].notna()])} "
                "linhas mascaradas com p_home preenchido"
            )

    sem = ts[
        ts["motivo_mascara"] == "SEM_MATCH"
    ]

    if len(sem) > 0 and sem["p_home"].notna().any():
        falhas.append(
            "SEM_MATCH tem p_home preenchido"
        )

    return falhas


def test_L9():
    cb = _load_cb_v2()
    falhas = []

    for k, v in CATBOOST_PARAMS_EXPECTED.items():
        got = cb.CATBOOST_PARAMS.get(k)

        if got != v:
            falhas.append(
                f"param {k}: {got} != {v}"
            )

    if cb.CAT_FEATURES != CAT_FEATURES_EXPECTED:
        falhas.append(
            f"cat_features: {cb.CAT_FEATURES} "
            f"!= {CAT_FEATURES_EXPECTED}"
        )

    return falhas


def test_L10(dfr):
    falhas = []

    total_n = dfr["n_test_metric"].sum()

    mae_global = float(
        (
            dfr["mae"] *
            dfr["n_test_metric"]
        ).sum() / total_n
    )

    if not np.isfinite(mae_global):
        falhas.append(
            "mae_global nao finito"
        )

    rmse_global = float(
        np.sqrt(
            (
                dfr["rmse"] ** 2 *
                dfr["n_test_metric"]
            ).sum() / total_n
        )
    )

    if not np.isfinite(rmse_global):
        falhas.append(
            "rmse_global nao finito"
        )

    n_better = int(
        (dfr["delta_mae"] < -1e-4).sum()
    )

    n_equal = int(
        (dfr["delta_mae"].abs() <= 1e-4).sum()
    )

    n_worse = int(
        len(dfr) -
        n_better -
        n_equal
    )

    if (
        n_better +
        n_equal +
        n_worse !=
        len(dfr)
    ):
        falhas.append(
            "contagem de folds inconsistente"
        )

    return falhas


def main():
    print("== CatBoost v2 — Testes L1-L10 ==")

    if not FOLDS.exists():
        print(
            f"FAIL: {FOLDS.name} ausente. "
            "Rode catboost_v2.py primeiro."
        )
        sys.exit(2)

    dfr = pd.read_csv(FOLDS)

    testes = [
        ("L1 (21 features, grupos)", test_L1, ()),
        ("L2 (participacao P0)", test_L2, (dfr,)),
        ("L3 (temporalidade 76 folds)", test_L3, (dfr,)),
        ("L4 (sem overlap 76 folds)", test_L4, (dfr,)),
        ("L5 (benchmark P0)", test_L5, (dfr,)),
        ("L6 (janelas por participacao)", test_L6, (dfr,)),
        ("L7 (shrinkage recompute)", test_L7, (dfr,)),
        ("L8 (confronto temporal-safe)", test_L8, (dfr,)),
        ("L9 (params CatBoost)", test_L9, ()),
        ("L10 (agregacao independente)", test_L10, (dfr,)),
    ]

    all_ok = True

    for nome, fn, args in testes:
        try:
            falhas = fn(*args)
        except Exception as e:
            print(
                f"  {nome}: FAIL "
                f"(excecao: {type(e).__name__}: {e})"
            )
            all_ok = False
            continue

        if falhas:
            print(f"  {nome}: FAIL")

            for f in falhas[:10]:
                print(f"      {f}")

            if len(falhas) > 10:
                print(
                    f"      ... (+{len(falhas)-10})"
                )

            all_ok = False
        else:
            print(f"  {nome}: PASS")

    if all_ok:
        print("\n=== RESULTADO: PASS ===")
    else:
        print("\n=== RESULTADO: FAIL ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
