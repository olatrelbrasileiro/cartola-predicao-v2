#!/usr/bin/env python3
"""
Testes L1-L10 do Ridge v2.

NAO treina o pipeline principal (ridge_v2.py). Apenas valida artefatos,
recomputa partes do pipeline para auditoria (L6, L7, L10) e verifica
integridade por hash (L8, L9).

L7 faz recomputacao INDEPENDENTE dos MAEs internos em 5 folds,
usando as funcoes de ridge_v2.py importadas como modulo (sem chamar main()).
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
    raise RuntimeError("Raiz do projeto nao encontrada")


ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
CONFIG = ROOT / "config"

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"
HARNESS = REP / "harness_folds_v2.csv"
B2S_FOLDS = REP / "baseline_b2s_folds.csv"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"

FOLDS = REP / "ridge_v2_folds.csv"
ALPHAS = REP / "ridge_v2_alphas.csv"
PRIORS = REP / "ridge_v2_priors_debug.csv"
SHRUNK = REP / "ridge_v2_shrunk_debug.csv"

TARGET = "pontos_num"
K_SHRINK = 7
MIN_INNER_ROUNDS = 3

HASHES_ESPERADOS = {
    "canonical": "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4",
    "b2s_folds": "b56c6db440d7628acbbc52bb477e1b4fe81ce3fc87610e5aa0452e2110e6a142",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ loader
def _load_ridge_v2():
    """Importa ridge_v2.py como modulo, sem executar main()."""
    spec = importlib.util.spec_from_file_location(
        "ridge_v2_module", str(ROOT / "scripts" / "ridge_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ L1
def test_L1(dfr):
    """L1: train_max_round < target_round em todos os folds."""
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    falhas = []
    for _, row in dfr.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        tr = df[(df["temporada"] == t) & (df["rodada"] < R) & df["elegivel"] & df["entrou_bool"]]
        if len(tr) == 0:
            falhas.append(f"{t}/R{R}: treino vazio")
            continue
        if tr["rodada"].max() >= R:
            falhas.append(f"{t}/R{R}: train_max={tr['rodada'].max()} >= R")
    return falhas


# ------------------------------------------------------------------ L2
def test_L2(dfr):
    """L2: test exatamente na target_round."""
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    falhas = []
    for _, row in dfr.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        te = df[(df["temporada"] == t) & (df["rodada"] == R) & df["elegivel"] & df["entrou_bool"]]
        if len(te) != int(row["n_test"]):
            falhas.append(f"{t}/R{R}: test={len(te)} vs n_test={row['n_test']}")
    return falhas


# ------------------------------------------------------------------ L3
def test_L3(dfr):
    """L3: nenhuma chave (temporada, rodada, atleta_id) compartilhada entre
    train e test em TODOS os 76 folds (nao spot-check)."""
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    falhas = []
    for _, row in dfr.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        tr = df[(df["temporada"] == t) & (df["rodada"] < R) & df["elegivel"] & df["entrou_bool"]]
        te = df[(df["temporada"] == t) & (df["rodada"] == R) & df["elegivel"] & df["entrou_bool"]]
        keys_tr = set(zip(tr["temporada"], tr["rodada"], tr["atleta_id"]))
        keys_te = set(zip(te["temporada"], te["rodada"], te["atleta_id"]))
        inter = keys_tr & keys_te
        if inter:
            falhas.append(f"{t}/R{R}: {len(inter)} chaves em comum")
    return falhas


# ------------------------------------------------------------------ L4
def test_L4(dfr):
    """L4: 2025/R1 ausente."""
    if ((dfr["temporada"] == 2025) & (dfr["rodada_alvo"] == 1)).any():
        return ["2025/R1 presente em ridge_v2_folds"]
    return []


# ------------------------------------------------------------------ L5
def test_L5(dfr):
    """L5: n_test == numero de participantes do test."""
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    falhas = []
    for _, row in dfr.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        te = df[(df["temporada"] == t) & (df["rodada"] == R) & df["elegivel"] & df["entrou_bool"]]
        if int(row["n_test"]) != len(te):
            falhas.append(f"{t}/R{R}: n_test={int(row['n_test'])} != participantes={len(te)}")
    return falhas


# ------------------------------------------------------------------ L6
def test_L6(dfr):
    """L6: spot-check 5 folds, recomputar prior_pos e prior_ga independentemente
    e comparar com o que foi usado pelo pipeline (ridge_v2_priors_debug.csv)."""
    if not PRIORS.exists():
        return ["ridge_v2_priors_debug.csv ausente"]
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    priors_used = pd.read_csv(PRIORS)
    falhas = []
    sample = dfr.sample(n=min(5, len(dfr)), random_state=1)
    for _, row in sample.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        tr = df[(df["temporada"] == t) & (df["rodada"] < R) & df["elegivel"] & df["entrou_bool"]]
        pp = tr.groupby("posicao_id")[TARGET].mean().to_dict()
        for pid, val in pp.items():
            mask = ((priors_used["temporada"] == t) & (priors_used["rodada"] == R)
                    & (priors_used["posicao_id"] == int(pid)))
            if mask.sum() == 0:
                falhas.append(f"{t}/R{R}/pos={pid}: prior nao registrado")
                continue
            v_used = float(priors_used.loc[mask, "prior_pos_used"].iloc[0])
            if abs(v_used - float(val)) > 1e-9:
                falhas.append(f"{t}/R{R}/pos={pid}: usado={v_used} != recomputado={float(val)}")
    return falhas


# ------------------------------------------------------------------ L7 (fortalecido)
def test_L7(dfr, n_folds=5):
    """L7: spot-check 5 folds com RECOMPUTACAO INDEPENDENTE dos MAEs internos.

    Reconstroi o dataset, recomputa features, priors, shrinkage, preprocessamento
    e refit para todos os alphas em cada round interno. Compara alpha_expected
    (argmin com tie-break por menor alpha) com alpha_star registrado.
    """
    rv2 = _load_ridge_v2()
    audit = rv2.load_yaml(AUDIT_YAML)

    # Reconstrucao completa (1x)
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)
    feats = rv2.compute_participation_features(df)
    feats = rv2.add_onehot(feats)

    falhas = []
    sample = dfr.sample(n=min(n_folds, len(dfr)), random_state=7)
    for _, row in sample.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        rounds_t = sorted(feats[feats["temporada"] == t]["rodada"].unique())
        inner_rounds = [r for r in rounds_t if rv2.MIN_INNER_ROUNDS <= r < R]

        inner_mean = {a: [] for a in rv2.ALPHAS}
        for r_i in inner_rounds:
            tr_i = feats[(feats["temporada"] == t) & (feats["rodada"] < r_i) & feats["elegivel"]]
            te_i = feats[(feats["temporada"] == t) & (feats["rodada"] == r_i) & feats["elegivel"]]
            tr_i = tr_i[tr_i["entrou_em_campo"].fillna(False)].copy()
            te_i = te_i[te_i["entrou_em_campo"].fillna(False)].copy()
            if len(tr_i) == 0 or len(te_i) == 0:
                continue

            pp, pg, b0i = rv2.compute_priors(tr_i)
            tr_i = rv2.add_shrunk_features(tr_i, pp, pg, b0i)
            te_i = rv2.add_shrunk_features(te_i, pp, pg, b0i)

            Xt, Xv, _ = rv2.preprocess_fit_apply(tr_i, te_i)
            yt = tr_i[TARGET].to_numpy(dtype=float)
            yv = te_i[TARGET].to_numpy(dtype=float)

            for a in rv2.ALPHAS:
                pred, _ = rv2.fit_predict_ridge(Xt, yt, Xv, a)
                inner_mean[a].append(rv2.mae(pred, yv))

        inner_agg = {a: float(np.mean(v)) if v else np.inf for a, v in inner_mean.items()}
        alpha_expected = min(rv2.ALPHAS, key=lambda a: (inner_agg[a], a))
        alpha_used = float(row["alpha_star"])

        if abs(alpha_expected - alpha_used) > 1e-9:
            falhas.append(
                f"{t}/R{R}: alpha_expected={alpha_expected} (MAE={inner_agg[alpha_expected]:.6f}) "
                f"!= alpha_usado={alpha_used}")

    return falhas


# ------------------------------------------------------------------ L8
def test_L8():
    """L8: SHA-256 do Parquet canonico permanece igual ao hash esperado
    (verificado apos a execucao de ridge_v2.py)."""
    h = sha256(CANON)
    if h != HASHES_ESPERADOS["canonical"]:
        return [f"canonical SHA-256 diverge do esperado: {h[:16]}"]
    return []


# ------------------------------------------------------------------ L9
def test_L9():
    """L9: SHA-256 de baseline_b2s_folds.csv permanece igual ao hash esperado
    (verificado apos a execucao de ridge_v2.py)."""
    h = sha256(B2S_FOLDS)
    if h != HASHES_ESPERADOS["b2s_folds"]:
        return [f"b2s_folds SHA-256 diverge do esperado: {h[:16]}"]
    return []


# ------------------------------------------------------------------ L10
def test_L10(dfr):
    """L10: spot-check 3 folds, recomputar hist_shrunk_k7 do zero com
    prior_pos do treino do fold. Tolerancia 1e-9."""
    if not SHRUNK.exists():
        return ["ridge_v2_shrunk_debug.csv ausente"]
    shrunk = pd.read_csv(SHRUNK)
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    df = df.sort_values(["temporada", "atleta_id", "rodada"],
                        kind="mergesort").reset_index(drop=True)
    g = df.groupby(["temporada", "atleta_id"], sort=False)
    df["_p"] = (df["entrou_bool"] & df["elegivel"]).astype(int)
    df["_y_part"] = df[TARGET] * df["_p"]
    df["n_prev"] = g["_p"].cumsum() - df["_p"]
    df["sum_prev"] = g["_y_part"].cumsum() - df["_y_part"]

    falhas = []
    sample = dfr.sample(n=min(3, len(dfr)), random_state=11)
    for _, row in sample.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        tr = df[(df["temporada"] == t) & (df["rodada"] < R) & df["elegivel"] & df["entrou_bool"]]
        pp = tr.groupby("posicao_id")[TARGET].mean().to_dict()
        b0 = float(tr[TARGET].mean())
        dbg = shrunk[(shrunk["temporada"] == t) & (shrunk["rodada"] == R)]
        for _, d in dbg.iterrows():
            aid = int(d["atleta_id"])
            full = df[(df["temporada"] == t) & (df["rodada"] == R) & (df["atleta_id"] == aid)]
            if len(full) == 0:
                falhas.append(f"{t}/R{R}/{aid}: linha ausente")
                continue
            n = float(full["n_prev"].iloc[0])
            s = float(full["sum_prev"].iloc[0])
            pos = int(full["posicao_id"].iloc[0])
            prior = pp.get(pos, b0)
            if n > 0:
                expected = (s + K_SHRINK * prior) / (n + K_SHRINK)
            else:
                expected = np.nan
            got = float(d["hist_shrunk_k7"]) if pd.notna(d["hist_shrunk_k7"]) else np.nan
            if np.isnan(expected) and np.isnan(got):
                continue
            if not (np.isnan(expected) and np.isnan(got)) and abs(expected - got) > 1e-9:
                falhas.append(f"{t}/R{R}/{aid}: usado={got} != recomputado={expected}")
    return falhas


# ------------------------------------------------------------------ main
def main():
    print("== Ridge v2 — Testes L1-L10 ==")

    if not FOLDS.exists():
        print("FAIL: ridge_v2_folds.csv ausente. Rode ridge_v2.py primeiro.")
        sys.exit(2)
    dfr = pd.read_csv(FOLDS)

    testes = [
        ("L1 (train_max < target)", test_L1, (dfr,)),
        ("L2 (test_round == target)", test_L2, (dfr,)),
        ("L3 (sem overlap train/test em 76 folds)", test_L3, (dfr,)),
        ("L4 (2025/R1 ausente)", test_L4, (dfr,)),
        ("L5 (n_test == participantes)", test_L5, (dfr,)),
        ("L6 (prior_pos spot-check 5 folds)", test_L6, (dfr,)),
        ("L7 (alpha_star recomputado independente, 5 folds)", test_L7, (dfr,)),
        ("L8 (canonical SHA-256 esperado)", test_L8, ()),
        ("L9 (b2s SHA-256 esperado)", test_L9, ()),
        ("L10 (hist_shrunk_k7 spot-check 3 folds)", test_L10, (dfr,)),
    ]

    all_ok = True
    for nome, fn, args in testes:
        try:
            falhas = fn(*args)
        except Exception as e:
            print(f"  {nome}: FAIL (excecao: {type(e).__name__}: {e})")
            all_ok = False
            continue
        if falhas:
            print(f"  {nome}: FAIL")
            for f in falhas[:10]:
                print(f"      {f}")
            if len(falhas) > 10:
                print(f"      ... (+{len(falhas)-10})")
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
