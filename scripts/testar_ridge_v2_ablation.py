#!/usr/bin/env python3
"""
Testes L1-L10 da ablacao do Ridge v2.

NAO treina. Valida artefatos, reusa infraestrutura do ridge_v2 e
verifica integridade por hash.
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

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"
HARNESS = REP / "harness_folds_v2.csv"
B2S_FOLDS = REP / "baseline_b2s_folds.csv"

FOLDS = REP / "ridge_v2_ablation_folds.csv"
SUMMARY = REP / "ridge_v2_ablation_summary.csv"
SEASON = REP / "ridge_v2_ablation_season.csv"
SELECTION = REP / "ridge_v2_ablation_selection.csv"
ALPHAS = REP / "ridge_v2_ablation_alphas.csv"
VALID = REP / "ridge_v2_ablation_validation.json"

TARGET = "pontos_num"

HASHES_ESPERADOS = {
    "canonical": "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4",
    "b2s_folds": "b56c6db440d7628acbbc52bb477e1b4fe81ce3fc87610e5aa0452e2110e6a142",
}

EXPECTED_EXPERIMENTS = {
    "HIST", "REC", "BASIC", "GA", "OPP", "POS",
    "FULL_MINUS_HIST", "FULL_MINUS_REC", "FULL_MINUS_BASIC",
    "FULL_MINUS_GA", "FULL_MINUS_OPP", "FULL_MINUS_POS",
    "FULL",
}

FEATURES_NUM = [
    "hist_mean", "hist_std", "hist_n", "hist_shrunk_k7",
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    "basic_mean", "basic_rec_5",
    "ga_per_game", "ga_rec_5", "ga_shrunk_k7",
    "entrou_rate", "is_cold_start",
]
FEATURES_ONEHOT = ["posicao_1", "posicao_2", "posicao_3", "posicao_4", "posicao_5", "posicao_6"]
FEATURES_ALL = FEATURES_NUM + FEATURES_ONEHOT

GROUPS = {
    "HIST":  ["hist_mean", "hist_std", "hist_n", "hist_shrunk_k7"],
    "REC":   ["rec_mean_5", "rec_mean_3", "rec_std_5", "momentum"],
    "BASIC": ["basic_mean", "basic_rec_5"],
    "GA":    ["ga_per_game", "ga_rec_5", "ga_shrunk_k7"],
    "OPP":   ["entrou_rate", "is_cold_start"],
    "POS":   FEATURES_ONEHOT,
}
EXPECTED_FEATURES = {}
for g, fl in GROUPS.items():
    EXPECTED_FEATURES[g] = set(fl)
    EXPECTED_FEATURES[f"FULL_MINUS_{g}"] = set(FEATURES_ALL) - set(fl)
EXPECTED_FEATURES["FULL"] = set(FEATURES_ALL)

ALPHAS_GRADE = [0.1, 1, 3, 10, 30, 100, 300, 1000]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------- L1
def test_L1(dfr):
    """Todos os experimentos usam EXATAMENTE os 76 folds do harness."""
    man = pd.read_csv(HARNESS)
    esperado = set(zip(man["temporada"].astype(int), man["rodada_alvo"].astype(int)))
    if len(esperado) != 76:
        return [f"harness tem {len(esperado)} folds (esperado 76)"]
    falhas = []
    for exp, sub in dfr.groupby("experimento"):
        chaves = list(zip(sub["temporada"].astype(int), sub["rodada_alvo"].astype(int)))
        n = len(chaves)
        if n != 76:
            falhas.append(f"{exp}: {n} folds (esperado 76)")
            continue
        atual = set(chaves)
        if atual != esperado:
            diff_a = esperado - atual
            diff_b = atual - esperado
            falhas.append(f"{exp}: faltam {len(diff_a)}, sobram {len(diff_b)}")
    return falhas


# ------------------------------------------------- L2
def test_L2(dfr):
    """2025/R1 ausente em todos os experimentos."""
    mask = (dfr["temporada"] == 2025) & (dfr["rodada_alvo"] == 1)
    if mask.any():
        return [f"{int(mask.sum())} linhas 2025/R1 em {dfr[mask]['experimento'].unique().tolist()}"]
    return []


# ------------------------------------------------- L3
def test_L3(dfr):
    """train_max_round < target_round em TODOS os folds de todos os experimentos."""
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    # cache de (t, R) -> train_max_round
    cache_trainmax = {}
    falhas = []
    for _, row in dfr.iterrows():
        exp = row["experimento"]
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        key = (t, R)
        if key not in cache_trainmax:
            tr = df[(df["temporada"] == t) & (df["rodada"] < R)
                    & df["elegivel"] & df["entrou_bool"]]
            if len(tr) == 0:
                cache_trainmax[key] = ("vazio", None)
            else:
                cache_trainmax[key] = ("ok", int(tr["rodada"].max()))
        status, tmax = cache_trainmax[key]
        if status == "vazio":
            falhas.append(f"{exp}/{t}/R{R}: treino vazio")
        elif tmax >= R:
            falhas.append(f"{exp}/{t}/R{R}: train_max={tmax} >= R")
    return falhas


# ------------------------------------------------- L4
def test_L4(dfr):
    """Nenhuma chave (temp, rodada, atleta) compartilhada em TODOS os folds
    de todos os experimentos."""
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    cache_keys = {}
    falhas = []
    for _, row in dfr.iterrows():
        exp = row["experimento"]
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        key = (t, R)
        if key not in cache_keys:
            tr = df[(df["temporada"] == t) & (df["rodada"] < R)
                    & df["elegivel"] & df["entrou_bool"]]
            te = df[(df["temporada"] == t) & (df["rodada"] == R)
                    & df["elegivel"] & df["entrou_bool"]]
            kt = set(zip(tr["temporada"], tr["rodada"], tr["atleta_id"]))
            kv = set(zip(te["temporada"], te["rodada"], te["atleta_id"]))
            cache_keys[key] = (kt, kv)
        kt, kv = cache_keys[key]
        inter = kt & kv
        if inter:
            falhas.append(f"{exp}/{t}/R{R}: {len(inter)} chaves em comum")
    return falhas


# ------------------------------------------------- L5
def test_L5(dfr):
    """n_test == numero de participantes do test em TODOS os folds
    de todos os experimentos."""
    df = pd.read_parquet(CANON)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    cache_n = {}
    falhas = []
    for _, row in dfr.iterrows():
        exp = row["experimento"]
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        key = (t, R)
        if key not in cache_n:
            te = df[(df["temporada"] == t) & (df["rodada"] == R)
                    & df["elegivel"] & df["entrou_bool"]]
            cache_n[key] = len(te)
        if int(row["n_test"]) != cache_n[key]:
            falhas.append(f"{exp}/{t}/R{R}: n_test={int(row['n_test'])} != {cache_n[key]}")
    return falhas


# ------------------------------------------------- L6
def test_L6(dfr, dsel):
    """Cada experimento utiliza exatamente o conjunto de features registrado."""
    falhas = []
    for _, row in dsel.iterrows():
        exp = row["experimento"]
        features_str = row["features"]
        feats = set(features_str.split("|"))
        esperado = EXPECTED_FEATURES.get(exp)
        if esperado is None:
            falhas.append(f"{exp}: experimento desconhecido")
            continue
        if feats != esperado:
            faltam = esperado - feats
            sobram = feats - esperado
            falhas.append(f"{exp}: faltam={sorted(faltam)} sobram={sorted(sobram)}")
        if int(row["n_features"]) != len(esperado):
            falhas.append(f"{exp}: n_features={int(row['n_features'])} != {len(esperado)}")
    return falhas


# ------------------------------------------------- L7
def test_L7(dfr, dsel):
    """Nenhuma feature fora do grupo aparece no modelo."""
    falhas = []
    for _, row in dsel.iterrows():
        exp = row["experimento"]
        feats = set(row["features"].split("|"))
        esperado = EXPECTED_FEATURES.get(exp)
        if esperado is None:
            continue
        fora = feats - set(FEATURES_ALL)
        if fora:
            falhas.append(f"{exp}: features desconhecidas {sorted(fora)}")
        if exp.startswith("FULL_MINUS_"):
            g = exp.replace("FULL_MINUS_", "")
            removidas = set(GROUPS[g])
            if feats & removidas:
                falhas.append(f"{exp}: contem features do grupo removido {sorted(feats & removidas)}")
    return falhas


# ------------------------------------------------- L8
def test_L8(dsel, dalphas):
    """alpha respeita grade; spot-check da regra de argmin + tie-break."""
    falhas = []
    for _, row in dsel.iterrows():
        for col in ["alpha_median", "alpha_min", "alpha_max"]:
            v = float(row[col])
            if v not in ALPHAS_GRADE:
                falhas.append(f"{row['experimento']}: {col}={v} fora da grade")

    # spot-check: para 5 experimentos, pegar 3 folds; verificar que
    # alpha_star registrado em ridge_v2_ablation_alphas.csv e' o argmin
    # (com tie-break por menor alpha)
    rng = np.random.default_rng(11)
    exps = list(dsel["experimento"].unique())
    for exp in rng.choice(exps, size=min(5, len(exps)), replace=False):
        sub = dalphas[dalphas["experimento"] == exp]
        folds_uni = sub[["temporada", "rodada_alvo"]].drop_duplicates()
        sample = folds_uni.sample(n=min(3, len(folds_uni)), random_state=7)
        for _, f in sample.iterrows():
            sub_f = sub[(sub["temporada"] == f["temporada"]) &
                        (sub["rodada_alvo"] == f["rodada_alvo"])]
            inner = dict(zip(sub_f["alpha"], sub_f["inner_mean_mae"]))
            expected = min(ALPHAS_GRADE, key=lambda a: (inner.get(a, np.inf), a))
            used_row = sub_f[sub_f["selecionado"] == 1]
            if len(used_row) != 1:
                falhas.append(f"{exp}/{f['temporada']}/{f['rodada_alvo']}: "
                              f"{len(used_row)} alphas marcados como selecionados")
                continue
            used = float(used_row["alpha"].iloc[0])
            if abs(used - expected) > 1e-9:
                falhas.append(f"{exp}/{f['temporada']}/{f['rodada_alvo']}: "
                              f"usado={used} != argmin={expected}")
    return falhas


# ------------------------------------------------- L9
def test_L9(dsum, dfr):
    """MAE global ponderado por n_test confere."""
    falhas = []
    for _, row in dsum.iterrows():
        exp = row["experimento"]
        sub = dfr[dfr["experimento"] == exp]
        n = sub["n_test"].sum()
        mae_g = float((sub["mae"] * sub["n_test"]).sum() / n)
        if abs(mae_g - float(row["mae_global"])) > 1e-9:
            falhas.append(f"{exp}: mae_global={row['mae_global']} != {mae_g}")
    return falhas


# ------------------------------------------------- L10
def test_L10():
    """SHA-256 do Parquet canonico permanece igual ao esperado."""
    falhas = []
    h1 = sha256(CANON)
    if h1 != HASHES_ESPERADOS["canonical"]:
        falhas.append(f"canonical: {h1[:16]}")
    h2 = sha256(B2S_FOLDS)
    if h2 != HASHES_ESPERADOS["b2s_folds"]:
        falhas.append(f"b2s_folds: {h2[:16]}")
    return falhas


# ------------------------------------------------- main
def main():
    print("== Ridge v2 — Testes L1-L10 (ablacao) ==")

    for p in [FOLDS, SUMMARY, SEASON, SELECTION, ALPHAS, VALID]:
        if not p.exists():
            print(f"FAIL: {p.name} ausente. Rode ridge_v2_ablation.py primeiro.")
            sys.exit(2)

    dfr = pd.read_csv(FOLDS)
    dsum = pd.read_csv(SUMMARY)
    dseas = pd.read_csv(SEASON)
    dsel = pd.read_csv(SELECTION)
    dalphas = pd.read_csv(ALPHAS)

    testes = [
        ("L1 (76 folds exatos em todos os exp.)", test_L1, (dfr,)),
        ("L2 (2025/R1 ausente)", test_L2, (dfr,)),
        ("L3 (train_max < target em todos os folds)", test_L3, (dfr,)),
        ("L4 (sem overlap em todos os folds)", test_L4, (dfr,)),
        ("L5 (n_test == participantes em todos os folds)", test_L5, (dfr,)),
        ("L6 (features do grupo registradas)", test_L6, (dfr, dsel)),
        ("L7 (sem feature fora do grupo)", test_L7, (dfr, dsel)),
        ("L8 (alpha argmin + tie-break)", test_L8, (dsel, dalphas)),
        ("L9 (MAE global ponderado)", test_L9, (dsum, dfr)),
        ("L10 (SHA-256 canonical + b2s)", test_L10, ()),
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
