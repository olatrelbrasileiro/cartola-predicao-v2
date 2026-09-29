#!/usr/bin/env python3
"""
Ridge v2 — features P0-corrected, walk-forward, intra-temporada.

Benchmark externo: B2S P0-corrected (MAE = 2.7485).
Spec: config/ridge_v2_spec.yaml

NAO executa o benchmark (le reports/baselines_p0_corrected_folds.csv).
NAO altera o Parquet canonico.
NAO usa Confronto, status_id, NDCG, xG/xA, embeddings, pooling.
"""
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ------------------------------------------------------------- paths
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
SPEC_YAML = CONFIG / "ridge_v2_spec.yaml"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"
B2S_FOLDS = REP / "baseline_b2s_folds.csv"               # antigo, mantido apenas para hash
B2S_P0_FOLDS = REP / "baselines_p0_corrected_folds.csv"  # benchmark oficial B2S P0-corrected
P0_FOLDS = REP / "baselines_p0_corrected_folds.csv"

OUT_FOLDS = REP / "ridge_v2_folds.csv"
OUT_ALPHAS = REP / "ridge_v2_alphas.csv"
OUT_COEF = REP / "ridge_v2_coefficients.csv"
OUT_SUMMARY = REP / "ridge_v2_summary.md"
OUT_SELECT = REP / "ridge_v2_selection.json"
OUT_VALID = REP / "ridge_v2_validation.json"
OUT_PRIORS = REP / "ridge_v2_priors_debug.csv"
OUT_SHRUNK = REP / "ridge_v2_shrunk_debug.csv"

TARGET = "pontos_num"
K_SHRINK = 7
MIN_INNER_ROUNDS = 3
ALPHAS = [0.1, 1, 3, 10, 30, 100, 300, 1000]
SEED = 42

FEATURES_NUM = [
    "hist_mean", "hist_std", "hist_n", "hist_shrunk_k7",
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    "basic_mean", "basic_rec_5",
    "ga_per_game", "ga_rec_5", "ga_shrunk_k7",
    "entrou_rate", "is_cold_start",
]
FEATURES_ONEHOT = ["posicao_1", "posicao_2", "posicao_3", "posicao_4", "posicao_5", "posicao_6"]
FEATURES = FEATURES_NUM + FEATURES_ONEHOT

HASHES_ESPERADOS = {
    "canonical": "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4",
    "b2s_folds": "b56c6db440d7628acbbc52bb477e1b4fe81ce3fc87610e5aa0452e2110e6a142",
    "harness":   "236b7421ed83860d6921131d708b0508601a8be495ac35f175f91eaefca803a5",
    "p0_folds":  "8b4f61851c227d8ee8c2c76aa38f8c6389656bbb8d1109ee909ccce2d77bb689",
}

FIT_COUNTER = {"n_ridge_fits": 0}


# ------------------------------------------------------------- utils
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mae(p, y):
    return float(np.mean(np.abs(p - y))) if len(y) else np.nan


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan


def spearman(p, y):
    if len(p) < 2:
        return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]):
        return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------- cumulative diffs
def apply_cumulative_diff(df, cols):
    out = df.sort_values(["temporada", "atleta_id", "rodada"],
                         kind="mergesort").reset_index(drop=True)
    for col in cols:
        ff = out.groupby(["temporada", "atleta_id"], sort=False)[col].ffill()
        chave = out[["temporada", "atleta_id"]].copy()
        chave["_ff"] = ff.values
        prev = chave.groupby(["temporada", "atleta_id"], sort=False)["_ff"].shift(1)
        out[f"diff_{col}"] = ff.fillna(0).to_numpy() - prev.fillna(0).to_numpy()
    return out


def compute_basic_score(df, audit):
    pesos = audit["pesos_por_temporada"]
    total = pd.Series(0.0, index=df.index)
    for s in audit["scouts_basicos"]:
        col = f"diff_scout_{s}"
        peso_linha = df["temporada"].map({int(t): pesos[t][s] for t in pesos})
        total = total + df[col].to_numpy() * peso_linha.to_numpy()
    return total.astype(float)


# ------------------------------------------------------------- features pre-computaveis
def compute_participation_features(df):
    """
    Para cada linha (temporada, atleta_id, rodada), computa features com base
    SOMENTE em observacoes anteriores elegiveis e participantes da mesma
    temporada/jogador.

    Retorna DataFrame indexado pela posicao original em `df` com colunas:
      hist_mean, hist_std, hist_n, rec_mean_5, rec_mean_3, rec_std_5,
      momentum, basic_mean, basic_rec_5, ga_per_game, ga_rec_5,
      entrou_rate, is_cold_start,
      n_prev, sum_prev, ga_rate_last
    """
    df = df.reset_index(drop=True).copy()
    df["_orig_idx"] = df.index

    df["_is_part"] = (df["entrou_em_campo"].fillna(False).astype(bool) & df["elegivel"]).astype(int)
    df["_is_eleg"] = df["elegivel"].astype(int)
    df["_y_part"] = df[TARGET] * df["_is_part"]
    df["_y2_part"] = (df[TARGET] ** 2) * df["_is_part"]
    df["_b_part"] = df["basic_score"] * df["_is_part"]

    df = df.sort_values(["temporada", "atleta_id", "rodada"],
                        kind="mergesort").reset_index(drop=True)

    g = df.groupby(["temporada", "atleta_id"], sort=False)
    df["_n_part_prev"] = g["_is_part"].cumsum() - df["_is_part"]
    df["_n_eleg_prev"] = g["_is_eleg"].cumsum() - df["_is_eleg"]
    df["_sum_y_prev"] = g["_y_part"].cumsum() - df["_y_part"]
    df["_sum_y2_prev"] = g["_y2_part"].cumsum() - df["_y2_part"]
    df["_sum_b_prev"] = g["_b_part"].cumsum() - df["_b_part"]

    n = df["_n_part_prev"].to_numpy(dtype=float)
    s = df["_sum_y_prev"].to_numpy(dtype=float)
    s2 = df["_sum_y2_prev"].to_numpy(dtype=float)
    sb = df["_sum_b_prev"].to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        hist_mean = np.where(n > 0, s / n, np.nan)
        var = np.where(n > 0, s2 / n - (s / n) ** 2, np.nan)
        var = np.where(var < 0, 0.0, var)
        hist_std = np.sqrt(var)
        basic_mean = np.where(n > 0, sb / n, np.nan)
        entrou_rate = np.where(
            df["_n_eleg_prev"].to_numpy() > 0,
            df["_n_part_prev"].to_numpy() / df["_n_eleg_prev"].to_numpy(),
            np.nan,
        )

    df["hist_mean"] = hist_mean
    df["hist_std"] = hist_std
    df["hist_n"] = df["_n_part_prev"].astype(int)
    df["basic_mean"] = basic_mean
    df["entrou_rate"] = entrou_rate
    df["is_cold_start"] = (df["_n_part_prev"] == 0).astype(int)

    # ---- features de "ultimas N PARTICIPACOES"
    rec_data = []
    for (t, aid), sub in df.groupby(["temporada", "atleta_id"], sort=False):
        sub = sub.sort_values("rodada").reset_index(drop=False)
        part = sub[sub["_is_part"] == 1].copy()
        if len(part) == 0:
            for _, r in sub.iterrows():
                rec_data.append({
                    "_orig_idx": int(r["_orig_idx"]),
                    "rec_mean_5": np.nan, "rec_mean_3": np.nan,
                    "rec_std_5": np.nan, "basic_rec_5": np.nan,
                    "ga_rate_last": np.nan, "ga_rec_5": np.nan,
                })
            continue

        part = part.reset_index(drop=True)
        part["_rec_mean_5"] = part[TARGET].rolling(5, min_periods=1).mean()
        part["_rec_mean_3"] = part[TARGET].rolling(3, min_periods=1).mean()
        part["_rec_std_5"] = part[TARGET].rolling(5, min_periods=2).std(ddof=0).fillna(0.0)
        part["_basic_rec_5"] = part["basic_score"].rolling(5, min_periods=1).mean()

        # (G+A) cumulativo da rodada (via diffs)
        part["_diff_ga"] = part["diff_scout_G"].fillna(0) + part["diff_scout_A"].fillna(0)
        part["_cum_ga"] = part["_diff_ga"].cumsum()
        part["_jogos"] = part["jogos_num"].fillna(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            part["_ga_rate"] = part["_cum_ga"] / part["_jogos"].clip(lower=1)
        part["_ga_diff_5"] = part["_diff_ga"].rolling(5, min_periods=1).sum()
        part["_n_part_5"] = part[TARGET].rolling(5, min_periods=1).count()
        part["_ga_rec_5"] = part["_ga_diff_5"] / part["_n_part_5"]

        part_rodadas = part["rodada"].to_numpy()
        target_rodadas = sub["rodada"].to_numpy()
        idx = np.searchsorted(part_rodadas, target_rodadas, side="left") - 1

        for i, (_, r) in enumerate(sub.iterrows()):
            j = idx[i]
            if j < 0:
                rec_data.append({
                    "_orig_idx": int(r["_orig_idx"]),
                    "rec_mean_5": np.nan, "rec_mean_3": np.nan,
                    "rec_std_5": np.nan, "basic_rec_5": np.nan,
                    "ga_rate_last": np.nan, "ga_rec_5": np.nan,
                })
            else:
                p = part.iloc[j]
                rec_data.append({
                    "_orig_idx": int(r["_orig_idx"]),
                    "rec_mean_5": float(p["_rec_mean_5"]),
                    "rec_mean_3": float(p["_rec_mean_3"]),
                    "rec_std_5": float(p["_rec_std_5"]),
                    "basic_rec_5": float(p["_basic_rec_5"]),
                    "ga_rate_last": float(p["_ga_rate"]),
                    "ga_rec_5": float(p["_ga_rec_5"]),
                })

    rec_df = pd.DataFrame(rec_data).set_index("_orig_idx")
    df = df.merge(rec_df, left_on="_orig_idx", right_index=True, how="left")
    df["momentum"] = df["rec_mean_5"] - df["hist_mean"]
    df["ga_per_game"] = df["ga_rate_last"]

    df["n_prev"] = df["_n_part_prev"].astype(int)
    df["sum_prev"] = df["_sum_y_prev"]

    df = df.sort_values("_orig_idx").reset_index(drop=True)
    return df[[
        "_orig_idx", "temporada", "rodada", "atleta_id", "posicao_id",
        "entrou_em_campo", "elegivel", TARGET,
        "hist_mean", "hist_std", "hist_n",
        "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
        "basic_mean", "basic_rec_5",
        "ga_per_game", "ga_rec_5", "ga_rate_last",
        "entrou_rate", "is_cold_start",
        "n_prev", "sum_prev",
    ]].copy()


def add_onehot(df):
    df = df.copy()
    for i in range(1, 7):
        df[f"posicao_{i}"] = (df["posicao_id"] == i).astype(float)
    return df


def add_shrunk_features(df, prior_pos_dict, prior_ga_dict, b0_const, k=K_SHRINK):
    """Adiciona hist_shrunk_k7 e ga_shrunk_k7. NaN quando n_prev == 0."""
    df = df.copy()
    n = df["n_prev"].to_numpy(dtype=float)
    s = df["sum_prev"].to_numpy(dtype=float)
    gr = df["ga_rate_last"].to_numpy(dtype=float)
    pos = df["posicao_id"].to_numpy()

    prior_pos_arr = np.array([prior_pos_dict.get(int(p), b0_const) for p in pos], dtype=float)
    prior_ga_arr = np.array([prior_ga_dict.get(int(p), 0.0) for p in pos], dtype=float)

    hist_shrunk = np.full(len(df), np.nan, dtype=float)
    mask = n > 0
    hist_shrunk[mask] = (s[mask] + k * prior_pos_arr[mask]) / (n[mask] + k)
    df["hist_shrunk_k7"] = hist_shrunk

    ga_shrunk = np.full(len(df), np.nan, dtype=float)
    mask_ga = (n > 0) & (~np.isnan(gr))
    ga_shrunk[mask_ga] = (n[mask_ga] * gr[mask_ga] + k * prior_ga_arr[mask_ga]) / (n[mask_ga] + k)
    df["ga_shrunk_k7"] = ga_shrunk
    return df


# ------------------------------------------------------------- preprocess + ridge
def preprocess_fit_apply(train_df, test_df):
    """Imputa NaN numericos por media do treino; standardiza no treino. One-hot intacto."""
    X_tr = train_df[FEATURES].to_numpy(dtype=float)
    X_te = test_df[FEATURES].to_numpy(dtype=float)
    n_num = len(FEATURES_NUM)

    X_tr_num = X_tr[:, :n_num]
    X_te_num = X_te[:, :n_num]
    X_tr_oh = X_tr[:, n_num:]
    X_te_oh = X_te[:, n_num:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(X_tr_num, axis=0)
    means = np.where(np.isnan(means), 0.0, means)

    X_tr_num_filled = np.where(np.isnan(X_tr_num), means, X_tr_num)
    X_te_num_filled = np.where(np.isnan(X_te_num), means, X_te_num)

    stds = X_tr_num_filled.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    X_tr_num_scaled = (X_tr_num_filled - means) / stds
    X_te_num_scaled = (X_te_num_filled - means) / stds

    X_tr_final = np.hstack([X_tr_num_scaled, X_tr_oh])
    X_te_final = np.hstack([X_te_num_scaled, X_te_oh])

    oh_std = X_tr_oh.std(axis=0)
    zero_var = [FEATURES_ONEHOT[i] for i in range(len(FEATURES_ONEHOT)) if oh_std[i] == 0]

    state = {"means": means.tolist(), "stds": stds.tolist(), "zero_var_onehot": zero_var}
    return X_tr_final, X_te_final, state


def fit_predict_ridge(X_tr, y_tr, X_te, alpha):
    """Closed-form numpy. Atualiza FIT_COUNTER."""
    y_mean = float(y_tr.mean())
    y_c = y_tr - y_mean
    n_feat = X_tr.shape[1]
    A = X_tr.T @ X_tr + alpha * np.eye(n_feat)
    w = np.linalg.solve(A, X_tr.T @ y_c)
    pred = X_te @ w + y_mean
    FIT_COUNTER["n_ridge_fits"] += 1
    return pred, w


# ------------------------------------------------------------- priors
def compute_priors(train_part):
    prior_pos = train_part.groupby("posicao_id")[TARGET].mean().to_dict()
    tmp = train_part[train_part["ga_rate_last"].notna()]
    if len(tmp) > 0:
        prior_ga = tmp.groupby("posicao_id")["ga_rate_last"].mean().to_dict()
    else:
        prior_ga = {}
    b0_const = float(train_part[TARGET].mean()) if len(train_part) else 0.0
    return prior_pos, prior_ga, b0_const


# ------------------------------------------------------------- main
def main():
    inicio = datetime.now()
    print("== Ridge v2 ==")

    # integridade antes
    h_antes = {}
    for nome, p in [("canonical", CANON), ("b2s_folds", B2S_FOLDS),
                    ("harness", HARNESS), ("p0_folds", P0_FOLDS)]:
        h = sha256(p)
        h_antes[nome] = h
        ok = "OK" if h == HASHES_ESPERADOS[nome] else "*** DIVERGE ***"
        print(f"  {nome:10s} {h[:16]}...  {ok}")
    if any(h_antes[k] != HASHES_ESPERADOS[k] for k in HASHES_ESPERADOS):
        print("ERRO: hash divergente. Abortando.", file=sys.stderr)
        sys.exit(2)

    audit = load_yaml(AUDIT_YAML)

    # canonical + diffs + basic_score
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("computando diffs...")
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = compute_basic_score(df, audit)

    print("computando features pre-computaveis...")
    feats = compute_participation_features(df)
    feats = add_onehot(feats)

    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")

    # B2S folds
    b2s_folds = pd.read_csv(B2S_P0_FOLDS).set_index(["temporada", "rodada_alvo"])

    rows_fold = []
    rows_alphas = []
    coeffs_by_fold = []
    priors_debug = []
    shrunk_debug = []

    for idx, (t, R) in enumerate(folds):
        mask_t = feats["temporada"] == t
        train_outer_all = feats[mask_t & (feats["rodada"] < R) & feats["elegivel"]]
        test_outer_all = feats[mask_t & (feats["rodada"] == R) & feats["elegivel"]]
        train_outer = train_outer_all[train_outer_all["entrou_em_campo"].fillna(False)].copy()
        test_outer = test_outer_all[test_outer_all["entrou_em_campo"].fillna(False)].copy()

        if len(train_outer) == 0 or len(test_outer) == 0:
            continue

        prior_pos_o, prior_ga_o, b0_o = compute_priors(train_outer)
        train_outer = add_shrunk_features(train_outer, prior_pos_o, prior_ga_o, b0_o)
        test_outer = add_shrunk_features(test_outer, prior_pos_o, prior_ga_o, b0_o)

        for pid, val in prior_pos_o.items():
            priors_debug.append({"temporada": t, "rodada": R, "posicao_id": int(pid),
                                 "prior_pos_used": float(val),
                                 "prior_ga_used": float(prior_ga_o.get(pid, np.nan))})

        # debug: primeiras 5 linhas do test
        dbg = test_outer.head(5)[["temporada", "rodada", "atleta_id", "posicao_id",
                                  "hist_shrunk_k7", "ga_shrunk_k7"]].copy()
        shrunk_debug.append(dbg)

        X_tr, X_te, state = preprocess_fit_apply(train_outer, test_outer)
        y_tr = train_outer[TARGET].to_numpy(dtype=float)
        y_te = test_outer[TARGET].to_numpy(dtype=float)

        # ---- inner walk-forward para alpha
        rounds_t = sorted(feats[mask_t]["rodada"].unique())
        inner_rounds = [r for r in rounds_t if MIN_INNER_ROUNDS <= r < R]
        inner_mean = {a: [] for a in ALPHAS}

        for r_i in inner_rounds:
            tr_i = feats[mask_t & (feats["rodada"] < r_i) & feats["elegivel"]]
            te_i = feats[mask_t & (feats["rodada"] == r_i) & feats["elegivel"]]
            tr_i = tr_i[tr_i["entrou_em_campo"].fillna(False)].copy()
            te_i = te_i[te_i["entrou_em_campo"].fillna(False)].copy()
            if len(tr_i) == 0 or len(te_i) == 0:
                continue
            pp, pg, b0i = compute_priors(tr_i)
            tr_i = add_shrunk_features(tr_i, pp, pg, b0i)
            te_i = add_shrunk_features(te_i, pp, pg, b0i)
            Xt, Xv, _ = preprocess_fit_apply(tr_i, te_i)
            yt = tr_i[TARGET].to_numpy(dtype=float)
            yv = te_i[TARGET].to_numpy(dtype=float)
            for a in ALPHAS:
                pred, _ = fit_predict_ridge(Xt, yt, Xv, a)
                inner_mean[a].append(mae(pred, yv))

        inner_agg = {a: float(np.mean(v)) if v else np.inf for a, v in inner_mean.items()}
        alpha_star = min(ALPHAS, key=lambda a: (inner_agg[a], a))

        for a in ALPHAS:
            rows_alphas.append({
                "temporada": t, "rodada_alvo": R, "alpha": a,
                "inner_mean_mae": inner_agg[a],
                "selecionado": int(a == alpha_star),
            })

        # ---- fit final + previsao
        pred_final, w = fit_predict_ridge(X_tr, y_tr, X_te, alpha_star)

        coeffs_by_fold.append({
            "temporada": t, "rodada_alvo": R,
            "alpha_star": alpha_star,
            **{f"w_{f}": float(w[i]) for i, f in enumerate(FEATURES)},
        })

        b2s_mae_fold = float(b2s_folds.loc[(t, R), "b2s_mae"])
        rows_fold.append({
            "temporada": t, "rodada_alvo": R,
            "n_train": int(len(train_outer)),
            "n_test": int(len(test_outer)),
            "alpha_star": float(alpha_star),
            "mae": mae(pred_final, y_te),
            "rmse": rmse(pred_final, y_te),
            "spearman": spearman(pred_final, y_te),
            "b2s_mae": b2s_mae_fold,
            "delta_mae": mae(pred_final, y_te) - b2s_mae_fold,
            "zero_var_onehot": ",".join(state["zero_var_onehot"]),
        })

        if (idx + 1) % 10 == 0:
            print(f"  fold {idx+1}/{len(folds)}  {t}/R{R}  "
                  f"MAE={rows_fold[-1]['mae']:.4f}  B2S={b2s_mae_fold:.4f}  alpha={alpha_star}")

    dfr = pd.DataFrame(rows_fold)
    dfa = pd.DataFrame(rows_alphas)
    dfc = pd.DataFrame(coeffs_by_fold)
    df_priors = pd.DataFrame(priors_debug)
    df_shrunk = pd.concat(shrunk_debug, ignore_index=True) if shrunk_debug else pd.DataFrame()

    dfr.to_csv(OUT_FOLDS, index=False)
    dfa.to_csv(OUT_ALPHAS, index=False)
    dfc.to_csv(OUT_COEF, index=False)
    df_priors.to_csv(OUT_PRIORS, index=False)
    df_shrunk.to_csv(OUT_SHRUNK, index=False)

    # ---- agregacao
    total_n = int(dfr["n_test"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"] ** 2 * dfr["n_test"]).sum() / total_n))
    sp_medio = float(dfr["spearman"].mean())
    b2s_mae_global = float((dfr["b2s_mae"] * dfr["n_test"]).sum() / total_n)

    n_melhor = int((dfr["delta_mae"] < -1e-4).sum())
    n_empate = int((dfr["delta_mae"].abs() <= 1e-4).sum())
    n_pior = int(len(dfr) - n_melhor - n_empate)
    med_delta = float(dfr["delta_mae"].median())

    por_temp = dfr.groupby("temporada").apply(lambda g: pd.Series({
        "n_folds": len(g),
        "n_test": int(g["n_test"].sum()),
        "mae_ridge": float((g["mae"] * g["n_test"]).sum() / g["n_test"].sum()),
        "mae_b2s": float((g["b2s_mae"] * g["n_test"]).sum() / g["n_test"].sum()),
        "delta": float(((g["mae"] - g["b2s_mae"]) * g["n_test"]).sum() / g["n_test"].sum()),
        "spearman_ridge": float(g["spearman"].mean()),
    }, dtype=float)).reset_index()

    # ---- criterios
    c1 = mae_global < b2s_mae_global - 1e-3
    c2 = n_melhor >= 50
    c3 = all((por_temp[por_temp["temporada"] == t]["mae_ridge"].iloc[0]
              < por_temp[por_temp["temporada"] == t]["mae_b2s"].iloc[0])
             for t in [2024, 2025, 2026])
    c4 = all((por_temp[por_temp["temporada"] == t]["mae_ridge"].iloc[0]
              <= 1.015 * por_temp[por_temp["temporada"] == t]["mae_b2s"].iloc[0])
             for t in [2024, 2025, 2026])
    c5 = med_delta < -1e-3

    criterios_ok = all([c1, c2, c3, c4, c5])
    if criterios_ok:
        decisao = "PROMOTE_RIDGE_V2"
        motivo = "todos os 5 criterios atendidos"
    elif any([not c1, not c2, not c3]) and not any([c1, c2, c3]):
        decisao = "KEEP_B2S"
        motivo = "criterios centrais nao atendidos"
    else:
        decisao = "INCONCLUSIVE"
        motivo = "criterios parcialmente atendidos"

    # ---- integridade depois
    h_depois = {nome: sha256(p) for nome, p in
                [("canonical", CANON), ("b2s_folds", B2S_FOLDS),
                 ("harness", HARNESS), ("p0_folds", P0_FOLDS)]}
    integridade_ok = all(h_antes[k] == h_depois[k] for k in h_antes)

    # ---- coeficientes agregados
    coef_cols = [f"w_{f}" for f in FEATURES]
    coef_agg = []
    for c in coef_cols:
        vals = dfc[c].to_numpy()
        coef_agg.append({
            "feature": c[2:],
            "coef_mean": float(vals.mean()),
            "coef_std": float(vals.std()),
            "pct_positive": float((vals > 0).mean()),
            "pct_negative": float((vals < 0).mean()),
            "pct_near_zero": float((np.abs(vals) < 1e-3).mean()),
        })
    pd.DataFrame(coef_agg).to_csv(OUT_COEF, index=False)

    # ---- summary
    fim = datetime.now()
    L = []
    L.append("# Ridge v2 — Summary\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- Folds: {len(dfr)}\n")
    L.append(f"- Obs no test: {total_n}\n")
    L.append(f"- n_ridge_fits (real): {FIT_COUNTER['n_ridge_fits']}\n\n")
    L.append("## Metricas globais\n\n")
    L.append(f"- Ridge v2 MAE ponderado: {mae_global:.4f}\n")
    L.append(f"- Ridge v2 RMSE ponderado: {rmse_global:.4f}\n")
    L.append(f"- Ridge v2 Spearman medio: {sp_medio:.4f}\n")
    L.append(f"- B2S MAE ponderado: {b2s_mae_global:.4f}\n")
    L.append(f"- Delta MAE (Ridge - B2S): {mae_global - b2s_mae_global:+.4f}\n\n")
    L.append("## Por fold\n\n")
    L.append(f"- Folds Ridge < B2S (tol 1e-4): {n_melhor}/{len(dfr)}\n")
    L.append(f"- Folds empate: {n_empate}\n")
    L.append(f"- Folds Ridge > B2S: {n_pior}\n")
    L.append(f"- Mediana do delta: {med_delta:+.4f}\n\n")
    L.append("## Por temporada\n\n")
    L.append(por_temp.to_string(index=False))
    L.append("\n\n## Criterios de promocao\n\n")
    L.append(f"- C1 (MAE global): {c1}\n")
    L.append(f"- C2 (>= 50/76 folds): {c2} ({n_melhor}/76)\n")
    L.append(f"- C3 (3 temporadas): {c3}\n")
    L.append(f"- C4 (nenhuma piora > 1.5%): {c4}\n")
    L.append(f"- C5 (mediana delta < -1e-3): {c5}\n\n")
    L.append(f"**Decisao: {decisao}**\n\n")
    L.append(f"Motivo: {motivo}\n\n")
    L.append("## Distribuicao de alpha_star\n\n")
    L.append(dfr["alpha_star"].value_counts().sort_index().to_string())
    L.append("\n\n## Integridade\n\n")
    for k in h_antes:
        L.append(f"- {k}: {'OK' if h_antes[k] == h_depois[k] else 'ALTERADO'}\n")
    OUT_SUMMARY.write_text("".join(L), encoding="utf-8")

    # ---- selection.json
    sel = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "decisao": decisao,
        "motivo": motivo,
        "criterios": {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5},
        "mae_global_ridge": mae_global,
        "mae_global_b2s": b2s_mae_global,
        "folds_melhor": n_melhor,
        "folds_empate": n_empate,
        "folds_pior": n_pior,
        "mediana_delta": med_delta,
        "por_temporada": por_temp.to_dict(orient="records"),
    }
    with open(OUT_SELECT, "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=2, default=str)

    # ---- validation.json
    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "n_folds": len(dfr),
        "n_test_total": total_n,
        "n_ridge_fits": FIT_COUNTER["n_ridge_fits"],
        "hashes_antes": h_antes,
        "hashes_depois": h_depois,
        "hashes_esperados": HASHES_ESPERADOS,
        "integridade_ok": integridade_ok,
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=== RESUMO ===")
    print(f"Ridge v2 MAE: {mae_global:.4f}  B2S: {b2s_mae_global:.4f}  "
          f"delta: {mae_global - b2s_mae_global:+.4f}")
    print(f"folds Ridge<B2S: {n_melhor}/76  empate: {n_empate}  pior: {n_pior}")
    print(f"n_ridge_fits: {FIT_COUNTER['n_ridge_fits']}")
    print(f"decisao: {decisao}")
    print(f"integridade: {integridade_ok}")

    if not integridade_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
