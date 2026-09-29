#!/usr/bin/env python3
"""
Ridge v2 — ablacao controlada por grupo de features.

Diagnostico do sinal de cada grupo sob protocolo temporal identico ao Ridge v2.
NAO compara com B2S em termos de PROMOTE/KEEP/INCONCLUSIVE.
NAO retreina o FULL: le reports/ridge_v2_folds.csv.

Reutiliza funcoes de scripts/ridge_v2.py importadas como modulo (sem main()).
"""
import hashlib
import json
import sys
import warnings
import importlib.util
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
SPEC = CONFIG / "ridge_v2_ablation_spec.yaml"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"
B2S_P0_FOLDS = REP / "baselines_p0_corrected_folds.csv"
RIDGE_V2_FOLDS = REP / "ridge_v2_folds.csv"

OUT_FOLDS = REP / "ridge_v2_ablation_folds.csv"
OUT_SUMMARY = REP / "ridge_v2_ablation_summary.csv"
OUT_SEASON = REP / "ridge_v2_ablation_season.csv"
OUT_SELECTION = REP / "ridge_v2_ablation_selection.csv"
OUT_ALPHAS = REP / "ridge_v2_ablation_alphas.csv"
OUT_VALID = REP / "ridge_v2_ablation_validation.json"

TARGET = "pontos_num"

HASHES_ESPERADOS = {
    "canonical": "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4",
    "b2s_folds": "b56c6db440d7628acbbc52bb477e1b4fe81ce3fc87610e5aa0452e2110e6a142",
    "harness":   "236b7421ed83860d6921131d708b0508601a8be495ac35f175f91eaefca803a5",
    "p0_folds":  "8b4f61851c227d8ee8c2c76aa38f8c6389656bbb8d1109ee909ccce2d77bb689",
}


# ------------------------------------------------------------- import ridge_v2
def _load_ridge_v2():
    spec = importlib.util.spec_from_file_location(
        "ridge_v2_module", str(ROOT / "scripts" / "ridge_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rv2 = _load_ridge_v2()

# reset do contador: o valor registrado em validation.json deve refletir
# SOMENTE os fits realizados pela ablacao atual.
rv2.FIT_COUNTER["n_ridge_fits"] = 0

FEATURES_NUM = rv2.FEATURES_NUM          # 15
FEATURES_ONEHOT = rv2.FEATURES_ONEHOT    # 6
FEATURES_ALL = FEATURES_NUM + FEATURES_ONEHOT

GROUPS = {
    "HIST":  ["hist_mean", "hist_std", "hist_n", "hist_shrunk_k7"],
    "REC":   ["rec_mean_5", "rec_mean_3", "rec_std_5", "momentum"],
    "BASIC": ["basic_mean", "basic_rec_5"],
    "GA":    ["ga_per_game", "ga_rec_5", "ga_shrunk_k7"],
    "OPP":   ["entrou_rate", "is_cold_start"],
    "POS":   list(FEATURES_ONEHOT),
}

EXPERIMENTS = {}
for g in ["HIST", "REC", "BASIC", "GA", "OPP", "POS"]:
    EXPERIMENTS[g] = list(GROUPS[g])
for g in ["HIST", "REC", "BASIC", "GA", "OPP", "POS"]:
    EXPERIMENTS[f"FULL_MINUS_{g}"] = [f for f in FEATURES_ALL if f not in GROUPS[g]]

EXPECTED_FEATURES = {}
for g, fl in GROUPS.items():
    EXPECTED_FEATURES[g] = set(fl)
    EXPECTED_FEATURES[f"FULL_MINUS_{g}"] = set(FEATURES_ALL) - set(fl)

# Pre-check estatico: conjunto de features de cada experimento deve bater
# exatamente com o grupo esperado. Nao adicionar/remover features.
for exp_name, feat_list in EXPERIMENTS.items():
    expected = EXPECTED_FEATURES[exp_name]
    if set(feat_list) != expected:
        raise RuntimeError(
            f"Conjunto de features invalido para {exp_name}: "
            f"faltam={sorted(expected - set(feat_list))}, "
            f"sobram={sorted(set(feat_list) - expected)}"
        )


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


# ------------------------------------------------------------- preprocess parametrizado
def preprocess_for_group(X_tr_raw, X_te_raw, features_exp):
    """
    X_tr_raw, X_te_raw: arrays (N, 21) na ordem FEATURES_ALL.
    features_exp: lista de strings do grupo, sempre na ordem:
      [num1, ..., numK, oh1, ..., ohM].
    Retorna (X_tr_proc, X_te_proc, meta).
    Imputa NaN numericos com media do treino.
    StandardScaler fit no treino.
    One-hot intacto (sem imputacao, sem padronizacao).
    """
    idx = [FEATURES_ALL.index(f) for f in features_exp]
    Xt = X_tr_raw[:, idx]
    Xv = X_te_raw[:, idx]

    n_num = sum(1 for f in features_exp if f in FEATURES_NUM)
    Xt_num = Xt[:, :n_num]
    Xv_num = Xv[:, :n_num]
    Xt_oh = Xt[:, n_num:] if n_num < Xt.shape[1] else np.zeros((len(Xt), 0))
    Xv_oh = Xv[:, n_num:] if n_num < Xv.shape[1] else np.zeros((len(Xv), 0))

    if n_num == 0:
        X_tr_final = Xt_oh
        X_te_final = Xv_oh
        return X_tr_final, X_te_final, {"n_num": 0, "zero_var_onehot": []}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(Xt_num, axis=0)
    means = np.where(np.isnan(means), 0.0, means)

    Xt_filled = Xt_num.copy()
    Xv_filled = Xv_num.copy()
    nan_t = np.isnan(Xt_filled)
    nan_v = np.isnan(Xv_filled)
    Xt_filled[nan_t] = np.take(means, np.where(nan_t)[1])
    Xv_filled[nan_v] = np.take(means, np.where(nan_v)[1])

    stds = Xt_filled.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    Xt_scaled = (Xt_filled - means) / stds
    Xv_scaled = (Xv_filled - means) / stds

    X_tr_final = np.hstack([Xt_scaled, Xt_oh]) if Xt_oh.shape[1] else Xt_scaled
    X_te_final = np.hstack([Xv_scaled, Xv_oh]) if Xv_oh.shape[1] else Xv_scaled

    zero_var = []
    if Xt_oh.shape[1]:
        oh_std = Xt_oh.std(axis=0)
        for i, c in enumerate(features_exp[n_num:]):
            if oh_std[i] == 0:
                zero_var.append(c)

    return X_tr_final, X_te_final, {"n_num": n_num, "zero_var_onehot": zero_var}


# ------------------------------------------------------------- main
def main():
    inicio = datetime.now()
    print("== Ridge v2 — Ablacao ==")

    # integridade
    h_antes = {}
    for nome, p in [("canonical", CANON), ("b2s_folds", rv2.B2S_FOLDS),
                    ("harness", HARNESS), ("p0_folds", B2S_P0_FOLDS)]:
        h = sha256(p)
        h_antes[nome] = h
        ok = "OK" if h == HASHES_ESPERADOS[nome] else "*** DIVERGE ***"
        print(f"  {nome:10s} {h[:16]}...  {ok}")
    if any(h_antes[k] != HASHES_ESPERADOS[k] for k in HASHES_ESPERADOS):
        print("ERRO: hash divergente. Abortando.", file=sys.stderr)
        sys.exit(2)

    audit = load_yaml(AUDIT_YAML)

    # base
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("computando diffs...")
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    print("computando features pre-computaveis...")
    feats = rv2.compute_participation_features(df)
    feats = rv2.add_onehot(feats)

    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")

    b2s_folds = pd.read_csv(B2S_P0_FOLDS).set_index(["temporada", "rodada_alvo"])

    rows_fold = []
    rows_alphas = []
    rows_selection_aux = []

    for idx, (t, R) in enumerate(folds):
        mask_t = feats["temporada"] == t
        tr_o_all = feats[mask_t & (feats["rodada"] < R) & feats["elegivel"]]
        te_o_all = feats[mask_t & (feats["rodada"] == R) & feats["elegivel"]]
        tr_o = tr_o_all[tr_o_all["entrou_em_campo"].fillna(False)].copy()
        te_o = te_o_all[te_o_all["entrou_em_campo"].fillna(False)].copy()
        if len(tr_o) == 0 or len(te_o) == 0:
            raise RuntimeError(
                f"Fold invalido: {t}/R{R} — treino ou teste vazio. "
                f"n_train={len(tr_o)} n_test={len(te_o)}"
            )

        # ----- prep do outer
        pp_o, pg_o, b0_o = rv2.compute_priors(tr_o)
        tr_o = rv2.add_shrunk_features(tr_o, pp_o, pg_o, b0_o)
        te_o = rv2.add_shrunk_features(te_o, pp_o, pg_o, b0_o)
        X_tr_o_raw = tr_o[FEATURES_ALL].to_numpy(dtype=float)
        X_te_o_raw = te_o[FEATURES_ALL].to_numpy(dtype=float)
        y_tr_o = tr_o[TARGET].to_numpy(dtype=float)
        y_te_o = te_o[TARGET].to_numpy(dtype=float)

        # ----- prep dos rounds internos (compartilhado entre experimentos)
        rounds_t = sorted(feats[mask_t]["rodada"].unique())
        inner_rounds = [r for r in rounds_t if rv2.MIN_INNER_ROUNDS <= r < R]
        inner_data = []
        for r_i in inner_rounds:
            tr_i = feats[mask_t & (feats["rodada"] < r_i) & feats["elegivel"]]
            te_i = feats[mask_t & (feats["rodada"] == r_i) & feats["elegivel"]]
            tr_i = tr_i[tr_i["entrou_em_campo"].fillna(False)].copy()
            te_i = te_i[te_i["entrou_em_campo"].fillna(False)].copy()
            if len(tr_i) == 0 or len(te_i) == 0:
                continue
            pp, pg, b0i = rv2.compute_priors(tr_i)
            tr_i = rv2.add_shrunk_features(tr_i, pp, pg, b0i)
            te_i = rv2.add_shrunk_features(te_i, pp, pg, b0i)
            inner_data.append({
                "Xt_raw": tr_i[FEATURES_ALL].to_numpy(dtype=float),
                "Xv_raw": te_i[FEATURES_ALL].to_numpy(dtype=float),
                "yt": tr_i[TARGET].to_numpy(dtype=float),
                "yv": te_i[TARGET].to_numpy(dtype=float),
            })

        b2s_mae_fold = float(b2s_folds.loc[(t, R), "b2s_mae"])

        # ----- por experimento
        for exp_name, feat_list in EXPERIMENTS.items():
            # Protecao adicional por experimento (redundante com o pre-check
            # global, mas garante integridade mesmo se a spec for editada).
            if set(feat_list) != EXPECTED_FEATURES[exp_name]:
                raise RuntimeError(
                    f"Features do experimento {exp_name} nao batem com a spec."
                )
            inner_mean = {a: [] for a in rv2.ALPHAS}
            for d_in in inner_data:
                Xt_i, Xv_i, _ = preprocess_for_group(d_in["Xt_raw"], d_in["Xv_raw"], feat_list)
                for a in rv2.ALPHAS:
                    pred, _ = rv2.fit_predict_ridge(Xt_i, d_in["yt"], Xv_i, a)
                    inner_mean[a].append(rv2.mae(pred, d_in["yv"]))

            inner_agg = {a: float(np.mean(v)) if v else np.inf for a, v in inner_mean.items()}
            alpha_star = min(rv2.ALPHAS, key=lambda a: (inner_agg[a], a))

            for a in rv2.ALPHAS:
                rows_alphas.append({
                    "experimento": exp_name,
                    "temporada": t, "rodada_alvo": R,
                    "alpha": a,
                    "inner_mean_mae": inner_agg[a],
                    "selecionado": int(a == alpha_star),
                })

            X_tr_fin, X_te_fin, meta = preprocess_for_group(X_tr_o_raw, X_te_o_raw, feat_list)
            pred_final, _ = rv2.fit_predict_ridge(X_tr_fin, y_tr_o, X_te_fin, alpha_star)
            mae_v = rv2.mae(pred_final, y_te_o)

            rows_fold.append({
                "experimento": exp_name,
                "temporada": t,
                "rodada_alvo": R,
                "n_test": int(len(y_te_o)),
                "mae": mae_v,
                "rmse": rv2.rmse(pred_final, y_te_o),
                "spearman": rv2.spearman(pred_final, y_te_o),
                "mae_b2s": b2s_mae_fold,
                "delta_mae": mae_v - b2s_mae_fold,
                "alpha_star": float(alpha_star),
                "n_features": int(len(feat_list)),
            })

        if (idx + 1) % 10 == 0:
            print(f"  fold {idx+1}/{len(folds)}  {t}/R{R}")

    dfr = pd.DataFrame(rows_fold)
    dfa = pd.DataFrame(rows_alphas)

    # ----- FULL: le do ridge_v2_folds.csv (resultado oficial, nao retreinado)
    if not RIDGE_V2_FOLDS.exists():
        raise RuntimeError(
            f"FULL nao pode ser incorporado: {RIDGE_V2_FOLDS} ausente. "
            f"Rode ridge_v2.py primeiro."
        )
    full = pd.read_csv(RIDGE_V2_FOLDS)
    if len(full) != 76:
        raise RuntimeError(
            f"FULL invalido: {len(full)} linhas em {RIDGE_V2_FOLDS.name} "
            f"(esperado 76)."
        )
    chaves_full = set(zip(full["temporada"].astype(int),
                          full["rodada_alvo"].astype(int)))
    chaves_harness = set(folds)
    if chaves_full != chaves_harness:
        faltam = chaves_harness - chaves_full
        sobram = chaves_full - chaves_harness
        raise RuntimeError(
            f"FULL invalido: chaves divergem do harness. "
            f"faltam={sorted(faltam)} sobram={sorted(sobram)}"
        )

    full_linhas = []
    for _, r in full.iterrows():
        full_linhas.append({
            "experimento": "FULL",
            "temporada": int(r["temporada"]),
            "rodada_alvo": int(r["rodada_alvo"]),
            "n_test": int(r["n_test"]),
            "mae": float(r["mae"]),
            "rmse": float(r["rmse"]),
            "spearman": float(r["spearman"]),
            "mae_b2s": float(r["b2s_mae"]),
            "delta_mae": float(r["mae"]) - float(r["b2s_mae"]),
            "alpha_star": float(r["alpha_star"]),
            "n_features": len(FEATURES_ALL),
        })
    print(f"  FULL: {len(full_linhas)} folds carregados de {RIDGE_V2_FOLDS.name}")

    dfr = pd.concat([dfr, pd.DataFrame(full_linhas)], ignore_index=True)

    # ----- summary
    def summarize(sub):
        n = sub["n_test"].sum()
        mae_g = float((sub["mae"] * sub["n_test"]).sum() / n)
        rmse_g = float(np.sqrt((sub["rmse"] ** 2 * sub["n_test"]).sum() / n))
        sp_g = float(sub["spearman"].mean())
        better = int((sub["delta_mae"] < -1e-4).sum())
        equal = int((sub["delta_mae"].abs() <= 1e-4).sum())
        worse = int(len(sub) - better - equal)
        med_delta = float(sub["delta_mae"].median())
        return {
            "n_features": int(sub["n_features"].iloc[0]),
            "mae_global": mae_g,
            "rmse_global": rmse_g,
            "spearman_global": sp_g,
            "folds_better": better,
            "folds_equal": equal,
            "folds_worse": worse,
            "median_delta_mae": med_delta,
        }

    sum_rows = []
    for exp, sub in dfr.groupby("experimento"):
        row = {"experimento": exp}
        row.update(summarize(sub))
        sum_rows.append(row)
    df_sum = pd.DataFrame(sum_rows).sort_values("mae_global").reset_index(drop=True)

    # ----- por temporada
    season_rows = []
    for (exp, t), sub in dfr.groupby(["experimento", "temporada"]):
        n = sub["n_test"].sum()
        season_rows.append({
            "experimento": exp,
            "temporada": int(t),
            "n_test": int(n),
            "mae": float((sub["mae"] * sub["n_test"]).sum() / n),
            "rmse": float(np.sqrt((sub["rmse"] ** 2 * sub["n_test"]).sum() / n)),
            "spearman": float(sub["spearman"].mean()),
        })
    df_season = pd.DataFrame(season_rows)

    # ----- selection
    sel_rows = []
    for exp, feat_list in EXPERIMENTS.items():
        sub_a = dfa[dfa["experimento"] == exp]
        if len(sub_a) == 0:
            continue
        selected = sub_a[sub_a["selecionado"] == 1]["alpha"].to_numpy()
        sel_rows.append({
            "experimento": exp,
            "n_features": len(feat_list),
            "features": "|".join(feat_list),
            "alpha_median": float(np.median(selected)),
            "alpha_min": float(selected.min()),
            "alpha_max": float(selected.max()),
            "alpha_distinct": int(len(np.unique(selected))),
        })
    if full_linhas:
        alphas_full = np.array([r["alpha_star"] for r in full_linhas])
        sel_rows.append({
            "experimento": "FULL",
            "n_features": len(FEATURES_ALL),
            "features": "|".join(FEATURES_ALL),
            "alpha_median": float(np.median(alphas_full)),
            "alpha_min": float(alphas_full.min()),
            "alpha_max": float(alphas_full.max()),
            "alpha_distinct": int(len(np.unique(alphas_full))),
        })
    df_sel = pd.DataFrame(sel_rows)

    # ----- integridade
    h_depois = {nome: sha256(p) for nome, p in
                [("canonical", CANON), ("b2s_folds", rv2.B2S_FOLDS),
                 ("harness", HARNESS), ("p0_folds", B2S_P0_FOLDS)]}
    integridade_ok = all(h_antes[k] == h_depois[k] for k in h_antes)

    # ----- escrever artefatos
    dfr.to_csv(OUT_FOLDS, index=False)
    df_sum.to_csv(OUT_SUMMARY, index=False)
    df_season.to_csv(OUT_SEASON, index=False)
    df_sel.to_csv(OUT_SELECTION, index=False)
    dfa.to_csv(OUT_ALPHAS, index=False)

    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "n_folds": len(folds),
        "n_experimentos": int(dfr["experimento"].nunique()),
        "experimentos": sorted(dfr["experimento"].unique().tolist()),
        "n_ridge_fits": rv2.FIT_COUNTER["n_ridge_fits"],
        "hashes_antes": h_antes,
        "hashes_depois": h_depois,
        "hashes_esperados": HASHES_ESPERADOS,
        "integridade_ok": integridade_ok,
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=== RESUMO ABLACAO ===")
    print(df_sum.to_string(index=False))
    print(f"\nn_ridge_fits: {rv2.FIT_COUNTER['n_ridge_fits']}")
    print(f"integridade: {integridade_ok}")
    print("artefatos:")
    for p in [OUT_FOLDS, OUT_SUMMARY, OUT_SEASON, OUT_SELECTION, OUT_ALPHAS, OUT_VALID]:
        print(f"  {p.relative_to(ROOT)}")

    if not integridade_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
