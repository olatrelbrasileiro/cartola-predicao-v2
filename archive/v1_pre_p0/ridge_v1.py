#!/usr/bin/env python3
"""
Ridge v1 — features pre-rodada, walk-forward expansivo intra-temporada.
Benchmark: B2S (MAE 1.4016, Spearman 0.4617).
Spec: config/ridge_v1_spec.yaml
"""
import hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path.home()/"cartola-predicao-v2"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
CONFIG = ROOT/"config.yaml"
SPEC = ROOT/"config"/"ridge_v1_spec.yaml"
AUDIT_YAML = ROOT/"config"/"pontuacao_scouts_auditoria.yaml"
MANIFEST = ROOT/"reports"/"harness_folds_v2.csv"
B2S_FOLDS = ROOT/"reports"/"baseline_b2s_folds.csv"
REP = ROOT/"reports"

CSV_FOLDS = REP/"ridge_v1_folds.csv"
CSV_ALPHAS = REP/"ridge_v1_alphas.csv"
SUMMARY = REP/"ridge_v1_summary.md"

TARGET = "pontos_num"
FEAT_COLS = ["hist_mean","hist_std","hist_n","rec_mean_5","rec_mean_3",
             "rec_std_5","momentum","basic_mean","basic_rec_5",
             "ga_per_game","ga_rec_5","entrou_rate","is_cold_start"]


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def load_yaml(p):
    with open(p,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cumulative_diff(df, scout_cols):
    out = df.sort_values(["temporada","atleta_id","rodada"],
                         kind="mergesort").reset_index(drop=True)
    for col in scout_cols:
        ff = out.groupby(["temporada","atleta_id"], sort=False)[col].ffill()
        chave = out[["temporada","atleta_id"]].copy()
        chave["_ff"] = ff.values
        prev = chave.groupby(["temporada","atleta_id"], sort=False)["_ff"].shift(1)
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


def compute_features(df):
    """Features walk-forward por (temporada, atleta_id)."""
    rows = []
    df = df.reset_index(drop=True)
    for (t, aid), sub in df.groupby(["temporada","atleta_id"], sort=False):
        sub = sub.sort_values("rodada").reset_index(drop=False)
        n = len(sub)
        for i in range(n):
            row = sub.iloc[i]
            hist = sub.iloc[:i]
            hist = hist[hist["elegivel"]]
            d = {"orig_idx": int(row["index"])}
            if len(hist) == 0:
                d.update(dict(hist_mean=np.nan, hist_std=np.nan, hist_n=0,
                              rec_mean_5=np.nan, rec_mean_3=np.nan,
                              rec_std_5=np.nan, momentum=np.nan,
                              basic_mean=np.nan, basic_rec_5=np.nan,
                              ga_per_game=np.nan, ga_rec_5=np.nan,
                              entrou_rate=np.nan, is_cold_start=1))
            else:
                p = hist["pontos_num"].to_numpy(dtype=float)
                b = hist["basic_score"].to_numpy(dtype=float)
                rec5 = hist.tail(5)
                rec3 = hist.tail(3)
                d["hist_mean"] = float(p.mean())
                d["hist_std"] = float(p.std(ddof=0))
                d["hist_n"] = int(len(hist))
                d["rec_mean_5"] = float(rec5["pontos_num"].mean())
                d["rec_mean_3"] = float(rec3["pontos_num"].mean())
                d["rec_std_5"] = float(rec5["pontos_num"].std(ddof=0)) if len(rec5) >= 2 else 0.0
                d["momentum"] = d["rec_mean_5"] - d["hist_mean"]
                d["basic_mean"] = float(b.mean())
                d["basic_rec_5"] = float(rec5["basic_score"].mean())
                last = hist.iloc[-1]
                jog = float(last["jogos_num"])
                gac = float(last["scout_G"]) + float(last["scout_A"])
                d["ga_per_game"] = gac / max(jog, 1.0)
                sumGA = float((rec5["diff_scout_G"].fillna(0)
                               + rec5["diff_scout_A"].fillna(0)).sum())
                d["ga_rec_5"] = sumGA / len(rec5)
                d["entrou_rate"] = float(hist["entrou_em_campo"].mean())
                d["is_cold_start"] = 0
            rows.append(d)
    return pd.DataFrame(rows).set_index("orig_idx")


def fit_and_predict(X_tr, y_tr, X_te, alpha):
    """Ridge + StandardScaler em numpy puro. Equivalente ao sklearn."""
    mu_col = X_tr.mean(numeric_only=True)
    X_tr_f = X_tr.fillna(mu_col).to_numpy(dtype=float)
    X_te_f = X_te.fillna(mu_col).to_numpy(dtype=float)
    mu = X_tr_f.mean(axis=0)
    sd = X_tr_f.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    X_tr_s = (X_tr_f - mu) / sd
    X_te_s = (X_te_f - mu) / sd
    y_mean = float(y_tr.mean())
    y_c = y_tr - y_mean
    A = X_tr_s.T @ X_tr_s + alpha * np.eye(X_tr_s.shape[1])
    w = np.linalg.solve(A, X_tr_s.T @ y_c)
    return X_te_s @ w + y_mean


def mae(p, y): return float(np.mean(np.abs(p - y)))
def rmse(p, y): return float(np.sqrt(np.mean((p - y)**2)))
def spearman(p, y):
    if len(p) < 2: return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])


def process_fold(df, feats, t, R, alphas, min_inner):
    def subsets(target_r, is_target):
        mask_t = df["temporada"] == t
        if is_target:
            sub = df[mask_t & (df["rodada"] == target_r) & df["elegivel"]]
            sub = sub[~((sub["entrou_em_campo"] == False) & (sub[TARGET] != 0))]
        else:
            sub = df[mask_t & (df["rodada"] < target_r) & df["elegivel"]]
        return sub

    outer_tr = subsets(R, False)
    outer_te = subsets(R, True)

    X_tr = feats.loc[outer_tr.index, FEAT_COLS]
    y_tr = outer_tr[TARGET].to_numpy(dtype=float)
    X_te = feats.loc[outer_te.index, FEAT_COLS]
    y_te = outer_te[TARGET].to_numpy(dtype=float)

    # inner walk-forward
    inner_rounds = sorted([r for r in df[df["temporada"]==t]["rodada"].unique()
                           if min_inner <= r <= R-1])
    inner_maes = {a: [] for a in alphas}
    for r_i in inner_rounds:
        in_tr = subsets(r_i, False)
        in_te = subsets(r_i, True)
        if len(in_tr) == 0 or len(in_te) == 0:
            continue
        Xi_tr = feats.loc[in_tr.index, FEAT_COLS]
        yi_tr = in_tr[TARGET].to_numpy(dtype=float)
        Xi_te = feats.loc[in_te.index, FEAT_COLS]
        yi_te = in_te[TARGET].to_numpy(dtype=float)
        for a in alphas:
            p = fit_and_predict(Xi_tr, yi_tr, Xi_te, a)
            inner_maes[a].append(mae(p, yi_te))

    inner_mean = {a: float(np.mean(v)) if v else np.inf
                  for a, v in inner_maes.items()}
    alpha_star = min(inner_mean, key=inner_mean.get)

    # fit final no outer_train
    p_te = fit_and_predict(X_tr, y_tr, X_te, alpha_star)

    return {
        "temporada": int(t), "rodada_alvo": int(R),
        "n_train": int(len(outer_tr)),
        "n_test": int(len(outer_te)),
        "alpha_star": float(alpha_star),
        "mae": mae(p_te, y_te),
        "rmse": rmse(p_te, y_te),
        "spearman": spearman(p_te, y_te),
        "n_inner_rounds": len(inner_rounds),
        "n_cold_start": int((feats.loc[outer_te.index, "is_cold_start"] == 1).sum()),
    }, inner_mean


def main():
    inicio = datetime.now()
    print("== RIDGE v1 ==")

    h_pq_antes = sha256(PARQUET)
    h_b2s_antes = sha256(B2S_FOLDS)

    cfg = load_yaml(CONFIG)
    mtr = cfg["validation"]["min_train_rounds"]
    spec = load_yaml(SPEC)
    alphas = spec["ridge"]["grade_alpha"]
    min_inner = int(spec["ridge"]["min_inner_rounds"])
    audit = load_yaml(AUDIT_YAML)

    df = pd.read_parquet(PARQUET)
    df = df.sort_values(["temporada","rodada","atleta_id"]).reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("computando diffs cumulativos...")
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    extra = ["scout_G","scout_A"]
    df = apply_cumulative_diff(df, scout_cols + extra)
    df["basic_score"] = compute_basic_score(df, audit)

    print("computando features walk-forward...")
    feats = compute_features(df)
    print(f"  features: {feats.shape}")

    man = pd.read_csv(MANIFEST)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")

    rows = []
    alpha_rows = []
    for (t, R) in folds:
        res, inner_mean = process_fold(df, feats, t, R, alphas, min_inner)
        rows.append(res)
        for a in alphas:
            alpha_rows.append({
                "temporada": t, "rodada_alvo": R, "alpha": a,
                "inner_mean_mae": inner_mean[a],
                "selecionado": int(a == res["alpha_star"]),
            })

    dfr = pd.DataFrame(rows)
    dfa = pd.DataFrame(alpha_rows)
    dfr.to_csv(CSV_FOLDS, index=False)
    dfa.to_csv(CSV_ALPHAS, index=False)

    # comparar com B2S
    b2s = pd.read_csv(B2S_FOLDS).rename(columns={"b2s_mae":"b2s_mae_ref",
                                                  "b2s_spearman":"b2s_sp_ref"})
    m = dfr.merge(b2s[["temporada","rodada_alvo","b2s_mae_ref","b2s_sp_ref",
                       "n_test"]].rename(columns={"n_test":"b2s_n_test"}),
                  on=["temporada","rodada_alvo"], how="left")
    m["delta_mae_vs_b2s"] = m["mae"] - m["b2s_mae_ref"]
    m["ridge_melhor_que_b2s"] = m["delta_mae_vs_b2s"] < 0

    n = int(dfr["n_test"].sum())
    ridge_mae_w = float((m["mae"]*m["n_test"]).sum()/n)
    ridge_rmse_w = float(np.sqrt((m["rmse"]**2 * m["n_test"]).sum()/n))
    ridge_sp = float(m["spearman"].mean())
    b2s_mae_w = float((m["b2s_mae_ref"]*m["n_test"]).sum()/n)
    b2s_sp = float(m["b2s_sp_ref"].mean())

    n_ridge_better = int((m["delta_mae_vs_b2s"] < 0).sum())
    n_b2s_better = int((m["delta_mae_vs_b2s"] > 0).sum())
    n_tie = int((m["delta_mae_vs_b2s"] == 0).sum())
    med_delta = float(m["delta_mae_vs_b2s"].median())

    alpha_counts = dfr["alpha_star"].value_counts().sort_index().to_dict()

    por_temp = m.groupby("temporada").apply(lambda g: pd.Series({
        "n_folds": len(g),
        "ridge_mae": float((g["mae"]*g["n_test"]).sum()/g["n_test"].sum()),
        "b2s_mae": float((g["b2s_mae_ref"]*g["n_test"]).sum()/g["n_test"].sum()),
        "delta": float(((g["mae"]-g["b2s_mae_ref"])*g["n_test"]).sum()/g["n_test"].sum()),
        "alpha_medio": float(g["alpha_star"].mean()),
    })).reset_index()

    h_pq_dep = sha256(PARQUET)
    h_b2s_dep = sha256(B2S_FOLDS)
    imut = (h_pq_antes == h_pq_dep) and (h_b2s_antes == h_b2s_dep)

    # criterio
    criterio_mae = ridge_mae_w < 1.4016
    criterio_folds = n_ridge_better >= 39
    criterio_sp = ridge_sp >= 0.4617

    fim = datetime.now()
    L = []
    L.append("# Ridge v1 — summary\n\n")
    L.append(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- Folds: {len(dfr)}  Obs: {n}\n")
    L.append(f"- Spec: config/ridge_v1_spec.yaml\n\n")

    L.append("## Metricas oficiais\n\n")
    L.append(f"- Ridge  MAE = {ridge_mae_w:.4f}  RMSE = {ridge_rmse_w:.4f}  Spearman = {ridge_sp:.4f}\n")
    L.append(f"- B2S    MAE = {b2s_mae_w:.4f}  Spearman = {b2s_sp:.4f}\n")
    L.append(f"- Delta MAE (Ridge - B2S): {ridge_mae_w - b2s_mae_w:+.4f}\n\n")

    L.append("## Comparacao com B2S\n\n")
    L.append(f"- Folds Ridge < B2S: {n_ridge_better}/{len(dfr)}\n")
    L.append(f"- Folds Ridge > B2S: {n_b2s_better}/{len(dfr)}\n")
    L.append(f"- Empates: {n_tie}\n")
    L.append(f"- Mediana delta MAE: {med_delta:+.4f}\n\n")

    L.append("## Alpha selecionado\n\n")
    L.append(f"- media: {dfr['alpha_star'].mean():.3f}\n")
    L.append(f"- mediana: {dfr['alpha_star'].median():.3f}\n")
    L.append(f"- contagem: {alpha_counts}\n\n")

    L.append("## Por temporada\n\n")
    L.append(por_temp.to_string(index=False))
    L.append("\n\n")

    L.append("## Criterio de sucesso\n\n")
    L.append(f"- MAE < 1.4016: {'PASS' if criterio_mae else 'FAIL'} "
             f"({ridge_mae_w:.4f})\n")
    L.append(f"- Folds melhor >= 39: {'PASS' if criterio_folds else 'FAIL'} "
             f"({n_ridge_better}/{len(dfr)})\n")
    L.append(f"- Spearman >= 0.4617: {'PASS' if criterio_sp else 'FAIL'} "
             f"({ridge_sp:.4f})\n\n")

    L.append("## Imutabilidade\n\n")
    L.append(f"- Parquet: {'OK' if h_pq_antes == h_pq_dep else 'MUDOU'}\n")
    L.append(f"- B2S folds: {'OK' if h_b2s_antes == h_b2s_dep else 'MUDOU'}\n")

    SUMMARY.write_text("".join(L), encoding="utf-8")

    print()
    print("=== RESUMO RIDGE v1 ===")
    print(f"folds: {len(dfr)}  obs: {n}")
    print(f"Ridge MAE: {ridge_mae_w:.4f}  Spearman: {ridge_sp:.4f}")
    print(f"B2S   MAE: {b2s_mae_w:.4f}  Spearman: {b2s_sp:.4f}")
    print(f"delta: {ridge_mae_w - b2s_mae_w:+.4f}")
    print(f"folds Ridge<B2S: {n_ridge_better}/{len(dfr)}")
    print(f"alpha medio: {dfr['alpha_star'].mean():.3f}")
    print(f"imutavel: {imut}")


if __name__ == "__main__":
    main()
