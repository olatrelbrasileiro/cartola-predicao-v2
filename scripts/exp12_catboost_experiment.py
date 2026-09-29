#!/usr/bin/env python
"""
EXP-12 F3 - CatBoost + contexto F2.
Protocolo pooled acumulado identico a catboost_pooling.py.
FEATURES_EXP12 = 21 FEATURES_COM + 20 F2 = 41.
Nao toca catboost_v2.py nem artefatos oficiais.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz nao encontrada")

ROOT = detect_root()

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cbv2 = _load("cbv2_f3", ROOT/"scripts"/"catboost_v2.py")
rv2  = cbv2.rv2

CANON         = cbv2.CANON
TEMPORAL_SAFE = cbv2.TEMPORAL_SAFE
AUDIT_YAML    = cbv2.AUDIT_YAML
HARNESS       = cbv2.HARNESS
P0_FOLDS      = cbv2.P0_FOLDS
HASH_CANON    = cbv2.HASH_CANON
HASH_HARNESS  = cbv2.HASH_HARNESS
HASH_P0       = cbv2.HASH_P0
TARGET        = cbv2.TARGET
REP           = cbv2.REP

DS_EXP12  = ROOT/"data/interim/cartola_jogador_rodada_exp12.parquet"
OUTD      = ROOT/"reports/exp12"
OUT_PREDS = OUTD/"catboost_exp12_predictions.csv"
OUT_FOLDS = OUTD/"catboost_exp12_folds.csv"
OUT_SUM   = OUTD/"catboost_exp12_summary.md"
OUT_V2CMP = OUTD/"catboost_exp12_vs_v2.csv"
OUT_PCMP  = OUTD/"catboost_exp12_vs_pooling.csv"
V2_FOLDS  = REP/"catboost_v2_folds.csv"
POOL_FOLDS= REP/"catboost_pooling_folds.csv"

F2_FEATURES = [
    "team_goals_scored_last3","team_goals_scored_last5",
    "team_goals_conceded_last3","team_goals_conceded_last5",
    "team_clean_sheet_rate_last3","team_clean_sheet_rate_last5",
    "team_shots_last3","team_shots_last5",
    "team_shots_on_target_last3","team_shots_on_target_last5",
    "opp_goals_scored_last3","opp_goals_scored_last5",
    "opp_goals_conceded_last3","opp_goals_conceded_last5",
    "opp_clean_sheet_rate_last3","opp_clean_sheet_rate_last5",
    "opp_shots_last3","opp_shots_last5",
    "opp_shots_on_target_last3","opp_shots_on_target_last5",
]
assert len(F2_FEATURES) == 20

EXPECTED_FOLDS = 76


def pooled_metrics(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    mae = float(np.mean(np.abs(y-p)))
    rmse = float(np.sqrt(np.mean((y-p)**2)))
    sp = cbv2.spearman(p, y)
    return mae, rmse, sp


def try_compare(folds_df, other: Path, label: str):
    if not other.exists():
        return None
    o = pd.read_csv(other)
    s_c = next((c for c in o.columns if c.lower() in ("temporada","target_season")), None)
    r_c = next((c for c in o.columns if c.lower() in ("rodada","rodada_alvo","target_round")), None)
    m_c = next((c for c in o.columns if c.lower() in ("mae","mae_com","mae_com_com")), None)
    if not (s_c and r_c and m_c):
        print(f"[aviso] {label}: colunas nao achadas em {other.name}: {list(o.columns)[:8]}")
        return None
    o = o[[s_c, r_c, m_c]].rename(
        columns={s_c:"temporada", r_c:"rodada_alvo", m_c:f"mae_{label}"})
    m = folds_df.merge(o, on=["temporada","rodada_alvo"], how="inner")
    m["delta"] = m["mae"] - m[f"mae_{label}"]
    return dict(
        label=label, n=len(m),
        better=int((m["delta"] < 0).sum()),
        worse=int((m["delta"] > 0).sum()),
        mean_delta=float(m["delta"].mean()),
        df=m,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)

    print("== EXP-12 F3 - CatBoost + contexto F2 ==")
    print(f"smoke={args.smoke}")

    # integridade
    for nome, p, exp in [("canonical", CANON, HASH_CANON),
                         ("harness",   HARNESS, HASH_HARNESS),
                         ("p0_folds",  P0_FOLDS, HASH_P0)]:
        h = cbv2.sha256(p)
        ok = "OK" if h == exp else "*** DIVERGE ***"
        print(f"  {nome:10s} {h[:16]}... {ok}")
        if h != exp:
            print("ERRO: hash divergente.", file=sys.stderr); return 2

    audit = cbv2.load_yaml(AUDIT_YAML)

    # pipeline identico ao v2/pooling
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada","rodada","atleta_id"],
                        kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("computando diffs...")
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G","scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    print("computando features de participacao...")
    feats = rv2.compute_participation_features(df)

    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada","rodada","atleta_id",
               "p_home","p_draw","p_away","mando","favoritismo_home_away"]
    n0 = len(feats)
    feats = feats.merge(ts[ts_cols], on=["temporada","rodada","atleta_id"], how="left")
    assert len(feats) == n0, f"confronto alterou cardinalidade: {n0} -> {len(feats)}"

    # merge F2
    ctx_path = ROOT/"data/interim/exp12/team_context_features.parquet"
    if not ctx_path.exists():
        print(f"ERRO: {ctx_path} ausente. Rode F2 antes.", file=sys.stderr); return 2
    if "clube_id" not in feats.columns:
        # feats vem de rv2.compute_participation_features (sem clube_id);
        # recuperar do df canonico pelo _orig_idx (indice pre-sort).
        idx = feats["_orig_idx"].to_numpy()
        feats = feats.copy()
        feats["clube_id"] = df.loc[idx, "clube_id"].to_numpy()
    ctx = pd.read_parquet(
        ctx_path,
        columns=["temporada","rodada","clube_id"] + F2_FEATURES)
    if ctx.duplicated(subset=["temporada","rodada","clube_id"]).any():
        print("ERRO: F2 tem duplicatas em (temporada,rodada,clube_id).",
              file=sys.stderr); return 2
    n0 = len(feats)
    feats = feats.merge(ctx, on=["temporada","rodada","clube_id"],
                        how="left", validate="many_to_one")
    assert len(feats) == n0, f"merge F2 alterou cardinalidade: {n0} -> {len(feats)}"

    feats["f2_available"] = feats[F2_FEATURES].notna().any(axis=1)
    n_av = int(feats["f2_available"].sum())
    print(f"f2_available pos-merge: {n_av}/{len(feats)} ({n_av/len(feats):.1%})")

    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")
    if len(folds) != EXPECTED_FOLDS:
        print(f"ERRO: {len(folds)} folds (esperado {EXPECTED_FOLDS}).",
              file=sys.stderr); return 2
    if args.smoke:
        folds = folds[:2]
        print(f"  SMOKE: {folds}")

    FEATURES_EXP12 = cbv2.FEATURES_COM + F2_FEATURES
    assert len(FEATURES_EXP12) == 41
    assert len(set(FEATURES_EXP12)) == 41
    # hist_shrunk_k7 e ga_shrunk_k7 sao derivadas per-fold (dependem dos priors
    # do treino). NAO existem em feats antes de add_shrunk_features.
    DERIVED_PER_FOLD = {"hist_shrunk_k7", "ga_shrunk_k7"}
    miss = [c for c in FEATURES_EXP12
            if c not in feats.columns and c not in DERIVED_PER_FOLD]
    if miss:
        print(f"ERRO: features ausentes: {miss}", file=sys.stderr); return 2
    print(f"FEATURES_EXP12 = 41 (21 COM + 20 F2; 2 derivadas per-fold)")

    all_preds = []
    rows_fold = []
    t0 = datetime.now()
    for idx, (t, R) in enumerate(folds, 1):
        te_all = feats[(feats["temporada"] == t) &
                       (feats["rodada"] == R) &
                       feats["elegivel"]]
        tr_all = feats[
            ((feats["temporada"] < t) |
             ((feats["temporada"] == t) & (feats["rodada"] < R)))
            & feats["elegivel"]
        ]
        tr = tr_all[tr_all["entrou_em_campo"].fillna(False)].copy()
        te_part = te_all[te_all["entrou_em_campo"].fillna(False)].copy()
        te_metric = te_part.copy()
        if len(tr) == 0 or len(te_metric) == 0:
            print(f"fold {t}/R{R} vazio: tr={len(tr)} te={len(te_metric)}",
                  file=sys.stderr); return 2

        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te_metric = rv2.add_shrunk_features(te_metric, prior_pos, prior_ga, b0_const)
        # sanity pos-derivacao per-fold
        miss_tr = [c for c in FEATURES_EXP12 if c not in tr.columns]
        miss_te = [c for c in FEATURES_EXP12 if c not in te_metric.columns]
        if miss_tr or miss_te:
            print(f"ERRO fold {t}/R{R}: faltam em tr={miss_tr} te={miss_te}",
                  file=sys.stderr)
            return 2

        y_tr = tr[TARGET].to_numpy(dtype=float)
        y_te = te_metric[TARGET].to_numpy(dtype=float)
        pred, _ = cbv2.fit_predict_catboost(
            tr[FEATURES_EXP12], y_tr, te_metric[FEATURES_EXP12])

        mae_ = cbv2.mae(pred, y_te)
        rmse_ = cbv2.rmse(pred, y_te)
        sp_ = cbv2.spearman(pred, y_te)
        rows_fold.append(dict(
            temporada=int(t), rodada_alvo=int(R),
            n_train=int(len(tr)),
            n_train_temp_atual=int((tr["temporada"] == t).sum()),
            n_train_ant=int((tr["temporada"] < t).sum()),
            n_test=int(len(te_metric)),
            mae=mae_, rmse=rmse_, spearman=sp_,
        ))
        all_preds.append(pd.DataFrame({
            "temporada": te_metric["temporada"].values,
            "rodada":    te_metric["rodada"].values,
            "atleta_id": te_metric["atleta_id"].values,
            "posicao_id": te_metric["posicao_id"].values,
            "y_true": y_te, "y_pred": pred,
            "fold_id": f"{t}_R{R}",
            "f2_available": te_metric["f2_available"].values,
        }))
        print(f"  fold {idx}/{len(folds)}: {t}/R{R} n_tr={len(tr)} "
              f"(ant={int((tr['temporada']<t).sum())}) n_te={len(te_metric)} "
              f"MAE={mae_:.4f} SP={sp_:.4f}")

    elapsed = (datetime.now() - t0).total_seconds()
    folds_df = pd.DataFrame(rows_fold)
    preds_df = pd.concat(all_preds, ignore_index=True)

    mae_p, rmse_p, sp_p = pooled_metrics(preds_df["y_true"], preds_df["y_pred"])
    print(f"\nelapsed={elapsed:.1f}s  fits={cbv2.FIT_COUNTER['n_catboost_fits']}  "
          f"preds={len(preds_df)}")
    print(f"MAE pooled={mae_p:.4f}  RMSE={rmse_p:.4f}  SP={sp_p:.4f}")

    folds_df.to_csv(OUT_FOLDS, index=False)
    preds_df.to_csv(OUT_PREDS, index=False)
    print(f"artefatos:\n  {OUT_FOLDS}\n  {OUT_PREDS}")

    cmp_v2   = try_compare(folds_df, V2_FOLDS, "v2")
    cmp_pool = try_compare(folds_df, POOL_FOLDS, "pooling")
    if cmp_v2:
        cmp_v2["df"].to_csv(OUT_V2CMP, index=False)
        print(f"vs v2: n={cmp_v2['n']} better={cmp_v2['better']} "
              f"worse={cmp_v2['worse']} delta={cmp_v2['mean_delta']:.4f}")
    if cmp_pool:
        cmp_pool["df"].to_csv(OUT_PCMP, index=False)
        print(f"vs pooling: n={cmp_pool['n']} better={cmp_pool['better']} "
              f"worse={cmp_pool['worse']} delta={cmp_pool['mean_delta']:.4f}")

    with OUT_SUM.open("w", encoding="utf-8") as fh:
        fh.write("# EXP-12 F3 - CatBoost + contexto F2\n\n")
        fh.write(f"- base: {DS_EXP12.name}\n")
        fh.write(f"- features: 21 FEATURES_COM + 20 F2 = 41\n")
        fh.write(f"- CAT_FEATURES = {cbv2.CAT_FEATURES}\n")
        fh.write(f"- protocolo: (temporada < t) OR (temporada == t AND rodada < R) & elegivel\n")
        fh.write(f"- folds={len(folds_df)} fits={cbv2.FIT_COUNTER['n_catboost_fits']} "
                 f"preds={len(preds_df)} tempo={elapsed:.1f}s\n\n")
        fh.write(f"## Pooled\n\n- MAE = {mae_p:.4f}\n- RMSE = {rmse_p:.4f}\n"
                 f"- Spearman = {sp_p:.4f}\n\n")

        fh.write("## Cobertura F2 no dataset (pos-merge)\n\n")
        for s, g in feats.groupby("temporada"):
            fh.write(f"- {s}: {int(g['f2_available'].sum())}/{len(g)} "
                     f"({g['f2_available'].mean():.1%})\n")
        fh.write("\n")

        n_av_p = int(preds_df["f2_available"].sum())
        fh.write("## f2_available nas previsoes\n\n")
        fh.write(f"- com F2: {n_av_p}/{len(preds_df)} ({n_av_p/len(preds_df):.1%})\n")
        fh.write(f"- sem F2: {len(preds_df)-n_av_p}/{len(preds_df)} "
                 f"({(len(preds_df)-n_av_p)/len(preds_df):.1%})\n\n")

        for label, mask in [("Com F2", preds_df["f2_available"] == True),
                            ("Sem F2", preds_df["f2_available"] == False)]:
            g = preds_df[mask]
            if g.empty:
                fh.write(f"### {label}\n\n(0 linhas)\n\n"); continue
            m_, r_, sp_ = pooled_metrics(g["y_true"], g["y_pred"])
            fh.write(f"### {label}\n\n- n = {len(g)}\n- MAE = {m_:.4f}\n"
                     f"- RMSE = {r_:.4f}\n- Spearman = {sp_:.4f}\n\n")

        fh.write("## Por temporada\n\n")
        for s, g in preds_df.groupby("temporada"):
            m_, r_, sp_ = pooled_metrics(g["y_true"], g["y_pred"])
            fh.write(f"- {s}: n={len(g)} MAE={m_:.4f} RMSE={r_:.4f} "
                     f"SP={sp_:.4f}\n")
        fh.write("\n## Por posicao\n\n")
        for p, g in preds_df.groupby("posicao_id"):
            m_, r_, sp_ = pooled_metrics(g["y_true"], g["y_pred"])
            fh.write(f"- posicao {p}: n={len(g)} MAE={m_:.4f} RMSE={r_:.4f} "
                     f"SP={sp_:.4f}\n")
        fh.write("\n## Comparacao fold a fold\n\n")
        if cmp_v2:
            fh.write(f"### vs catboost_v2\n\n- folds comuns: {cmp_v2['n']}\n"
                     f"- exp12 melhor: {cmp_v2['better']}\n"
                     f"- exp12 pior: {cmp_v2['worse']}\n"
                     f"- delta medio (exp12 - v2): {cmp_v2['mean_delta']:.4f}\n\n")
        if cmp_pool:
            fh.write(f"### vs catboost_pooling (mesmo protocolo, sem F2)\n\n"
                     f"- folds comuns: {cmp_pool['n']}\n"
                     f"- exp12 melhor: {cmp_pool['better']}\n"
                     f"- exp12 pior: {cmp_pool['worse']}\n"
                     f"- delta medio (exp12 - pooling): {cmp_pool['mean_delta']:.4f}\n\n")
        fh.write("\n## Notas\n\n")
        fh.write("- Protocolo pooled acumulado identico a catboost_pooling.py.\n")
        fh.write("- Comparacao com v2 confunde protocolo (single-season vs pooled) + F2.\n")
        fh.write("- Comparacao com pooling isola o efeito das 20 F2 (se disponivel).\n")
        fh.write("- Sem decisao de promocao.\n")

    print(f"summary: {OUT_SUM}")
    for p in [OUT_FOLDS, OUT_PREDS, OUT_SUM]:
        print(f"sha256 {p.name} = {hashlib.sha256(p.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
