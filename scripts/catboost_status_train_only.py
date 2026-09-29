#!/usr/bin/env python3
"""
CatBoost status-train-only — EXPERIMENTO DE ABLACAO (import-based).

Unica diferenca vs catboost_v2.main():
    tr = tr_all[tr_all["entrou_em_campo"].fillna(False)].copy()
    tr = tr[tr["status_id"].isin([7, 2])].copy()   <<< UNICA LINHA NOVA

status_id NAO entra em FEATURES_COM/SEM.
status_id e' recuperado como METADATA apos compute_participation_features
via merge 1:1 com lookup construido sobre o canonico.
Todos os outros objetos vem de catboost_v2 (importados, nao copiados).

Uso:
    python scripts/catboost_status_train_only.py --smoke    # 1 fold
    python scripts/catboost_status_train_only.py            # 76 folds
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------- bootstrap
def detect_root():
    for c in [Path.home() / "cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c / "reports" / "harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz nao encontrada")


ROOT = detect_root()


def load_cbv2():
    spec = importlib.util.spec_from_file_location(
        "cbv2_ablation", str(ROOT / "scripts" / "catboost_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cbv2 = load_cbv2()

CANON         = cbv2.CANON
TEMPORAL_SAFE = cbv2.TEMPORAL_SAFE
AUDIT_YAML    = cbv2.AUDIT_YAML
HARNESS       = cbv2.HARNESS
P0_FOLDS      = cbv2.P0_FOLDS
HASH_CANON    = cbv2.HASH_CANON
HASH_HARNESS  = cbv2.HASH_HARNESS
HASH_P0       = cbv2.HASH_P0
TARGET        = cbv2.TARGET
rv2           = cbv2.rv2

REP = cbv2.REP
OUT_FOLDS   = REP / "catboost_status_train_only_folds.csv"
OUT_SUMMARY = REP / "catboost_status_train_only_summary.md"
OUT_VALID   = REP / "catboost_status_train_only_validation.json"

STATUS_TRAIN_OK = [7, 2]
STATUS_VALIDOS  = {2, 3, 5, 6, 7}
V2_FOLDS = REP / "catboost_v2_folds.csv"


# ------------------------------------------------------------- asserts
def sanity_asserts():
    assert "status_id" not in cbv2.FEATURES_COM, "status_id em FEATURES_COM"
    assert "status_id" not in cbv2.FEATURES_SEM, "status_id em FEATURES_SEM"
    assert "elegivel" not in cbv2.FEATURES_COM, "elegivel em FEATURES_COM"
    assert "elegivel" not in cbv2.FEATURES_SEM, "elegivel em FEATURES_SEM"
    assert cbv2.FIT_COUNTER.get("n_catboost_fits", -1) == 0, \
        "import disparou fits — revisar"


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="executa apenas o primeiro fold")
    args = ap.parse_args()

    inicio = datetime.now()
    print("== CatBoost status-train-only (import-based ablation) ==")
    print(f"smoke={args.smoke}")

    sanity_asserts()

    # ---- 1. integridade
    h_antes = {}
    for nome, p, esperado in [("canonical", CANON, HASH_CANON),
                              ("harness", HARNESS, HASH_HARNESS),
                              ("p0_folds", P0_FOLDS, HASH_P0)]:
        h = cbv2.sha256(p)
        h_antes[nome] = h
        ok = "OK" if h == esperado else "*** DIVERGE ***"
        print(f"  {nome:10s} {h[:16]}... {ok}")
    if any(h_antes[k] != esp for k, esp in
           [("canonical", HASH_CANON), ("harness", HASH_HARNESS),
            ("p0_folds", HASH_P0)]):
        print("ERRO: hash divergente. Abortando.", file=sys.stderr)
        sys.exit(2)

    audit = cbv2.load_yaml(AUDIT_YAML)

    # ---- 2. base canonica (mesmo pipeline do v2)
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)

    assert "status_id" in df.columns, "status_id ausente no canonico"
    assert df["status_id"].notna().all(), "status_id com nulos no canonico"
    assert set(df["status_id"].unique()).issubset(STATUS_VALIDOS), \
        f"status_id desconhecidos no canonico: {sorted(set(df['status_id'].unique()))}"
    print(f"  status_id no canonico: 100% presente, "
          f"codigos={sorted(df['status_id'].unique())}")

    # ---- 2b. LOOKUP de status ANTES de compute_participation_features
    #     (metadata pura; nao entra nas features)
    status_lookup = df[
        ["temporada", "rodada", "atleta_id", "status_id"]
    ].copy()
    assert len(status_lookup) == len(df), "lookup com cardinalidade errada"
    assert not status_lookup.duplicated(
        ["temporada", "rodada", "atleta_id"]).any(), \
        "lookup com chaves duplicadas"
    print(f"  status_lookup: {len(status_lookup)} linhas, "
          f"chaves unicas OK")

    # mesma linha do v2
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("computando diffs...")
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    print("computando features de participacao...")
    feats = rv2.compute_participation_features(df)

    # ---- 2c. RE-INJETAR status_id como metadata com checagem de cardinalidade
    n_before = len(feats)
    feats = feats.merge(
        status_lookup,
        on=["temporada", "rodada", "atleta_id"],
        how="left",
        validate="one_to_one",
    )
    assert len(feats) == n_before, (
        f"merge do status_lookup alterou cardinalidade: "
        f"{n_before} -> {len(feats)}")
    assert "status_id" in feats.columns, \
        "status_id nao entrou em feats"
    assert feats["status_id"].notna().all(), \
        "status_id com nulos apos merge"
    assert set(feats["status_id"].unique()).issubset(STATUS_VALIDOS), \
        f"status_id desconhecidos apos merge: {sorted(feats['status_id'].unique())}"
    print(f"  status_id re-injetado como metadata: {len(feats)} linhas, "
          f"sem alteracao de cardinalidade")

    # ---- 2d. testes negativos: status_id NAO e feature
    assert "status_id" not in cbv2.FEATURES_COM, "status_id em FEATURES_COM"
    assert "status_id" not in cbv2.FEATURES_SEM, "status_id em FEATURES_SEM"

    # merge do confronto temporal-safe (identico ao v2)
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id",
               "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    n_before_ts = len(feats)
    feats = feats.merge(ts[ts_cols],
                        on=["temporada", "rodada", "atleta_id"], how="left")
    # confronto e' left join: cardinalidade deve se manter
    assert len(feats) == n_before_ts, \
        f"merge do confronto alterou cardinalidade: {n_before_ts} -> {len(feats)}"

    # ---- 3. harness (mesmos 76 folds)
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"],
                                              man["rodada_alvo"])]
    print(f"folds: {len(folds)}")
    if len(folds) != 76:
        print(f"ERRO: {len(folds)} folds (esperado 76).", file=sys.stderr)
        sys.exit(2)
    if args.smoke:
        folds = folds[:1]
        print(f"  SMOKE: rodando apenas {folds}")

    # ---- 4. loop (espelha catboost_v2.main(), com 1 linha nova)
    rows_fold = []
    for idx, (t, R) in enumerate(folds):
        mask_t = feats["temporada"] == t
        tr_all = feats[mask_t & (feats["rodada"] < R) & feats["elegivel"]]
        te_all = feats[mask_t & (feats["rodada"] == R) & feats["elegivel"]]

        # identico ao v2
        tr = tr_all[tr_all["entrou_em_campo"].fillna(False)].copy()
        te_part = te_all[te_all["entrou_em_campo"].fillna(False)].copy()

        # <<< UNICA LINHA DO EXPERIMENTO >>>
        n_train_original = int(len(tr))
        tr = tr[tr["status_id"].isin(STATUS_TRAIN_OK)].copy()
        n_train_7_2 = int(len(tr))

        te_metric = te_part.copy()
        n_B_round = int(((te_all["entrou_em_campo"].fillna(False) == False)
                         & (te_all[TARGET] != 0)).sum())

        if n_train_7_2 == 0 or len(te_metric) == 0:
            raise RuntimeError(
                f"Fold invalido: {t}/R{R} — treino_7_2={n_train_7_2} "
                f"test={len(te_metric)}")

        # priors SOBRE O tr FILTRADO (parte do efeito experimental)
        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te_metric = rv2.add_shrunk_features(te_metric, prior_pos, prior_ga, b0_const)

        y_tr = tr[TARGET].to_numpy(dtype=float)
        y_te = te_metric[TARGET].to_numpy(dtype=float)

        pred_com, _ = cbv2.fit_predict_catboost(
            tr[cbv2.FEATURES_COM], y_tr, te_metric[cbv2.FEATURES_COM])
        pred_sem, _ = cbv2.fit_predict_catboost(
            tr[cbv2.FEATURES_SEM], y_tr, te_metric[cbv2.FEATURES_SEM])

        rows_fold.append({
            "temporada": int(t),
            "rodada": int(R),
            "n_train_original": n_train_original,
            "n_train_7_2": n_train_7_2,
            "pct_retido": round(n_train_7_2 / n_train_original * 100, 4)
                          if n_train_original else None,
            "n_test_metric": int(len(te_metric)),
            "n_B_round": n_B_round,
            "mae_com": cbv2.mae(pred_com, y_te),
            "rmse_com": cbv2.rmse(pred_com, y_te),
            "sp_com": cbv2.spearman(pred_com, y_te),
            "mae_sem": cbv2.mae(pred_sem, y_te),
            "rmse_sem": cbv2.rmse(pred_sem, y_te),
            "sp_sem": cbv2.spearman(pred_sem, y_te),
        })
        print(f"  fold {idx+1}/{len(folds)}: {t}/R{R} "
              f"n_tr={n_train_original}->{n_train_7_2} "
              f"({rows_fold[-1]['pct_retido']}%) "
              f"MAE={rows_fold[-1]['mae_com']:.4f}")

    folds_df = pd.DataFrame(rows_fold)

    if args.smoke:
        print("\n== SMOKE OK ==")
        print(folds_df.to_string(index=False))
        print(f"fits: {cbv2.FIT_COUNTER['n_catboost_fits']} (esperado 2)")
        return

    folds_df.to_csv(OUT_FOLDS, index=False)
    print(f"artefato: {OUT_FOLDS}")

    mae_exp = float(np.average(folds_df["mae_com"]))
    sp_exp = float(np.average(folds_df["sp_com"]))
    rmse_exp = float(np.average(folds_df["rmse_com"]))
    mae_sem = float(np.average(folds_df["mae_sem"]))
    sp_sem = float(np.average(folds_df["sp_sem"]))

    summary = {
        "catboost_status_train_only": {
            "mae_com": mae_exp, "rmse_com": rmse_exp, "spearman_com": sp_exp,
            "mae_sem": mae_sem, "spearman_sem": sp_sem,
            "n_folds": int(len(folds_df)),
            "n_fits": cbv2.FIT_COUNTER["n_catboost_fits"],
        },
        "retencao_treino": {
            "n_train_original_total": int(folds_df["n_train_original"].sum()),
            "n_train_7_2_total": int(folds_df["n_train_7_2"].sum()),
            "pct_retido_medio": float(folds_df["pct_retido"].mean()),
            "pct_retido_min": float(folds_df["pct_retido"].min()),
            "pct_retido_max": float(folds_df["pct_retido"].max()),
        },
    }

    comparacao = None
    if V2_FOLDS.exists():
        v2 = pd.read_csv(V2_FOLDS)
        key = [k for k in ["temporada", "rodada"] if k in v2.columns]
        mae_col = next((c for c in v2.columns if c in
                        ("mae_com", "mae", "mae_com_com")), None)
        sp_col = next((c for c in v2.columns if c in
                       ("sp_com", "spearman_com", "spearman")), None)
        if len(key) == 2 and mae_col:
            merged = folds_df.merge(
                v2[key + [mae_col] + ([sp_col] if sp_col else [])],
                on=key, suffixes=("_exp", "_v2"))
            d_mae = merged["mae_com"] - merged[mae_col]
            comparacao = {
                "n_folds": int(len(merged)),
                "mae_v2": float(v2[mae_col].mean()),
                "mae_exp": float(merged["mae_com"].mean()),
                "delta_mae_medio": float(d_mae.mean()),
                "delta_mae_mediano": float(d_mae.median()),
                "delta_mae_min": float(d_mae.min()),
                "delta_mae_max": float(d_mae.max()),
                "delta_mae_std": float(d_mae.std()),
                "folds_exp_melhor_mae": int((d_mae < 0).sum()),
                "folds_exp_pior_mae": int((d_mae > 0).sum()),
                "folds_empate_mae": int((d_mae == 0).sum()),
            }
            if sp_col:
                d_sp = merged["sp_com"] - merged[sp_col]
                comparacao.update({
                    "sp_v2": float(v2[sp_col].mean()),
                    "sp_exp": float(merged["sp_com"].mean()),
                    "delta_sp_medio": float(d_sp.mean()),
                    "folds_exp_melhor_sp": int((d_sp > 0).sum()),
                    "folds_exp_pior_sp": int((d_sp < 0).sum()),
                    "folds_empate_sp": int((d_sp == 0).sum()),
                })
        else:
            summary["aviso_comparacao"] = (
                f"v2_folds encontrado mas colunas-chave ausentes. "
                f"presentes={list(v2.columns)}")
    else:
        summary["aviso_comparacao"] = \
            f"{V2_FOLDS.name} nao encontrado — comparacao por fold nao executada."

    validation = {
        "timestamp": inicio.isoformat(),
        "modo": "import-based ablation",
        "catboost_v2_importado": str(ROOT / "scripts" / "catboost_v2.py"),
        "hash_canonical": h_antes["canonical"],
        "hash_harness": h_antes["harness"],
        "hash_p0_folds": h_antes["p0_folds"],
        "hash_canonical_ok": h_antes["canonical"] == HASH_CANON,
        "hash_harness_ok": h_antes["harness"] == HASH_HARNESS,
        "hash_p0_folds_ok": h_antes["p0_folds"] == HASH_P0,
        "features_com": list(cbv2.FEATURES_COM),
        "features_sem": list(cbv2.FEATURES_SEM),
        "cat_features": list(cbv2.CAT_FEATURES),
        "catboost_params": dict(cbv2.CATBOOST_PARAMS),
        "target": TARGET,
        "status_id_em_features": ("status_id" in cbv2.FEATURES_COM
                                  or "status_id" in cbv2.FEATURES_SEM),
        "status_treino": STATUS_TRAIN_OK,
        "n_folds": int(len(folds_df)),
        "summary": summary,
        "comparacao_v2": comparacao,
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)

    L = ["# CatBoost status-train-only — ablacao (import-based)\n\n",
         f"- Data: {inicio.isoformat()}\n",
         f"- Pipeline base: `scripts/catboost_v2.py` (importado, nao copiado)\n",
         f"- Unica divergencia: `tr = tr[tr['status_id'].isin([7, 2])]`\n",
         f"- status_id recuperado como METADATA (merge 1:1), nao como feature\n",
         f"- Harness: 76 folds\n\n",
         "## Retencao do treino\n\n| metrica | valor |\n|---|---|\n",
         f"| n_train_original | {summary['retencao_treino']['n_train_original_total']} |\n",
         f"| n_train_7_2 | {summary['retencao_treino']['n_train_7_2_total']} |\n",
         f"| pct_retido medio | {summary['retencao_treino']['pct_retido_medio']:.2f}% |\n\n",
         "## Performance\n\n| Modelo | MAE | RMSE | Spearman |\n|---|---|---|---|\n",
         f"| status-train (COM) | {mae_exp:.4f} | {rmse_exp:.4f} | {sp_exp:.4f} |\n",
         f"| status-train (SEM) | {mae_sem:.4f} | — | {sp_sem:.4f} |\n"]
    if comparacao:
        L.append(f"| CatBoost v2 (COM) | {comparacao['mae_v2']:.4f} | — | "
                 f"{comparacao.get('sp_v2', float('nan')):.4f} |\n")
    L.append(f"| B2S P0-corrected | 2.7485 | — | 0.2552 |\n\n")
    if comparacao:
        L.append("## Comparacao por fold vs v2 (COM)\n\n")
        L.append(f"- Folds: {comparacao['n_folds']}\n")
        L.append(f"- ΔMAE medio:   {comparacao['delta_mae_medio']:+.4f}\n")
        L.append(f"- ΔMAE mediano: {comparacao['delta_mae_mediano']:+.4f}\n")
        L.append(f"- ΔMAE min:     {comparacao['delta_mae_min']:+.4f}\n")
        L.append(f"- ΔMAE max:     {comparacao['delta_mae_max']:+.4f}\n")
        L.append(f"- ΔMAE std:     {comparacao['delta_mae_std']:.4f}\n\n")
        L.append(f"- Exp melhor MAE: {comparacao['folds_exp_melhor_mae']}/{comparacao['n_folds']}\n")
        L.append(f"- Exp pior MAE:   {comparacao['folds_exp_pior_mae']}/{comparacao['n_folds']}\n")
        L.append(f"- Empates MAE:    {comparacao['folds_empate_mae']}/{comparacao['n_folds']}\n")
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.writelines(L)

    print(f"artefato: {OUT_SUMMARY}")
    print(f"artefato: {OUT_VALID}")
    print(f"MAE exp (COM) = {mae_exp:.4f}")
    print(f"Spearman exp (COM) = {sp_exp:.4f}")
    if comparacao:
        print(f"ΔMAE medio vs v2 = {comparacao['delta_mae_medio']:+.4f}")
    print(f"tempo total: {datetime.now() - inicio}")


if __name__ == "__main__":
    main()
