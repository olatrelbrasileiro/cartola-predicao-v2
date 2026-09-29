#!/usr/bin/env python3
"""
CatBoost pooling cumulativo entre temporadas — EXPERIMENTO DE ABLACAO.

Unica diferenca vs catboost_v2.main():
    tr_all passa a incluir temporadas ANTERIORES (< t), com features
    INALTERADAS (resetam por temporada, como no v2).

Regra do treino por fold (t, R):
    (temporada < t)                          -> todas as rodadas disponiveis
    (temporada == t) & (rodada < R)          -> rodadas anteriores da atual
    & elegivel                               -> mesma regra do v2

Teste: identico ao v2 (temporada==t & rodada==R & elegivel).
Features: identicas (nao cruzam temporada — reset por (temporada, atleta_id)).
Priors: calculados sobre o tr POOLED (parte do efeito experimental).
status_id: fora das features; nao usado como filtro.

Para folds de 2024: nao existe t' < 2024 no canonical -> pool == v2.
Isso permite checagem bit-a-bit como sanidade.

Uso:
    python scripts/catboost_pooling.py --smoke    # 2024/R10 apenas
    python scripts/catboost_pooling.py            # 76 folds
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
        "cbv2_pooling", str(ROOT / "scripts" / "catboost_v2.py"))
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
OUT_FOLDS   = REP / "catboost_pooling_folds.csv"
OUT_SUMMARY = REP / "catboost_pooling_summary.md"
OUT_VALID   = REP / "catboost_pooling_validation.json"

V2_FOLDS = REP / "catboost_v2_folds.csv"


# ------------------------------------------------------------- asserts
def sanity_asserts():
    assert "status_id" not in cbv2.FEATURES_COM, "status_id em FEATURES_COM"
    assert "status_id" not in cbv2.FEATURES_SEM, "status_id em FEATURES_SEM"
    assert "elegivel" not in cbv2.FEATURES_COM, "elegivel em FEATURES_COM"
    assert "elegivel" not in cbv2.FEATURES_SEM, "elegivel em FEATURES_SEM"
    assert "status_id" not in cbv2.CAT_FEATURES, "status_id em CAT_FEATURES"
    assert cbv2.FIT_COUNTER.get("n_catboost_fits", -1) == 0,         "import disparou fits — revisar module-level de catboost_v2"


# ------------------------------------------------------------- helpers
def build_tr_all_v2(feats, t, R):
    """Treino single-season do v2 (diagnostico)."""
    return feats[(feats["temporada"] == t) &
                 (feats["rodada"] < R) &
                 feats["elegivel"]]


def build_tr_all_pooling(feats, t, R):
    """Treino pooled: temporadas < t (todas) + temporada t com rodada < R."""
    mask = (
        ((feats["temporada"] < t) |
         ((feats["temporada"] == t) & (feats["rodada"] < R)))
        & feats["elegivel"]
    )
    return feats[mask]


def load_v2_fold_row(t, R):
    """Le a linha do fold (t,R) de catboost_v2_folds.csv, se existir."""
    if not V2_FOLDS.exists():
        return None
    v2 = pd.read_csv(V2_FOLDS)
    key_t = next((c for c in v2.columns if c.lower() in ("temporada",)), None)
    key_r = next((c for c in v2.columns
                  if c.lower() in ("rodada", "rodada_alvo")), None)
    if not (key_t and key_r):
        return None
    row = v2[(v2[key_t] == t) & (v2[key_r] == R)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def extract_metrics_from_v2_row(row):
    """Extrai MAE/RMSE/Spearman do dict do CSV v2 com nomes tolerantes."""
    if row is None:
        return None
    def pick(names):
        for n in names:
            if n in row:
                try:
                    return float(row[n])
                except (TypeError, ValueError):
                    return None
        return None
    return {
        "mae": pick(["mae_com", "mae_com_com", "mae"]),
        "rmse": pick(["rmse_com", "rmse"]),
        "spearman": pick(["sp_com", "spearman_com", "spearman"]),
    }


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="executa apenas 2024/R10")
    args = ap.parse_args()

    inicio = datetime.now()
    print("== CatBoost pooling cumulativo entre temporadas ==")
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

    # mesma linha do v2
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("computando diffs...")
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    print("computando features de participacao...")
    feats = rv2.compute_participation_features(df)

    # features resetam por (temporada, atleta_id) -> nada a fazer aqui.
    # status_id NUNCA entra em feats para este experimento.

    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id",
               "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    n_before_ts = len(feats)
    feats = feats.merge(ts[ts_cols],
                        on=["temporada", "rodada", "atleta_id"], how="left")
    assert len(feats) == n_before_ts,         f"merge do confronto alterou cardinalidade: {n_before_ts} -> {len(feats)}"

    # ---- 3. harness
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"],
                                              man["rodada_alvo"])]
    print(f"folds: {len(folds)}")
    if len(folds) != 76:
        print(f"ERRO: {len(folds)} folds (esperado 76).", file=sys.stderr)
        sys.exit(2)
    if args.smoke:
        folds = [(2024, 10)]
        print(f"  SMOKE: rodando apenas {folds}")

    # ---- 4. loop
    rows_fold = []
    for idx, (t, R) in enumerate(folds):
        te_all = feats[(feats["temporada"] == t) &
                       (feats["rodada"] == R) &
                       feats["elegivel"]]

        tr_all_v2  = build_tr_all_v2(feats, t, R)
        tr_all_pool = build_tr_all_pooling(feats, t, R)

        # participantes (identico ao v2)
        tr_v2 = tr_all_v2[tr_all_v2["entrou_em_campo"].fillna(False)].copy()
        tr    = tr_all_pool[tr_all_pool["entrou_em_campo"].fillna(False)].copy()
        te_part = te_all[te_all["entrou_em_campo"].fillna(False)].copy()
        te_metric = te_part.copy()

        # para 2024: pool deve ser identico ao v2
        if t == 2024:
            assert len(tr) == len(tr_v2), (
                f"pool != v2 em 2024/R{R}: {len(tr)} vs {len(tr_v2)}")

        n_train_v2      = int(len(tr_v2))
        n_train_pooling = int(len(tr))
        n_train_temp_atual = int((tr["temporada"] == t).sum())
        n_train_ant        = int((tr["temporada"] < t).sum())

        n_B_round = int(((te_all["entrou_em_campo"].fillna(False) == False)
                         & (te_all[TARGET] != 0)).sum())

        if n_train_pooling == 0 or len(te_metric) == 0:
            raise RuntimeError(
                f"Fold invalido: {t}/R{R} — tr_pool={n_train_pooling} "
                f"te={len(te_metric)}")

        # priors SOBRE O tr pooled (efeito experimental)
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
            "rodada_alvo": int(R),
            "n_train_v2": n_train_v2,
            "n_train_pooling": n_train_pooling,
            "n_train_temporada_atual": n_train_temp_atual,
            "n_train_temporadas_anteriores": n_train_ant,
            "n_test_metric": int(len(te_metric)),
            "n_ambigua_B_test": n_B_round,
            "mae_com": cbv2.mae(pred_com, y_te),
            "rmse_com": cbv2.rmse(pred_com, y_te),
            "sp_com": cbv2.spearman(pred_com, y_te),
            "mae_sem_confronto": cbv2.mae(pred_sem, y_te),
            "rmse_sem_confronto": cbv2.rmse(pred_sem, y_te),
            "sp_sem_confronto": cbv2.spearman(pred_sem, y_te),
        })

        print(f"  fold {idx+1}/{len(folds)}: {t}/R{R} "
              f"n_tr_v2={n_train_v2} pool={n_train_pooling} "
              f"(ant={n_train_ant}) "
              f"n_te={rows_fold[-1]['n_test_metric']} "
              f"MAE={rows_fold[-1]['mae_com']:.4f} "
              f"SP={rows_fold[-1]['sp_com']:.4f}")

    folds_df = pd.DataFrame(rows_fold)

    # ---- 5. smoke: verificar compatibilidade com v2 em 2024/R10
    if args.smoke:
        print("\n== SMOKE OK ==")
        print(folds_df.to_string(index=False))
        print(f"fits: {cbv2.FIT_COUNTER['n_catboost_fits']} (esperado 2)")

        v2_row = load_v2_fold_row(2024, 10)
        if v2_row is not None:
            m = extract_metrics_from_v2_row(v2_row)
            print("\nComparacao com v2 (2024/R10) se disponivel:")
            print(f"  v2  : {m}")
            print(f"  pool: MAE={folds_df.iloc[0]['mae_com']:.4f} "
                  f"RMSE={folds_df.iloc[0]['rmse_com']:.4f} "
                  f"SP={folds_df.iloc[0]['sp_com']:.4f}")
        else:
            print("\n(v2_folds.csv nao encontrado ou fold ausente; "
                  "comparacao manual recomendada)")
        return

    # ---- 6. artefatos (full run)
    folds_df.to_csv(OUT_FOLDS, index=False)
    print(f"artefato: {OUT_FOLDS}")

    mae_exp  = float(np.average(folds_df["mae_com"]))
    sp_exp   = float(np.average(folds_df["sp_com"]))
    rmse_exp = float(np.average(folds_df["rmse_com"]))
    mae_sem  = float(np.average(folds_df["mae_sem_confronto"]))
    rmse_sem = float(np.average(folds_df["rmse_sem_confronto"]))
    sp_sem   = float(np.average(folds_df["sp_sem_confronto"]))

    # efeito por temporada
    por_temp = (folds_df.groupby("temporada")
                .agg(n_folds=("rodada_alvo", "count"),
                     mae_com=("mae_com", "mean"),
                     sp_com=("sp_com", "mean"),
                     n_train_pooling=("n_train_pooling", "mean"),
                     n_train_v2=("n_train_v2", "mean"))
                .reset_index())

    summary = {
        "catboost_pooling": {
            "mae_com": mae_exp, "rmse_com": rmse_exp, "spearman_com": sp_exp,
            "mae_sem": mae_sem, "rmse_sem": rmse_sem, "spearman_sem": sp_sem,
            "n_folds": int(len(folds_df)),
            "n_fits": cbv2.FIT_COUNTER["n_catboost_fits"],
        },
        "treino": {
            "n_train_pooling_medio": float(folds_df["n_train_pooling"].mean()),
            "n_train_v2_medio": float(folds_df["n_train_v2"].mean()),
            "n_train_temp_atual_medio":
                float(folds_df["n_train_temporada_atual"].mean()),
            "n_train_temporadas_anteriores_medio":
                float(folds_df["n_train_temporadas_anteriores"].mean()),
        },
        "por_temporada": por_temp.to_dict(orient="records"),
    }

    # ---- 7. comparacao com v2
    comparacao = None
    if V2_FOLDS.exists():
        v2 = pd.read_csv(V2_FOLDS)
        key_t = next((c for c in v2.columns if c.lower() == "temporada"), None)
        key_r = next((c for c in v2.columns
                      if c.lower() in ("rodada", "rodada_alvo")), None)
        mae_col = next((c for c in v2.columns
                        if c in ("mae_com", "mae_com_com", "mae")), None)
        sp_col  = next((c for c in v2.columns
                        if c in ("sp_com", "spearman_com", "spearman")), None)
        if key_t and key_r and mae_col:
            v2_r = v2.rename(columns={key_t: "temporada", key_r: "rodada_alvo",
                                      mae_col: "mae_v2"})
            cols = ["temporada", "rodada_alvo", "mae_v2"]
            if sp_col:
                v2_r = v2_r.rename(columns={sp_col: "sp_v2"})
                cols.append("sp_v2")
            merged = folds_df.merge(v2_r[cols],
                                    on=["temporada", "rodada_alvo"],
                                    how="inner")
            d_mae = merged["mae_com"] - merged["mae_v2"]
            comparacao = {
                "n_folds": int(len(merged)),
                "mae_v2": float(merged["mae_v2"].mean()),
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
            if "sp_v2" in merged.columns:
                d_sp = merged["sp_com"] - merged["sp_v2"]
                comparacao.update({
                    "sp_v2": float(merged["sp_v2"].mean()),
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
        summary["aviso_comparacao"] =             f"{V2_FOLDS.name} nao encontrado — comparacao nao executada."

    # ---- 8. validation.json
    validation = {
        "timestamp": inicio.isoformat(),
        "modo": "import-based pooling cumulativo",
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
        "n_folds": int(len(folds_df)),
        "summary": summary,
        "comparacao_v2": comparacao,
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)

    # ---- 9. summary.md
    L = [
        "# CatBoost pooling cumulativo entre temporadas — ablacao\n\n",
        f"- Data: {inicio.isoformat()}\n",
        "- Pipeline base: `scripts/catboost_v2.py` (importado)\n",
        "- Unica divergencia: `tr_all` inclui temporadas < t\n",
        "- Features: inalteradas (reset por temporada, atleta_id)\n",
        "- Teste: identico ao v2\n",
        "- status_id: fora das features, sem filtro\n",
        "- Harness: 76 folds\n\n",
        "## Treino (medias)\n\n",
        "| metrica | valor |\n",
        "|---|---|\n",
        f"| n_train_v2 medio | {summary['treino']['n_train_v2_medio']:.1f} |\n",
        f"| n_train_pooling medio | {summary['treino']['n_train_pooling_medio']:.1f} |\n",
        f"| n_train_temp_atual medio | {summary['treino']['n_train_temp_atual_medio']:.1f} |\n",
        f"| n_train_temporadas_anteriores medio | {summary['treino']['n_train_temporadas_anteriores_medio']:.1f} |\n\n",
        "## Performance\n\n",
        "| Modelo | MAE | RMSE | Spearman |\n",
        "|---|---|---|---|\n",
        f"| pooling (COM) | {mae_exp:.4f} | {rmse_exp:.4f} | {sp_exp:.4f} |\n",
        f"| pooling (SEM) | {mae_sem:.4f} | {rmse_sem:.4f} | {sp_sem:.4f} |\n",
    ]

    if comparacao:
        L.append(
            f"| CatBoost v2 (COM) | {comparacao['mae_v2']:.4f} | — | "
            f"{comparacao.get('sp_v2', float('nan')):.4f} |\n"
        )

    L.append("| B2S P0-corrected | 2.7485 | — | 0.2552 |\n\n")

    L.append("## Por temporada\n\n")
    L.append(
        "| temporada | n_folds | MAE pool | SP pool | "
        "n_train_v2 | n_train_pool |\n"
    )
    L.append("|---|---|---|---|---|---|\n")

    for r in summary["por_temporada"]:
        L.append(
            f"| {int(r['temporada'])} | {int(r['n_folds'])} | "
            f"{r['mae_com']:.4f} | {r['sp_com']:.4f} | "
            f"{r['n_train_v2']:.0f} | {r['n_train_pooling']:.0f} |\n"
        )

    if comparacao:
        L.append("\n## Comparacao por fold vs v2 (COM)\n\n")
        L.append(f"- Folds: {comparacao['n_folds']}\n")
        L.append(f"- ΔMAE medio:   {comparacao['delta_mae_medio']:+.4f}\n")
        L.append(f"- ΔMAE mediano: {comparacao['delta_mae_mediano']:+.4f}\n")
        L.append(f"- ΔMAE min:     {comparacao['delta_mae_min']:+.4f}\n")
        L.append(f"- ΔMAE max:     {comparacao['delta_mae_max']:+.4f}\n")
        L.append(f"- ΔMAE std:     {comparacao['delta_mae_std']:.4f}\n\n")
        L.append(
            f"- Exp melhor MAE: "
            f"{comparacao['folds_exp_melhor_mae']}/{comparacao['n_folds']}\n"
        )
        L.append(
            f"- Exp pior MAE:   "
            f"{comparacao['folds_exp_pior_mae']}/{comparacao['n_folds']}\n"
        )
        L.append(
            f"- Empates MAE:    "
            f"{comparacao['folds_empate_mae']}/{comparacao['n_folds']}\n"
        )

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.writelines(L)

    print(f"artefato: {OUT_SUMMARY}")
    print(f"artefato: {OUT_VALID}")
    print(f"MAE pool (COM)     = {mae_exp:.4f}")
    print(f"RMSE pool (COM)    = {rmse_exp:.4f}")
    print(f"Spearman pool (COM)= {sp_exp:.4f}")
    if comparacao:
        print(f"ΔMAE medio vs v2   = {comparacao['delta_mae_medio']:+.4f}")
    print(f"tempo total: {datetime.now() - inicio}")


if __name__ == "__main__":
    main()
