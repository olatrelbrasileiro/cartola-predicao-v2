#!/usr/bin/env python3
"""
Materializa predicoes por observacao — CatBoost v2 (COM/SEM confronto) e B2S.

Objetivo: reproduzir numericamente (dentro da tolerancia 1e-9) o experimento
oficial ja registrado em:
  - reports/catboost_v2_folds.csv
  - reports/baselines_p0_corrected_folds.csv

e salvar reports/catboost_v2_predictions.csv com:
  temporada, rodada, atleta_id, posicao_id,
  y_true, y_pred_catboost, y_pred_catboost_sem, y_pred_b2s, entrou_em_campo

OBS: y_pred_catboost_sem e' uma SAIDA ADICIONAL de diagnostico
     (ablation COM vs SEM confronto). Nao faz parte da especificacao
     minima.

GARANTIAS
- Reutiliza as FUNCOES OFICIAIS:
    baselines_p0_corrected.precompute_b2_all_rounds
    baselines_p0_corrected.calibrate_alpha_from_precomp
    baselines_p0_corrected.apply_cumulative_diff / compute_basic_score /
      compute_participation_features
    catboost_v2.FEATURES_COM / FEATURES_SEM / CATBOOST_PARAMS /
      fit_predict_catboost
    ridge_v2.compute_priors / add_shrunk_features
- NAO reimplementa B2S.
- NAO altera nenhum artefato oficial.
- ABORTA se mae, spearman, mae_sem, b2s_mae, b2s_spearman ou
  b2s_alpha_star divergirem > 1e-9 em qualquer fold.
"""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ paths
def detect_root():
    for c in [Path.home() / "cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path("/data/data/com.termux/files/home/cartola-predicao-v2")]:
        if (c / "data" / "interim" / "cartola_jogador_rodada_v2.parquet").exists():
            return c
    raise RuntimeError("Raiz do projeto nao encontrada")


ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"

HARNESS = REP / "harness_folds_v2.csv"
CB_V2_FOLDS = REP / "catboost_v2_folds.csv"
B2S_FOLDS = REP / "baselines_p0_corrected_folds.csv"

OUT_PRED = REP / "catboost_v2_predictions.csv"
OUT_VALID = REP / "catboost_v2_predictions_validation.json"

TARGET = "pontos_num"
TOL = 1e-9

HASH_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"


# ------------------------------------------------------------------ import oficial
def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("importando modulos oficiais...")
b0 = _load_mod("baselines_p0_corrected", SCRIPTS / "baselines_p0_corrected.py")
cb2 = _load_mod("catboost_v2", SCRIPTS / "catboost_v2.py")  # ja importa rv2 internamente
rv2 = cb2.rv2  # reusar a mesma referencia

print(f"  b0.K_SHRINK={b0.K_SHRINK}  b0.MIN_INNER_ROUNDS={b0.MIN_INNER_ROUNDS}")
print(f"  b0.ALPHAS={b0.ALPHAS}")
print(f"  cb2.FEATURES_COM: {len(cb2.FEATURES_COM)} features")
print(f"  cb2.FEATURES_SEM: {len(cb2.FEATURES_SEM)} features")
print(f"  cb2.CATBOOST_PARAMS: {cb2.CATBOOST_PARAMS}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ metricas
def mae(p, y): return float(np.mean(np.abs(p - y))) if len(y) else np.nan


def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan


def spearman(p, y):
    if len(p) < 2: return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ main
def main():
    inicio = datetime.now()
    print()
    print("== Persistencia de predicoes — CatBoost v2 + B2S ==")

    h_antes = sha256(CANON)
    print(f"Parquet SHA-256: {h_antes[:16]}...")
    if h_antes != HASH_CANON:
        print("ERRO: hash do Parquet diverge. Abortando.", file=sys.stderr)
        sys.exit(2)

    audit = b0.load_yaml(b0.AUDIT_YAML)

    # ---------------------------------------------------------------- B2S
    print()
    print("[1/2] construindo pipeline B2S com funcoes oficiais...")
    df_b = pd.read_parquet(CANON)
    df_b = df_b.sort_values(["temporada", "rodada", "atleta_id"],
                            kind="mergesort").reset_index(drop=True)
    df_b["elegivel"] = ~((df_b["temporada"] == 2025) & (df_b["rodada"] == 1))
    df_b["entrou_bool"] = df_b["entrou_em_campo"].fillna(False).astype(bool)

    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df_b = b0.apply_cumulative_diff(df_b, scout_cols)
    df_b["basic_score"] = b0.compute_basic_score(df_b, audit)

    feats_b = b0.compute_participation_features(df_b)
    df_b = df_b.merge(feats_b, left_index=True, right_index=True, how="left")
    print(f"  df_b: {df_b.shape}  colunas b0: "
          f"{[c for c in df_b.columns if c.endswith('_p')]}")

    precomp = b0.precompute_b2_all_rounds(df_b)
    print(f"  precomp (t,r): {len(precomp)} rodadas")

    # ---------------------------------------------------------------- CatBoost
    print()
    print("[2/2] construindo pipeline CatBoost com funcoes oficiais...")
    df_c = pd.read_parquet(CANON)
    df_c = df_c.sort_values(["temporada", "rodada", "atleta_id"],
                            kind="mergesort").reset_index(drop=True)
    df_c["elegivel"] = ~((df_c["temporada"] == 2025) & (df_c["rodada"] == 1))
    df_c = rv2.apply_cumulative_diff(df_c, scout_cols + ["scout_G", "scout_A"])
    df_c["basic_score"] = rv2.compute_basic_score(df_c, audit)

    feats_c = rv2.compute_participation_features(df_c)
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id",
               "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    feats_c = feats_c.merge(ts[ts_cols], on=["temporada", "rodada", "atleta_id"],
                            how="left")
    print(f"  feats_c: {feats_c.shape}")

    # ---------------------------------------------------------------- harness
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    if len(folds) != 76:
        print(f"ERRO: {len(folds)} folds != 76.", file=sys.stderr); sys.exit(2)
    print(f"  folds: {len(folds)}")

    # ---------------------------------------------------------------- loop
    print()
    print("rodando 76 folds...")
    pred_rows = []
    per_fold = []

    for idx, (t, R) in enumerate(folds):
        # ------- B2S (usa funcoes oficiais)
        entry = precomp[(t, R)]
        alpha_star, _ = b0.calibrate_alpha_from_precomp(t, R, precomp)
        pred_b2s = alpha_star * entry["b2"] + (1 - alpha_star) * entry["anchor"]
        y_b = entry["y"]

        # ------- CatBoost (usa rv2 + cb2)
        mask_t = feats_c["temporada"] == t
        tr_all = feats_c[mask_t & (feats_c["rodada"] < R) & feats_c["elegivel"]]
        te_all = feats_c[mask_t & (feats_c["rodada"] == R) & feats_c["elegivel"]]
        tr = tr_all[tr_all["entrou_em_campo"].fillna(False)].copy()
        te = te_all[te_all["entrou_em_campo"].fillna(False)].copy()

        if len(tr) == 0 or len(te) == 0:
            raise RuntimeError(f"fold invalido {t}/R{R}")

        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te = rv2.add_shrunk_features(te, prior_pos, prior_ga, b0_const)

        y_te = te[TARGET].to_numpy(dtype=float)
        y_tr = tr[TARGET].to_numpy(dtype=float)

        pred_com, _ = cb2.fit_predict_catboost(tr[cb2.FEATURES_COM], y_tr,
                                                te[cb2.FEATURES_COM])
        pred_sem, _ = cb2.fit_predict_catboost(tr[cb2.FEATURES_SEM], y_tr,
                                                te[cb2.FEATURES_SEM])

        # -- checagem de alinhamento B2S x CatBoost: n_test_metric deve ser o mesmo
        if len(y_b) != len(y_te):
            raise RuntimeError(
                f"{t}/R{R}: n B2S={len(y_b)} != n CatBoost={len(y_te)}")

        # -- B2S deve ser construido sobre o MESMO conjunto de participantes.
        #    O precomp foi feito com df_b (uma versao paralela). Vamos validar
        #    que o y_b efetivamente corresponde ao mesmo (atleta) no round.
        #    Para isso, reconstruimos atleta_ids do B2S via mascara equivalente.
        test_b = df_b[mask_t.rename("temporada").to_frame().assign(
            rodada=df_b["rodada"])["temporada"] == t]  # placeholder; ver abaixo
        # Extracao direta:
        mask_b = (df_b["temporada"] == t) & (df_b["rodada"] == R) & \
                 df_b["elegivel"] & df_b["entrou_bool"]
        ids_b = df_b.loc[mask_b, "atleta_id"].astype(int).to_numpy()

        # -- CatBoost: ids de test
        ids_te = te["atleta_id"].astype(int).to_numpy()

        # -- devem ser conjuntos iguais (mesma regra P0). Senao, algo esta errado.
        if set(ids_b) != set(ids_te):
            raise RuntimeError(
                f"{t}/R{R}: IDs B2S != IDs CatBoost "
                f"(faltam {len(set(ids_b) - set(ids_te))}, "
                f"sobram {len(set(ids_te) - set(ids_b))})")

        # -- ordena ambos pela mesma ordem de atleta_id para salvar consistente
        order = np.argsort(ids_te)
        ids_te = ids_te[order]
        pred_com = pred_com[order]
        pred_sem = pred_sem[order]
        y_te = y_te[order]
        pos_te = te["posicao_id"].astype(int).to_numpy()[order]

        # -- reordena B2S pela mesma chave de atleta
        order_b = np.argsort(ids_b)
        ids_b = ids_b[order_b]
        pred_b2s = pred_b2s[order_b]
        y_b = y_b[order_b]

        # -- checagem de identidade
        if not np.array_equal(ids_te, ids_b):
            raise RuntimeError(f"{t}/R{R}: ordem de atleta divergente")

        # -- sanity: y_true deve ser identico nos dois pipelines
        if not np.allclose(y_te, y_b, atol=1e-12):
            raise RuntimeError(f"{t}/R{R}: y_true divergente entre pipelines")

        for i in range(len(ids_te)):
            pred_rows.append({
                "temporada": int(t),
                "rodada": int(R),
                "atleta_id": int(ids_te[i]),
                "posicao_id": int(pos_te[i]),
                "y_true": float(y_te[i]),
                "y_pred_catboost": float(pred_com[i]),
                "y_pred_catboost_sem": float(pred_sem[i]),
                "y_pred_b2s": float(pred_b2s[i]),
                "entrou_em_campo": True,
            })

        per_fold.append({
            "temporada": t, "rodada_alvo": R, "n": int(len(ids_te)),
            "mae": mae(pred_com, y_te),
            "rmse": rmse(pred_com, y_te),
            "spearman": spearman(pred_com, y_te),
            "mae_sem_confronto": mae(pred_sem, y_te),
            "b2s_mae": mae(pred_b2s, y_te),
            "b2s_spearman": spearman(pred_b2s, y_te),
            "b2s_alpha_star": float(alpha_star),
        })

        if (idx + 1) % 10 == 0:
            print(f"  fold {idx+1}/{len(folds)}  {t}/R{R}  n={len(ids_te)}")

    df_pred = pd.DataFrame(pred_rows)
    df_m = pd.DataFrame(per_fold)

    # ---------------------------------------------------------------- validacao oficial
    print()
    print("=== VALIDACAO DA REPRODUCAO NUMERICA ===")
    oficial_cb = pd.read_csv(CB_V2_FOLDS).set_index(["temporada", "rodada_alvo"])
    oficial_b2s = pd.read_csv(B2S_FOLDS).set_index(["temporada", "rodada_alvo"])

    falhas = []
    metricas_checadas = 0
    for _, row in df_m.iterrows():
        t, R = int(row["temporada"]), int(row["rodada_alvo"])
        key = (t, R)

        checks = [
            ("mae",               "mae",               oficial_cb),
            ("spearman",          "spearman",          oficial_cb),
            ("mae_sem_confronto", "mae_sem_confronto", oficial_cb),
            ("b2s_mae",           "b2s_mae",           oficial_b2s),
            ("b2s_spearman",      "b2s_spearman",      oficial_b2s),
            ("b2s_alpha_star",    "b2s_alpha_star",    oficial_b2s),
        ]
        for col_of, col_my, src in checks:
            exp = float(src.loc[key, col_of])
            got = float(row[col_my])
            metricas_checadas += 1
            if abs(exp - got) > TOL:
                falhas.append(
                    f"{t}/R{R} [{col_of}]: exp={exp!r} got={got!r} "
                    f"diff={abs(exp-got):.2e}")

    if falhas:
        print(f"FALHAS: {len(falhas)} de {metricas_checadas} metricas")
        for f in falhas[:20]: print(f"  - {f}")
        if len(falhas) > 20: print(f"  ... (+{len(falhas)-20})")
        print()
        print("=== RESULTADO: ABORT ===")
        print("Predicoes NAO foram persistidas. Investigar.")
        sys.exit(3)

    print(f"OK: {metricas_checadas} metricas reproduzidas com |diff| <= {TOL}")
    print(f"     ({len(df_m)} folds x 6 metricas)")

    # ---------------------------------------------------------------- salvar
    df_pred.to_csv(OUT_PRED, index=False)
    print(f"salvo: {OUT_PRED.relative_to(ROOT)}  ({len(df_pred)} linhas)")

    h_depois = sha256(CANON)
    integridade_ok = (h_antes == h_depois == HASH_CANON)
    print(f"Parquet SHA-256 inalterado: {integridade_ok}")

    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "parquet_sha256": h_antes,
        "parquet_sha_esperado": HASH_CANON,
        "parquet_inalterado": bool(integridade_ok),
        "n_folds": int(len(df_m)),
        "n_observacoes": int(len(df_pred)),
        "tolerancia": TOL,
        "metricas_checadas": metricas_checadas,
        "folds_com_divergencia": 0,
        "resultado": "PASS",
        "arquivos_oficiais_sobrescritos": [],
        "metodo_reproducao": (
            "Reutiliza funcoes oficiais de baselines_p0_corrected.py "
            "(precompute_b2_all_rounds, calibrate_alpha_from_precomp) "
            "e catboost_v2.py (fit_predict_catboost, FEATURES_COM/SEM, "
            "CATBOOST_PARAMS). Nao reimplementa B2S."
        ),
        "y_pred_catboost_sem_nota": (
            "Saida adicional (ablation COM vs SEM confronto). "
            "Nao faz parte da especificacao minima."
        ),
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)
    print(f"salvo: {OUT_VALID.relative_to(ROOT)}")

    print()
    print(f"=== RESULTADO: PASS ===  "
          f"n_obs={len(df_pred)}  n_folds={len(df_m)}")


if __name__ == "__main__":
    main()
