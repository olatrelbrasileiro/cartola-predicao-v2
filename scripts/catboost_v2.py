#!/usr/bin/env python3
"""
CatBoost v2 — challenger nao linear sob metodologia P0-corrected.

Benchmark externo: B2S P0-corrected (reports/baselines_p0_corrected_folds.csv).
Spec: config/ridge_v2_spec.yaml (features identicas) + regras P0 do projeto.

Diferencas em relacao ao catboost_v1:
- treino e teste filtrados para entrou_em_campo == True;
- flag_ambigua_B removida da metrica primaria;
- janelas 3/5 usam ULTIMAS PARTICIPACOES, nao ultimas linhas;
- hist_n conta participacoes;
- priors calculados sobre participantes do treino;
- benchmark = P0-corrected, nao o antigo baseline_b2s_folds.csv;
- posicao_id nativa (categorical), sem one-hot;
- ablation COM vs SEM confronto em um unico artefato.

NAO executa tuning. NAO altera v1. NAO altera Parquet/harness/B2S.
"""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor


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
TEMPORAL_SAFE = INTERIM / "features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"
HARNESS = REP / "harness_folds_v2.csv"
P0_FOLDS = REP / "baselines_p0_corrected_folds.csv"

OUT_FOLDS = REP / "catboost_v2_folds.csv"
OUT_SUMMARY = REP / "catboost_v2_summary.md"
OUT_FI = REP / "catboost_v2_feature_importance.csv"
OUT_GI = REP / "catboost_v2_group_importance.csv"
OUT_ABL = REP / "catboost_v2_ablation.csv"
OUT_VALID = REP / "catboost_v2_validation.json"

TARGET = "pontos_num"

HASH_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
HASH_HARNESS = "236b7421ed83860d6921131d708b0508601a8be495ac35f175f91eaefca803a5"
HASH_P0 = "8b4f61851c227d8ee8c2c76aa38f8c6389656bbb8d1109ee909ccce2d77bb689"

CATBOOST_PARAMS = dict(
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

FEATURES_COM = [
    # Historico
    "hist_mean", "hist_std", "hist_n", "hist_shrunk_k7",
    # Recente
    "rec_mean_5", "rec_mean_3", "rec_std_5", "momentum",
    # Basico
    "basic_mean", "basic_rec_5",
    # GA
    "ga_per_game", "ga_rec_5", "ga_shrunk_k7",
    # Oportunidade
    "entrou_rate", "is_cold_start",
    # Posicao (categorica nativa)
    "posicao_id",
    # Confronto
    "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away",
]
CONFRONTO_FEATURES = {"p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"}
FEATURES_SEM = [f for f in FEATURES_COM if f not in CONFRONTO_FEATURES]

CAT_FEATURES = ["posicao_id"]

GRUPO_FEAT = {
    "Historico":    ["hist_mean", "hist_std", "hist_n", "hist_shrunk_k7"],
    "Recente":      ["rec_mean_5", "rec_mean_3", "rec_std_5", "momentum"],
    "Basico":       ["basic_mean", "basic_rec_5"],
    "GA":           ["ga_per_game", "ga_rec_5", "ga_shrunk_k7"],
    "Oportunidade": ["entrou_rate", "is_cold_start"],
    "Posicao":      ["posicao_id"],
    "Confronto":    ["p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"],
}

FIT_COUNTER = {"n_catboost_fits": 0}


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


def _load_ridge_v2():
    """Importa infraestrutura auditada do ridge_v2 (features de participacao,
    priors, shrinkage). Nao executa main()."""
    spec = importlib.util.spec_from_file_location(
        "ridge_v2_module", str(ROOT / "scripts" / "ridge_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rv2 = _load_ridge_v2()


# ------------------------------------------------------------- metricas
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


# ------------------------------------------------------------- fit
def fit_predict_catboost(X_tr, y_tr, X_te):
    """Treina CatBoost e devolve (pred, modelo). Conta o fit."""
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(X_tr, y_tr, cat_features=CAT_FEATURES)
    FIT_COUNTER["n_catboost_fits"] += 1
    return m.predict(X_te), m


# ------------------------------------------------------------- main
def main():
    inicio = datetime.now()
    print("== CatBoost v2 ==")

    # ---- 1. integridade antes
    h_antes = {}
    for nome, p, esperado in [
        ("canonical", CANON, HASH_CANON),
        ("harness",   HARNESS, HASH_HARNESS),
        ("p0_folds",  P0_FOLDS, HASH_P0),
    ]:
        h = sha256(p)
        h_antes[nome] = h
        ok = "OK" if h == esperado else "*** DIVERGE ***"
        print(f"  {nome:10s} {h[:16]}...  {ok}")
    if any(h_antes[k] != esp for k, esp in
           [("canonical", HASH_CANON), ("harness", HASH_HARNESS), ("p0_folds", HASH_P0)]):
        print("ERRO: hash divergente. Abortando.", file=sys.stderr)
        sys.exit(2)

    audit = load_yaml(AUDIT_YAML)

    # ---- 2. base canonica + diffs + basic_score
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("computando diffs...")
    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    # ---- 3. features de participacao (row-wise, so participacoes)
    print("computando features de participacao...")
    feats = rv2.compute_participation_features(df)

    # ---- 4. confronto temporal-safe
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id",
               "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    feats = feats.merge(ts[ts_cols],
                        on=["temporada", "rodada", "atleta_id"], how="left")

    # ---- 5. harness
    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")
    if len(folds) != 76:
        print(f"ERRO: {len(folds)} folds (esperado 76).", file=sys.stderr)
        sys.exit(2)

    # ---- 6. B2S P0-corrected
    b2s_folds = pd.read_csv(P0_FOLDS).set_index(["temporada", "rodada_alvo"])

    # ---- 7. loop
    rows_fold = []
    fi_com_acc = np.zeros(len(FEATURES_COM))
    n_folds_fi = 0

    for idx, (t, R) in enumerate(folds):
        mask_t = feats["temporada"] == t
        tr_all = feats[mask_t & (feats["rodada"] < R) & feats["elegivel"]]
        te_all = feats[mask_t & (feats["rodada"] == R) & feats["elegivel"]]

        # P0: treino e teste SO com participantes
        tr = tr_all[tr_all["entrou_em_campo"].fillna(False)].copy()
        te_part = te_all[te_all["entrou_em_campo"].fillna(False)].copy()

        # metrica: apenas participantes (ambiguas B sao nao participantes -> excluidas)
        te_metric = te_part.copy()
        n_B_round = int(((te_all["entrou_em_campo"].fillna(False) == False)
                         & (te_all[TARGET] != 0)).sum())

        if len(tr) == 0 or len(te_metric) == 0:
            raise RuntimeError(
                f"Fold invalido: {t}/R{R} — treino={len(tr)} test={len(te_metric)}")

        # ---- priors so do treino (participantes)
        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te_metric = rv2.add_shrunk_features(te_metric, prior_pos, prior_ga, b0_const)

        y_tr = tr[TARGET].to_numpy(dtype=float)
        y_te = te_metric[TARGET].to_numpy(dtype=float)

        # ---- CatBoost COM confronto
        X_tr_com = tr[FEATURES_COM]
        X_te_com = te_metric[FEATURES_COM]
        pred_com, model_com = fit_predict_catboost(X_tr_com, y_tr, X_te_com)
        fi_com_acc += np.array(model_com.get_feature_importance())
        n_folds_fi += 1
        mae_com = mae(pred_com, y_te)
        rmse_com = rmse(pred_com, y_te)
        sp_com = spearman(pred_com, y_te)

        # ---- CatBoost SEM confronto
        X_tr_sem = tr[FEATURES_SEM]
        X_te_sem = te_metric[FEATURES_SEM]
        pred_sem, _ = fit_predict_catboost(X_tr_sem, y_tr, X_te_sem)
        mae_sem = mae(pred_sem, y_te)
        rmse_sem = rmse(pred_sem, y_te)
        sp_sem = spearman(pred_sem, y_te)

        b2s_mae_fold = float(b2s_folds.loc[(t, R), "b2s_mae"])

        rows_fold.append({
            "temporada": t,
            "rodada_alvo": R,
            "n_train": int(len(tr)),
            "n_test_metric": int(len(te_metric)),
            "n_ambigua_B_test": n_B_round,
            "mae": mae_com,
            "rmse": rmse_com,
            "spearman": sp_com,
            "b2s_mae": b2s_mae_fold,
            "delta_mae": mae_com - b2s_mae_fold,
            "improvement": b2s_mae_fold - mae_com,
            "mae_sem_confronto": mae_sem,
            "rmse_sem_confronto": rmse_sem,
            "spearman_sem_confronto": sp_sem,
            "delta_mae_sem": mae_sem - b2s_mae_fold,
            "improvement_sem": b2s_mae_fold - mae_sem,
        })

        if (idx + 1) % 10 == 0:
            print(f"  fold {idx+1}/{len(folds)}  {t}/R{R}  "
                  f"MAE_COM={mae_com:.4f}  MAE_SEM={mae_sem:.4f}  B2S={b2s_mae_fold:.4f}")

    dfr = pd.DataFrame(rows_fold)
    dfr.to_csv(OUT_FOLDS, index=False)

    # ---- 8. ablation csv
    dfa = dfr[["temporada", "rodada_alvo",
               "mae", "mae_sem_confronto"]].rename(columns={
        "mae": "mae_com_confronto",
        "mae_sem_confronto": "mae_sem_confronto",
    })
    dfa["delta_sem_menos_com"] = dfa["mae_sem_confronto"] - dfa["mae_com_confronto"]
    dfa.to_csv(OUT_ABL, index=False)

    # ---- 9. agregacao global
    total_n = int(dfr["n_test_metric"].sum())
    mae_global = float((dfr["mae"] * dfr["n_test_metric"]).sum() / total_n)
    rmse_global = float(np.sqrt((dfr["rmse"]**2 * dfr["n_test_metric"]).sum() / total_n))
    sp_global = float(dfr["spearman"].mean())
    mae_sem_global = float((dfr["mae_sem_confronto"] * dfr["n_test_metric"]).sum() / total_n)
    b2s_mae_global = float((dfr["b2s_mae"] * dfr["n_test_metric"]).sum() / total_n)

    n_better = int((dfr["delta_mae"] < -1e-4).sum())
    n_equal = int((dfr["delta_mae"].abs() <= 1e-4).sum())
    n_worse = int(len(dfr) - n_better - n_equal)
    med_delta = float(dfr["delta_mae"].median())

    # ---- 10. por temporada
    por_temp_rows = []
    for t_ in sorted(dfr["temporada"].unique()):
        g = dfr[dfr["temporada"] == t_]
        n_t = g["n_test_metric"].sum()
        por_temp_rows.append({
            "temporada": int(t_),
            "n_test": int(n_t),
            "mae": float((g["mae"] * g["n_test_metric"]).sum() / n_t),
            "mae_sem": float((g["mae_sem_confronto"] * g["n_test_metric"]).sum() / n_t),
            "b2s_mae": float((g["b2s_mae"] * g["n_test_metric"]).sum() / n_t),
            "delta": float(((g["mae"] - g["b2s_mae"]) * g["n_test_metric"]).sum() / n_t),
            "spearman": float(g["spearman"].mean()),
        })
    por_temp = pd.DataFrame(por_temp_rows)

    # ---- 11. criterios de promocao (pre-registrados)
    c1 = mae_global < b2s_mae_global - 1e-3
    c2 = n_better >= 50
    c3 = all(
        por_temp[por_temp["temporada"] == t]["mae"].iloc[0]
        < por_temp[por_temp["temporada"] == t]["b2s_mae"].iloc[0]
        for t in [2024, 2025, 2026]
    )
    c4 = all(
        por_temp[por_temp["temporada"] == t]["mae"].iloc[0]
        <= 1.015 * por_temp[por_temp["temporada"] == t]["b2s_mae"].iloc[0]
        for t in [2024, 2025, 2026]
    )
    c5 = med_delta < -1e-3

    # ---- 11.5 integridade DEPOIS (computada ANTES da decisao)
    h_depois = {nome: sha256(p) for nome, p in
                [("canonical", CANON), ("harness", HARNESS), ("p0_folds", P0_FOLDS)]}
    integridade_ok = all(h_antes[k] == h_depois[k] for k in h_antes)

    # ---- 11.6 decisao final (nao alteravel depois de ver resultados)
    if all([c1, c2, c3, c4, c5]):
        decisao = "PROMOTE_CATBOOST_V2"
        motivo = "todos os 5 criterios atendidos"
    elif integridade_ok:
        decisao = "KEEP_B2S"
        motivo = "criterios de promocao nao foram todos atendidos"
    else:
        decisao = "INCONCLUSIVE"
        motivo = "falha de integridade/auditoria impede decisao confiavel"

    # ---- 12. feature importance
    fi_com = fi_com_acc / max(n_folds_fi, 1)
    fi_df = pd.DataFrame({"feature": FEATURES_COM, "importance": fi_com})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(OUT_FI, index=False)

    grp_rows = []
    for grupo, fgs in GRUPO_FEAT.items():
        imp = float(fi_df[fi_df["feature"].isin(fgs)]["importance"].sum())
        grp_rows.append({"grupo": grupo, "importance_total": imp,
                         "features": ", ".join(fgs)})
    pd.DataFrame(grp_rows).sort_values("importance_total", ascending=False).to_csv(
        OUT_GI, index=False)

    # ---- 14. validation.json
    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "folds": len(dfr),
        "n_test_total": total_n,
        "n_catboost_fits": FIT_COUNTER["n_catboost_fits"],
        "hashes_antes": h_antes,
        "hashes_depois": h_depois,
        "hashes_esperados": {
            "canonical": HASH_CANON, "harness": HASH_HARNESS, "p0_folds": HASH_P0,
        },
        "integridade_ok": integridade_ok,
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    # ---- 15. summary
    fim = datetime.now()
    L = []
    L.append("# CatBoost v2 — Summary\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- CatBoost: {__import__('catboost').__version__}\n")
    L.append(f"- Folds: {len(dfr)}\n")
    L.append(f"- Obs avaliadas (metric): {total_n}\n")
    L.append(f"- n_catboost_fits: {FIT_COUNTER['n_catboost_fits']}\n\n")

    L.append("## Parametros (pre-registrados)\n\n")
    for k, v in CATBOOST_PARAMS.items():
        L.append(f"- {k}: {v}\n")
    L.append(f"- cat_features: {CAT_FEATURES}\n\n")

    L.append("## Metricas globais\n\n")
    L.append(f"- **CatBoost v2 MAE ponderado: {mae_global:.4f}**\n")
    L.append(f"- CatBoost v2 RMSE ponderado: {rmse_global:.4f}\n")
    L.append(f"- CatBoost v2 Spearman medio: {sp_global:.4f}\n")
    L.append(f"- **B2S P0-corrected MAE: {b2s_mae_global:.4f}**\n")
    L.append(f"- Delta MAE (CB - B2S): {mae_global - b2s_mae_global:+.4f}\n")
    L.append(f"- Improvement: {b2s_mae_global - mae_global:+.4f}\n\n")

    L.append("## Consistencia por fold\n\n")
    L.append(f"- Folds CB < B2S: {n_better}/{len(dfr)}\n")
    L.append(f"- Folds empate: {n_equal}\n")
    L.append(f"- Folds CB > B2S: {n_worse}\n")
    L.append(f"- Mediana do delta: {med_delta:+.4f}\n\n")

    L.append("## Por temporada\n\n")
    L.append(por_temp.to_string(index=False))
    L.append("\n\n")

    L.append("## Criterios de promocao\n\n")
    L.append(f"- C1 (MAE global < B2S - 1e-3): {c1}\n")
    L.append(f"- C2 (>= 50/76 folds melhores): {c2} ({n_better}/76)\n")
    L.append(f"- C3 (3/3 temporadas melhores): {c3}\n")
    L.append(f"- C4 (nenhuma temporada > 1.5%): {c4}\n")
    L.append(f"- C5 (mediana delta < -1e-3): {c5}\n\n")
    L.append(f"**Decisao: {decisao}**\n\n")
    L.append(f"Motivo: {motivo}\n\n")

    L.append("## Ablation — COM vs SEM confronto\n\n")
    L.append(f"- MAE com confronto: {mae_global:.4f}\n")
    L.append(f"- MAE sem confronto: {mae_sem_global:.4f}\n")
    L.append(f"- Delta (sem - com): {mae_sem_global - mae_global:+.4f}\n\n")

    L.append("## Feature importance (media dos folds)\n\n")
    L.append(fi_df.to_string(index=False))
    L.append("\n\n## Importancia por grupo\n\n")
    L.append(pd.DataFrame(grp_rows).sort_values("importance_total", ascending=False).to_string(index=False))
    L.append("\n\n## Integridade\n\n")
    for k in h_antes:
        L.append(f"- {k}: {'OK' if h_antes[k] == h_depois[k] else 'ALTERADO'}\n")

    OUT_SUMMARY.write_text("".join(L), encoding="utf-8")

    print()
    print("=== RESUMO ===")
    print(f"CatBoost v2 MAE: {mae_global:.4f}  B2S: {b2s_mae_global:.4f}  "
          f"delta: {mae_global - b2s_mae_global:+.4f}")
    print(f"folds CB<B2S: {n_better}/76  igual: {n_equal}  pior: {n_worse}")
    print(f"n_catboost_fits: {FIT_COUNTER['n_catboost_fits']}")
    print(f"decisao: {decisao}")
    print(f"integridade: {integridade_ok}")
    print("artefatos:")
    for p in [OUT_FOLDS, OUT_SUMMARY, OUT_FI, OUT_GI, OUT_ABL, OUT_VALID]:
        print(f"  {p.relative_to(ROOT)}")

    if not integridade_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
