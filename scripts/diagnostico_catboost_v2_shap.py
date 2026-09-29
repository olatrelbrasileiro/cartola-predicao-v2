#!/usr/bin/env python3
"""
C4 — Diagnostico SHAP do CatBoost v2.

Pergunta: quais grupos de informacao o CatBoost v2 utiliza, e como esse uso se
distribui entre temporadas, posicoes e modelos (COM vs SEM confronto)?

Escopo:
  - SHAP nativo (model.get_feature_importance(type="ShapValues", data=Pool))
  - calculado SOMENTE nas observacoes de teste de cada fold (23.691 obs)
  - por modelo oficial (COM e SEM), com seed=42 e mesmos hiperparametros
  - agregacao GLOBAL ponderada por observacao (nao media de media de folds)
  - 7 grupos de features, nomes em portugues

Garantias:
  - NAO salva modelos
  - NAO faz subamostragem
  - NAO faz selecao de features
  - NAO altera Parquet, harness, benchmark, features, hiperparametros
  - NAO altera previsoes oficiais
  - ABORTA se tempo total > 30 min
  - ABORTA se qualquer predicao reproduzida divergir > 1e-9 do persistido
"""
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


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
PRED = REP / "catboost_v2_predictions.csv"
HARNESS = REP / "harness_folds_v2.csv"
AUDIT_YAML = CONFIG / "pontuacao_scouts_auditoria.yaml"

# Formato wide: 1 linha por (temporada, rodada, atleta_id).
# Colunas: com_<feature> (21 COM) + sem_<feature> (16 SEM) + base values.
# Total ~23.691 linhas (nao ~500k).
OUT_OBS = REP / "catboost_v2_shap_by_observation.parquet"
OUT_FI = REP / "catboost_v2_shap_feature_importance.csv"
OUT_GI = REP / "catboost_v2_shap_group_importance.csv"
OUT_CS = REP / "catboost_v2_shap_com_vs_sem.csv"
OUT_MD = REP / "catboost_v2_shap.md"
OUT_VALID = REP / "catboost_v2_shap_validation.json"

HASH_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
TOL = 1e-9
TIMEOUT_S = 1800  # 30 min

GRUPOS = {
    "historico":     ["hist_mean", "hist_std", "hist_n", "hist_shrunk_k7"],
    "recente":       ["rec_mean_5", "rec_mean_3", "rec_std_5", "momentum"],
    "basico_scouts": ["basic_mean", "basic_rec_5"],
    "g_a":           ["ga_per_game", "ga_rec_5", "ga_shrunk_k7"],
    "oportunidade":  ["entrou_rate", "is_cold_start"],
    "posicao":       ["posicao_id"],
    "confronto":     ["p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"],
}

POS_NAMES = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro",
             4: "Meia",    5: "Atacante", 6: "Tecnico"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ import
print("importando modulos oficiais...")
cb2 = _load_mod("catboost_v2", SCRIPTS / "catboost_v2.py")
rv2 = cb2.rv2

FEATURES_COM = list(cb2.FEATURES_COM)
FEATURES_SEM = list(cb2.FEATURES_SEM)
CATBOOST_PARAMS = dict(cb2.CATBOOST_PARAMS)
CAT_FEATURES = list(cb2.CAT_FEATURES)

print(f"  FEATURES_COM: {len(FEATURES_COM)}")
print(f"  FEATURES_SEM: {len(FEATURES_SEM)}")
print(f"  CAT_FEATURES: {CAT_FEATURES}")
print(f"  CATBOOST_PARAMS: {CATBOOST_PARAMS}")


# ------------------------------------------------------------------ agregacao
def agregar(shap_matrix, features, ids_df):
    """
    shap_matrix: (n, n_features+1) - ultima coluna = base value
    Retorna dois DataFrames:
      - per_obs: (n, 4 + n_features + 2) wide com shap por feature + base value
      - nothing else
    """
    n, m = shap_matrix.shape
    assert m == len(features) + 1, f"SHAP shape {m} != {len(features)+1}"
    shap_vals = shap_matrix[:, :-1]
    base_vals = shap_matrix[:, -1]
    per_obs = ids_df.copy().reset_index(drop=True)
    for i, f in enumerate(features):
        per_obs[f] = shap_vals[:, i]
    per_obs["_base_value"] = base_vals
    return per_obs


def estatisticas_por_feature(df_wide, features, modelo_prefix):
    """
    df_wide tem colunas (temporada, rodada, atleta_id, posicao_id, <features>, ...)
    Retorna lista de dicts: modelo | escopo | escopo_valor | feature | mean_abs | mean | std | p25 | p75
    """
    rows = []
    for f in features:
        s = df_wide[f].to_numpy(dtype=float)
        rows.append({
            "modelo": modelo_prefix,
            "escopo": "global",
            "escopo_valor": "all",
            "feature": f,
            "n_obs": int(len(s)),
            "mean_abs_shap": float(np.mean(np.abs(s))),
            "mean_shap": float(np.mean(s)),
            "std_shap": float(np.std(s)),
            "p25_shap": float(np.percentile(s, 25)),
            "p75_shap": float(np.percentile(s, 75)),
            "min_shap": float(np.min(s)),
            "max_shap": float(np.max(s)),
        })
    for f in features:
        for t, sub in df_wide.groupby("temporada"):
            s = sub[f].to_numpy(dtype=float)
            rows.append({
                "modelo": modelo_prefix,
                "escopo": "temporada",
                "escopo_valor": str(int(t)),
                "feature": f,
                "n_obs": int(len(s)),
                "mean_abs_shap": float(np.mean(np.abs(s))),
                "mean_shap": float(np.mean(s)),
                "std_shap": float(np.std(s)),
                "p25_shap": float(np.percentile(s, 25)),
                "p75_shap": float(np.percentile(s, 75)),
                "min_shap": float(np.min(s)),
                "max_shap": float(np.max(s)),
            })
    for f in features:
        for p, sub in df_wide.groupby("posicao_id"):
            s = sub[f].to_numpy(dtype=float)
            rows.append({
                "modelo": modelo_prefix,
                "escopo": "posicao",
                "escopo_valor": POS_NAMES.get(int(p), str(int(p))),
                "feature": f,
                "n_obs": int(len(s)),
                "mean_abs_shap": float(np.mean(np.abs(s))),
                "mean_shap": float(np.mean(s)),
                "std_shap": float(np.std(s)),
                "p25_shap": float(np.percentile(s, 25)),
                "p75_shap": float(np.percentile(s, 75)),
                "min_shap": float(np.min(s)),
                "max_shap": float(np.max(s)),
            })
    return rows


def agregar_grupos(df_fi):
    """
    Recebe o DataFrame de feature importance e adiciona agregacao por grupo.
    """
    rows = []
    for (modelo, escopo, escopo_valor), sub in df_fi.groupby(
            ["modelo", "escopo", "escopo_valor"]):
        total = float(sub["mean_abs_shap"].sum())
        for grupo, feats in GRUPOS.items():
            sub_g = sub[sub["feature"].isin(feats)]
            if len(sub_g) == 0:
                continue
            soma = float(sub_g["mean_abs_shap"].sum())
            rows.append({
                "modelo": modelo,
                "escopo": escopo,
                "escopo_valor": escopo_valor,
                "grupo": grupo,
                "n_features": int(len(sub_g)),
                "soma_mean_abs_shap": soma,
                "fracao_sobre_total": (soma / total) if total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    inicio = datetime.now()
    t0 = time.time()
    print("== C4 — SHAP CatBoost v2 ==")

    # -------- validacao de integridade
    if not CANON.exists():
        print("ERRO: Parquet ausente.", file=sys.stderr); sys.exit(2)
    if not PRED.exists():
        print("ERRO: catboost_v2_predictions.csv ausente.", file=sys.stderr)
        sys.exit(2)
    h_antes = sha256(CANON)
    print(f"Parquet SHA-256: {h_antes[:16]}...")
    if h_antes != HASH_CANON:
        print("ERRO: SHA do Parquet diverge.", file=sys.stderr); sys.exit(2)

    # -------- carregar pred persistido
    pred_df = pd.read_csv(PRED)
    print(f"predicoes persistidas: {len(pred_df)} linhas")
    if len(pred_df) != 23691:
        print(f"AVISO: esperado 23691, obtido {len(pred_df)}")

    # -------- reconstruir dataframe (mesmo pipeline oficial)
    audit = rv2.load_yaml(AUDIT_YAML)
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = rv2.apply_cumulative_diff(df, scout_cols + ["scout_G", "scout_A"])
    df["basic_score"] = rv2.compute_basic_score(df, audit)

    feats = rv2.compute_participation_features(df)
    ts = pd.read_parquet(TEMPORAL_SAFE)
    ts_cols = ["temporada", "rodada", "atleta_id",
               "p_home", "p_draw", "p_away", "mando", "favoritismo_home_away"]
    feats = feats.merge(ts[ts_cols], on=["temporada", "rodada", "atleta_id"],
                        how="left")
    print(f"  feats: {feats.shape}")

    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"  folds: {len(folds)}")

    # -------- loop dos folds
    obs_com = []
    obs_sem = []

    for idx, (t, R) in enumerate(folds):
        mask_t = feats["temporada"] == t
        tr_all = feats[mask_t & (feats["rodada"] < R) & feats["elegivel"]]
        te_all = feats[mask_t & (feats["rodada"] == R) & feats["elegivel"]]
        tr = tr_all[tr_all["entrou_em_campo"].fillna(False)].copy()
        te = te_all[te_all["entrou_em_campo"].fillna(False)].copy()

        if len(tr) == 0 or len(te) == 0:
            print(f"ERRO: fold vazio {t}/R{R}", file=sys.stderr); sys.exit(3)

        prior_pos, prior_ga, b0_const = rv2.compute_priors(tr)
        tr = rv2.add_shrunk_features(tr, prior_pos, prior_ga, b0_const)
        te = rv2.add_shrunk_features(te, prior_pos, prior_ga, b0_const)

        # ordenar por atleta_id (mesma ordem da persistencia)
        te = te.sort_values("atleta_id").reset_index(drop=True)
        ids_te = te["atleta_id"].astype(int).to_numpy()

        y_tr = tr[TARGET := "pontos_num"].to_numpy(dtype=float)

        # ---- model COM
        m_com = CatBoostRegressor(**CATBOOST_PARAMS)
        m_com.fit(tr[FEATURES_COM], y_tr, cat_features=CAT_FEATURES)
        pred_com = m_com.predict(te[FEATURES_COM])

        # ---- model SEM
        m_sem = CatBoostRegressor(**CATBOOST_PARAMS)
        m_sem.fit(tr[FEATURES_SEM], y_tr, cat_features=CAT_FEATURES)
        pred_sem = m_sem.predict(te[FEATURES_SEM])

        # ---- validar contra persistido
        persistido = pred_df[(pred_df["temporada"] == t) & (pred_df["rodada"] == R)].copy()
        persistido = persistido.set_index("atleta_id").sort_index()
        idx_ord = np.argsort(ids_te)
        ids_sorted = ids_te[idx_ord]

        if not np.array_equal(ids_sorted, persistido.index.to_numpy()):
            print(f"ERRO: ordem de atleta divergente {t}/R{R}", file=sys.stderr)
            sys.exit(3)

        pred_com_sorted = pred_com[idx_ord]
        pred_sem_sorted = pred_sem[idx_ord]

        ref_com = persistido["y_pred_catboost"].to_numpy()
        ref_sem = persistido["y_pred_catboost_sem"].to_numpy()
        diff_com = np.abs(pred_com_sorted - ref_com)
        diff_sem = np.abs(pred_sem_sorted - ref_sem)

        max_diff_com = float(diff_com.max())
        max_diff_sem = float(diff_sem.max())
        if max_diff_com > TOL or max_diff_sem > TOL:
            print(f"ERRO: fold {t}/R{R} nao reproduz predicoes.", file=sys.stderr)
            print(f"  max diff COM: {max_diff_com:.3e}", file=sys.stderr)
            print(f"  max diff SEM: {max_diff_sem:.3e}", file=sys.stderr)
            sys.exit(3)

        # ---- SHAP COM
        pool_com = Pool(te[FEATURES_COM], cat_features=CAT_FEATURES)
        shap_com = m_com.get_feature_importance(type="ShapValues", data=pool_com)
        if shap_com.shape != (len(te), len(FEATURES_COM) + 1):
            print(f"ERRO: SHAP COM shape {shap_com.shape}", file=sys.stderr)
            sys.exit(3)
        if not np.isfinite(shap_com).all():
            print(f"ERRO: SHAP COM contem NaN/inf no fold {t}/R{R}",
                  file=sys.stderr)
            sys.exit(3)

        # ---- SHAP SEM
        pool_sem = Pool(te[FEATURES_SEM], cat_features=CAT_FEATURES)
        shap_sem = m_sem.get_feature_importance(type="ShapValues", data=pool_sem)
        if shap_sem.shape != (len(te), len(FEATURES_SEM) + 1):
            print(f"ERRO: SHAP SEM shape {shap_sem.shape}", file=sys.stderr)
            sys.exit(3)
        if not np.isfinite(shap_sem).all():
            print(f"ERRO: SHAP SEM contem NaN/inf no fold {t}/R{R}",
                  file=sys.stderr)
            sys.exit(3)

        # ---- ids
        # A posicao real e metadado da observacao e NAO pode ser
        # sobrescrita pelo SHAP da feature "posicao_id".
        ids_df = pd.DataFrame({
            "temporada": int(t),
            "rodada": int(R),
            "atleta_id": ids_sorted,
            "posicao_real_id": te["posicao_id"].astype(int).to_numpy()[idx_ord],
        })

        # ---- agregar COM
        df_com_obs = pd.DataFrame({
            "temporada": ids_df["temporada"],
            "rodada": ids_df["rodada"],
            "atleta_id": ids_df["atleta_id"],
            "posicao_real_id": ids_df["posicao_real_id"],
        })
        for i, f in enumerate(FEATURES_COM):
            df_com_obs[f] = shap_com[:, i]
        df_com_obs["base_value"] = shap_com[:, -1]
        obs_com.append(df_com_obs)

        # ---- agregar SEM
        df_sem_obs = pd.DataFrame({
            "temporada": ids_df["temporada"],
            "rodada": ids_df["rodada"],
            "atleta_id": ids_df["atleta_id"],
            "posicao_real_id": ids_df["posicao_real_id"],
        })
        for i, f in enumerate(FEATURES_SEM):
            df_sem_obs[f] = shap_sem[:, i]
        df_sem_obs["base_value"] = shap_sem[:, -1]
        obs_sem.append(df_sem_obs)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  fold {idx+1}/{len(folds)}  {t}/R{R}  "
                  f"n={len(te)}  elapsed={elapsed:.1f}s  "
                  f"max_diff_com={max_diff_com:.1e}")

        # ---- watchdog
        elapsed = time.time() - t0
        if elapsed > TIMEOUT_S:
            print(f"ABORT: tempo excedido ({elapsed:.1f}s > {TIMEOUT_S}s) "
                  f"no fold {idx+1}/{len(folds)}.", file=sys.stderr)
            print("Nenhum resultado parcial persistido.", file=sys.stderr)
            sys.exit(4)

    # -------- concatenar
    df_com_all = pd.concat(obs_com, ignore_index=True)
    df_sem_all = pd.concat(obs_sem, ignore_index=True)
    print(f"obs COM: {len(df_com_all)}  SEM: {len(df_sem_all)}")

    # -------- persistir per-observation
    df_com_all_w = df_com_all.copy()
    df_com_all_w.columns = (
        ["temporada", "rodada", "atleta_id", "posicao_real_id"]
        + [f"com_{c}" for c in df_com_all.columns[4:]]
    )

    df_sem_all_w = df_sem_all.copy()
    df_sem_all_w.columns = (
        ["temporada", "rodada", "atleta_id", "posicao_real_id"]
        + [f"sem_{c}" for c in df_sem_all.columns[4:]]
    )

    df_obs = df_com_all_w.merge(
        df_sem_all_w,
        on=["temporada", "rodada", "atleta_id"],
        how="inner",
        suffixes=("_com", "_sem"),
    )

    # A chave da observacao e (temporada, rodada, atleta_id).
    # posicao_real_id e metadado da observacao e deve ser identico
    # entre os modelos COM e SEM.
    posicao_consistente = (
        df_obs["posicao_real_id_com"] == df_obs["posicao_real_id_sem"]
    )
    if not bool(posicao_consistente.all()):
        n_inconsistentes = int((~posicao_consistente).sum())
        print(
            f"ERRO: posicao_real_id divergente entre COM e SEM "
            f"em {n_inconsistentes} observacoes.",
            file=sys.stderr,
        )
        sys.exit(3)

    df_obs["posicao_id"] = df_obs["posicao_real_id_com"]
    df_obs.drop(
        columns=["posicao_real_id_com", "posicao_real_id_sem"],
        inplace=True,
    )

    df_obs.to_parquet(OUT_OBS, index=False)
    print(f"  {OUT_OBS.relative_to(ROOT)}")

    # -------- feature importance por escopo/modelo
    rows_fi = []
    rows_fi.extend(estatisticas_por_feature(df_com_all, FEATURES_COM, "COM"))
    rows_fi.extend(estatisticas_por_feature(df_sem_all, FEATURES_SEM, "SEM"))
    df_fi = pd.DataFrame(rows_fi)
    df_fi.to_csv(OUT_FI, index=False)

    # -------- grupos
    df_gi = agregar_grupos(df_fi)
    df_gi.to_csv(OUT_GI, index=False)

    # -------- COM vs SEM (features comuns)
    feats_comuns = [f for f in FEATURES_COM if f in FEATURES_SEM]
    rows_cs = []
    for escopo in ["global", "temporada", "posicao"]:
        for ev in df_fi[df_fi["escopo"] == escopo]["escopo_valor"].unique():
            for f in feats_comuns:
                c = df_fi[(df_fi["modelo"] == "COM")
                          & (df_fi["escopo"] == escopo)
                          & (df_fi["escopo_valor"] == ev)
                          & (df_fi["feature"] == f)]
                s = df_fi[(df_fi["modelo"] == "SEM")
                          & (df_fi["escopo"] == escopo)
                          & (df_fi["escopo_valor"] == ev)
                          & (df_fi["feature"] == f)]
                if len(c) == 0 or len(s) == 0:
                    continue
                rows_cs.append({
                    "escopo": escopo,
                    "escopo_valor": ev,
                    "feature": f,
                    "com_mean_abs_shap": float(c["mean_abs_shap"].iloc[0]),
                    "sem_mean_abs_shap": float(s["mean_abs_shap"].iloc[0]),
                    "delta_abs": float(c["mean_abs_shap"].iloc[0] - s["mean_abs_shap"].iloc[0]),
                })
    df_cs = pd.DataFrame(rows_cs)
    df_cs.to_csv(OUT_CS, index=False)

    # -------- validacao final
    h_depois = sha256(CANON)
    integridade = (h_depois == HASH_CANON)
    elapsed = time.time() - t0

    # -------- MD
    L = []
    L.append("# C4 — SHAP CatBoost v2\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {elapsed:.1f}s\n")
    L.append(f"- Observacoes SHAP: {len(df_obs)}\n")
    L.append(f"- Folds: {len(folds)}\n")
    L.append(f"- Parquet SHA-256: `{h_depois}`\n\n")

    L.append("## 1. Metodologia\n\n")
    L.append("- SHAP nativo: `get_feature_importance(type=\"ShapValues\", data=Pool(...))`.\n")
    L.append("- Calculado somente nas observacoes de teste de cada fold.\n")
    L.append("- Modelos treinados com `seed=42`, mesmos hiperparametros.\n")
    L.append("- Reproducao validada contra `catboost_v2_predictions.csv` "
             "em TODAS as observacoes dos 76 folds, tolerancia 1e-9.\n")
    L.append("- Agregacao global ponderada por observacao (nao media de medias).\n")
    L.append("- `posicao_id` categórica em `cat_features`.\n")
    L.append("- Ultima coluna do output SHAP e o base value (separado).\n\n")

    L.append("## 2. Importancia global por feature (COM)\n\n")
    sub = df_fi[(df_fi["modelo"] == "COM") & (df_fi["escopo"] == "global")]
    sub = sub.sort_values("mean_abs_shap", ascending=False)
    L.append("| feature | mean \\|SHAP\\| | mean SHAP | std |\n")
    L.append("|---|---|---|---|\n")
    for _, r in sub.iterrows():
        L.append(f"| {r['feature']} | {r['mean_abs_shap']:.4f} | "
                 f"{r['mean_shap']:+.4f} | {r['std_shap']:.4f} |\n")
    L.append("\n")

    L.append("## 3. Importancia global por feature (SEM)\n\n")
    sub = df_fi[(df_fi["modelo"] == "SEM") & (df_fi["escopo"] == "global")]
    sub = sub.sort_values("mean_abs_shap", ascending=False)
    L.append("| feature | mean \\|SHAP\\| | mean SHAP | std |\n")
    L.append("|---|---|---|---|\n")
    for _, r in sub.iterrows():
        L.append(f"| {r['feature']} | {r['mean_abs_shap']:.4f} | "
                 f"{r['mean_shap']:+.4f} | {r['std_shap']:.4f} |\n")
    L.append("\n")

    L.append("## 4. Importancia por grupo (global)\n\n")
    sub = df_gi[df_gi["escopo"] == "global"].sort_values(
        ["modelo", "soma_mean_abs_shap"], ascending=[True, False])
    L.append("| modelo | grupo | n_features | soma mean\\|SHAP\\| | fracao |\n")
    L.append("|---|---|---|---|---|\n")
    for _, r in sub.iterrows():
        L.append(f"| {r['modelo']} | {r['grupo']} | {r['n_features']} | "
                 f"{r['soma_mean_abs_shap']:.4f} | {r['fracao_sobre_total']:.4f} |\n")
    L.append("\n")

    L.append("## 5. Grupos por temporada\n\n")
    sub = df_gi[df_gi["escopo"] == "temporada"].sort_values(
        ["modelo", "escopo_valor", "soma_mean_abs_shap"],
        ascending=[True, True, False])
    L.append("| modelo | temporada | grupo | fracao |\n")
    L.append("|---|---|---|---|\n")
    for _, r in sub.iterrows():
        L.append(f"| {r['modelo']} | {r['escopo_valor']} | {r['grupo']} | "
                 f"{r['fracao_sobre_total']:.4f} |\n")
    L.append("\n")

    L.append("## 6. Grupos por posicao (COM)\n\n")
    sub = df_gi[(df_gi["escopo"] == "posicao")
                & (df_gi["modelo"] == "COM")].sort_values(
        ["escopo_valor", "soma_mean_abs_shap"], ascending=[True, False])
    L.append("| posicao | grupo | fracao |\n")
    L.append("|---|---|---|\n")
    for _, r in sub.iterrows():
        L.append(f"| {r['escopo_valor']} | {r['grupo']} | "
                 f"{r['fracao_sobre_total']:.4f} |\n")
    L.append("\n")

    L.append("## 7. COM vs SEM (features comuns, global)\n\n")
    sub = df_cs[df_cs["escopo"] == "global"].sort_values("delta_abs", ascending=False)
    L.append("| feature | mean\\|SHAP\\| COM | mean\\|SHAP\\| SEM | delta |\n")
    L.append("|---|---|---|---|\n")
    for _, r in sub.iterrows():
        L.append(f"| {r['feature']} | {r['com_mean_abs_shap']:.4f} | "
                 f"{r['sem_mean_abs_shap']:.4f} | {r['delta_abs']:+.4f} |\n")
    L.append("\n")

    L.append("## 8. Limitacoes\n\n")
    L.append("- SHAP reflete o uso das features pelo modelo, nao causalidade.\n")
    L.append("- Correlacao entre features divide contribuicao; SHAP nao a desfaz.\n")
    L.append("- Comparacoes COM vs SEM mostram reajuste do modelo, nao importancia causal.\n")
    L.append("- Nada aqui autoriza selecao, remocao ou tuning.\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    # -------- criterios PASS/FAIL
    sem_duplicatas_com = not df_com_all.duplicated(
        subset=["temporada", "rodada", "atleta_id"]).any()
    sem_duplicatas_sem = not df_sem_all.duplicated(
        subset=["temporada", "rodada", "atleta_id"]).any()

    features_grupos = [f for grupo in GRUPOS.values() for f in grupo]
    grupos_ok = (
        len(features_grupos) == len(set(features_grupos))
        and set(features_grupos) == set(FEATURES_COM)
    )
    features_grupos_sem = [
        f for grupo, feats in GRUPOS.items()
        if grupo != "confronto" for f in feats
    ]
    grupos_sem_ok = (
        len(features_grupos_sem) == len(set(features_grupos_sem))
        and set(features_grupos_sem) == set(FEATURES_SEM)
    )

    def _artefato_nao_vazio(path):
        return path.exists() and path.stat().st_size > 0

    criterios = {
        "parquet_sha_ok": bool(integridade),
        "predicoes_presentes": PRED.exists() and len(pred_df) > 0,
        "n_obs_shap_ok": len(df_obs) == 23691,
        "folds_76_ok": len(folds) == 76,
        "shap_com_21_features": len(FEATURES_COM) == 21,
        "shap_sem_16_features": len(FEATURES_SEM) == 16,
        "reproducao_predicoes_ok": True,
        "shap_com_finito": True,
        "shap_sem_finito": True,
        "sem_duplicatas_com": bool(sem_duplicatas_com),
        "sem_duplicatas_sem": bool(sem_duplicatas_sem),
        "participantes_only": True,
        "posicoes_validas": set(df_obs["posicao_id"].unique()).issubset(
            set(POS_NAMES.keys())),
        "grupos_cobrem_features_com": bool(grupos_ok),
        "grupos_cobrem_features_sem": bool(grupos_sem_ok),
        "tempo_ok": elapsed <= TIMEOUT_S,
        "artefato_obs_nao_vazio": _artefato_nao_vazio(OUT_OBS),
        "artefato_fi_nao_vazio": _artefato_nao_vazio(OUT_FI),
        "artefato_gi_nao_vazio": _artefato_nao_vazio(OUT_GI),
        "artefato_cs_nao_vazio": _artefato_nao_vazio(OUT_CS),
        "artefato_md_nao_vazio": _artefato_nao_vazio(OUT_MD),
        "sem_feature_selection": True,
    }
    pass_geral = all(criterios.values())

    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "duracao_s": float(elapsed),
        "parquet_sha256": h_depois,
        "parquet_sha_esperado": HASH_CANON,
        "n_observacoes_shap": int(len(df_obs)),
        "n_folds": int(len(folds)),
        "features_com": FEATURES_COM,
        "features_sem": FEATURES_SEM,
        "grupos": GRUPOS,
        "cat_features": CAT_FEATURES,
        "catboost_params": CATBOOST_PARAMS,
        "criterios": criterios,
        "resultado": "PASS" if pass_geral else "FAIL",
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=== RESUMO ===")
    print(f"observacoes SHAP: {len(df_obs)}")
    print(f"folds: {len(folds)}")
    print(f"duracao: {elapsed:.1f}s")
    print(f"resultado: {val['resultado']}")
    print("artefatos:")
    for p in [OUT_OBS, OUT_FI, OUT_GI, OUT_CS, OUT_MD, OUT_VALID]:
        if p.exists():
            print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")

    if not pass_geral:
        sys.exit(1)


if __name__ == "__main__":
    main()
