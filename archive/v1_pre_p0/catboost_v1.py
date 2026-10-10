#!/usr/bin/env python3
"""
CatBoost v1 — desafiante do B2S no harness walk-forward oficial.

21 features em 7 grupos. Config fixa pre-registrada. Sem tuning.
Ablation: com Confronto vs sem Confronto.

CACHE em 3 niveis:
  N1: dataframe preparado (diffs + features walk-forward)  -> .parquet
  N2: resultado por fold                                    -> .json
  N3: agregacoes                                            -> re-derivadas
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

# ------------------------------------------------------------------ paths
def _detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz do projeto nao encontrada")

ROOT = _detect_root()
INTERIM = ROOT/"data"/"interim"
CACHE = INTERIM/"cache"
CACHE.mkdir(parents=True, exist_ok=True)
REP = ROOT/"reports"
CONFIG = ROOT/"config"

CANON = INTERIM/"cartola_jogador_rodada_v2.parquet"
TEMPORAL_SAFE = INTERIM/"features_confronto_v1_temporal_safe.parquet"
AUDIT_YAML = CONFIG/"pontuacao_scouts_auditoria.yaml"
MANIFEST = REP/"harness_folds_v2.csv"
B2S_FOLDS = REP/"baseline_b2s_folds.csv"

CSV_FOLDS = REP/"catboost_v1_folds.csv"
CSV_SUMMARY = REP/"catboost_v1_summary.md"
CSV_FEAT_IMP = REP/"catboost_v1_feature_importance.csv"
CSV_GRP_IMP = REP/"catboost_v1_group_importance.csv"
CSV_ABL = REP/"catboost_v1_ablation.csv"

CACHE_N1 = CACHE/"catboost_v1_prepared.parquet"
CACHE_N2_DIR = CACHE/"catboost_v1_folds"
CACHE_N2_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MANIFEST = CACHE/"catboost_v1_cache_manifest.json"

CACHE_VERSION = "v1"

TARGET = "pontos_num"
K_SHRINK = 7
RECENT_WINDOW = 5
SEED = 42

FEATURES_COM = [
    "hist_mean","hist_std","hist_n","hist_shrunk",
    "rec_mean_5","rec_mean_3","rec_std_5","momentum",
    "basic_mean","basic_rec_5",
    "ga_per_game","ga_rec_5","ga_shrunk",
    "entrou_rate","is_cold_start",
    "posicao_id",
    "p_home","p_draw","p_away","mando","favoritismo_home_away",
]
FEATURES_SEM = [f for f in FEATURES_COM
                if f not in {"p_home","p_draw","p_away","mando","favoritismo_home_away"}]
CAT_FEATURES = ["posicao_id"]

GRUPO_FEAT = {
    "Histórico":     ["hist_mean","hist_std","hist_n","hist_shrunk"],
    "Recente":       ["rec_mean_5","rec_mean_3","rec_std_5","momentum"],
    "Básico":        ["basic_mean","basic_rec_5"],
    "G+A":           ["ga_per_game","ga_rec_5","ga_shrunk"],
    "Oportunidade":  ["entrou_rate","is_cold_start"],
    "Posição":       ["posicao_id"],
    "Confronto":     ["p_home","p_draw","p_away","mando","favoritismo_home_away"],
}

CATBOOST_PARAMS = dict(
    loss_function="MAE",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    thread_count=-1,
    verbose=0,
    allow_writing_files=False,
)


# ------------------------------------------------------------------ utils
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def hash_dict(d):
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ pipeline
def apply_cumulative_diff(df, cols):
    out = df.sort_values(["temporada","atleta_id","rodada"],
                         kind="mergesort").reset_index(drop=True)
    for col in cols:
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
                d.update(dict(
                    hist_mean=np.nan, hist_std=np.nan, hist_n=0,
                    rec_mean_5=np.nan, rec_mean_3=np.nan, rec_std_5=np.nan,
                    momentum=np.nan,
                    basic_mean=np.nan, basic_rec_5=np.nan,
                    ga_per_game=np.nan, ga_rec_5=np.nan,
                    ga_rate_last=np.nan,
                    entrou_rate=np.nan, is_cold_start=1,
                ))
            else:
                p = hist["pontos_num"].to_numpy(dtype=float)
                b = hist["basic_score"].to_numpy(dtype=float)
                rec5 = hist.tail(RECENT_WINDOW)
                rec3 = hist.tail(3)
                d["hist_mean"] = float(p.mean())
                d["hist_std"]  = float(p.std(ddof=0))
                d["hist_n"]    = int(len(hist))
                d["rec_mean_5"] = float(rec5["pontos_num"].mean())
                d["rec_mean_3"] = float(rec3["pontos_num"].mean())
                d["rec_std_5"]  = float(rec5["pontos_num"].std(ddof=0)) if len(rec5) >= 2 else 0.0
                d["momentum"]   = d["rec_mean_5"] - d["hist_mean"]
                d["basic_mean"] = float(b.mean())
                d["basic_rec_5"] = float(rec5["basic_score"].mean())
                last = hist.iloc[-1]
                jog = float(last["jogos_num"]) if pd.notna(last["jogos_num"]) else 0.0
                gac = float(last["scout_G"]) + float(last["scout_A"])
                ga_rate = gac / max(jog, 1.0)
                d["ga_per_game"] = ga_rate
                d["ga_rate_last"] = ga_rate
                sumGA = float((rec5["diff_scout_G"].fillna(0)
                               + rec5["diff_scout_A"].fillna(0)).sum())
                d["ga_rec_5"] = sumGA / len(rec5) if len(rec5) else np.nan
                d["entrou_rate"] = float(hist["entrou_em_campo"].mean())
                d["is_cold_start"] = 0
            rows.append(d)
    return pd.DataFrame(rows).set_index("orig_idx")


def prepare_dataframe():
    """Nivel 1 de cache: carrega se hash de entradas bate."""
    h_canon = sha256(CANON)
    h_temporal = sha256(TEMPORAL_SAFE)
    h_audit = sha256(AUDIT_YAML)
    chave = {
        "cache_version": CACHE_VERSION,
        "h_canon": h_canon,
        "h_temporal": h_temporal,
        "h_audit": h_audit,
    }
    chave_hash = hash_dict(chave)

    if CACHE_N1.exists():
        with open(CACHE_MANIFEST, "r") as f:
            man = json.load(f)
        if man.get("n1_hash") == chave_hash:
            print(f"[CACHE HIT N1] carregando {CACHE_N1.name}")
            return pd.read_parquet(CACHE_N1), chave

    print("[CACHE MISS N1] computando dataframe (pode levar minutos)...")
    audit = load_yaml(AUDIT_YAML)
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada","rodada","atleta_id"], kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["flag_ambigua_B"] = (df["entrou_em_campo"] == False) & (df[TARGET] != 0)

    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    extra = ["scout_G","scout_A"]
    df = apply_cumulative_diff(df, scout_cols + extra)
    df["basic_score"] = compute_basic_score(df, audit)

    feats = compute_features(df)
    df = df.merge(feats, left_index=True, right_index=True, how="left")

    ts = pd.read_parquet(TEMPORAL_SAFE)
    cols_ts = ["temporada","rodada","atleta_id",
               "p_home","p_draw","p_away","mando","favoritismo_home_away",
               "motivo_mascara"]
    df = df.merge(ts[cols_ts], on=["temporada","rodada","atleta_id"], how="left")

    df.to_parquet(CACHE_N1, index=False)
    with open(CACHE_MANIFEST, "w") as f:
        json.dump({**chave, "n1_hash": chave_hash}, f, indent=2)
    print(f"[CACHE MISS N1] salvo {CACHE_N1.name}")
    return df, chave


# ------------------------------------------------------------------ metricas
def mae(p, y): return float(np.mean(np.abs(p - y)))
def rmse(p, y): return float(np.sqrt(np.mean((p - y)**2)))
def spearman(p, y):
    if len(p) < 2: return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])


def fit_predict_cat(train_df, test_df, features, cat_features):
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(train_df[features], train_df[TARGET], cat_features=cat_features)
    pred = m.predict(test_df[features])
    return pred, m


# ------------------------------------------------------------------ fold cache
def run_fold(df, t, R, b2s, cache_key_base):
    """Nivel 2 de cache: por fold."""
    cache_file = CACHE_N2_DIR/f"fold_{t}_R{R}_{cache_key_base}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    mask_t = df["temporada"] == t
    train_all = df[mask_t & (df["rodada"] < R) & df["elegivel"]].copy()
    test_all  = df[mask_t & (df["rodada"] == R) & df["elegivel"]].copy()

    prior_pos = train_all.groupby("posicao_id")[TARGET].mean()
    last_obs = (train_all.sort_values("rodada")
                .groupby("atleta_id").tail(1)[["atleta_id","posicao_id","ga_rate_last"]]
                .dropna(subset=["ga_rate_last"]))
    prior_ga = last_obs.groupby("posicao_id")["ga_rate_last"].mean()
    global_mean = float(train_all[TARGET].mean())
    global_ga = float(last_obs["ga_rate_last"].mean()) if len(last_obs) else 0.0

    def add_shrunk(s):
        s = s.copy()
        n = s["hist_n"].fillna(0).to_numpy(dtype=float)
        hm = s["hist_mean"].fillna(0).to_numpy(dtype=float)
        pp = s["posicao_id"].map(prior_pos).fillna(global_mean).to_numpy(dtype=float)
        s["hist_shrunk"] = np.where(n > 0, (n*hm + K_SHRINK*pp)/(n+K_SHRINK), np.nan)
        gr = s["ga_rate_last"].fillna(0).to_numpy(dtype=float)
        pg = s["posicao_id"].map(prior_ga).fillna(global_ga).to_numpy(dtype=float)
        s["ga_shrunk"] = np.where(n > 0, (n*gr + K_SHRINK*pg)/(n+K_SHRINK), np.nan)
        return s

    train_all = add_shrunk(train_all)
    test_all  = add_shrunk(test_all)
    test_metric = test_all[~test_all["flag_ambigua_B"]].copy()
    y = test_metric[TARGET].to_numpy(dtype=float)

    pred_c, model_c = fit_predict_cat(train_all, test_metric, FEATURES_COM, CAT_FEATURES)
    mae_c = mae(pred_c, y); rmse_c = rmse(pred_c, y); sp_c = spearman(pred_c, y)
    fi = model_c.get_feature_importance().tolist()

    pred_s, _ = fit_predict_cat(train_all, test_metric, FEATURES_SEM, CAT_FEATURES)
    mae_s = mae(pred_s, y); sp_s = spearman(pred_s, y)

    b2s_mae = float(b2s.loc[(t,R),"b2s_mae"])

    result = {
        "temporada": t, "rodada_alvo": R,
        "n_train": int(len(train_all)),
        "n_test": int(len(test_all)),
        "n_test_metric": int(len(test_metric)),
        "n_B_test": int(len(test_all) - len(test_metric)),
        "cb_mae": mae_c, "cb_rmse": rmse_c, "cb_spearman": sp_c,
        "cb_sem_confronto_mae": mae_s, "cb_sem_confronto_spearman": sp_s,
        "b2s_mae": b2s_mae,
        "delta_mae": mae_c - b2s_mae,
        "improvement": b2s_mae - mae_c,
        "feature_importance": fi,
    }
    with open(cache_file, "w") as f:
        json.dump(result, f)
    return result


# ------------------------------------------------------------------ main
def main():
    inicio = datetime.now()
    print("== CatBoost v1 ==")

    hash_canon_antes = sha256(CANON)
    hash_temporal_antes = sha256(TEMPORAL_SAFE)
    hash_b2s_antes = sha256(B2S_FOLDS)
    print(f"Parquet canonico: {hash_canon_antes[:16]}...")
    print(f"Temporal-safe:    {hash_temporal_antes[:16]}...")
    print(f"B2S folds:        {hash_b2s_antes[:16]}...")

    # N1
    df, chave_n1 = prepare_dataframe()
    print(f"features totais no dataframe: {df.shape[1]} colunas")

    man = pd.read_csv(MANIFEST)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")

    b2s = pd.read_csv(B2S_FOLDS).set_index(["temporada","rodada_alvo"])

    # chave N2: inclui params + feature set
    cache_key_base = hash_dict({
        "cache_version": CACHE_VERSION,
        "params": CATBOOST_PARAMS,
        "feat_com": FEATURES_COM,
        "feat_sem": FEATURES_SEM,
        "k_shrink": K_SHRINK,
        "seed": SEED,
    })
    print(f"N2 cache key: {cache_key_base}")

    # verificar quantos folds ja' estao em cache
    ja_cacheados = sum(
        1 for (t,R) in folds
        if (CACHE_N2_DIR/f"fold_{t}_R{R}_{cache_key_base}.json").exists()
    )
    print(f"folds ja em cache: {ja_cacheados}/{len(folds)}")

    rows = []
    feat_imp = np.zeros(len(FEATURES_COM))
    for i, (t, R) in enumerate(folds):
        r = run_fold(df, t, R, b2s, cache_key_base)
        rows.append({k: v for k, v in r.items() if k != "feature_importance"})
        feat_imp += np.array(r["feature_importance"])
        if (i+1) % 10 == 0:
            print(f"  fold {i+1}/{len(folds)}  {t}/R{R}  CB={r['cb_mae']:.4f} B2S={r['b2s_mae']:.4f}")
    feat_imp = feat_imp / len(folds)

    dfr = pd.DataFrame(rows)
    dfa = dfr[["temporada","rodada_alvo","cb_mae","cb_sem_confronto_mae"]].rename(
        columns={"cb_mae":"mae_com_confronto","cb_sem_confronto_mae":"mae_sem_confronto"})
    dfa["delta_sem_menos_com"] = dfa["mae_sem_confronto"] - dfa["mae_com_confronto"]

    hash_canon_depois = sha256(CANON)
    hash_temporal_depois = sha256(TEMPORAL_SAFE)
    hash_b2s_depois = sha256(B2S_FOLDS)
    integridade_ok = (
        hash_canon_antes == hash_canon_depois
        and hash_temporal_antes == hash_temporal_depois
        and hash_b2s_antes == hash_b2s_depois
    )

    total_n = int(dfr["n_test_metric"].sum())
    cb_mae_global = float((dfr["cb_mae"]*dfr["n_test_metric"]).sum()/total_n)
    cb_rmse_global = float(np.sqrt((dfr["cb_rmse"]**2*dfr["n_test_metric"]).sum()/total_n))
    cb_sem_mae_global = float((dfr["cb_sem_confronto_mae"]*dfr["n_test_metric"]).sum()/total_n)
    b2s_mae_global = float((dfr["b2s_mae"]*dfr["n_test_metric"]).sum()/total_n)

    sp_cb_medio = float(dfr["cb_spearman"].mean())
    sp_sem_medio = float(dfr["cb_sem_confronto_spearman"].mean())
    b2s_sp_medio = float(b2s["b2s_spearman"].mean())

    n_melhor = int((dfr["delta_mae"] < 0).sum())
    n_pior   = int((dfr["delta_mae"] > 0).sum())
    n_empate = int((dfr["delta_mae"] == 0).sum())
    mediana_delta = float(dfr["delta_mae"].median())
    media_delta = float(dfr["delta_mae"].mean())
    melhoria_abs = b2s_mae_global - cb_mae_global
    melhoria_pct = (melhoria_abs/b2s_mae_global)*100

    por_temp_rows = []
    for t_ in sorted(dfr["temporada"].unique()):
        g = dfr[dfr["temporada"] == t_].copy()
        keys = [(int(a), int(b)) for a, b in zip(g["temporada"], g["rodada_alvo"])]
        m = b2s.loc[keys, "b2s_mae"].to_numpy()
        n = g["n_test_metric"].to_numpy()
        b2s_mae_t = float((m*n).sum()/n.sum())
        b2s_sp_t = float(np.nanmean(b2s.loc[keys, "b2s_spearman"].to_numpy()))
        por_temp_rows.append({
            "temporada": int(t_),
            "n_folds": int(len(g)),
            "cb_mae": float((g["cb_mae"]*g["n_test_metric"]).sum()/g["n_test_metric"].sum()),
            "cb_sem_mae": float((g["cb_sem_confronto_mae"]*g["n_test_metric"]).sum()/g["n_test_metric"].sum()),
            "b2s_mae": b2s_mae_t,
            "delta": float(((g["cb_mae"]-g["b2s_mae"])*g["n_test_metric"]).sum()/g["n_test_metric"].sum()),
            "cb_spearman": float(g["cb_spearman"].mean()),
            "b2s_spearman": b2s_sp_t,
        })
    por_temp = pd.DataFrame(por_temp_rows)

    dfr.to_csv(CSV_FOLDS, index=False)
    dfa.to_csv(CSV_ABL, index=False)

    fi_df = pd.DataFrame({"feature": FEATURES_COM, "importance": feat_imp}).sort_values(
        "importance", ascending=False)
    fi_df.to_csv(CSV_FEAT_IMP, index=False)

    grp_rows = []
    for g, fs in GRUPO_FEAT.items():
        grp_rows.append({"grupo": g,
                         "importance_total": float(fi_df[fi_df["feature"].isin(fs)]["importance"].sum()),
                         "features": ", ".join(fs)})
    pd.DataFrame(grp_rows).sort_values("importance_total", ascending=False).to_csv(
        CSV_GRP_IMP, index=False)

    fim = datetime.now()
    L = []
    L.append("# CatBoost v1 — Summary\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- CatBoost: {__import__('catboost').__version__}\n")
    L.append(f"- Folds: {len(dfr)}\n")
    L.append(f"- Obs avaliadas: {total_n}\n")
    L.append(f"- Cache N2: {ja_cacheados}/{len(folds)} folds ja em cache antes da execucao\n\n")

    L.append("## Parametros (pre-registrados)\n\n")
    for k, v in CATBOOST_PARAMS.items():
        L.append(f"- {k}: {v}\n")
    L.append("\n## Metricas oficiais\n\n")
    L.append(f"- **CatBoost MAE global ponderado: {cb_mae_global:.4f}**\n")
    L.append(f"- CatBoost RMSE: {cb_rmse_global:.4f}\n")
    L.append(f"- CatBoost Spearman medio: {sp_cb_medio:.4f}\n")
    L.append(f"- B2S MAE: {b2s_mae_global:.4f}\n")
    L.append(f"- B2S Spearman medio: {b2s_sp_medio:.4f}\n")
    L.append(f"- Delta MAE (CB - B2S): {cb_mae_global - b2s_mae_global:+.4f}\n")
    L.append(f"- Melhoria absoluta: {melhoria_abs:+.4f}\n")
    L.append(f"- Melhoria percentual: {melhoria_pct:+.2f}%\n\n")

    L.append("## Consistencia por fold\n\n")
    L.append(f"- Folds CB < B2S: {n_melhor}/{len(dfr)}\n")
    L.append(f"- Folds CB > B2S: {n_pior}/{len(dfr)}\n")
    L.append(f"- Empates: {n_empate}\n")
    L.append(f"- Mediana delta: {mediana_delta:+.4f}\n")
    L.append(f"- Media delta: {media_delta:+.4f}\n\n")

    L.append("## Por temporada\n\n")
    L.append(por_temp.to_string(index=False))
    L.append("\n\n## Ablation — sem Confronto\n\n")
    L.append(f"- CatBoost com Confronto: {cb_mae_global:.4f}\n")
    L.append(f"- CatBoost sem Confronto: {cb_sem_mae_global:.4f}\n")
    L.append(f"- Delta (sem - com): {cb_sem_mae_global - cb_mae_global:+.4f}\n")
    L.append(f"- Spearman com: {sp_cb_medio:.4f}\n")
    L.append(f"- Spearman sem: {sp_sem_medio:.4f}\n\n")

    L.append("## Feature importance (media dos 76 folds, nao e causalidade)\n\n")
    L.append(fi_df.to_string(index=False))
    L.append("\n\n## Importancia por grupo\n\n")
    L.append(pd.DataFrame(grp_rows).sort_values("importance_total", ascending=False).to_string(index=False))
    L.append("\n\n## Integridade\n\n")
    L.append(f"- Parquet canonico inalterado: {hash_canon_antes == hash_canon_depois}\n")
    L.append(f"- Temporal-safe inalterado: {hash_temporal_antes == hash_temporal_depois}\n")
    L.append(f"- B2S folds inalterado: {hash_b2s_antes == hash_b2s_depois}\n")
    CSV_SUMMARY.write_text("".join(L), encoding="utf-8")

    print()
    print("=== RESUMO ===")
    print(f"CatBoost MAE: {cb_mae_global:.4f}  B2S MAE: {b2s_mae_global:.4f}")
    print(f"delta: {cb_mae_global-b2s_mae_global:+.4f}  melhoria: {melhoria_pct:+.2f}%")
    print(f"folds CB<B2S: {n_melhor}/{len(dfr)}  CB>B2S: {n_pior}  empates: {n_empate}")
    print(f"CatBoost Spearman: {sp_cb_medio:.4f}  B2S: {b2s_sp_medio:.4f}")
    print(f"sem Confronto: {cb_sem_mae_global:.4f}  delta sem-com: {cb_sem_mae_global-cb_mae_global:+.4f}")
    print(f"integridade OK: {integridade_ok}")
    print(f"summary: {CSV_SUMMARY.relative_to(ROOT)}")

    if not integridade_ok:
        print("ERRO: algum artefato de entrada foi alterado.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
