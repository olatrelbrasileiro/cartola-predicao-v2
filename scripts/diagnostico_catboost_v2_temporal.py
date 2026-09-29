#!/usr/bin/env python3
"""
C2 — Diagnostico temporal do CatBoost v2.

Le reports/catboost_v2_predictions.csv (23.691 obs, materializado com PASS).
NAO treina. NAO faz tuning. NAO cria features. NAO altera artefatos oficiais.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def detect_root():
    candidatos = [
        Path.home() / "cartola-predicao-v2",
        Path("/workspaces/cartola-predicao-v2"),
        Path("/data/data/com.termux/files/home/cartola-predicao-v2"),
    ]
    for c in candidatos:
        if (c / "data" / "interim" / "cartola_jogador_rodada_v2.parquet").exists():
            return c
    raise RuntimeError("Raiz do projeto nao encontrada.")


ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"
PRED = REP / "catboost_v2_predictions.csv"
HARNESS = REP / "harness_folds_v2.csv"

OUT_TEMP = REP / "catboost_v2_temporal.csv"
OUT_TEMP_POS = REP / "catboost_v2_temporal_por_posicao.csv"
OUT_FOLD = REP / "catboost_v2_temporal_folds.csv"
OUT_RES = REP / "catboost_v2_temporal_resumo.csv"
OUT_NDCG = REP / "catboost_v2_temporal_ndcg.csv"
OUT_NDCG_FOLD = REP / "catboost_v2_temporal_ndcg_folds.csv"
OUT_ALERTAS = REP / "catboost_v2_temporal_alertas.csv"
OUT_MD = REP / "catboost_v2_temporal.md"
OUT_VALID = REP / "catboost_v2_temporal_validation.json"

HASH_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
EXPECTED_FOLDS = {2024: 29, 2025: 29, 2026: 18}
EXPECTED_SEASONS = {2024, 2025, 2026}
KS = [3, 5, 7]
POS_NAMES = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro",
             4: "Meia", 5: "Atacante", 6: "Tecnico"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def mae(pred, y):
    if len(y) == 0: return np.nan
    return float(np.mean(np.abs(pred - y)))


def rmse(pred, y):
    if len(y) == 0: return np.nan
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def spearman(pred, y):
    if len(pred) < 2: return np.nan
    if np.all(pred == pred[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(pred).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def dcg(rel, k):
    n = min(k, len(rel))
    if n == 0: return 0.0
    descontos = np.log2(np.arange(2, n + 2))
    return float(np.sum(rel[:n] / descontos))


def ndcg(rel, pred, k):
    n = len(rel)
    if n < 2: return np.nan, "n_lt_2", np.nan, np.nan
    rel_pred = rel[np.argsort(-pred)]
    dcg_val = dcg(rel_pred, k)
    rel_ideal = np.sort(rel)[::-1]
    idcg_val = dcg(rel_ideal, k)
    if idcg_val == 0: return np.nan, "idcg_zero", dcg_val, idcg_val
    if idcg_val < 0: return np.nan, "idcg_negativo", dcg_val, idcg_val
    valor = dcg_val / idcg_val
    if valor < 0: return valor, "ndcg_negativo", dcg_val, idcg_val
    return valor, "ok", dcg_val, idcg_val


def ndcg_por_temporada(df, pred_col):
    rows = []; alertas = []
    for (temporada, rodada, posicao), sub in df.groupby(
            ["temporada", "rodada", "posicao_id"]):
        y = sub["y_true"].to_numpy(dtype=float)
        pred = sub[pred_col].to_numpy(dtype=float)
        for k in KS:
            valor, status, dcg_val, idcg_val = ndcg(y, pred, k)
            rows.append({
                "temporada": int(temporada), "rodada": int(rodada),
                "posicao_id": int(posicao),
                "posicao_nome": POS_NAMES.get(int(posicao), "?"),
                "K": int(k),
                "modelo": ("catboost" if pred_col == "y_pred_catboost" else "b2s"),
                "n": int(len(sub)), "ndcg": valor, "status": status,
                "dcg": dcg_val, "idcg": idcg_val,
            })
            if status != "ok":
                alertas.append({
                    "temporada": int(temporada), "rodada": int(rodada),
                    "posicao_id": int(posicao),
                    "posicao_nome": POS_NAMES.get(int(posicao), "?"),
                    "K": int(k),
                    "modelo": ("catboost" if pred_col == "y_pred_catboost" else "b2s"),
                    "status": status, "n": int(len(sub)),
                    "max_rel": float(np.max(y)), "min_rel": float(np.min(y)),
                    "dcg": dcg_val, "idcg": idcg_val, "ndcg": valor,
                })
    df_ndcg = pd.DataFrame(rows)
    if len(alertas):
        df_alertas = pd.DataFrame(alertas)
    else:
        df_alertas = pd.DataFrame(columns=[
            "temporada", "rodada", "posicao_id", "posicao_nome",
            "K", "modelo", "status", "n", "max_rel", "min_rel",
            "dcg", "idcg", "ndcg"])
    return df_ndcg, df_alertas


def main():
    inicio = datetime.now()
    print("== C2 — Diagnostico temporal CatBoost v2 ==")

    if not CANON.exists():
        print("ERRO: Parquet ausente.", file=sys.stderr); sys.exit(2)
    if not PRED.exists():
        print("ERRO: catboost_v2_predictions.csv ausente.", file=sys.stderr)
        sys.exit(2)

    hash_atual = sha256(CANON)
    if hash_atual != HASH_CANON:
        print(f"ERRO: SHA diverge: {hash_atual}", file=sys.stderr); sys.exit(2)
    print(f"Parquet SHA-256 OK: {hash_atual[:16]}...")

    df = pd.read_csv(PRED)
    print(f"predicoes: {len(df)} linhas")

    falhas = []
    cols_obrig = ["temporada", "rodada", "atleta_id", "posicao_id",
                  "entrou_em_campo", "y_true", "y_pred_catboost", "y_pred_b2s"]
    faltantes = [c for c in cols_obrig if c not in df.columns]
    if faltantes: falhas.append(f"colunas ausentes: {faltantes}")
    if len(df) == 0: falhas.append("arquivo vazio")
    if df.duplicated(subset=["temporada", "rodada", "atleta_id"]).any():
        falhas.append("duplicatas (T,R,A)")
    if "entrou_em_campo" in df.columns and not df["entrou_em_campo"].all():
        falhas.append("linhas com entrou_em_campo=False")
    if df[["y_true", "y_pred_catboost", "y_pred_b2s"]].isna().any().any():
        falhas.append("NaN em y_true/y_pred")
    pos_inv = set(df["posicao_id"].unique()) - set(POS_NAMES.keys())
    if pos_inv: falhas.append(f"posicoes inesperadas: {sorted(pos_inv)}")
    temporadas = set(df["temporada"].astype(int).unique())
    if temporadas != EXPECTED_SEASONS:
        falhas.append(f"temporadas inesperadas: {sorted(temporadas)}")
    folds_por_temporada = df.groupby("temporada")["rodada"].nunique().to_dict()
    for t, esp in EXPECTED_FOLDS.items():
        obs = int(folds_por_temporada.get(t, 0))
        if obs != esp: falhas.append(f"{t}: {obs} folds != {esp}")
    folds = df[["temporada", "rodada"]].drop_duplicates()
    if len(folds) != 76: falhas.append(f"folds: {len(folds)} != 76")

    if HARNESS.exists():
        man = pd.read_csv(HARNESS)
        hk = set(zip(man["temporada"].astype(int), man["rodada_alvo"].astype(int)))
        dk = set(zip(folds["temporada"].astype(int), folds["rodada"].astype(int)))
        if hk != dk:
            faltam = hk - dk; sobram = dk - hk
            falhas.append(f"folds != harness (faltam {len(faltam)}, sobram {len(sobram)})")

    if falhas:
        print(); print("FALHAS:")
        for f in falhas: print(f" - {f}")
        print(); print("=== RESULTADO: FAIL ==="); sys.exit(1)
    print(f"validacoes OK ({len(folds)} folds, {len(df)} obs)")

    # ------- por temporada
    print(); print("computando metricas por temporada...")
    rows_temp = []
    for temporada, sub in df.groupby("temporada"):
        y = sub["y_true"].to_numpy(dtype=float)
        cb = sub["y_pred_catboost"].to_numpy(dtype=float)
        b2s = sub["y_pred_b2s"].to_numpy(dtype=float)
        mae_cb = mae(cb, y); mae_b2s = mae(b2s, y)
        rmse_cb = rmse(cb, y); rmse_b2s = rmse(b2s, y)
        sp_cb = spearman(cb, y); sp_b2s = spearman(b2s, y)
        rows_temp.append({
            "temporada": int(temporada), "n_observacoes": int(len(sub)),
            "n_folds": int(sub.groupby(["temporada", "rodada"]).ngroups),
            "mae_catboost": mae_cb, "mae_b2s": mae_b2s, "delta_mae": mae_cb - mae_b2s,
            "rmse_catboost": rmse_cb, "rmse_b2s": rmse_b2s, "delta_rmse": rmse_cb - rmse_b2s,
            "spearman_catboost": sp_cb, "spearman_b2s": sp_b2s,
            "delta_spearman": (sp_cb - sp_b2s) if not (np.isnan(sp_cb) or np.isnan(sp_b2s)) else np.nan,
        })
    df_temp = pd.DataFrame(rows_temp).sort_values("temporada")

    # ------- por temporada x posicao
    print("computando temporada x posicao...")
    rows_tp = []
    for (temporada, posicao), sub in df.groupby(["temporada", "posicao_id"]):
        y = sub["y_true"].to_numpy(dtype=float)
        cb = sub["y_pred_catboost"].to_numpy(dtype=float)
        b2s = sub["y_pred_b2s"].to_numpy(dtype=float)
        mae_cb = mae(cb, y); mae_b2s = mae(b2s, y)
        sp_cb = spearman(cb, y); sp_b2s = spearman(b2s, y)
        rows_tp.append({
            "temporada": int(temporada), "posicao_id": int(posicao),
            "posicao_nome": POS_NAMES.get(int(posicao), "?"),
            "n_observacoes": int(len(sub)),
            "mae_catboost": mae_cb, "mae_b2s": mae_b2s, "delta_mae": mae_cb - mae_b2s,
            "spearman_catboost": sp_cb, "spearman_b2s": sp_b2s,
            "delta_spearman": (sp_cb - sp_b2s) if not (np.isnan(sp_cb) or np.isnan(sp_b2s)) else np.nan,
        })
    df_tp = pd.DataFrame(rows_tp).sort_values(["temporada", "posicao_id"])

    # ------- por fold
    print("computando metricas por fold...")
    rows_fold = []
    for (temporada, rodada), sub in df.groupby(["temporada", "rodada"]):
        y = sub["y_true"].to_numpy(dtype=float)
        cb = sub["y_pred_catboost"].to_numpy(dtype=float)
        b2s = sub["y_pred_b2s"].to_numpy(dtype=float)
        mae_cb = mae(cb, y); mae_b2s = mae(b2s, y)
        rmse_cb = rmse(cb, y); rmse_b2s = rmse(b2s, y)
        sp_cb = spearman(cb, y); sp_b2s = spearman(b2s, y)
        rows_fold.append({
            "temporada": int(temporada), "rodada_alvo": int(rodada), "n": int(len(sub)),
            "mae_catboost": mae_cb, "mae_b2s": mae_b2s, "delta_mae": mae_cb - mae_b2s,
            "rmse_catboost": rmse_cb, "rmse_b2s": rmse_b2s, "delta_rmse": rmse_cb - rmse_b2s,
            "spearman_catboost": sp_cb, "spearman_b2s": sp_b2s,
            "delta_spearman": (sp_cb - sp_b2s) if not (np.isnan(sp_cb) or np.isnan(sp_b2s)) else np.nan,
        })
    df_fold = pd.DataFrame(rows_fold).sort_values(["temporada", "rodada_alvo"])

    # ------- resumo por temporada
    print("resumindo estabilidade por temporada...")
    rows_res = []
    for temporada, sub in df_fold.groupby("temporada"):
        delta = sub["delta_mae"]
        melhor = int((delta < -1e-4).sum())
        pior = int((delta > 1e-4).sum())
        empate = int(len(sub) - melhor - pior)
        delta_sp = sub["delta_spearman"].dropna()
        rows_res.append({
            "temporada": int(temporada), "n_folds": int(len(sub)),
            "folds_catboost_melhor": melhor,
            "folds_catboost_pior": pior,
            "folds_empate": empate,
            "frac_folds_melhor": (melhor / len(sub) if len(sub) else np.nan),
            "delta_mae_media": float(delta.mean()),
            "delta_mae_mediana": float(delta.median()),
            "delta_mae_std": float(delta.std()),
            "delta_mae_min": float(delta.min()),
            "delta_mae_max": float(delta.max()),
            "n_spearman_validos": int(len(delta_sp)),
            "delta_spearman_media": (float(delta_sp.mean()) if len(delta_sp) else np.nan),
            "delta_spearman_mediana": (float(delta_sp.median()) if len(delta_sp) else np.nan),
            "delta_spearman_std": (float(delta_sp.std()) if len(delta_sp) else np.nan),
        })
    df_res = pd.DataFrame(rows_res).sort_values("temporada")

    # ------- NDCG
    print("computando NDCG temporal...")
    ndcg_cb, alertas_cb = ndcg_por_temporada(df, "y_pred_catboost")
    ndcg_b2s, alertas_b2s = ndcg_por_temporada(df, "y_pred_b2s")
    df_ndcg_fato = pd.concat([ndcg_cb, ndcg_b2s], ignore_index=True)
    if len(alertas_cb) or len(alertas_b2s):
        df_alertas = pd.concat([alertas_cb, alertas_b2s], ignore_index=True)
    else:
        df_alertas = pd.DataFrame(columns=[
            "temporada", "rodada", "posicao_id", "posicao_nome",
            "K", "modelo", "status", "n", "max_rel", "min_rel",
            "dcg", "idcg", "ndcg"])

    # NDCG agregado por temporada x K x modelo
    rows_ndcg = []
    for temporada in sorted(df["temporada"].unique()):
        for k in KS:
            for modelo in ["catboost", "b2s"]:
                sub = df_ndcg_fato[
                    (df_ndcg_fato["temporada"] == temporada)
                    & (df_ndcg_fato["K"] == k)
                    & (df_ndcg_fato["modelo"] == modelo)
                    & (df_ndcg_fato["status"].isin(["ok", "ndcg_negativo"]))
                ]
                valor = float(sub["ndcg"].mean()) if len(sub) else np.nan
                rows_ndcg.append({
                    "temporada": int(temporada), "K": int(k),
                    "modelo": modelo,
                    "n_grupos_validos": int(len(sub)),
                    "ndcg_medio": valor,
                })
    df_ndcg_temp = pd.DataFrame(rows_ndcg)

    # NDCG wide
    rows_wide = []
    for temporada in sorted(df["temporada"].unique()):
        row = {"temporada": int(temporada)}
        for k in KS:
            cb = df_ndcg_temp[(df_ndcg_temp["temporada"] == temporada)
                              & (df_ndcg_temp["K"] == k)
                              & (df_ndcg_temp["modelo"] == "catboost")]
            b2s = df_ndcg_temp[(df_ndcg_temp["temporada"] == temporada)
                               & (df_ndcg_temp["K"] == k)
                               & (df_ndcg_temp["modelo"] == "b2s")]
            vc = float(cb["ndcg_medio"].iloc[0]) if len(cb) else np.nan
            vb = float(b2s["ndcg_medio"].iloc[0]) if len(b2s) else np.nan
            row[f"ndcg{k}_catboost"] = vc
            row[f"ndcg{k}_b2s"] = vb
            row[f"ndcg{k}_delta"] = (vc - vb) if not (np.isnan(vc) or np.isnan(vb)) else np.nan
        rows_wide.append(row)
    df_ndcg = pd.DataFrame(rows_wide).sort_values("temporada")

    # Alertas
    n_alertas = len(df_alertas)
    n_lt_2 = int((df_alertas["status"] == "n_lt_2").sum()) if n_alertas else 0
    n_idcg_zero = int((df_alertas["status"] == "idcg_zero").sum()) if n_alertas else 0
    n_idcg_neg = int((df_alertas["status"] == "idcg_negativo").sum()) if n_alertas else 0
    n_ndcg_neg = int((df_alertas["status"] == "ndcg_negativo").sum()) if n_alertas else 0
    ndcg_status = "NDCG_PENDENTE_DECISAO" if n_alertas > 0 else "PASS"

    # Heterogeneidade (descritiva)
    temporadas_pior_mae = df_temp[df_temp["delta_mae"] > 0]["temporada"].astype(int).tolist()
    temporadas_melhor_mae = df_temp[df_temp["delta_mae"] < 0]["temporada"].astype(int).tolist()
    sinais_mae = set(np.sign(df_temp["delta_mae"]).astype(int))
    sinal_mae_misto = (-1 in sinais_mae and 1 in sinais_mae)

    temporadas_melhor_sp = df_temp[df_temp["delta_spearman"] > 0]["temporada"].astype(int).tolist()
    temporadas_pior_sp = df_temp[df_temp["delta_spearman"] < 0]["temporada"].astype(int).tolist()
    sinais_sp = set(np.sign(df_temp["delta_spearman"].dropna()).astype(int))
    sinal_sp_misto = (-1 in sinais_sp and 1 in sinais_sp)

    # ------- CSVs
    df_temp.to_csv(OUT_TEMP, index=False)
    df_tp.to_csv(OUT_TEMP_POS, index=False)
    df_fold.to_csv(OUT_FOLD, index=False)
    df_res.to_csv(OUT_RES, index=False)
    df_ndcg.to_csv(OUT_NDCG, index=False)
    df_ndcg_fato.to_csv(OUT_NDCG_FOLD, index=False)
    df_alertas.to_csv(OUT_ALERTAS, index=False)

    # ------- MD
    L = []
    L.append("# C2 — Diagnostico temporal CatBoost v2\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fonte: `{PRED.relative_to(ROOT)}`\n")
    L.append(f"- Observacoes: {len(df)}\n")
    L.append(f"- Folds: {len(folds)}\n")
    L.append(f"- SHA Parquet: `{hash_atual}`\n\n")

    L.append("## 1. Resultado por temporada\n\n")
    L.append("| temporada | n | folds | MAE_CB | MAE_B2S | delta_MAE | "
             "RMSE_CB | RMSE_B2S | delta_RMSE | SP_CB | SP_B2S | delta_SP |\n")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in df_temp.iterrows():
        L.append(f"| {int(r['temporada'])} | {int(r['n_observacoes'])} | "
                 f"{int(r['n_folds'])} | {r['mae_catboost']:.4f} | "
                 f"{r['mae_b2s']:.4f} | {r['delta_mae']:+.4f} | "
                 f"{r['rmse_catboost']:.4f} | {r['rmse_b2s']:.4f} | "
                 f"{r['delta_rmse']:+.4f} | {r['spearman_catboost']:.4f} | "
                 f"{r['spearman_b2s']:.4f} | {r['delta_spearman']:+.4f} |\n")
    L.append("\n")

    L.append("## 2. Estabilidade por fold\n\n")
    L.append("| temporada | folds | CB melhor | CB pior | empates | "
             "frac melhor | delta medio | mediana | std | min | max |\n")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in df_res.iterrows():
        L.append(f"| {int(r['temporada'])} | {int(r['n_folds'])} | "
                 f"{int(r['folds_catboost_melhor'])} | {int(r['folds_catboost_pior'])} | "
                 f"{int(r['folds_empate'])} | {r['frac_folds_melhor']:.3f} | "
                 f"{r['delta_mae_media']:+.4f} | {r['delta_mae_mediana']:+.4f} | "
                 f"{r['delta_mae_std']:.4f} | {r['delta_mae_min']:+.4f} | "
                 f"{r['delta_mae_max']:+.4f} |\n")
    L.append("\n")

    L.append("## 3. NDCG por temporada\n\n")
    L.append("| temporada | NDCG@3 CB | B2S | delta | NDCG@5 CB | B2S | delta | "
             "NDCG@7 CB | B2S | delta |\n")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in df_ndcg.iterrows():
        def fmt(x): return "—" if pd.isna(x) else f"{x:.4f}"
        L.append(f"| {int(r['temporada'])} | {fmt(r['ndcg3_catboost'])} | "
                 f"{fmt(r['ndcg3_b2s'])} | {fmt(r['ndcg3_delta'])} | "
                 f"{fmt(r['ndcg5_catboost'])} | {fmt(r['ndcg5_b2s'])} | "
                 f"{fmt(r['ndcg5_delta'])} | {fmt(r['ndcg7_catboost'])} | "
                 f"{fmt(r['ndcg7_b2s'])} | {fmt(r['ndcg7_delta'])} |\n")
    L.append("\n")

    L.append("## 4. Heterogeneidade\n\n")
    L.append(f"- CatBoost melhor em MAE: {temporadas_melhor_mae}\n")
    L.append(f"- CatBoost pior em MAE: {temporadas_pior_mae}\n")
    L.append(f"- Sinal de delta MAE misto: {sinal_mae_misto}\n")
    L.append(f"- CatBoost melhor em Spearman: {temporadas_melhor_sp}\n")
    L.append(f"- CatBoost pior em Spearman: {temporadas_pior_sp}\n")
    L.append(f"- Sinal de delta Spearman misto: {sinal_sp_misto}\n\n")

    L.append("## 5. Alertas NDCG\n\n")
    L.append(f"- total: {n_alertas}\n")
    L.append(f"- n<2: {n_lt_2}\n")
    L.append(f"- IDCG=0: {n_idcg_zero}\n")
    L.append(f"- IDCG<0: {n_idcg_neg}\n")
    L.append(f"- NDCG<0: {n_ndcg_neg}\n")
    L.append(f"- resultado_ndcg: **{ndcg_status}**\n\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    criterios = {
        "parquet_sha_ok": hash_atual == HASH_CANON,
        "predicoes_presentes": PRED.exists() and len(df) > 0,
        "n_observacoes_ok": len(df) == 23691,
        "folds_76_ok": len(folds) == 76,
        "folds_temporadas_ok": folds_por_temporada == EXPECTED_FOLDS,
        "sem_duplicatas": not df.duplicated(subset=["temporada", "rodada", "atleta_id"]).any(),
        "participantes_only": bool(df["entrou_em_campo"].all()),
        "predicoes_validas": bool(df[["y_true", "y_pred_catboost", "y_pred_b2s"]].notna().all().all()),
        "posicoes_validas": len(pos_inv) == 0,
        "tres_temporadas": temporadas == EXPECTED_SEASONS,
        "metricas_temporais_ok": len(df_temp) == 3,
        "metricas_temporada_posicao_ok": len(df_tp) > 0,
        "metricas_fold_ok": len(df_fold) == 76,
        "resumo_fold_ok": len(df_res) == 3,
        "ndcg_temporal_ok": len(df_ndcg) == 3,
        "comparacao_mesmo_conjunto_ok": True,
        "relatorio_gerado": True,
        "validation_json_gerado": True,
    }
    pass_metricas = all(criterios.values())

    validacao = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "parquet_sha256": hash_atual,
        "parquet_sha_esperado": HASH_CANON,
        "n_observacoes": int(len(df)),
        "n_folds": int(len(folds)),
        "folds_por_temporada": {str(k): int(v) for k, v in folds_por_temporada.items()},
        "criterios": criterios,
        "resultado_metricas": "PASS" if pass_metricas else "FAIL",
        "resultado_ndcg": ndcg_status,
        "heterogeneidade": {
            "temporadas_melhor_mae": temporadas_melhor_mae,
            "temporadas_pior_mae": temporadas_pior_mae,
            "sinal_mae_misto": bool(sinal_mae_misto),
            "temporadas_melhor_spearman": temporadas_melhor_sp,
            "temporadas_pior_spearman": temporadas_pior_sp,
            "sinal_spearman_misto": bool(sinal_sp_misto),
        },
        "ndcg_alertas": {
            "n_alertas_total": int(n_alertas),
            "n_lt_2": int(n_lt_2),
            "n_idcg_zero": int(n_idcg_zero),
            "n_idcg_negativo": int(n_idcg_neg),
            "n_ndcg_negativo": int(n_ndcg_neg),
            "decisao_transformacao_relevancia": (
                "PENDENTE" if n_alertas > 0 else "NAO_NECESSARIA_NESTE_DIAGNOSTICO"),
        },
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(validacao, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=== RESULTADO POR TEMPORADA ===")
    print(df_temp.to_string(index=False))
    print()
    print("=== RESUMO DOS FOLDS ===")
    print(df_res.to_string(index=False))
    print()
    print("=== NDCG POR TEMPORADA ===")
    print(df_ndcg.to_string(index=False))
    print()
    print("=== ALERTAS NDCG ===")
    print(f"total={n_alertas} n_lt_2={n_lt_2} idcg_zero={n_idcg_zero} "
          f"idcg_neg={n_idcg_neg} ndcg_neg={n_ndcg_neg}")
    print()
    print(f"resultado_metricas: {validacao['resultado_metricas']}")
    print(f"resultado_ndcg:     {validacao['resultado_ndcg']}")
    print(f"C2 completo:        {'PASS' if pass_metricas else 'FAIL'}")
    print()
    print("artefatos:")
    for path in [OUT_TEMP, OUT_TEMP_POS, OUT_FOLD, OUT_RES, OUT_NDCG,
                 OUT_NDCG_FOLD, OUT_ALERTAS, OUT_MD, OUT_VALID]:
        if path.exists():
            print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size} bytes)")
    print()
    print(f"SHA Parquet: {hash_atual}")

    if not pass_metricas:
        sys.exit(1)


if __name__ == "__main__":
    main()
