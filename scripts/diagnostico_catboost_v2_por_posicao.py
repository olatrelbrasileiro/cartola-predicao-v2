#!/usr/bin/env python3
"""
C1 — Diagnostico CatBoost v2 por posicao.

Le reports/catboost_v2_predictions.csv (produzido com PASS pela etapa de
persistencia numerica).

NAO treina nada. NAO altera nenhum artefato oficial.

Metricas por posicao:
  - MAE   (CatBoost, B2S, delta)
  - Spearman (CatBoost, B2S, delta)
  - NDCG@3, @5, @7 (CatBoost, B2S, delta)

NDCG — definicao congelada:
  grupo    = (temporada, rodada, posicao_id)
  relev.   = pontos_num  BRUTO (sem transformacao)
  ranking  = y_pred_catboost / y_pred_b2s
  K        = {3, 5, 7}
  ganho    = linear: DCG@K = sum_{i=1..K} rel_i / log2(i+1)
  n < 2    -> status 'n_lt_2'      -> grupo invalido, excluido da media
  IDCG == 0 -> status 'idcg_zero'  -> grupo invalido, excluido da media
  IDCG <  0 -> status 'idcg_negativo' -> grupo invalido, excluido da media
  NDCG <  0 -> status 'ndcg_negativo' -> INCLUIDO na media, mas flagrado
  caso contrario -> status 'ok'    -> incluido na media
  agregacao = media simples dos valores nao-NaN (ok + ndcg_negativo)

ALERTAS DE NDCG
  Como pontos_num pode ser negativo, grupos com max(pontos_num) <= 0
  tendem a produzir IDCG <= 0. Nesses casos o NDCG e' indefinido.
  O script NAO decide transformacao de relevancia — apenas conta e reporta.
  Se houver qualquer alerta, resultado_ndcg = NDCG_PENDENTE_DECISAO.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def detect_root():
    for c in [Path.home() / "cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path("/data/data/com.termux/files/home/cartola-predicao-v2")]:
        if (c / "data" / "interim" / "cartola_jogador_rodada_v2.parquet").exists():
            return c
    raise RuntimeError("Raiz nao encontrada")


ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"
PRED = REP / "catboost_v2_predictions.csv"

OUT_POS = REP / "catboost_v2_por_posicao.csv"
OUT_TEMP = REP / "catboost_v2_por_posicao_temporada.csv"
OUT_FOLD = REP / "catboost_v2_por_posicao_folds.csv"
OUT_FOLD_RES = REP / "catboost_v2_por_posicao_fold_resumo.csv"
OUT_NDCG = REP / "catboost_v2_por_posicao_ndcg.csv"
OUT_NDCG_AL = REP / "catboost_v2_por_posicao_ndcg_alertas.csv"
OUT_MD = REP / "catboost_v2_por_posicao.md"
OUT_VALID = REP / "catboost_v2_por_posicao_validation.json"

HASH_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
KS = [3, 5, 7]
POS_NAMES = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro",
             4: "Meia",    5: "Atacante", 6: "Tecnico"}

STATUS_VALIDOS_NA_MEDIA = {"ok", "ndcg_negativo"}
STATUS_INVALIDOS = {"n_lt_2", "idcg_zero", "idcg_negativo"}


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


# ------------------------------------------------------------------ ndcg
def _dcg(rel, k):
    n = min(k, len(rel))
    if n == 0: return 0.0
    disc = np.log2(np.arange(2, n + 2))
    return float(np.sum(rel[:n] / disc))


def _ndcg(rel, y_pred, k):
    """
    Retorna (ndcg, status, dcg, idcg).
    status em {ok, n_lt_2, idcg_zero, idcg_negativo, ndcg_negativo}.
    """
    n = len(rel)
    if n < 2:
        return np.nan, "n_lt_2", np.nan, np.nan
    rel_pred = rel[np.argsort(-y_pred)]
    dcg = _dcg(rel_pred, k)
    rel_ideal = np.sort(rel)[::-1]
    idcg = _dcg(rel_ideal, k)
    if idcg == 0:
        return np.nan, "idcg_zero", dcg, idcg
    if idcg < 0:
        return np.nan, "idcg_negativo", dcg, idcg
    ndcg = dcg / idcg
    if ndcg < 0:
        return ndcg, "ndcg_negativo", dcg, idcg
    return ndcg, "ok", dcg, idcg


def ndcg_por_grupo(df, pred_col):
    """
    Computa NDCG@k para cada grupo (temporada, rodada, posicao_id).

    Retorna dict:
      ndcg_media:    {k: float}  media simples dos valores em {ok, ndcg_negativo}
      n_ok:          {k: int}    grupos ok + ndcg_negativo (contribuintes da media)
      n_invalidos:   {k: int}    n_lt_2 + idcg_zero + idcg_negativo
      resumo_status: {k: {status: count}}
      alertas:       DataFrame com TODOS os grupos cujo status != 'ok'
    """
    ndcg_vals = {k: [] for k in KS}
    resumo_status = {
        k: {"ok": 0, "n_lt_2": 0, "idcg_zero": 0,
            "idcg_negativo": 0, "ndcg_negativo": 0}
        for k in KS
    }
    alertas = []
    for (t, R, p), sub in df.groupby(["temporada", "rodada", "posicao_id"]):
        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub[pred_col].to_numpy(dtype=float)
        max_rel = float(np.max(yt)) if len(yt) else 0.0
        min_rel = float(np.min(yt)) if len(yt) else 0.0
        for k in KS:
            v, status, dcg, idcg = _ndcg(yt, yp, k)
            resumo_status[k][status] += 1
            if status in STATUS_VALIDOS_NA_MEDIA:
                ndcg_vals[k].append(v)
            if status != "ok":
                alertas.append({
                    "temporada": int(t), "rodada": int(R),
                    "posicao_id": int(p),
                    "posicao_nome": POS_NAMES.get(int(p), "?"),
                    "K": int(k),
                    "status": status,
                    "n": int(len(yt)),
                    "max_rel": max_rel,
                    "min_rel": min_rel,
                    "dcg": dcg,
                    "idcg": idcg,
                    "ndcg": v,
                })
    ndcg_media = {k: (float(np.mean(ndcg_vals[k])) if ndcg_vals[k] else np.nan)
                  for k in KS}
    n_ok = {k: resumo_status[k]["ok"] + resumo_status[k]["ndcg_negativo"]
            for k in KS}
    n_invalidos = {k: (resumo_status[k]["n_lt_2"]
                       + resumo_status[k]["idcg_zero"]
                       + resumo_status[k]["idcg_negativo"])
                   for k in KS}
    return {
        "ndcg_media": ndcg_media,
        "n_ok": n_ok,
        "n_invalidos": n_invalidos,
        "resumo_status": resumo_status,
        "alertas": pd.DataFrame(alertas),
    }


# ------------------------------------------------------------------ main
def main():
    inicio = datetime.now()
    print("== C1 — Diagnostico CatBoost v2 por posicao ==")

    if not PRED.exists():
        print(f"ERRO: {PRED.name} ausente. Rode persistir_predicoes primeiro.",
              file=sys.stderr)
        sys.exit(2)

    h = sha256(CANON)
    if h != HASH_CANON:
        print(f"ERRO: hash do Parquet diverge: {h[:16]}", file=sys.stderr)
        sys.exit(2)
    print(f"Parquet SHA-256 OK: {h[:16]}...")

    df = pd.read_csv(PRED)
    print(f"predicoes: {len(df)} linhas")

    # ---------------------------------------------------------------- validacoes
    falhas = []
    if df.duplicated(subset=["temporada", "rodada", "atleta_id"]).any():
        falhas.append("duplicatas (T,R,A)")
    folds_uni = df[["temporada", "rodada"]].drop_duplicates()
    if len(folds_uni) != 76:
        falhas.append(f"folds: {len(folds_uni)} != 76")
    if "entrou_em_campo" in df.columns and not df["entrou_em_campo"].all():
        falhas.append("linhas com entrou_em_campo=False")
    pos_inv = set(df["posicao_id"].unique()) - set(POS_NAMES.keys())
    if pos_inv:
        falhas.append(f"posicoes inesperadas: {sorted(pos_inv)}")
    if df[["y_true", "y_pred_catboost", "y_pred_b2s"]].isna().any().any():
        falhas.append("NaN em y_true/y_pred")
    if falhas:
        print("FALHAS:")
        for f in falhas[:20]: print(f"  - {f}")
        print("=== RESULTADO: FAIL ===")
        sys.exit(1)
    print(f"validacoes OK  ({len(folds_uni)} folds, {len(df)} obs)")

    # ---------------------------------------------------------------- por posicao
    print("computando MAE/Spearman por posicao...")
    rows_pos = []
    for p, sub in df.groupby("posicao_id"):
        yt = sub["y_true"].to_numpy()
        yc = sub["y_pred_catboost"].to_numpy()
        yb = sub["y_pred_b2s"].to_numpy()
        sp_c = spearman(yc, yt); sp_b = spearman(yb, yt)
        rows_pos.append({
            "posicao_id": int(p),
            "posicao_nome": POS_NAMES.get(int(p), "?"),
            "n_observacoes": int(len(sub)),
            "n_folds": int(sub.groupby(["temporada", "rodada"]).ngroups),
            "mae_catboost": mae(yc, yt),
            "mae_b2s": mae(yb, yt),
            "delta_mae": mae(yc, yt) - mae(yb, yt),
            "spearman_catboost": sp_c,
            "spearman_b2s": sp_b,
            "delta_spearman": (sp_c - sp_b)
                if not (np.isnan(sp_c) or np.isnan(sp_b)) else np.nan,
        })
    df_pos = pd.DataFrame(rows_pos).sort_values("posicao_id")

    # ---------------------------------------------------------------- por temporada
    print("computando por temporada x posicao...")
    rows_temp = []
    for (t, p), sub in df.groupby(["temporada", "posicao_id"]):
        yt = sub["y_true"].to_numpy()
        yc = sub["y_pred_catboost"].to_numpy()
        yb = sub["y_pred_b2s"].to_numpy()
        sp_c = spearman(yc, yt); sp_b = spearman(yb, yt)
        rows_temp.append({
            "temporada": int(t),
            "posicao_id": int(p),
            "posicao_nome": POS_NAMES.get(int(p), "?"),
            "n_observacoes": int(len(sub)),
            "mae_catboost": mae(yc, yt),
            "mae_b2s": mae(yb, yt),
            "delta_mae": mae(yc, yt) - mae(yb, yt),
            "spearman_catboost": sp_c,
            "spearman_b2s": sp_b,
            "delta_spearman": (sp_c - sp_b)
                if not (np.isnan(sp_c) or np.isnan(sp_b)) else np.nan,
        })
    df_temp = pd.DataFrame(rows_temp).sort_values(["temporada", "posicao_id"])

    # ---------------------------------------------------------------- por fold
    print("computando por fold x posicao...")
    rows_fold = []
    for (t, R, p), sub in df.groupby(["temporada", "rodada", "posicao_id"]):
        yt = sub["y_true"].to_numpy()
        yc = sub["y_pred_catboost"].to_numpy()
        yb = sub["y_pred_b2s"].to_numpy()
        rows_fold.append({
            "temporada": int(t), "rodada_alvo": int(R),
            "posicao_id": int(p), "posicao_nome": POS_NAMES.get(int(p), "?"),
            "n": int(len(sub)),
            "mae_catboost": mae(yc, yt),
            "mae_b2s": mae(yb, yt),
            "delta_mae": mae(yc, yt) - mae(yb, yt),
        })
    df_fold = pd.DataFrame(rows_fold)

    rows_res = []
    for p, sub in df_fold.groupby("posicao_id"):
        melhor = int((sub["delta_mae"] < -1e-4).sum())
        pior = int((sub["delta_mae"] > 1e-4).sum())
        empate = int(len(sub) - melhor - pior)
        rows_res.append({
            "posicao_id": int(p),
            "posicao_nome": POS_NAMES.get(int(p), "?"),
            "n_folds": int(len(sub)),
            "folds_catboost_melhor": melhor,
            "folds_catboost_pior": pior,
            "folds_empate": empate,
            "delta_mae_media": float(sub["delta_mae"].mean()),
            "delta_mae_mediana": float(sub["delta_mae"].median()),
            "delta_mae_std": float(sub["delta_mae"].std()),
        })
    df_fold_res = pd.DataFrame(rows_res).sort_values("posicao_id")

    # ---------------------------------------------------------------- NDCG
    print("computando NDCG@K por posicao (chamada unica por modelo)...")
    ndcg_rows = []
    alertas_por_pos = []
    for p in sorted(df["posicao_id"].unique()):
        sub = df[df["posicao_id"] == p]
        r_cb = ndcg_por_grupo(sub, "y_pred_catboost")
        r_b2s = ndcg_por_grupo(sub, "y_pred_b2s")

        row = {"posicao_id": int(p),
               "posicao_nome": POS_NAMES.get(int(p), "?"),
               "n_observacoes": int(len(sub))}
        for k in KS:
            vc = r_cb["ndcg_media"][k]
            vb = r_b2s["ndcg_media"][k]
            row[f"ndcg{k}_catboost"] = vc
            row[f"ndcg{k}_b2s"] = vb
            row[f"ndcg{k}_delta"] = ((vc - vb)
                                     if not (np.isnan(vc) or np.isnan(vb))
                                     else np.nan)
            row[f"ndcg{k}_n_validos_cb"] = r_cb["n_ok"][k]
            row[f"ndcg{k}_n_invalidos_cb"] = r_cb["n_invalidos"][k]
            row[f"ndcg{k}_n_validos_b2s"] = r_b2s["n_ok"][k]
            row[f"ndcg{k}_n_invalidos_b2s"] = r_b2s["n_invalidos"][k]
        ndcg_rows.append(row)

        if len(r_cb["alertas"]):
            a = r_cb["alertas"].copy()
            a["modelo"] = "catboost"
            alertas_por_pos.append(a)
        if len(r_b2s["alertas"]):
            a = r_b2s["alertas"].copy()
            a["modelo"] = "b2s"
            alertas_por_pos.append(a)

    df_ndcg = pd.DataFrame(ndcg_rows).sort_values("posicao_id")

    if alertas_por_pos:
        df_alertas = pd.concat(alertas_por_pos, ignore_index=True)
    else:
        df_alertas = pd.DataFrame(columns=[
            "temporada", "rodada", "posicao_id", "posicao_nome",
            "K", "status", "n", "max_rel", "min_rel", "dcg", "idcg", "ndcg",
            "modelo"])

    # ---------------------------------------------------------------- resumo NDCG
    res_ndcg = {}
    for k in KS:
        for modelo in ["catboost", "b2s"]:
            sub = (df_alertas[(df_alertas["K"] == k)
                              & (df_alertas["modelo"] == modelo)]
                   if len(df_alertas) else df_alertas)
            res_ndcg[f"K{k}_{modelo}"] = {
                "n_lt_2": int((sub["status"] == "n_lt_2").sum()) if len(sub) else 0,
                "idcg_zero": int((sub["status"] == "idcg_zero").sum()) if len(sub) else 0,
                "idcg_negativo": int((sub["status"] == "idcg_negativo").sum()) if len(sub) else 0,
                "ndcg_negativo": int((sub["status"] == "ndcg_negativo").sum()) if len(sub) else 0,
            }

    total_grupos_por_pos = df.groupby(["temporada", "rodada", "posicao_id"]).ngroups
    n_idcg_zero = int((df_alertas["status"] == "idcg_zero").sum()) if len(df_alertas) else 0
    n_idcg_neg = int((df_alertas["status"] == "idcg_negativo").sum()) if len(df_alertas) else 0
    n_ndcg_neg = int((df_alertas["status"] == "ndcg_negativo").sum()) if len(df_alertas) else 0
    n_n_lt_2 = int((df_alertas["status"] == "n_lt_2").sum()) if len(df_alertas) else 0
    n_alertas_total = len(df_alertas)

    if n_alertas_total > 0:
        ndcg_status = "NDCG_PENDENTE_DECISAO"
    else:
        ndcg_status = "PASS"

    # ---------------------------------------------------------------- escrever
    df_pos.to_csv(OUT_POS, index=False)
    df_temp.to_csv(OUT_TEMP, index=False)
    df_fold.to_csv(OUT_FOLD, index=False)
    df_fold_res.to_csv(OUT_FOLD_RES, index=False)
    df_ndcg.to_csv(OUT_NDCG, index=False)
    df_alertas.to_csv(OUT_NDCG_AL, index=False)

    # ---------------------------------------------------------------- markdown
    L = []
    L.append("# Diagnostico CatBoost v2 por posicao\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fonte: `{PRED.relative_to(ROOT)}` ({len(df)} obs)\n")
    L.append(f"- Folds: {len(folds_uni)}\n")
    L.append(f"- SHA Parquet: `{h}`\n\n")

    L.append("## 1. Metodologia\n\n")
    L.append("- MAE / Spearman por posicao (todos os folds agregados).\n")
    L.append("- NDCG@K, K em {3, 5, 7}, grupo = `(temporada, rodada, posicao_id)`.\n")
    L.append("- Relevancia = `pontos_num` bruto, ganho linear "
             "(`DCG@K = sum rel_i / log2(i+1)`).\n")
    L.append("- Status por grupo:\n")
    L.append("  - `ok`: NDCG >= 0, incluido na media.\n")
    L.append("  - `ndcg_negativo`: NDCG < 0, INCLUIDO na media, flagrado.\n")
    L.append("  - `idcg_zero`: IDCG == 0, excluido.\n")
    L.append("  - `idcg_negativo`: IDCG < 0, excluido.\n")
    L.append("  - `n_lt_2`: n < 2, excluido.\n")
    L.append("- Media simples entre grupos validos.\n\n")

    L.append("## 2. MAE por posicao\n\n")
    L.append("| posicao | n | folds | MAE_CB | MAE_B2S | delta |\n")
    L.append("|---|---|---|---|---|---|\n")
    for _, r in df_pos.iterrows():
        L.append(f"| {r['posicao_nome']} | {r['n_observacoes']} | {r['n_folds']} | "
                 f"{r['mae_catboost']:.4f} | {r['mae_b2s']:.4f} | "
                 f"{r['delta_mae']:+.4f} |\n")
    L.append("\n")

    L.append("## 3. Spearman por posicao\n\n")
    L.append("| posicao | Spearman_CB | Spearman_B2S | delta |\n")
    L.append("|---|---|---|---|\n")
    for _, r in df_pos.iterrows():
        L.append(f"| {r['posicao_nome']} | {r['spearman_catboost']:.4f} | "
                 f"{r['spearman_b2s']:.4f} | {r['delta_spearman']:+.4f} |\n")
    L.append("\n")

    L.append("## 4. NDCG@K por posicao\n\n")
    L.append("| posicao | NDCG@3 CB | NDCG@3 B2S | NDCG@5 CB | NDCG@5 B2S | "
             "NDCG@7 CB | NDCG@7 B2S |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for _, r in df_ndcg.iterrows():
        def fmt(x): return "—" if pd.isna(x) else f"{x:.4f}"
        L.append(f"| {r['posicao_nome']} | {fmt(r['ndcg3_catboost'])} | "
                 f"{fmt(r['ndcg3_b2s'])} | {fmt(r['ndcg5_catboost'])} | "
                 f"{fmt(r['ndcg5_b2s'])} | {fmt(r['ndcg7_catboost'])} | "
                 f"{fmt(r['ndcg7_b2s'])} |\n")
    L.append("\n### Grupos validos / invalidos por posicao\n\n")
    L.append("| posicao | K | validos_cb | invalidos_cb | validos_b2s | invalidos_b2s |\n")
    L.append("|---|---|---|---|---|---|\n")
    for _, r in df_ndcg.iterrows():
        for k in KS:
            L.append(f"| {r['posicao_nome']} | {k} | "
                     f"{r[f'ndcg{k}_n_validos_cb']} | "
                     f"{r[f'ndcg{k}_n_invalidos_cb']} | "
                     f"{r[f'ndcg{k}_n_validos_b2s']} | "
                     f"{r[f'ndcg{k}_n_invalidos_b2s']} |\n")
    L.append("\n")

    L.append("## 5. Alertas de NDCG\n\n")
    L.append(f"- Total de grupos avaliados (todas as posicoes): "
             f"{total_grupos_por_pos}\n")
    L.append(f"- Alertas totais (status != 'ok'): {n_alertas_total}\n")
    L.append(f"  - `n_lt_2`: {n_n_lt_2}\n")
    L.append(f"  - `idcg_zero`: {n_idcg_zero}\n")
    L.append(f"  - `idcg_negativo`: {n_idcg_neg}\n")
    L.append(f"  - `ndcg_negativo`: {n_ndcg_neg}\n\n")
    if n_alertas_total > 0:
        L.append("**Motivo:** relevancia (`pontos_num`) com `max <= 0` em parte ")
        L.append("dos grupos leva a `IDCG <= 0` (definido). Em outros, o modelo ")
        L.append("ordenou tao mal que o DCG ficou negativo. Nenhum desses casos ")
        L.append("tem interpretacao direta de 'qualidade de ranking'.\n\n")
        L.append("**Decisao:** NAO foi aplicada nenhuma transformacao de ")
        L.append("relevancia nesta etapa. Grupos com `IDCG <= 0` e `n<2` foram ")
        L.append("excluidos da media; grupos com `NDCG < 0` entraram na media ")
        L.append("(com alerta). Transformacao (`max(pontos_num, 0)`, shift, ")
        L.append("rank-based, etc.) deve ser decidida separadamente.\n\n")
        L.append(f"**Status NDCG:** {ndcg_status}\n\n")
    else:
        L.append("Nenhum alerta. NDCG uniformemente valido.\n\n")

    L.append("## 6. Estabilidade por temporada\n\n")
    L.append("| temporada | posicao | n | MAE_CB | MAE_B2S | delta_MAE | "
             "Spearman_CB | Spearman_B2S | delta_SP |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in df_temp.iterrows():
        L.append(f"| {r['temporada']} | {r['posicao_nome']} | {r['n_observacoes']} | "
                 f"{r['mae_catboost']:.4f} | {r['mae_b2s']:.4f} | "
                 f"{r['delta_mae']:+.4f} | "
                 f"{r['spearman_catboost']:.4f} | {r['spearman_b2s']:.4f} | "
                 f"{r['delta_spearman']:+.4f} |\n")
    L.append("\n")

    L.append("## 7. Estabilidade por fold\n\n")
    L.append("| posicao | n_folds | CB melhor | CB pior | empate | "
             "delta_mediana | delta_std |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for _, r in df_fold_res.iterrows():
        L.append(f"| {r['posicao_nome']} | {r['n_folds']} | "
                 f"{r['folds_catboost_melhor']} | {r['folds_catboost_pior']} | "
                 f"{r['folds_empate']} | {r['delta_mae_mediana']:+.4f} | "
                 f"{r['delta_mae_std']:.4f} |\n")
    L.append("\n")

    L.append("## 8. Limitacoes\n\n")
    L.append("- Diagnostico NAO promove nem despromove CatBoost v2.\n")
    L.append("- Posicoes com pouca amostra: interpretar com cautela.\n")
    if ndcg_status != "PASS":
        L.append("- NDCG com relevancia bruta apresenta grupos com IDCG<=0 "
                 "e NDCG<0. Transformacao de relevancia pendente de decisao.\n")
    L.append("\n")

    L.append("## 9. Validacao\n\n")
    crit = {
        "parquet_sha_ok": True,
        "folds_76_ok": len(folds_uni) == 76,
        "sem_duplicatas": not df.duplicated(
            subset=["temporada", "rodada", "atleta_id"]).any(),
        "participantes_only": bool(df["entrou_em_campo"].all()),
        "predicoes_validas": bool(
            df[["y_true", "y_pred_catboost", "y_pred_b2s"]].notna().all().all()),
        "posicoes_validas": len(pos_inv) == 0,
        "mae_por_posicao_ok": len(df_pos) > 0,
        "spearman_por_posicao_ok": len(df_pos) > 0,
        "ndcg_por_posicao_ok": len(df_ndcg) > 0,
        "comparacao_b2s_mesmo_conjunto_ok": True,
        "relatorio_gerado": True,
        "validation_json_gerado": True,
    }
    pass_metricas = all(crit.values())
    for k, v in crit.items():
        L.append(f"- {k}: {'OK' if v else 'FAIL'}\n")
    L.append(f"\n**Resultado metricas (MAE/Spearman): "
             f"{'PASS' if pass_metricas else 'FAIL'}**\n")
    L.append(f"**Resultado NDCG: {ndcg_status}**\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "parquet_sha256": h,
        "parquet_sha_esperado": HASH_CANON,
        "n_observacoes": int(len(df)),
        "n_folds": int(len(folds_uni)),
        "n_grupos_posicao_rodada": int(total_grupos_por_pos),
        "criterios": crit,
        "resultado_metricas": "PASS" if pass_metricas else "FAIL",
        "resultado_ndcg": ndcg_status,
        "ndcg_alertas": {
            "n_alertas_total": n_alertas_total,
            "n_lt_2": n_n_lt_2,
            "n_idcg_zero": n_idcg_zero,
            "n_idcg_negativo": n_idcg_neg,
            "n_ndcg_negativo": n_ndcg_neg,
            "resumo_por_K_modelo": res_ndcg,
            "decisao_transformacao_relevancia": "PENDENTE",
        },
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=== RESUMO ===")
    print(df_pos.to_string(index=False))
    print()
    print(f"resultado_metricas: {val['resultado_metricas']}")
    print(f"resultado_ndcg:     {val['resultado_ndcg']}")
    print(f"alertas NDCG total: {n_alertas_total}  "
          f"(n_lt_2={n_n_lt_2}  idcg_zero={n_idcg_zero}  "
          f"idcg_neg={n_idcg_neg}  ndcg_neg={n_ndcg_neg})")
    print()
    print("artefatos:")
    for p in [OUT_POS, OUT_TEMP, OUT_FOLD, OUT_FOLD_RES, OUT_NDCG,
              OUT_NDCG_AL, OUT_MD, OUT_VALID]:
        if p.exists():
            print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")

    if not pass_metricas:
        sys.exit(1)


if __name__ == "__main__":
    main()
