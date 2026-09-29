#!/usr/bin/env python3
"""Analise fold-a-fold CatBoost v1 vs B2S: Spearman, MAE, correlacoes.

Le apenas os CSVs ja existentes. Nao treina, nao altera nada.
"""
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def _detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz nao encontrada")


ROOT = _detect_root()
REP = ROOT/"reports"

CB_FOLDS = REP/"catboost_v1_folds.csv"
B2S_FOLDS = REP/"baseline_b2s_folds.csv"
OUT_MD = REP/"catboost_v1_spearman_analysis.md"
OUT_CSV = REP/"catboost_v1_spearman_folds.csv"

cb = pd.read_csv(CB_FOLDS)
b2s = pd.read_csv(B2S_FOLDS)

# canonicalizar B2S (nome oficial das colunas: b2s_mae, b2s_spearman)
b2s_sel = b2s[["temporada","rodada_alvo","b2s_mae","b2s_spearman"]].rename(
    columns={"b2s_mae":"b2s_mae_oficial","b2s_spearman":"b2s_sp_oficial"})

# cb ja' tem b2s_mae; remover para evitar conflito
df = cb.drop(columns=["b2s_mae"], errors="ignore").merge(
    b2s_sel, on=["temporada","rodada_alvo"], how="left")

df["delta_mae"]      = df["cb_mae"] - df["b2s_mae_oficial"]
df["delta_spearman"] = df["cb_spearman"] - df["b2s_sp_oficial"]
df["delta_sem_mae"]  = df["cb_sem_confronto_mae"] - df["b2s_mae_oficial"]

df["mae_win"]  = df["delta_mae"] < 0
df["mae_loss"] = df["delta_mae"] > 0
df["mae_tie"]  = df["delta_mae"] == 0
df["sp_win"]   = df["delta_spearman"] > 0
df["sp_loss"]  = df["delta_spearman"] < 0
df["sp_tie"]   = df["delta_spearman"] == 0

df = df.sort_values(["temporada","rodada_alvo"]).reset_index(drop=True)

n = len(df)
n_sp_win  = int(df["sp_win"].sum())
n_sp_loss = int(df["sp_loss"].sum())
n_sp_tie  = int(df["sp_tie"].sum())
n_mae_win  = int(df["mae_win"].sum())
n_mae_loss = int(df["mae_loss"].sum())
n_mae_tie  = int(df["mae_tie"].sum())

sp_stats  = df["delta_spearman"].describe(percentiles=[.1,.25,.5,.75,.9])
mae_stats = df["delta_mae"].describe(percentiles=[.1,.25,.5,.75,.9])
corr      = float(df["delta_mae"].corr(df["delta_spearman"]))

ct = pd.crosstab(df["mae_win"], df["sp_win"], margins=True)

por_temp = []
for t in sorted(df["temporada"].unique()):
    sub = df[df["temporada"] == t]
    por_temp.append({
        "temporada": t,
        "n_folds": len(sub),
        "sp_cb_wins": int(sub["sp_win"].sum()),
        "sp_b2s_wins": int(sub["sp_loss"].sum()),
        "sp_ties": int(sub["sp_tie"].sum()),
        "sp_delta_mean":   float(sub["delta_spearman"].mean()),
        "sp_delta_median": float(sub["delta_spearman"].median()),
        "mae_cb_wins":  int(sub["mae_win"].sum()),
        "mae_b2s_wins": int(sub["mae_loss"].sum()),
        "mae_delta_mean": float(sub["delta_mae"].mean()),
    })
por_temp = pd.DataFrame(por_temp)

top_losses = df.nsmallest(10, "delta_spearman")[[
    "temporada","rodada_alvo","n_test_metric",
    "cb_spearman","b2s_sp_oficial","delta_spearman",
    "cb_mae","b2s_mae_oficial","delta_mae"]].copy()
top_gains  = df.nlargest(10, "delta_spearman")[[
    "temporada","rodada_alvo","n_test_metric",
    "cb_spearman","b2s_sp_oficial","delta_spearman",
    "cb_mae","b2s_mae_oficial","delta_mae"]].copy()

losing = df[df["mae_loss"]].sort_values("delta_mae", ascending=False)

df.to_csv(OUT_CSV, index=False)

L = []
L.append("# CatBoost v1 — Analise fold-a-fold (Spearman e MAE)\n\n")
L.append(f"- Gerado em: {datetime.now().isoformat(timespec='seconds')}\n")
L.append(f"- Folds: {n}\n\n")

L.append("## Como o CatBoost se comporta em Spearman?\n\n")
L.append(f"- CB ganha Spearman: **{n_sp_win}/{n}**\n")
L.append(f"- B2S ganha Spearman: **{n_sp_loss}/{n}**\n")
L.append(f"- Empates Spearman: {n_sp_tie}\n\n")
L.append("Distribuicao de delta_spearman (cb - b2s):\n\n")
L.append("| estat | valor |\n|---|---|\n")
for k in ["count","mean","std","min","10%","25%","50%","75%","90%","max"]:
    L.append(f"| {k} | {sp_stats[k]:.4f} |\n")
L.append("\n")

L.append("## Como o CatBoost se comporta em MAE?\n\n")
L.append(f"- CB ganha MAE: **{n_mae_win}/{n}**\n")
L.append(f"- B2S ganha MAE: **{n_mae_loss}/{n}**\n")
L.append(f"- Empates MAE: {n_mae_tie}\n\n")
L.append("Distribuicao de delta_mae (cb - b2s):\n\n")
L.append("| estat | valor |\n|---|---|\n")
for k in ["count","mean","std","min","10%","25%","50%","75%","90%","max"]:
    L.append(f"| {k} | {mae_stats[k]:.4f} |\n")
L.append("\n")

L.append("## Correlation entre delta_mae e delta_spearman\n\n")
L.append(f"corr(delta_mae, delta_spearman) = **{corr:.4f}**\n\n")
L.append("Cross-tab (MAE win x Spearman win):\n\n")
L.append("```\n" + str(ct) + "\n```\n\n")

L.append("## Por temporada\n\n")
L.append(por_temp.to_string(index=False))
L.append("\n\n")

L.append("## Top 10 maiores perdas de Spearman\n\n")
L.append(top_losses.to_string(index=False))
L.append("\n\n")

L.append("## Top 10 maiores ganhos de Spearman\n\n")
L.append(top_gains.to_string(index=False))
L.append("\n\n")

L.append(f"## Os {len(losing)} folds onde CatBoost perdeu MAE\n\n")
L.append(losing[["temporada","rodada_alvo","n_test_metric",
                  "cb_mae","b2s_mae_oficial","delta_mae",
                  "cb_spearman","b2s_sp_oficial","delta_spearman",
                  "cb_sem_confronto_mae"]].to_string(index=False))
L.append("\n\n")

OUT_MD.write_text("".join(L), encoding="utf-8")

print("=== Spearman ===")
print(f"CB ganha: {n_sp_win}/{n}   B2S ganha: {n_sp_loss}/{n}   empates: {n_sp_tie}")
print(f"delta_spearman: mean={sp_stats['mean']:+.4f}  median={sp_stats['50%']:+.4f}  std={sp_stats['std']:.4f}")
print(f"  min={sp_stats['min']:+.4f}  p25={sp_stats['25%']:+.4f}  p75={sp_stats['75%']:+.4f}  max={sp_stats['max']:+.4f}")
print()
print("=== MAE ===")
print(f"CB ganha: {n_mae_win}/{n}   B2S ganha: {n_mae_loss}/{n}   empates: {n_mae_tie}")
print(f"delta_mae: mean={mae_stats['mean']:+.4f}  median={mae_stats['50%']:+.4f}")
print()
print(f"corr(delta_mae, delta_spearman) = {corr:+.4f}")
print()
print("Cross-tab MAE win x Spearman win:")
print(ct)
print()
print("Por temporada:")
print(por_temp.to_string(index=False))
print()
print("Top 5 maiores perdas de Spearman:")
print(top_losses.head(5)[["temporada","rodada_alvo","cb_spearman","b2s_sp_oficial","delta_spearman","delta_mae"]].to_string(index=False))
print()
print(f"Artefatos:")
print(f"  {OUT_CSV.relative_to(ROOT)}")
print(f"  {OUT_MD.relative_to(ROOT)}")
