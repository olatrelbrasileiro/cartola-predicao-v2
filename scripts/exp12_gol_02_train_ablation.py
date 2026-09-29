#!/usr/bin/env python
"""
scripts/exp12_gol_02_train_ablation.py

Ablação walk-forward para goleiros. Compara 4 braços nas MESMAS linhas,
mesmos folds e mesmo algoritmo:

  A = gk_* + mando_C                            (baseline individual)
  B = A + team_* (novo contexto)                (+ próprio time)
  C = B + opp_* (novo contexto)                 (+ adversário)
  D = C + odds pré-jogo                         (+ mercado)

Regras:
  - Lê data/interim/exp12/gol_pooling_2024_2026.parquet (output do Script 01).
  - NÃO altera o Script 01, NÃO reconstrói features.
  - NÃO existe gk_GS; não é procurado, não é sintetizado.
  - NÃO usa as features antigas 2026-only (team_goals_*, team_shots_*, etc.).
  - Walk-forward: train = (T'<T) | (T'==T & R'<R) ; test = (T,R).
  - Sem drop de linhas por NaN entre braços; CatBoost lida nativamente.
  - Hiperparâmetros fixos (mesmos em A/B/C/D).
  - Sem tuning, sem ensemble, sem feature selection.

Robustez:
  - Features obrigatórias de A, B e C ausentes -> RuntimeError.
  - Odds canônicas não resolvidas (faltando qualquer de home/draw/away) -> RuntimeError.
  - Qualquer inf/-inf em qualquer feature usada -> RuntimeError.
  - Target NaN em linha elegível -> RuntimeError.
  - Chave duplicada -> RuntimeError.

Saídas:
  reports/exp12/exp12_gol_ablation_summary.csv
  reports/exp12/exp12_gol_ablation_by_fold.csv
  reports/exp12/exp12_gol_ablation_by_season.csv
  reports/exp12/exp12_gol_ablation_report.md
  reports/exp12/exp12_gol_ablation_manifest.json
  data/interim/exp12/gol_ablation_predictions.parquet
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parent.parent
IN_PARQUET = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"

REPORTS_DIR = ROOT / "reports" / "exp12"
OUT_SUMMARY  = REPORTS_DIR / "exp12_gol_ablation_summary.csv"
OUT_BY_FOLD  = REPORTS_DIR / "exp12_gol_ablation_by_fold.csv"
OUT_BY_SEASON = REPORTS_DIR / "exp12_gol_ablation_by_season.csv"
OUT_REPORT   = REPORTS_DIR / "exp12_gol_ablation_report.md"
OUT_MANIFEST = REPORTS_DIR / "exp12_gol_ablation_manifest.json"
OUT_PREDS    = ROOT / "data" / "interim" / "exp12" / "gol_ablation_predictions.parquet"

TARGET = "pontos_num"
ELIGIBLE_COL = "elegivel_modelo"

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 4,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "loss_function": "RMSE",
}

MIN_TRAIN_ROWS = 50
MIN_TEST_ROWS  = 3


# ============================================================================
# Candidatos
# ============================================================================

A_CANDIDATES = [
    "gk_pontos_last1",
    "gk_pontos_mean_3",       "gk_pontos_mean_3_n_obs",
    "gk_pontos_mean_5",       "gk_pontos_mean_5_n_obs",
    "gk_pontos_mean_hist",
    "gk_DE_mean_3",           "gk_DE_mean_3_n_obs",
    "gk_DE_mean_5",           "gk_DE_mean_5_n_obs",
    "mando_C",
]

B_CANDIDATES = [
    "team_gols_sofridos_mean_3",       "team_gols_sofridos_mean_3_n_obs",
    "team_gols_sofridos_mean_5",       "team_gols_sofridos_mean_5_n_obs",
    "team_clean_sheet_rate_mean_3",    "team_clean_sheet_rate_mean_3_n_obs",
    "team_clean_sheet_rate_mean_5",    "team_clean_sheet_rate_mean_5_n_obs",
]

C_CANDIDATES = [
    "opp_gols_marcados_mean_3",  "opp_gols_marcados_mean_3_n_obs",
    "opp_FTT_mean_3",            "opp_FTT_mean_3_n_obs",
    "opp_FTT_mean_5",            "opp_FTT_mean_5_n_obs",
]

# Aliases aceitos para odds. Todas as 3 canônicas são OBRIGATÓRIAS.
D_CANDIDATES_ALIASES = {
    "canon_odd_home": ["odds_home", "odd_p_home", "odd_home", "p_home"],
    "canon_odd_draw": ["odds_draw", "odd_p_draw", "odd_draw", "p_draw"],
    "canon_odd_away": ["odds_away", "odd_p_away", "odd_away", "p_away"],
}

# Apenas para AVISO (não bloqueia).
OLD_FEATURES_PREFIX = (
    "team_goals_scored_last", "team_goals_conceded_last",
    "team_clean_sheet_rate_last",
    "team_shots_last", "team_shots_on_target_last",
    "opp_goals_scored_last", "opp_goals_conceded_last",
    "opp_clean_sheet_rate_last",
    "opp_shots_last", "opp_shots_on_target_last",
)


# ============================================================================
# Util
# ============================================================================

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _spearman(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 3 or len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan")
    try:
        return float(spearmanr(y, p).statistic)
    except Exception:
        return float("nan")


def _descriptive(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append({"feature": c, "present": False, "dtype": None,
                         "n": 0, "nan_pct": np.nan, "inf_count": np.nan,
                         "min": np.nan, "p01": np.nan, "median": np.nan,
                         "p99": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan})
            continue
        raw = df[c]
        s = pd.to_numeric(raw, errors="coerce").astype(float)
        arr = s.to_numpy()
        n = len(arr)
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        valid = s[~nan_mask & ~inf_mask]
        rows.append({
            "feature": c,
            "present": True,
            "dtype": str(raw.dtype),
            "n": n,
            "nan_pct": 100.0 * float(nan_mask.sum()) / n,
            "inf_count": int(inf_mask.sum()),
            "min": float(valid.min()) if len(valid) else np.nan,
            "p01": float(valid.quantile(0.01)) if len(valid) else np.nan,
            "median": float(valid.median()) if len(valid) else np.nan,
            "p99": float(valid.quantile(0.99)) if len(valid) else np.nan,
            "max": float(valid.max()) if len(valid) else np.nan,
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "std": float(valid.std()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Discovery (com obrigatoriedade estrita)
# ============================================================================

def discover(df: pd.DataFrame) -> dict:
    cols = set(df.columns)

    a_present = [c for c in A_CANDIDATES if c in cols]
    a_missing = [c for c in A_CANDIDATES if c not in cols]

    b_present = [c for c in B_CANDIDATES if c in cols]
    b_missing = [c for c in B_CANDIDATES if c not in cols]

    c_present = [c for c in C_CANDIDATES if c in cols]
    c_missing = [c for c in C_CANDIDATES if c not in cols]

    d_map: dict[str, str | None] = {}
    d_present: list[str] = []
    d_missing: list[str] = []
    for canon, aliases in D_CANDIDATES_ALIASES.items():
        chosen = next((a for a in aliases if a in cols), None)
        d_map[canon] = chosen
        if chosen is not None:
            d_present.append(canon)
        else:
            d_missing.append(canon)

    old_present = sorted([
        c for c in cols
        if any(c.startswith(p) for p in OLD_FEATURES_PREFIX)
    ])

    return {
        "A_present": a_present, "A_missing": a_missing,
        "B_present": b_present, "B_missing": b_missing,
        "C_present": c_present, "C_missing": c_missing,
        "D_present_canon": d_present, "D_missing_canon": d_missing,
        "D_map": d_map,
        "old_f2_features_present": old_present,
    }


def enforce_required_features(dis: dict) -> None:
    """
    Erro se QUALQUER feature candidata de A, B, C estiver ausente,
    OU se qualquer uma das 3 odds canônicas não foi resolvida.
    Sem fallback, sem braço reduzido.
    """
    problemas: list[str] = []
    if dis["A_missing"]:
        problemas.append(f"A ausentes: {dis['A_missing']}")
    if dis["B_missing"]:
        problemas.append(f"B ausentes: {dis['B_missing']}")
    if dis["C_missing"]:
        problemas.append(f"C ausentes: {dis['C_missing']}")
    if dis["D_missing_canon"]:
        problemas.append(f"D canônicas não resolvidas: {dis['D_missing_canon']}")
    if problemas:
        raise RuntimeError(
            "Features obrigatórias ausentes no parquet do Script 01. "
            "Não há fallback. Corrija o Script 01 antes de rodar:\n  - "
            + "\n  - ".join(problemas)
        )


def enforce_no_inf(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Erro se QUALQUER feature usada tiver inf/-inf.
    """
    inf_report: list[tuple[str, int]] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()
        n_inf = int(np.isinf(s).sum())
        if n_inf > 0:
            inf_report.append((c, n_inf))
    if inf_report:
        msg = "\n".join(f"  - {c}: {n}" for c, n in inf_report)
        raise RuntimeError(
            "inf/-inf detectado em features usadas. Corrija o Script 01 "
            "antes de rodar. Features com inf:\n" + msg
        )


def print_discovery(dis: dict) -> None:
    print("=" * 78)
    print("FEATURE DISCOVERY")
    print("=" * 78)

    for name, present_key, missing_key in [
        ("A (individual)", "A_present", "A_missing"),
        ("B (team)",       "B_present", "B_missing"),
        ("C (opponent)",   "C_present", "C_missing"),
    ]:
        print(f"\n{name} — {len(dis[present_key])} presentes:")
        for f in dis[present_key]:
            print(f"    + {f}")
        if dis[missing_key]:
            print(f"  AUSENTES ({len(dis[missing_key])}):")
            for f in dis[missing_key]:
                print(f"    - {f}")

    print(f"\nD (odds) — {len(dis['D_present_canon'])} canônicas resolvidas:")
    for canon in dis["D_present_canon"]:
        real = dis["D_map"][canon]
        print(f"    + {canon}  <-  coluna real '{real}'")
    if dis["D_missing_canon"]:
        print(f"  AUSENTES ({len(dis['D_missing_canon'])}):")
        for canon in dis["D_missing_canon"]:
            print(f"    - {canon}  (nenhum alias encontrado)")

    if dis["old_f2_features_present"]:
        print("\n[AVISO] Features F2 antigas presentes no parquet (NÃO usadas nesta EXP):")
        for f in dis["old_f2_features_present"]:
            print(f"    ! {f}")
    print()


def rename_d_aliases(df: pd.DataFrame, dis: dict) -> pd.DataFrame:
    df = df.copy()
    for canon, real in dis["D_map"].items():
        if real is None:
            continue
        df[canon] = pd.to_numeric(df[real], errors="coerce").astype(float)
    return df


# ============================================================================
# Walk-forward
# ============================================================================

def build_folds(df_elig: pd.DataFrame) -> list[tuple[int, int]]:
    key = (df_elig[["temporada", "rodada"]]
           .drop_duplicates()
           .sort_values(["temporada", "rodada"], kind="mergesort"))
    return [tuple(map(int, x)) for x in key.values]


def train_mask(df: pd.DataFrame, T: int, R: int) -> pd.Series:
    return (df["temporada"] < T) | ((df["temporada"] == T) & (df["rodada"] < R))


def run_ablation(df: pd.DataFrame,
                 feats: dict[str, list[str]],
                 elig_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_folds(df[elig_mask])
    print(f"Folds (T,R) detectados: {len(folds)}  —  de {folds[0]} a {folds[-1]}")

    fold_rows: list[dict] = []
    preds_all: list[pd.DataFrame] = []

    for T, R in folds:
        tr = train_mask(df, T, R) & elig_mask
        te = (df["temporada"] == T) & (df["rodada"] == R) & elig_mask

        n_tr = int(tr.sum())
        n_te = int(te.sum())
        if n_tr < MIN_TRAIN_ROWS or n_te < MIN_TEST_ROWS:
            continue

        row: dict = {"temporada": int(T), "rodada": int(R),
                     "n_train": n_tr, "n_test": n_te}

        preds_fold = pd.DataFrame({
            "temporada": T,
            "rodada": R,
            "atleta_id": df.loc[te, "atleta_id"].values,
            "pontos_real": df.loc[te, TARGET].astype(float).values,
        })

        for arm in ("A", "B", "C", "D"):
            fcols = feats[arm]
            Xtr = df.loc[tr, fcols]
            ytr = df.loc[tr, TARGET].astype(float)
            Xte = df.loc[te, fcols]
            yte = df.loc[te, TARGET].astype(float)

            model = CatBoostRegressor(**CATBOOST_PARAMS)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            mae = float(np.mean(np.abs(yte.values - pred)))
            rmse = float(np.sqrt(np.mean((yte.values - pred) ** 2)))
            sp = _spearman(yte.values, pred)

            row[f"MAE_{arm}"] = mae
            row[f"RMSE_{arm}"] = rmse
            row[f"Spearman_{arm}"] = sp

            preds_fold[f"pred_{arm}"] = pred

        preds_all.append(preds_fold)
        fold_rows.append(row)

        print(f"  {T} R{R:>2d}  n_train={n_tr:>4d} n_test={n_te:>3d}  "
              f"MAE A/B/C/D = {row['MAE_A']:.4f} / {row['MAE_B']:.4f} / "
              f"{row['MAE_C']:.4f} / {row['MAE_D']:.4f}")

    preds = pd.concat(preds_all, ignore_index=True)
    by_fold = pd.DataFrame(fold_rows)
    return preds, by_fold


# ============================================================================
# Sumário
# ============================================================================

def summarize_pooled(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ("A", "B", "C", "D"):
        y = preds["pontos_real"].astype(float).values
        p = preds[f"pred_{arm}"].astype(float).values
        rows.append({
            "arm": arm,
            "n": int(len(preds)),
            "MAE": float(np.mean(np.abs(y - p))),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "Spearman_pooled": _spearman(y, p),
        })
    return pd.DataFrame(rows)


def summarize_by_season(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for T, g in preds.groupby("temporada"):
        for arm in ("A", "B", "C", "D"):
            y = g["pontos_real"].astype(float).values
            p = g[f"pred_{arm}"].astype(float).values
            rows.append({
                "temporada": int(T),
                "arm": arm,
                "n": int(len(g)),
                "MAE": float(np.mean(np.abs(y - p))),
                "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
                "Spearman": _spearman(y, p),
            })
    return pd.DataFrame(rows)


def summarize_spearman_mean_folds(by_fold: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for arm in ("A", "B", "C", "D"):
        col = f"Spearman_{arm}"
        if col in by_fold.columns:
            vals = by_fold[col].replace([np.inf, -np.inf], np.nan).dropna()
            out[arm] = float(vals.mean()) if len(vals) else float("nan")
        else:
            out[arm] = float("nan")
    return out


def paired_win_rates(preds: pd.DataFrame, by_fold: pd.DataFrame) -> dict:
    out: dict = {}
    for new_arm, base_arm in [("B", "A"), ("C", "B"), ("D", "C")]:
        a_err = np.abs(preds["pontos_real"].values - preds[f"pred_{base_arm}"].values)
        b_err = np.abs(preds["pontos_real"].values - preds[f"pred_{new_arm}"].values)
        diff = b_err - a_err

        fold_win = None
        mae_new = f"MAE_{new_arm}"
        mae_base = f"MAE_{base_arm}"
        if mae_new in by_fold.columns and mae_base in by_fold.columns:
            mask = by_fold[mae_new].notna() & by_fold[mae_base].notna()
            if mask.any():
                fold_win = float((by_fold.loc[mask, mae_new] < by_fold.loc[mask, mae_base]).mean())

        out[f"{new_arm}_vs_{base_arm}"] = {
            "n_obs": int(len(diff)),
            "win_rate_obs": float(np.mean(diff < -1e-12)),
            "tie_rate_obs": float(np.mean(np.abs(diff) <= 1e-12)),
            "lose_rate_obs": float(np.mean(diff > 1e-12)),
            "mean_delta_MAE_obs": float(np.mean(diff)),
            "median_delta_MAE_obs": float(np.median(diff)),
            "win_rate_fold": fold_win,
        }
    return out


# ============================================================================
# Auditoria
# ============================================================================

def audit_dataset(df: pd.DataFrame) -> pd.Series:
    print("=" * 78)
    print("DATASET AUDIT")
    print("=" * 78)
    print(f"shape: {df.shape}")
    print(f"columns: {len(df.columns)}")
    print(f"temporadas: {sorted(df['temporada'].dropna().unique().tolist())}")
    print("rodadas por temporada:")
    print(df.groupby("temporada")["rodada"].agg(["min", "max", "nunique"]).to_string())

    if ELIGIBLE_COL not in df.columns:
        raise RuntimeError(f"Coluna '{ELIGIBLE_COL}' ausente.")
    if TARGET not in df.columns:
        raise RuntimeError(f"Coluna target '{TARGET}' ausente.")

    elig = df[ELIGIBLE_COL].fillna(False).astype(bool)
    print(f"\nelegíveis: {int(elig.sum())} / {len(df)}")
    print("elegíveis por temporada:")
    print(df[elig].groupby("temporada").size().to_string())

    n_target_na = int(df.loc[elig, TARGET].isna().sum())
    if n_target_na > 0:
        raise RuntimeError(f"{n_target_na} linhas elegíveis com target NaN.")

    dups = int(df.duplicated(subset=["temporada", "rodada", "atleta_id"]).sum())
    if dups > 0:
        raise RuntimeError(f"{dups} duplicatas em (temporada, rodada, atleta_id).")
    print("\nchave (temporada, rodada, atleta_id) única: OK")
    print()
    return elig


def print_descriptive(title: str, table: pd.DataFrame) -> None:
    print(f"--- {title} ---")
    cols = ["feature", "present", "dtype", "n", "nan_pct", "inf_count",
            "min", "p01", "median", "p99", "max", "mean", "std"]
    with pd.option_context("display.max_rows", 200, "display.width", 220,
                           "display.float_format", "{:.4f}".format):
        print(table[cols].to_string(index=False))
    print()


# ============================================================================
# Relatório
# ============================================================================

def write_report(path: Path,
                 dis: dict,
                 desc_a: pd.DataFrame, desc_b: pd.DataFrame,
                 desc_c: pd.DataFrame, desc_d: pd.DataFrame,
                 summary: pd.DataFrame,
                 by_season: pd.DataFrame,
                 by_fold: pd.DataFrame,
                 wins: dict,
                 sp_mean_folds: dict[str, float],
                 manifest: dict) -> None:
    lines: list[str] = []
    lines.append("# EXP-12 GOL — Ablation report")
    lines.append("")
    lines.append(f"_Gerado: {manifest['timestamp']}_")
    lines.append("")

    lines.append("## Feature blocks")
    lines.append("")
    for label, present_key, missing_key in [
        ("A — individual", "A_present", "A_missing"),
        ("B — + team", "B_present", "B_missing"),
        ("C — + opponent", "C_present", "C_missing"),
    ]:
        lines.append(f"**{label}** ({len(dis[present_key])} features):")
        for f in dis[present_key]:
            lines.append(f"- `{f}`")
        if dis[missing_key]:
            lines.append(f"  _ausentes: {dis[missing_key]}_")
        lines.append("")
    lines.append(f"**D — + odds** ({len(dis['D_present_canon'])} features):")
    for canon in dis["D_present_canon"]:
        lines.append(f"- `{canon}` (real: `{dis['D_map'][canon]}`)")
    if dis["D_missing_canon"]:
        lines.append(f"  _ausentes: {dis['D_missing_canon']}_")
    lines.append("")

    if dis["old_f2_features_present"]:
        lines.append("## F2 antigo — detectado mas NÃO usado")
        lines.append("")
        for f in dis["old_f2_features_present"]:
            lines.append(f"- `{f}`")
        lines.append("")

    lines.append("## Descriptive stats (por bloco)")
    lines.append("")
    for label, table in [("A", desc_a), ("B", desc_b), ("C", desc_c), ("D", desc_d)]:
        lines.append(f"### Bloco {label}")
        lines.append("")
        lines.append(table.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Pooled results")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("Spearman médio por fold:")
    lines.append("")
    for arm, v in sp_mean_folds.items():
        lines.append(f"- {arm}: {v:.4f}" if np.isfinite(v) else f"- {arm}: NaN")
    lines.append("")

    lines.append("## Por temporada")
    lines.append("")
    lines.append(by_season.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Deltas pareados (por observação)")
    lines.append("")
    lines.append("| par | n | win% | tie% | lose% | ΔMAE médio | ΔMAE mediano | win% folds |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, v in wins.items():
        fw = v["win_rate_fold"]
        fw_s = f"{100*fw:.2f}" if fw is not None else "n/a"
        lines.append(
            f"| {k} | {v['n_obs']} | {100*v['win_rate_obs']:.2f} | "
            f"{100*v['tie_rate_obs']:.2f} | {100*v['lose_rate_obs']:.2f} | "
            f"{v['mean_delta_MAE_obs']:+.4f} | {v['median_delta_MAE_obs']:+.4f} | "
            f"{fw_s} |"
        )
    lines.append("")

    lines.append("## Por fold")
    lines.append("")
    lines.append(by_fold.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Manifest")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(manifest, indent=2, default=str))
    lines.append("```")

    path.write_text("\n".join(lines))


# ============================================================================
# Main
# ============================================================================

def main():
    if not IN_PARQUET.exists():
        raise FileNotFoundError(f"Parquet do Script 01 não encontrado: {IN_PARQUET}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] lendo {IN_PARQUET} ...")
    df = pd.read_parquet(IN_PARQUET)
    print(f"     shape: {df.shape}")

    print("\n[2/7] auditoria do dataset ...")
    elig_mask = audit_dataset(df)

    print("[3/7] feature discovery ...")
    dis = discover(df)
    print_discovery(dis)

    # --- Correção 1: features obrigatórias ---
    enforce_required_features(dis)

    # Normaliza aliases de D
    df = rename_d_aliases(df, dis)

    # Listas finais canônicas
    FEATS: dict[str, list[str]] = {
        "A": list(dis["A_present"]),
        "B": list(dis["A_present"]) + list(dis["B_present"]),
        "C": list(dis["A_present"]) + list(dis["B_present"]) + list(dis["C_present"]),
        "D": list(dis["A_present"]) + list(dis["B_present"]) + list(dis["C_present"])
             + list(dis["D_present_canon"]),
    }
    print("Tamanho dos braços:", {k: len(v) for k, v in FEATS.items()})

    # Checagem extra: nenhuma feature "gk_GS" foi introduzida
    for arm, fs in FEATS.items():
        assert not any("gk_GS" in f for f in fs), \
            f"gk_GS detectado no braço {arm} — não deveria existir."

    # --- Correção 2: inf/-inf em qualquer feature usada ---
    all_feat_cols = sorted({c for fs in FEATS.values() for c in fs})
    enforce_no_inf(df, all_feat_cols)
    print(f"Verificação de inf/-inf: OK em {len(all_feat_cols)} features.")

    print("\n[4/7] estatística descritiva por bloco ...")
    desc_a = _descriptive(df, FEATS["A"])
    desc_b = _descriptive(df, [c for c in FEATS["B"] if c not in FEATS["A"]])
    desc_c = _descriptive(df, [c for c in FEATS["C"] if c not in FEATS["B"]])
    desc_d = _descriptive(df, [c for c in FEATS["D"] if c not in FEATS["C"]])

    print_descriptive("A — individual", desc_a)
    print_descriptive("B — team (features novas)", desc_b)
    print_descriptive("C — opponent (features novas)", desc_c)
    print_descriptive("D — odds", desc_d)

    for canon in dis["D_present_canon"]:
        if canon in df.columns:
            sub = df.loc[elig_mask, canon]
            print(f"  odds coverage: {canon}  NaN% = "
                  f"{100*sub.isna().mean():.2f}  (n={len(sub)})")

    print("\n[5/7] walk-forward A/B/C/D ...")
    preds, by_fold = run_ablation(df, FEATS, elig_mask)
    if len(preds) == 0:
        raise RuntimeError("Nenhum fold válido. Confira MIN_TRAIN_ROWS/MIN_TEST_ROWS.")

    print(f"\n     Total de folds executados: {len(by_fold)}")
    print(f"     Total de observações pooled: {len(preds)}")
    print("     Mesma amostra em A/B/C/D: garantida por construção (mesma mask por fold).")

    print("\n[6/7] sumarizando ...")
    summary = summarize_pooled(preds)
    by_season = summarize_by_season(preds)
    sp_mean_folds = summarize_spearman_mean_folds(by_fold)
    wins = paired_win_rates(preds, by_fold)

    print("=" * 78)
    print("POOLED RESULTS")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    print()
    print("Spearman médio por fold:")
    for arm, v in sp_mean_folds.items():
        print(f"  {arm}: {v:.4f}" if np.isfinite(v) else f"  {arm}: NaN")

    print("\n" + "=" * 78)
    print("DELTAS PAREADOS")
    print("=" * 78)
    for k, v in wins.items():
        fw = v["win_rate_fold"]
        fw_s = f"{100*fw:.2f}%" if fw is not None else "n/a"
        print(f"  {k}:  "
              f"win_obs={100*v['win_rate_obs']:.2f}%  "
              f"tie_obs={100*v['tie_rate_obs']:.2f}%  "
              f"lose_obs={100*v['lose_rate_obs']:.2f}%  "
              f"ΔMAE_mean={v['mean_delta_MAE_obs']:+.4f}  "
              f"ΔMAE_med={v['median_delta_MAE_obs']:+.4f}  "
              f"win_fold={fw_s}")

    print("\n[7/7] gravando outputs ...")

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(IN_PARQUET),
        "input_dataset_sha256": _sha(IN_PARQUET),
        "input_dataset_shape": list(df.shape),
        "target": TARGET,
        "eligibility_col": ELIGIBLE_COL,
        "catboost_params": CATBOOST_PARAMS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "min_test_rows": MIN_TEST_ROWS,
        "n_folds": int(len(by_fold)),
        "n_predictions_pooled": int(len(preds)),
        "features_A": FEATS["A"],
        "features_B": FEATS["B"],
        "features_C": FEATS["C"],
        "features_D": FEATS["D"],
        "features_missing": {
            "A": dis["A_missing"], "B": dis["B_missing"], "C": dis["C_missing"],
            "D": dis["D_missing_canon"],
        },
        "old_f2_features_present": dis["old_f2_features_present"],
        "seasons": sorted(df["temporada"].dropna().unique().tolist()),
        "rounds_per_season": (
            df.groupby("temporada")["rodada"].nunique().astype(int).to_dict()
        ),
        "pooled_summary": summary.to_dict(orient="records"),
        "paired_win_rates": wins,
        "spearman_mean_folds": sp_mean_folds,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))

    preds.to_parquet(OUT_PREDS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    by_fold.to_csv(OUT_BY_FOLD, index=False)
    by_season.to_csv(OUT_BY_SEASON, index=False)

    write_report(OUT_REPORT,
                 dis,
                 desc_a, desc_b, desc_c, desc_d,
                 summary, by_season, by_fold,
                 wins, sp_mean_folds, manifest)

    print(f"     -> {OUT_SUMMARY}")
    print(f"     -> {OUT_BY_FOLD}")
    print(f"     -> {OUT_BY_SEASON}")
    print(f"     -> {OUT_REPORT}")
    print(f"     -> {OUT_MANIFEST}")
    print(f"     -> {OUT_PREDS}")


if __name__ == "__main__":
    main()
