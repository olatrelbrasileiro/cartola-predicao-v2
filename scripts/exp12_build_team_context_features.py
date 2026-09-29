#!/usr/bin/env python
"""
EXP-12 F2 - Team context temporal features.
Granularidade: (temporada, rodada, clube_id)
Eixo temporal: RODADA OFICIAL DO CARTOLA.
Historico permitido: (temporada < T) OR (temporada == T AND rodada < R).
Data fisica NUNCA e usada como cutoff.
Mask de roster ausente: FTT/FA = NaN. gols/CS nao mascarados.
Fonte do mask: reports/exp12/f2_missing_roster_raw_audit.csv (GATE 2).
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST_PATH   = ROOT / "data" / "interim" / "exp12" / "team_match_history.parquet"
OUT_PARQUET = ROOT / "data" / "interim" / "exp12" / "team_context_features.parquet"
OUT_CSV     = ROOT / "data" / "interim" / "exp12" / "team_context_features.csv"
AUDIT_MD    = ROOT / "reports" / "exp12" / "exp12_team_context_features_audit.md"
AUDIT_CSV   = ROOT / "reports" / "exp12" / "f2_missing_roster_raw_audit.csv"

KEY_COLS = ["temporada", "rodada", "clube_id"]

TEAM_FEATURES = [
    "team_goals_scored_last3",   "team_goals_scored_last5",
    "team_goals_conceded_last3", "team_goals_conceded_last5",
    "team_clean_sheet_rate_last3","team_clean_sheet_rate_last5",
    "team_shots_last3",          "team_shots_last5",
    "team_shots_on_target_last3","team_shots_on_target_last5",
]
OPP_FEATURES = [
    "opp_goals_scored_last3",   "opp_goals_scored_last5",
    "opp_goals_conceded_last3", "opp_goals_conceded_last5",
    "opp_clean_sheet_rate_last3","opp_clean_sheet_rate_last5",
    "opp_shots_last3",          "opp_shots_last5",
    "opp_shots_on_target_last3","opp_shots_on_target_last5",
]
ALL_FEATURES = TEAM_FEATURES + OPP_FEATURES
assert len(ALL_FEATURES) == 20

OPP_COL_CANDIDATES = (
    "adversario_canon",
    "adversario_id", "clube_adversario_id", "id_adversario",
    "opp_id", "opponent_id", "opponent",
)


def load_roster_absence_keys() -> set:
    """Le o CSV do GATE 2 e devolve set de (temporada, rodada, clube_id)
    onde n_jogadores_raw == 0."""
    if not AUDIT_CSV.exists():
        raise RuntimeError(
            f"CSV de auditoria RAW nao encontrado: {AUDIT_CSV}. "
            "Rode scripts/exp12_audit_missing_roster_raw.py primeiro."
        )
    a = pd.read_csv(AUDIT_CSV)
    for c in ("temporada", "rodada", "clube_id", "n_jogadores_raw"):
        if c not in a.columns:
            raise RuntimeError(f"CSV de auditoria sem coluna obrigatoria: {c}")
    mask = a[a["n_jogadores_raw"] == 0][["temporada", "rodada", "clube_id"]]
    keys = set(map(tuple, mask.astype(int).values.tolist()))
    print(f"[F2][mask] {len(keys)} club-rounds vindos do GATE 2 (n_jogadores_raw == 0)")
    return keys


def detect_opp_col(df: pd.DataFrame) -> str:
    present = [c for c in OPP_COL_CANDIDATES if c in df.columns]
    if not present:
        raise RuntimeError(
            "Coluna de adversario canonica nao encontrada. "
            f"Esperado um de {OPP_COL_CANDIDATES}. Colunas: {list(df.columns)}"
        )
    if "adversario_canon" in present:
        return "adversario_canon"
    print(f"[F2][aviso] 'adversario_canon' ausente. Usando '{present[0]}'.")
    return present[0]


def apply_roster_absence_mask(df: pd.DataFrame, absence_keys: set) -> pd.DataFrame:
    """FTT/FA -> NaN onde (temporada,rodada,clube_id) in absence_keys.
    Nao toca gols_marcados, gols_sofridos, clean_sheet."""
    if not absence_keys:
        print("[F2][mask] conjunto vazio — nenhuma linha mascarada.")
        return df.copy()
    df = df.copy()
    is_abs = df.apply(
        lambda r: (int(r["temporada"]), int(r["rodada"]), int(r["clube_id"])) in absence_keys,
        axis=1,
    )
    df.loc[is_abs, "FTT"] = np.nan
    df.loc[is_abs, "FA"]  = np.nan
    print(f"[F2][mask] {int(is_abs.sum())} linhas mascaradas (roster ausente).")
    return df


def _ensure_ftt_fa(df: pd.DataFrame) -> pd.DataFrame:
    if "FTT" not in df.columns:
        comps = ["FD", "FF", "FT", "G", "PP"]
        miss = [c for c in comps if c not in df.columns]
        if miss:
            raise RuntimeError(f"FTT ausente e componentes faltando: {miss}.")
        df["FTT"] = df[comps].sum(axis=1, min_count=len(comps))
    if "FA" not in df.columns:
        comps = ["FD", "G"]
        miss = [c for c in comps if c not in df.columns]
        if miss:
            raise RuntimeError(f"FA ausente e componentes faltando: {miss}.")
        df["FA"] = df[comps].sum(axis=1, min_count=len(comps))
    return df


def _validate_history(df: pd.DataFrame) -> None:
    for c in ["temporada", "rodada", "clube_id", "gols_marcados", "gols_sofridos"]:
        if c not in df.columns:
            raise RuntimeError(f"Coluna obrigatoria ausente: {c}")
    n = int(df.duplicated(subset=KEY_COLS).sum())
    if n:
        raise RuntimeError(f"{n} duplicatas em {KEY_COLS}. Abortando.")


def _shift_mean(s: pd.Series, k: int) -> pd.Series:
    return s.shift(1).rolling(k, min_periods=1).mean()


def _shift_cs_rate(s: pd.Series, k: int) -> pd.Series:
    """CS observados / partidas observadas. NaN se count == 0."""
    prior = s.shift(1)
    win = prior.rolling(k, min_periods=1)
    n_obs = win.count()
    total = win.sum()
    rate = total / n_obs
    return rate.where(n_obs > 0, np.nan)


def build_team_features(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    _validate_history(df)
    opp_col = detect_opp_col(df)
    df = _ensure_ftt_fa(df)

    gs = df["gols_sofridos"]
    df["clean_sheet"] = np.where(gs.isna(), np.nan, (gs == 0).astype(float))

    df = (df.sort_values(["clube_id", "temporada", "rodada"], kind="mergesort")
            .reset_index(drop=True))

    def per_group(col, fn):
        return df.groupby("clube_id", sort=False)[col].transform(fn)

    for k in (3, 5):
        df[f"team_goals_scored_last{k}"]     = per_group("gols_marcados",  lambda s, k=k: _shift_mean(s, k))
        df[f"team_goals_conceded_last{k}"]   = per_group("gols_sofridos",  lambda s, k=k: _shift_mean(s, k))
        df[f"team_clean_sheet_rate_last{k}"] = per_group("clean_sheet",    lambda s, k=k: _shift_cs_rate(s, k))
        df[f"team_shots_last{k}"]            = per_group("FTT",            lambda s, k=k: _shift_mean(s, k))
        df[f"team_shots_on_target_last{k}"]  = per_group("FA",             lambda s, k=k: _shift_mean(s, k))

    lookup = df[["temporada", "rodada", "clube_id"] + TEAM_FEATURES].rename(
        columns={"clube_id": "_opp_clube_id",
                 **{c: c.replace("team_", "opp_") for c in TEAM_FEATURES}}
    )
    df["_opp_clube_id"] = df[opp_col]

    merged = df.merge(
        lookup,
        on=["temporada", "rodada", "_opp_clube_id"],
        how="left",
        validate="many_to_one",
    )

    out = merged[KEY_COLS + ["_opp_clube_id"] + ALL_FEATURES].rename(
        columns={"_opp_clube_id": "opp_clube_id"}
    )
    out["temporada"] = out["temporada"].astype(int)
    out["rodada"]    = out["rodada"].astype(int)
    out["clube_id"]  = out["clube_id"].astype(int)
    return out.reset_index(drop=True)


def _prior_window(hist, T, R, club, k=5):
    m = ((hist.clube_id == club) &
         ((hist.temporada < T) | ((hist.temporada == T) & (hist.rodada < R))))
    return hist.loc[m].sort_values(["temporada", "rodada"]).tail(k)


def _fmt_example(title, hist, out, T, R, club):
    row = out.query("temporada==@T and rodada==@R and clube_id==@club")
    if row.empty:
        return f"### {title}\n\n(alvo nao encontrado)\n"
    row = row.iloc[0]
    win = _prior_window(hist, T, R, club, 5)
    lines = [f"### {title}",
             f"**Alvo**: T={T} R={R} clube_id={club} opp={row['opp_clube_id']}", "",
             "| temporada | rodada | gm | gs | FTT | FA | cs |",
             "|---|---|---|---|---|---|---|"]
    for _, r in win.iterrows():
        cs = "?" if pd.isna(r["gols_sofridos"]) else int(r["gols_sofridos"] == 0)
        lines.append(f"| {int(r['temporada'])} | {int(r['rodada'])} | "
                     f"{r['gols_marcados']} | {r['gols_sofridos']} | "
                     f"{r.get('FTT','')} | {r.get('FA','')} | {cs} |")
    lines += ["", "Features finais:", ""]
    for c in TEAM_FEATURES:
        lines.append(f"- `{c}` = {row[c]}")
    for c in OPP_FEATURES:
        lines.append(f"- `{c}` = {row[c]}")
    return "\n".join(lines) + "\n"


def write_audit(out: pd.DataFrame, hist: pd.DataFrame) -> None:
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(OUT_PARQUET.read_bytes()).hexdigest()

    seasons = sorted(hist["temporada"].unique().tolist())
    rounds  = sorted(hist["rodada"].unique().tolist())
    n_clubs = hist["clube_id"].nunique()
    n_dups  = int(out.duplicated(subset=KEY_COLS).sum())
    nan_count = out[ALL_FEATURES].isna().sum()

    first_T = seasons[0]
    first_R = int(hist.loc[hist["temporada"] == first_T, "rodada"].min())
    mid_T   = seasons[len(seasons) // 2]
    mid_R   = int(hist.loc[hist["temporada"] == mid_T, "rodada"].median())

    any_first = out.query("temporada==@first_T and rodada==@first_R").iloc[0]
    mid_pick  = out.query("temporada==@mid_T and rodada==@mid_R").iloc[0]

    miss_row = None
    if "FTT" in hist.columns:
        mask = hist["FTT"].isna() & hist["gols_sofridos"].notna()
        if mask.any():
            mr = hist.loc[mask].iloc[0]
            miss_row = (int(mr["temporada"]), int(mr["rodada"]), int(mr["clube_id"]))

    diff = out.dropna(subset=["team_shots_last3", "team_shots_last5"])
    diff = diff[diff["team_shots_last3"] != diff["team_shots_last5"]]
    diff_row = None
    if not diff.empty:
        dr = diff.iloc[0]
        diff_row = (int(dr["temporada"]), int(dr["rodada"]), int(dr["clube_id"]))

    opp_hist = out.dropna(subset=["opp_shots_last3"])
    opp_row = None
    if not opp_hist.empty:
        orr = opp_hist.iloc[0]
        opp_row = (int(orr["temporada"]), int(orr["rodada"]), int(orr["clube_id"]))

    with AUDIT_MD.open("w", encoding="utf-8") as fh:
        fh.write("# EXP-12 - F2 - Auditoria de Team Context Features\n\n")
        fh.write(f"**SHA256 parquet**: `{sha}`\n\n")

        fh.write("## 1. Estrutura de entrada\n\n")
        fh.write(f"- arquivo: `{HIST_PATH.relative_to(ROOT)}`\n")
        fh.write(f"- shape: {hist.shape}\n")
        fh.write(f"- colunas: {list(hist.columns)}\n")
        fh.write(f"- temporadas: {seasons}\n")
        fh.write(f"- rodadas unicas: {len(rounds)} min={min(rounds)} max={max(rounds)}\n")
        fh.write(f"- clubes: {n_clubs}\n")
        fh.write(f"- duplicatas (T,R,clube): {int(hist.duplicated(subset=KEY_COLS).sum())}\n\n")

        fh.write("## 2. Saida\n\n")
        fh.write(f"- shape: {out.shape}\n")
        fh.write(f"- colunas: {list(out.columns)}\n")
        fh.write(f"- duplicatas (T,R,clube): {n_dups}\n\n")

        fh.write("## 3. NaN por feature\n\n")
        for c in ALL_FEATURES:
            fh.write(f"- `{c}`: {int(nan_count[c])}\n")
        fh.write("\n")

        fh.write("## 4. Exemplos manuais auditados\n\n")
        fh.write(_fmt_example("Rodada inicial da temporada",
                              hist, out, first_T, first_R, int(any_first["clube_id"])))
        fh.write(_fmt_example("Rodada intermediaria",
                              hist, out, mid_T, mid_R, int(mid_pick["clube_id"])))
        if miss_row:
            fh.write(_fmt_example("Partida com missing roster", hist, out, *miss_row))
        else:
            fh.write("### Partida com missing roster\n\n(nenhuma encontrada)\n\n")
        if diff_row:
            fh.write(_fmt_example("last3 != last5", hist, out, *diff_row))
        if opp_row:
            fh.write(_fmt_example("Adversario com historico", hist, out, *opp_row))

        fh.write("\n## 5. Notas de implementacao\n\n")
        fh.write("- Eixo temporal = (temporada, rodada). Data fisica ignorada.\n")
        fh.write("- last3/last5 = ultimas 3/5 PARTIDAS anteriores (posicional).\n")
        fh.write("- clean_sheet = (gols_sofridos == 0). Nao usa SG.\n")
        fh.write("- FTT = FD+FF+FT+G+PP ; FA = FD+G. Roster ausente -> NaN.\n")
        fh.write("- CS rate = CS observados / partidas observadas. NaN se 0 obs.\n")
        fh.write("- Mask de ausencia de roster aplicada no layer de features, "
                 "sem alteracao do F1.\n")
        fh.write("- Fonte do mask: `reports/exp12/f2_missing_roster_raw_audit.csv` "
                 "(GATE 2, n_jogadores_raw == 0).\n")

    print(f"[audit] escrito em {AUDIT_MD}")


def main() -> None:
    if not HIST_PATH.exists():
        raise SystemExit(f"[F2] historico nao encontrado: {HIST_PATH}")

    hist = pd.read_parquet(HIST_PATH)
    print(f"[F2] historico: {hist.shape}")

    absence_keys = load_roster_absence_keys()
    hist = apply_roster_absence_mask(hist, absence_keys)

    out = build_team_features(hist)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"[F2] saida: {OUT_PARQUET} shape={out.shape}")

    write_audit(out, hist)
    sha = hashlib.sha256(OUT_PARQUET.read_bytes()).hexdigest()
    print(f"[F2] sha256 = {sha}")


if __name__ == "__main__":
    main()
