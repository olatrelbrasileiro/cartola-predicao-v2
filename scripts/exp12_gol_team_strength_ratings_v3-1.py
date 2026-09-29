#!/usr/bin/env python3
"""
EXP-12 — Team Strength Ratings v3

Objetivo:
  Construir força ofensiva/defensiva de times de forma estritamente temporal,
  usando um modelo Poisson com mando e shrinkage L2, e derivar contexto
  pré-jogo para goleiros.

Principais garantias metodológicas:
  - nenhum jogo da rodada-alvo entra no rating da própria rodada;
  - treino temporal usa (temporada menor) OU (mesma temporada e rodada menor);
  - identidade dos jogos é (temporada, rodada, partida_id), não apenas partida_id;
  - cada partida precisa ter exatamente 1 mandante e 1 visitante;
  - restrição mean(attack)=0 e mean(defense)=0 é imposta DURANTE a otimização;
  - comparação de features usa apenas dados pré-jogo;
  - probabilidades de eventos são derivadas do lambda Poisson, sem min-max artificial;
  - avaliação de gols usa NLL Poisson e métricas de probabilidade legítimas;
  - coverage e missingness são reportados explicitamente.

Execução:
  python -m pip install --quiet pandas numpy scipy pyarrow
  python scripts/exp12_gol_team_strength_ratings_v3.py
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim" / "exp12"
REPORTS_DIR = ROOT / "reports" / "exp12"

GOL_POOLING_PATH = INTERIM_DIR / "gol_pooling_2024_2026.parquet"
OUT_RATINGS = INTERIM_DIR / "team_strength_ratings_v3.parquet"
OUT_FEATURES = INTERIM_DIR / "gol_team_strength_features_v3.parquet"
OUT_EVAL = REPORTS_DIR / "exp12_gol_team_strength_eval_v3.csv"
OUT_BY_SEASON = REPORTS_DIR / "exp12_gol_team_strength_eval_by_season_v3.csv"
OUT_BY_PHASE = REPORTS_DIR / "exp12_gol_team_strength_eval_by_phase_v3.csv"
OUT_REPORT = REPORTS_DIR / "exp12_gol_team_strength_report_v3.md"

REQUIRED_SEASONS = [2024, 2025, 2026]
MIN_HISTORY_MATCHES = 50
L2_REG = 1.0
ESTIMATE_HOME_ADV = True
FALLBACK_HOME_ADV = 0.25
MAXITER = 500

MATCH_CANDIDATES = [
    INTERIM_DIR / "team_match_history.parquet",
    INTERIM_DIR / "team_match_history.csv",
    INTERIM_DIR / "team_match_history_r1.parquet",
    INTERIM_DIR / "team_match_history_r1.csv",
]


# ---------------------------------------------------------------------------
# IO / schema
# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato não suportado: {path}")


def find_match_source() -> Path:
    for p in MATCH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Não encontrei team_match_history. Procurei:\n" +
        "\n".join(f"  - {p}" for p in MATCH_CANDIDATES)
    )


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label}: colunas ausentes: {missing}")


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_team_history(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "temporada", "rodada", "partida_id", "clube_id", "adversario_id",
        "gols_marcados", "gols_sofridos", "mando",
    ]
    require_columns(df, required, "team_match_history")
    df = df.copy()
    df = numeric(df, ["temporada", "rodada", "partida_id", "clube_id",
                      "adversario_id", "gols_marcados", "gols_sofridos"])
    df = df.dropna(subset=required).copy()
    df["temporada"] = df["temporada"].astype(int)
    df["rodada"] = df["rodada"].astype(int)
    df["partida_id"] = df["partida_id"].astype(int)
    df["clube_id"] = df["clube_id"].astype(int)
    df["adversario_id"] = df["adversario_id"].astype(int)

    raw = df["mando"].astype(str).str.strip().str.upper()
    home_values = {"C", "CASA", "HOME", "H", "MANDANTE", "M"}
    away_values = {"F", "FORA", "AWAY", "A", "VISITANTE", "V"}
    df["is_home"] = raw.map(lambda x: True if x in home_values else (False if x in away_values else np.nan))
    if df["is_home"].isna().any():
        bad = sorted(raw[df["is_home"].isna()].dropna().unique().tolist())[:30]
        raise RuntimeError(f"Valores de mando não reconhecidos: {bad}")
    df["is_home"] = df["is_home"].astype(bool)

    key = ["temporada", "rodada", "partida_id", "clube_id"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        sample = df.loc[dup, key].head(10).to_dict("records")
        raise RuntimeError(f"Duplicidade na identidade time-jogo: {sample}")

    # IDs das partidas precisam ser únicos pelo trio temporal. Reutilização do
    # partida_id em temporadas distintas é permitida; colisão dentro do trio não.
    return df.sort_values(["temporada", "rodada", "partida_id", "is_home"], kind="mergesort").reset_index(drop=True)


def build_match_table(team_hist: pd.DataFrame) -> pd.DataFrame:
    keys = ["temporada", "rodada", "partida_id"]
    rows = []
    bad = []
    for k, g in team_hist.groupby(keys, sort=False):
        if len(g) != 2:
            bad.append((k, "n_rows", len(g)))
            continue
        homes = g[g["is_home"]]
        aways = g[~g["is_home"]]
        if len(homes) != 1 or len(aways) != 1:
            bad.append((k, "home_away", len(homes), len(aways)))
            continue
        h = homes.iloc[0]
        a = aways.iloc[0]
        if int(h["adversario_id"]) != int(a["clube_id"]) or int(a["adversario_id"]) != int(h["clube_id"]):
            bad.append((k, "opponent_mismatch"))
            continue
        if int(h["gols_marcados"]) != int(a["gols_sofridos"]) or int(a["gols_marcados"]) != int(h["gols_sofridos"]):
            bad.append((k, "score_mismatch"))
            continue
        rows.append({
            "temporada": int(k[0]),
            "rodada": int(k[1]),
            "partida_id": int(k[2]),
            "home_team": int(h["clube_id"]),
            "away_team": int(a["clube_id"]),
            "home_goals": float(h["gols_marcados"]),
            "away_goals": float(a["gols_marcados"]),
        })
    if bad:
        sample = bad[:15]
        raise RuntimeError(
            f"Falhas na reconstrução dos jogos: {len(bad)} partidas problemáticas. Exemplo: {sample}"
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Nenhuma partida válida foi reconstruída.")
    if out.duplicated(["temporada", "rodada", "partida_id"]).any():
        raise RuntimeError("Partidas duplicadas após reconstrução.")
    return out.sort_values(["temporada", "rodada", "partida_id"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Poisson ratings with exact centering inside objective
# ---------------------------------------------------------------------------

def _center(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def fit_poisson_ratings(matches: pd.DataFrame, l2: float = L2_REG) -> dict:
    teams = np.sort(np.unique(np.concatenate([matches["home_team"].values,
                                               matches["away_team"].values]))).astype(int)
    idx = {int(t): i for i, t in enumerate(teams)}
    hi = matches["home_team"].map(idx).to_numpy(dtype=int)
    ai = matches["away_team"].map(idx).to_numpy(dtype=int)
    yh = matches["home_goals"].to_numpy(dtype=float)
    ya = matches["away_goals"].to_numpy(dtype=float)
    n = len(teams)

    # theta = [mu, home_adv, raw_attack(n), raw_defense(n)]
    x0 = np.zeros(2 + 2 * n, dtype=float)
    x0[0] = math.log(max((yh.sum() + ya.sum()) / max(2 * len(matches), 1), 0.25))
    x0[1] = 0.20

    def unpack(theta):
        mu = theta[0]
        home_adv = theta[1]
        attack = _center(theta[2:2+n])
        defense = _center(theta[2+n:2+2*n])
        return mu, home_adv, attack, defense

    def objective(theta, need_grad=False):
        mu, home_adv, attack, defense = unpack(theta)
        eta_h = mu + home_adv + attack[hi] - defense[ai]
        eta_a = mu + attack[ai] - defense[hi]
        # Avoid numerical overflow while keeping gradient consistent with the
        # centered parameterization. Realistic values should stay well inside.
        eta_h = np.clip(eta_h, -12.0, 12.0)
        eta_a = np.clip(eta_a, -12.0, 12.0)
        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)
        loss = np.sum(lam_h - yh * eta_h) + np.sum(lam_a - ya * eta_a)
        loss += l2 * (np.dot(attack, attack) + np.dot(defense, defense))
        if not need_grad:
            return float(loss)

        rh = lam_h - yh
        ra = lam_a - ya
        grad = np.zeros_like(theta)
        grad[0] = rh.sum() + ra.sum()
        grad[1] = rh.sum()

        ga = np.bincount(hi, weights=rh, minlength=n) + np.bincount(ai, weights=ra, minlength=n)
        gd = -np.bincount(ai, weights=rh, minlength=n) - np.bincount(hi, weights=ra, minlength=n)
        # Jacobian of center(x): I - 11'/n
        ga = ga - ga.mean() + 2.0 * l2 * attack
        gd = gd - gd.mean() + 2.0 * l2 * defense
        grad[2:2+n] = ga
        grad[2+n:2+2*n] = gd
        return float(loss), grad

    def fun(theta):
        return objective(theta, need_grad=True)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "ftol": 1e-10, "gtol": 1e-7})
    if not res.success:
        # A non-successful optimizer result is not silently accepted. The
        # objective is still deterministic, but promotion needs a valid fit.
        raise RuntimeError(f"Falha L-BFGS-B: {res.message}")

    mu, home_adv, attack, defense = unpack(res.x)
    return {
        "teams": teams,
        "mu": float(mu),
        "home_adv": float(home_adv) if ESTIMATE_HOME_ADV else float(FALLBACK_HOME_ADV),
        "attack": attack,
        "defense": defense,
        "n_matches": int(len(matches)),
        "objective": float(res.fun),
        "iterations": int(getattr(res, "nit", -1)),
    }


def ratings_to_frame(fit: dict, season: int, round_: int) -> pd.DataFrame:
    return pd.DataFrame({
        "rating_temporada": season,
        "rating_rodada": round_,
        "clube_id": fit["teams"].astype(int),
        "rating_attack": fit["attack"].astype(float),
        "rating_defense": fit["defense"].astype(float),
        "rating_mu": float(fit["mu"]),
        "rating_home_adv": float(fit["home_adv"]),
        "rating_n_history_matches": int(fit["n_matches"]),
        "rating_objective": float(fit["objective"]),
        "rating_iterations": int(fit["iterations"]),
    })


def build_temporal_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    target_rounds = matches[["temporada", "rodada"]].drop_duplicates().sort_values(["temporada", "rodada"])
    pieces = []
    total = len(target_rounds)
    for i, row in enumerate(target_rounds.itertuples(index=False), start=1):
        season, round_ = int(row.temporada), int(row.rodada)
        mask = (matches["temporada"] < season) | ((matches["temporada"] == season) & (matches["rodada"] < round_))
        hist = matches.loc[mask]
        if len(hist) < MIN_HISTORY_MATCHES:
            continue
        fit = fit_poisson_ratings(hist)
        pieces.append(ratings_to_frame(fit, season, round_))
    if not pieces:
        raise RuntimeError("Nenhuma rodada atingiu MIN_HISTORY_MATCHES.")
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["rating_temporada", "rating_rodada", "clube_id"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Join ratings with goalkeeper rows
# ---------------------------------------------------------------------------

def locate_opponent_col(df: pd.DataFrame) -> str:
    for c in ["adversario_id", "opp_clube_id", "opponent_id"]:
        if c in df.columns:
            return c
    raise RuntimeError("Gol pooling não possui coluna de adversário reconhecível.")


def build_home_lookup(team_hist: pd.DataFrame) -> pd.DataFrame:
    cols = ["temporada", "rodada", "clube_id", "partida_id", "is_home"]
    return team_hist[cols].copy().drop_duplicates(["temporada", "rodada", "clube_id"])


def join_goalkeepers(gk: pd.DataFrame, ratings: pd.DataFrame, team_hist: pd.DataFrame) -> pd.DataFrame:
    require_columns(gk, ["temporada", "rodada", "clube_id"], "gol_pooling")
    opp_col = locate_opponent_col(gk)
    gk = gk.copy()
    gk = numeric(gk, ["temporada", "rodada", "clube_id", opp_col])
    gk = gk.dropna(subset=["temporada", "rodada", "clube_id", opp_col]).copy()
    gk["temporada"] = gk["temporada"].astype(int)
    gk["rodada"] = gk["rodada"].astype(int)
    gk["clube_id"] = gk["clube_id"].astype(int)
    gk[opp_col] = gk[opp_col].astype(int)

    own = ratings.rename(columns={
        "rating_temporada": "temporada",
        "rating_rodada": "rodada",
        "clube_id": "clube_id",
        "rating_attack": "rating_attack_own",
        "rating_defense": "rating_defense_own",
        "rating_mu": "rating_mu_own",
        "rating_home_adv": "rating_home_adv_own",
        "rating_n_history_matches": "rating_n_history_matches_own",
    })[["temporada", "rodada", "clube_id", "rating_attack_own", "rating_defense_own", "rating_mu_own", "rating_home_adv_own", "rating_n_history_matches_own"]]

    opp = ratings.rename(columns={
        "rating_temporada": "temporada",
        "rating_rodada": "rodada",
        "clube_id": opp_col,
        "rating_attack": "rating_attack_opp",
        "rating_defense": "rating_defense_opp",
        "rating_n_history_matches": "rating_n_history_matches_opp",
    })[["temporada", "rodada", opp_col, "rating_attack_opp", "rating_defense_opp", "rating_n_history_matches_opp"]]

    out = gk.merge(own, on=["temporada", "rodada", "clube_id"], how="left", validate="many_to_one")
    out = out.merge(opp, on=["temporada", "rodada", opp_col], how="left", validate="many_to_one")

    home = build_home_lookup(team_hist).rename(columns={"clube_id": "clube_id", "is_home": "own_is_home"})
    out = out.merge(home[["temporada", "rodada", "clube_id", "own_is_home"]],
                    on=["temporada", "rodada", "clube_id"], how="left", validate="many_to_one")

    # Ratings exist before the target round; home_adv is global for the round.
    valid = out["rating_mu_own"].notna() & out["rating_attack_opp"].notna() & out["rating_defense_own"].notna()
    if valid.any():
        # For the goalkeeper's goals-against lambda, home advantage belongs to the opponent.
        # Therefore it is added when the goalkeeper's team is AWAY.
        opp_home_component = np.where(out["own_is_home"] == False, out["rating_home_adv_own"], 0.0)
        away_eta = out["rating_mu_own"] + out["rating_attack_opp"] - out["rating_defense_own"]
        out["log_lambda_against"] = away_eta + opp_home_component
        out["expected_goals_against"] = np.exp(np.clip(out["log_lambda_against"], -12, 12))

        # Own scoring context: useful as offensive team context, not a GK target.
        # For the goalkeeper team's own scoring lambda, home advantage belongs to own team.
        own_home_component = np.where(out["own_is_home"] == True, out["rating_home_adv_own"], 0.0)
        out["log_lambda_for"] = out["rating_mu_own"] + out["rating_attack_own"] - out["rating_defense_opp"] + own_home_component
        out["expected_goals_for"] = np.exp(np.clip(out["log_lambda_for"], -12, 12))

        out["attack_vs_defense"] = out["rating_attack_opp"] - out["rating_defense_own"]
        lam = out["expected_goals_against"]
        out["p_sg"] = np.exp(-lam)
        out["p_sofrer_1plus"] = 1.0 - out["p_sg"]
        out["p_sofrer_2plus"] = 1.0 - np.exp(-lam) * (1.0 + lam)
        out["p_sofrer_3plus"] = 1.0 - np.exp(-lam) * (1.0 + lam + (lam ** 2) / 2.0)

    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def corr_row(df: pd.DataFrame, feature: str, target: str = "pontos_num") -> dict | None:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 10:
        return None
    xr, yr = x[m].to_numpy(), y[m].to_numpy()
    sp = spearmanr(xr, yr)
    pe = pearsonr(xr, yr)
    return {"feature": feature, "target": target, "n": n, "coverage": n / max(len(df), 1),
            "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "pearson": float(pe.statistic), "pearson_p": float(pe.pvalue)}


def evaluate_correlations(df: pd.DataFrame, features: list[str], by: str | None = None) -> pd.DataFrame:
    frames = []
    groups = [(None, df)] if by is None else list(df.groupby(by, sort=True))
    for g, part in groups:
        for f in features:
            if f not in part.columns:
                continue
            r = corr_row(part, f)
            if r is not None:
                if by is not None:
                    r[by] = g
                frames.append(r)
    return pd.DataFrame(frames)


def poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    from scipy.special import gammaln
    eta = np.log(np.clip(lam, 1e-12, None))
    return float(np.mean(lam - y * eta + gammaln(y + 1.0)))


def event_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def evaluate_probability_outputs(df: pd.DataFrame) -> pd.DataFrame:
    if "gols_sofridos" not in df.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df["gols_sofridos"], errors="coerce")
    lam = pd.to_numeric(df.get("expected_goals_against"), errors="coerce")
    m = y.notna() & lam.notna() & (lam > 0)
    if m.sum() < 10:
        return pd.DataFrame()
    yv = y[m].to_numpy(dtype=float)
    lv = lam[m].to_numpy(dtype=float)
    p0 = np.exp(-lv)
    p1plus = 1.0 - p0
    p2plus = 1.0 - np.exp(-lv) * (1.0 + lv)
    p3plus = 1.0 - np.exp(-lv) * (1.0 + lv + lv**2 / 2.0)
    return pd.DataFrame([
        {"metric": "poisson_nll_goals_against", "value": poisson_nll(yv, lv), "n": int(m.sum())},
        {"metric": "brier_clean_sheet", "value": event_brier((yv == 0).astype(float), p0), "n": int(m.sum())},
        {"metric": "brier_sofrer_1plus", "value": event_brier((yv >= 1).astype(float), p1plus), "n": int(m.sum())},
        {"metric": "brier_sofrer_2plus", "value": event_brier((yv >= 2).astype(float), p2plus), "n": int(m.sum())},
        {"metric": "brier_sofrer_3plus", "value": event_brier((yv >= 3).astype(float), p3plus), "n": int(m.sum())},
    ])


def make_phase(df: pd.DataFrame) -> pd.Series:
    r = pd.to_numeric(df["rodada"], errors="coerce")
    return pd.cut(r, bins=[-np.inf, 5, 10, 15, 20, 25, np.inf], labels=["R1-5", "R6-10", "R11-15", "R16-20", "R21-25", "R26+"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not GOL_POOLING_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {GOL_POOLING_PATH}")

    match_path = find_match_source()
    raw_hist = load_table(match_path)
    hist = normalize_team_history(raw_hist)
    matches = build_match_table(hist)

    seasons = sorted(set(matches["temporada"]).intersection(REQUIRED_SEASONS))
    missing = [s for s in REQUIRED_SEASONS if s not in seasons]
    if missing:
        raise RuntimeError(f"Histórico de jogos não contém todas as temporadas exigidas: {missing}")

    # Do not silently change the requested universe.
    gk = pd.read_parquet(GOL_POOLING_PATH)
    require_columns(gk, ["temporada", "rodada", "clube_id"], "gol_pooling")
    gk_seasons = sorted(pd.to_numeric(gk["temporada"], errors="coerce").dropna().astype(int).unique().tolist())
    missing_gk = [s for s in REQUIRED_SEASONS if s not in gk_seasons]
    if missing_gk:
        raise RuntimeError(f"gol_pooling não contém todas as temporadas exigidas: {missing_gk}")

    ratings = build_temporal_ratings(matches)
    ratings.to_parquet(OUT_RATINGS, index=False)

    joined = join_goalkeepers(gk, ratings, hist)
    joined.to_parquet(OUT_FEATURES, index=False)

    rating_features = [
        "rating_attack_own", "rating_defense_own", "rating_attack_opp", "rating_defense_opp",
        "attack_vs_defense", "log_lambda_against", "expected_goals_against", "expected_goals_for",
        "p_sg", "p_sofrer_1plus", "p_sofrer_2plus", "p_sofrer_3plus",
    ]
    baseline_features = [
        "team_gols_sofridos_mean_3", "team_gols_sofridos_mean_5",
        "opp_gols_marcados_mean_3", "opp_gols_marcados_mean_5",
        "gk_pontos_mean_5",
    ]
    all_features = [c for c in baseline_features + rating_features if c in joined.columns]
    eval_df = joined[pd.to_numeric(joined["temporada"], errors="coerce").isin(REQUIRED_SEASONS)].copy()

    # Fair group comparison: both groups are evaluated on their own valid rows,
    # and a common-sample table is reported separately.
    eval_total = evaluate_correlations(eval_df, all_features)
    eval_total.to_csv(OUT_EVAL, index=False)

    eval_season = evaluate_correlations(eval_df, all_features, by="temporada")
    eval_season.to_csv(OUT_BY_SEASON, index=False)

    eval_phase_df = eval_df.copy()
    eval_phase_df["fase"] = make_phase(eval_phase_df)
    eval_phase = evaluate_correlations(eval_phase_df, all_features, by="fase")
    eval_phase.to_csv(OUT_BY_PHASE, index=False)

    prob_eval = evaluate_probability_outputs(eval_df)
    prob_path = REPORTS_DIR / "exp12_gol_team_strength_probability_eval_v3.csv"
    prob_eval.to_csv(prob_path, index=False)

    # Aggregate group correlations on a genuinely common sample.
    base_valid = [c for c in baseline_features if c in eval_df.columns]
    rating_valid = [c for c in rating_features if c in eval_df.columns]
    common = pd.Series(True, index=eval_df.index)
    for c in base_valid + rating_valid:
        common &= eval_df[c].notna()
    common_target = eval_df["pontos_num"].notna() if "pontos_num" in eval_df.columns else pd.Series(False, index=eval_df.index)
    common &= common_target
    common_df = eval_df.loc[common].copy()

    common_rows = []
    for c in base_valid + rating_valid:
        r = corr_row(common_df, c)
        if r:
            r["sample_type"] = "common_all_features"
            common_rows.append(r)
    common_eval = pd.DataFrame(common_rows)
    common_path = REPORTS_DIR / "exp12_gol_team_strength_eval_common_sample_v3.csv"
    common_eval.to_csv(common_path, index=False)

    # Match-level diagnostics of the underlying team model, independent of GK points.
    match_pred_rows = []
    rating_keys = [(int(r.temporada), int(r.rodada)) for r in ratings[["rating_temporada", "rating_rodada"]].drop_duplicates().itertuples()]
    for season, round_ in rating_keys:
        rr = ratings[(ratings.rating_temporada == season) & (ratings.rating_rodada == round_)]
        lookup = rr.set_index("clube_id")
        ms = matches[(matches.temporada == season) & (matches.rodada == round_)]
        for m in ms.itertuples(index=False):
            if m.home_team not in lookup.index or m.away_team not in lookup.index:
                continue
            h = lookup.loc[m.home_team]
            a = lookup.loc[m.away_team]
            lh = math.exp(np.clip(h.rating_mu + h.rating_home_adv + h.rating_attack - a.rating_defense, -12, 12))
            la = math.exp(np.clip(a.rating_mu + a.rating_attack - h.rating_defense, -12, 12))
            match_pred_rows.append((season, round_, m.home_goals, lh, m.away_goals, la))
    md = pd.DataFrame(match_pred_rows, columns=["temporada", "rodada", "home_goals", "home_lambda", "away_goals", "away_lambda"])
    if not md.empty:
        md["nll_home"] = md.apply(lambda x: poisson_nll(np.array([x.home_goals]), np.array([x.home_lambda])), axis=1)
        md["nll_away"] = md.apply(lambda x: poisson_nll(np.array([x.away_goals]), np.array([x.away_lambda])), axis=1)
        md["brier_home_0"] = (np.exp(-md["home_lambda"]) - (md["home_goals"] == 0).astype(float)) ** 2
        md["brier_away_0"] = (np.exp(-md["away_lambda"]) - (md["away_goals"] == 0).astype(float)) ** 2
        md.to_csv(REPORTS_DIR / "exp12_team_strength_match_eval_v3.csv", index=False)

    n_gk = len(eval_df)
    n_with_rating = int(eval_df["expected_goals_against"].notna().sum()) if "expected_goals_against" in eval_df else 0
    rounds_rated = ratings[["rating_temporada", "rating_rodada"]].drop_duplicates().shape[0]

    report = []
    report.append("# EXP-12 — Team Strength Ratings v3\n")
    report.append(f"- Match source: `{match_path.relative_to(ROOT)}`")
    report.append(f"- GK source: `{GOL_POOLING_PATH.relative_to(ROOT)}`")
    report.append(f"- Temporadas exigidas: {REQUIRED_SEASONS}")
    report.append(f"- Partidas válidas reconstruídas: {len(matches)}")
    report.append(f"- Rodadas com rating disponível: {rounds_rated}")
    report.append(f"- GK rows avaliadas: {n_gk}")
    report.append(f"- GK rows com rating completo: {n_with_rating} ({n_with_rating/max(n_gk,1):.1%})")
    report.append("")
    report.append("## Garantias temporais\n")
    report.append("Ratings da rodada (T,R) usam somente partidas com T menor ou, na mesma temporada, R menor. Partidas da própria rodada são excluídas.")
    report.append("Identidade da partida = (temporada, rodada, partida_id). Cada jogo precisa ter exatamente um mandante e um visitante, com adversário e placar consistentes.")
    report.append("")
    report.append("## Definição dos ratings\n")
    report.append("`log(lambda_home) = mu + home_adv + attack_home - defense_away`")
    report.append("`log(lambda_away) = mu + attack_away - defense_home`")
    report.append("`mean(attack)=0` e `mean(defense)=0` são impostos durante a otimização; L2 atua sobre esses parâmetros centrados.")
    report.append("")
    report.append("## Probabilidades derivadas\n")
    report.append("A probabilidade de SG é `P(GA=0)=exp(-lambda_against)`. Também são calculadas `P(GA>=1)`, `P(GA>=2)` e `P(GA>=3)`. Essas probabilidades não são min-max de features arbitrárias.")
    report.append("")
    report.append("## Avaliação\n")
    report.append("A força do time deve primeiro demonstrar valor no nível da partida (NLL Poisson/Brier) antes de ser promovida como contexto do goleiro.")
    report.append(f"Arquivos gerados: `{OUT_RATINGS.relative_to(ROOT)}`, `{OUT_FEATURES.relative_to(ROOT)}`, `{OUT_EVAL.relative_to(ROOT)}`, `{OUT_BY_SEASON.relative_to(ROOT)}`, `{OUT_BY_PHASE.relative_to(ROOT)}`, `{prob_path.relative_to(ROOT)}`, `{common_path.relative_to(ROOT)}`.")
    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("=" * 70)
    print("EXP-12 — TEAM STRENGTH RATINGS V3")
    print("=" * 70)
    print(f"Match source : {match_path.relative_to(ROOT)}")
    print(f"GK source    : {GOL_POOLING_PATH.relative_to(ROOT)}")
    print(f"Partidas     : {len(matches)}")
    print(f"Ratings      : {rounds_rated} rodadas")
    print(f"GK avaliadas : {n_gk}")
    print(f"Rating full  : {n_with_rating} ({n_with_rating/max(n_gk,1):.1%})")
    print(f"\nRelatório: {OUT_REPORT.relative_to(ROOT)}")
    if not prob_eval.empty:
        print("\nAvaliação probabilística:")
        print(prob_eval.to_string(index=False))
    if not eval_total.empty:
        print("\nTop correlações Spearman:")
        print(eval_total.sort_values("spearman", ascending=False).head(15).to_string(index=False))
    print("\nSHA256 ratings:", sha256(OUT_RATINGS))
    print("SHA256 features:", sha256(OUT_FEATURES))


if __name__ == "__main__":
    main()
