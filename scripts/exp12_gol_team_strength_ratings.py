#!/usr/bin/env python3
"""
exp12_gol_team_strength_ratings.py

Experimento Mínimo - Opção B
Ratings de Força Ofensiva e Defensiva (Poisson / Maher style)

- Ratings usados na rodada R só enxergam jogos estritamente anteriores a R
- Estimação por temporada (mais estável)
- Shrinkage para a média da liga
- Avaliação de poder preditivo nos goleiros participantes

Uso:
    python scripts/exp12_gol_team_strength_ratings.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

MATCH_HISTORY = ROOT / "data" / "interim" / "exp12" / "team_match_history.parquet"
GOL_POOL = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"

OUT_DIR = ROOT / "data" / "interim" / "exp12"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports" / "exp12"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_RATINGS = OUT_DIR / "team_strength_ratings.parquet"
OUT_EVAL = REPORTS_DIR / "exp12_gol_team_strength_eval.csv"
OUT_REPORT = REPORTS_DIR / "exp12_gol_team_strength_report.md"

# ------------------------------------------------------------------ parâmetros
MIN_MATCHES = 5
SHRINKAGE_K = 10.0
HOME_ADV = 0.28          # vantagem de mando em log-espaço (aproximação clássica)
MAX_ITER = 150

def poisson_neg_loglik(params, home_idx, away_idx, gh, ga, n_teams, home_adv):
    attacks = params[:n_teams]
    defenses = params[n_teams:2*n_teams]
    mu = params[-1]

    log_lh = mu + home_adv + attacks[home_idx] - defenses[away_idx]
    log_la = mu + attacks[away_idx] - defenses[home_idx]

    log_lh = np.clip(log_lh, -8, 8)
    log_la = np.clip(log_la, -8, 8)

    lh = np.exp(log_lh)
    la = np.exp(log_la)

    ll = (gh * log_lh - lh) + (ga * log_la - la)
    return -np.sum(ll)

def estimate_ratings(matches: pd.DataFrame) -> dict:
    """Estima attack/defense para um conjunto de partidas (já no formato home/away)."""
    if len(matches) < 12:
        return {}

    teams = sorted(set(matches["home_id"]).union(set(matches["away_id"])))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = matches["home_id"].map(team_to_idx).values
    away_idx = matches["away_id"].map(team_to_idx).values
    gh = matches["gols_home"].values.astype(float)
    ga = matches["gols_away"].values.astype(float)

    x0 = np.zeros(2 * n + 1)
    x0[-1] = np.log(max((gh.mean() + ga.mean()) / 2, 0.15))

    res = minimize(
        poisson_neg_loglik,
        x0,
        args=(home_idx, away_idx, gh, ga, n, HOME_ADV),
        method="L-BFGS-B",
        options={"maxiter": MAX_ITER, "disp": False},
    )

    if not res.success:
        return {}

    attacks = res.x[:n]
    defenses = res.x[n:2*n]

    # Shrinkage para zero (média da liga)
    factor = len(matches) / (len(matches) + SHRINKAGE_K)
    attacks *= factor
    defenses *= factor

    return {
        t: {"attack": float(attacks[i]), "defense": float(defenses[i])}
        for t, i in team_to_idx.items()
    }

def build_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma o formato longo (um row por time) em home/away."""
    home = df[df["mando"] == "C"][[
        "temporada", "rodada", "partida_id", "clube_id", "adversario_id",
        "gols_marcados", "gols_sofridos"
    ]].copy()
    home = home.rename(columns={
        "clube_id": "home_id",
        "adversario_id": "away_id",
        "gols_marcados": "gols_home",
        "gols_sofridos": "gols_away",
    })
    return home.drop_duplicates(subset=["partida_id"])

def main():
    print("=" * 70)
    print("EXP-12 — Team Strength Ratings (Poisson / Maher)")
    print("=" * 70)

    # ------------------------------------------------------------------ 1. Carregar e preparar partidas
    print("\n[1/6] Carregando e preparando partidas...")
    raw = pd.read_parquet(MATCH_HISTORY)
    matches = build_home_away(raw)
    matches = matches.sort_values(["temporada", "rodada", "partida_id"]).reset_index(drop=True)

    print(f"     Partidas únicas: {len(matches):,}")
    print(f"     Temporadas: {sorted(matches['temporada'].unique())}")

    # ------------------------------------------------------------------ 2. Gerar ratings temporais (sem leakage)
    print("\n[2/6] Estimando ratings com segurança temporal...")

    ratings_rows = []

    for temporada in sorted(matches["temporada"].unique()):
        seas = matches[matches["temporada"] == temporada]
        rodadas = sorted(seas["rodada"].unique())

        for R in rodadas:
            # Apenas jogos estritamente anteriores a R
            hist = seas[seas["rodada"] < R]
            if len(hist) < 20:
                continue

            ratings = estimate_ratings(hist)
            if not ratings:
                continue

            for clube_id, vals in ratings.items():
                ratings_rows.append({
                    "temporada": temporada,
                    "rodada": R,          # rating disponível PARA esta rodada
                    "clube_id": clube_id,
                    "rating_attack": vals["attack"],
                    "rating_defense": vals["defense"],
                })

        print(f"     Temporada {temporada}: {len(rodadas)} rodadas processadas")

    ratings_df = pd.DataFrame(ratings_rows)
    print(f"\n     Total de ratings gerados: {len(ratings_df):,}")

    if len(ratings_df) == 0:
        raise RuntimeError("Nenhum rating gerado. Verifique dados.")

    ratings_df.to_parquet(OUT_RATINGS, index=False)
    print(f"     Salvo em: {OUT_RATINGS}")

    # ------------------------------------------------------------------ 3. Join com goleiros
    print("\n[3/6] Fazendo join com goleiros participantes...")
    gol = pd.read_parquet(GOL_POOL)
    eleg = gol[gol["elegivel_modelo"] == True].copy()

    # Rating do próprio time (defesa) e do adversário (ataque)
    eleg = eleg.merge(
        ratings_df.rename(columns={
            "clube_id": "clube_id",
            "rating_attack": "rating_attack_own",
            "rating_defense": "rating_defense_own",
        }),
        on=["temporada", "rodada", "clube_id"],
        how="left",
    )

    eleg = eleg.merge(
        ratings_df.rename(columns={
            "clube_id": "adversario_id",
            "rating_attack": "rating_attack_opp",
            "rating_defense": "rating_defense_opp",
        }),
        on=["temporada", "rodada", "adversario_id"],
        how="left",
    )

    # Features derivadas
    eleg["dif_forca"] = eleg["rating_attack_opp"] - eleg["rating_defense_own"]
    eleg["soma_forca"] = eleg["rating_attack_opp"] + eleg["rating_defense_own"]

    print(f"     Linhas elegíveis após join: {len(eleg):,}")
    print(f"     Cobertura rating_defense_own: {eleg['rating_defense_own'].notna().mean():.1%}")
    print(f"     Cobertura rating_attack_opp : {eleg['rating_attack_opp'].notna().mean():.1%}")

    # ------------------------------------------------------------------ 4. Poder preditivo
    print("\n[4/6] Calculando poder preditivo...")

    features = [
        "rating_attack_opp",
        "rating_defense_own",
        "rating_attack_own",
        "rating_defense_opp",
        "dif_forca",
        "soma_forca",
        # baselines atuais para comparação
        "team_gols_sofridos_mean_3",
        "team_gols_sofridos_mean_5",
        "opp_gols_marcados_mean_3",
        "opp_gols_marcados_mean_5",
        "gk_pontos_mean_5",
    ]

    results = []
    for col in features:
        if col not in eleg.columns:
            continue
        sub = eleg[[col, "pontos_num"]].dropna()
        if len(sub) < 80:
            continue
        sp = sub[col].corr(sub["pontos_num"], method="spearman")
        results.append({
            "feature": col,
            "spearman": sp,
            "n": len(sub),
            "coverage": eleg[col].notna().mean(),
        })

    eval_df = pd.DataFrame(results).sort_values("spearman", key=abs, ascending=False)
    eval_df.to_csv(OUT_EVAL, index=False)

    print("\n=== PODER PREDITIVO (Spearman com pontos_num) ===")
    print(eval_df.to_string(index=False))

    # ------------------------------------------------------------------ 5. Relatório
    print("\n[5/6] Gerando relatório...")

    report = f"""# EXP-12 — Team Strength Ratings (Poisson/Maher)

## Resultado do poder preditivo

{eval_df.to_markdown(index=False)}

## Interpretação

- Se `dif_forca` ou `rating_attack_opp` tiverem Spearman claramente melhor que as médias simples atuais,
  os ratings estão capturando sinal útil.
- Valores absolutos ainda devem ser baixos (problema difícil), mas a comparação relativa importa.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"\n[6/6] Arquivos salvos:")
    print(f"  {OUT_RATINGS}")
    print(f"  {OUT_EVAL}")
    print(f"  {OUT_REPORT}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
