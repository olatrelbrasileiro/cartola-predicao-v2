#!/usr/bin/env python3
"""
exp12_gol_team_strength_ratings_v3.py

Experimento Mínimo - Opção B+ (Versão Corrigida com Histórico Completo)
Ratings de Força Ofensiva e Defensiva baseados em Finalizações (FTT/FS) + Gols.

Correção: Concatena team_scout_history_2024_2025.parquet com team_match_history.parquet 
para garantir cobertura de 2024 a 2026, resolvendo o problema de cobertura de 25%.
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
MATCH_HISTORY_24_25 = ROOT / "data" / "interim" / "exp12" / "team_scout_history_2024_2025.parquet"
GOL_POOL = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"

OUT_DIR = ROOT / "data" / "interim" / "exp12"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports" / "exp12"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_RATINGS = OUT_DIR / "team_strength_ratings_v3.parquet"
OUT_EVAL = REPORTS_DIR / "exp12_gol_team_strength_eval_v3.csv"
OUT_REPORT = REPORTS_DIR / "exp12_gol_team_strength_report_v3.md"

# ------------------------------------------------------------------ parâmetros
MIN_MATCHES = 4               
SHRINKAGE_K = 8.0             
HOME_ADV = 0.25               
MAX_ITER = 150

def poisson_neg_loglik(params, home_idx, away_idx, events_home, events_away, n_teams, home_adv):
    attacks = params[:n_teams]
    defenses = params[n_teams:2*n_teams]
    mu = params[-1]

    log_lh = mu + home_adv + attacks[home_idx] - defenses[away_idx]
    log_la = mu + attacks[away_idx] - defenses[home_idx]

    log_lh = np.clip(log_lh, -8, 8)
    log_la = np.clip(log_la, -8, 8)

    lh = np.exp(log_lh)
    la = np.exp(log_la)

    ll = (events_home * log_lh - lh) + (events_away * log_la - la)
    return -np.sum(ll)

def estimate_ratings(matches: pd.DataFrame, target_home: str, target_away: str) -> dict:
    if len(matches) < MIN_MATCHES:
        return {}

    teams = sorted(set(matches["home_id"]).union(set(matches["away_id"])))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = matches["home_id"].map(team_to_idx).values
    away_idx = matches["away_id"].map(team_to_idx).values
    
    eh = np.maximum(matches[target_home].fillna(0).values.astype(float), 0.0)
    ea = np.maximum(matches[target_away].fillna(0).values.astype(float), 0.0)

    x0 = np.zeros(2 * n + 1)
    x0[-1] = np.log(max((eh.mean() + ea.mean()) / 2, 0.5))

    res = minimize(
        poisson_neg_loglik,
        x0,
        args=(home_idx, away_idx, eh, ea, n, HOME_ADV),
        method="L-BFGS-B",
        options={"maxiter": MAX_ITER, "disp": False},
    )

    if not res.success:
        return {}

    attacks = res.x[:n]
    defenses = res.x[n:2*n]

    factor = len(matches) / (len(matches) + SHRINKAGE_K)
    attacks *= factor
    defenses *= factor

    return {
        t: {"attack": float(attacks[i]), "defense": float(defenses[i])}
        for t, i in team_to_idx.items()
    }

def build_home_away(df: pd.DataFrame) -> pd.DataFrame:
    home = df[df["mando"] == "C"][[
        "temporada", "rodada", "partida_id", "clube_id", "adversario_id",
        "gols_marcados", "gols_sofridos", "FTT", "FS"
    ]].copy()
    
    home = home.rename(columns={
        "clube_id": "home_id",
        "adversario_id": "away_id",
        "gols_marcados": "gols_home",
        "gols_sofridos": "gols_away",
        "FTT": "ftt_home",
        "FS": "fs_away"
    })
    
    away = df[df["mando"] == "F"][[
        "partida_id", "FTT", "FS"
    ]].copy()
    away = away.rename(columns={"FTT": "ftt_away", "FS": "fs_home"})
    
    home = home.merge(away, on="partida_id", how="left")
    return home.drop_duplicates(subset=["partida_id"])

def main():
    print("=" * 70)
    print("EXP-12 — Team Strength Ratings V3 (Histórico Completo 2024-2026)")
    print("=" * 70)

    # ------------------------------------------------------------------ 1. Carregar e preparar partidas (CORREÇÃO AQUI)
    print("\n[1/6] Carregando e preparando partidas...")
    
    hist_2026 = pd.read_parquet(MATCH_HISTORY)
    
    if MATCH_HISTORY_24_25.exists():
        hist_24_25 = pd.read_parquet(MATCH_HISTORY_24_25)
        # Concatena garantindo que as colunas essenciais estejam presentes
        matches = pd.concat([hist_24_25, hist_2026], ignore_index=True)
        print("     ✅ Histórico 2024-2025 carregado e concatenado com sucesso.")
    else:
        matches = hist_2026.copy()
        print("     ⚠️ Aviso: team_scout_history_2024_2025.parquet não encontrado.")

    matches = build_home_away(matches)
    matches = matches.sort_values(["temporada", "rodada", "partida_id"]).reset_index(drop=True)

    print(f"     Partidas únicas totais: {len(matches):,}")
    print(f"     Temporadas cobertas: {sorted(matches['temporada'].unique())}")

    # ------------------------------------------------------------------ 2. Gerar ratings temporais (sem leakage)
    print("\n[2/6] Estimando ratings com segurança temporal...")
    ratings_rows = []

    for temporada in sorted(matches["temporada"].unique()):
        seas = matches[matches["temporada"] == temporada]
        rodadas = sorted(seas["rodada"].unique())

        for R in rodadas:
            hist = seas[seas["rodada"] < R]
            if len(hist) < MIN_MATCHES:
                continue

            ratings = estimate_ratings(hist, target_home="ftt_home", target_away="fs_away")
            if not ratings:
                continue

            for clube_id, vals in ratings.items():
                ratings_rows.append({
                    "temporada": temporada,
                    "rodada": R,
                    "clube_id": clube_id,
                    "rating_attack_ftt": vals["attack"],
                    "rating_defense_fs": vals["defense"],
                })

        print(f"     Temporada {temporada}: {len(rodadas)} rodadas processadas")

    ratings_df = pd.DataFrame(ratings_rows)
    print(f"\n     Total de ratings gerados: {len(ratings_df):,}")

    if len(ratings_df) == 0:
        raise RuntimeError("Nenhum rating gerado. Verifique os dados.")

    ratings_df.to_parquet(OUT_RATINGS, index=False)
    print(f"     Salvo em: {OUT_RATINGS}")

    # ------------------------------------------------------------------ 3. Join com goleiros
    print("\n[3/6] Fazendo join com goleiros participantes...")
    gol = pd.read_parquet(GOL_POOL)
    eleg = gol[gol["elegivel_modelo"] == True].copy()

    eleg = eleg.merge(
        ratings_df.rename(columns={
            "clube_id": "clube_id",
            "rating_attack_ftt": "rating_attack_own",
            "rating_defense_fs": "rating_defense_own",
        }),
        on=["temporada", "rodada", "clube_id"],
        how="left",
    )

    eleg = eleg.merge(
        ratings_df.rename(columns={
            "clube_id": "adversario_id",
            "rating_attack_ftt": "rating_attack_opp",
            "rating_defense_fs": "rating_defense_opp",
        }),
        on=["temporada", "rodada", "adversario_id"],
        how="left",
    )

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

    report = f"""# EXP-12 — Team Strength Ratings V3 (FTT/FS Proxy + Histórico Completo)

## Resultado do poder preditivo

{eval_df.to_markdown(index=False)}

## Interpretação
- A cobertura deve ter saltado de ~25% para >90% ao incluir os dados de 2024 e 2025.
- Se o Spearman de `soma_forca` e `rating_attack_opp` se mantiver superior às médias simples 
  com essa amostra muito maior, temos uma feature candidata definitiva para o modelo.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"\n[6/6] Arquivos salvos:")
    print(f"  {OUT_RATINGS}")
    print(f"  {OUT_EVAL}")
    print(f"  {OUT_REPORT}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
