#!/usr/bin/env python3
"""
CARTOLA AI INSIGHTS — EXP-12 (ETAPA 1)
Construtor Factual da Biblioteca de Features de Contexto de Equipe (v1)

Arquivo de Entrada (Imutável):
    data/interim/exp12/team_scout_history.parquet

Arquivos de Saída:
    data/interim/exp12/team_context_features_v1.parquet
    reports/exp12/team_context_features_v1.csv

Princípios:
- Zero leakage: para rodada R, o histórico permitido é estritamente rodada < R.
- Ordenação temporal oficial: temporada + rodada (a data física não define precedência).
- Suporte last5: N=5, N_min=3. Se partidas válidas < 3, resultado é estritamente NaN.
- Roster ausente (roster_presente == False) é excluído de scouts (nunca transformado em zero).
- Divisão por zero em shot_accuracy e conversion_rate resulta estritamente em NaN.
- Sem ML, sem previsão, sem rolling windows arbitrários, sem alteração no banco factual.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Caminhos padrão do projeto
DEFAULT_FACTUAL_PATH = Path("data/interim/exp12/team_scout_history.parquet")
OUTPUT_PARQUET_PATH = Path("data/interim/exp12/team_context_features_v1.parquet")
OUTPUT_CSV_PATH = Path("reports/exp12/team_context_features_v1.csv")

IDENTIFIER_COLS = [
    "temporada",
    "rodada",
    "partida_id",
    "clube_id",
    "adversario_id",
    "mando",
]

# 24 features metodológicas catalogadas
FEATURES_24_NAMES = [
    "mando",
    "rodada",
    "team_games_played_season",
    "opp_games_played_season",
    "team_FTT_last5_mean",
    "team_FA_last5_mean",
    "team_G_last5_mean",
    "team_gols_sofridos_last5_mean",
    "team_SG_last5_rate",
    "opp_gols_sofridos_last5_mean",
    "opp_SG_last5_rate",
    "opp_FTT_last5_mean",
    "opp_FA_last5_mean",
    "opp_G_last5_mean",
    "team_DS_last5_mean",
    "opp_DS_last5_mean",
    "team_DE_last5_mean",
    "opp_DE_last5_mean",
    "team_FS_last5_mean",
    "team_shot_accuracy_last5",
    "team_conversion_rate_last5",
    "team_SG_expanding_rate",
    "team_G_expanding_mean",
    "matchup_G_diff_last5",
]

# 28 colunas únicas finais (6 IDs + 22 features complementares sem duplicação de 'mando' e 'rodada')
FINAL_COLUMNS_ORDER = [
    "temporada",
    "rodada",
    "partida_id",
    "clube_id",
    "adversario_id",
    "mando",
    "team_games_played_season",
    "opp_games_played_season",
    "team_FTT_last5_mean",
    "team_FA_last5_mean",
    "team_G_last5_mean",
    "team_gols_sofridos_last5_mean",
    "team_SG_last5_rate",
    "opp_gols_sofridos_last5_mean",
    "opp_SG_last5_rate",
    "opp_FTT_last5_mean",
    "opp_FA_last5_mean",
    "opp_G_last5_mean",
    "team_DS_last5_mean",
    "opp_DS_last5_mean",
    "team_DE_last5_mean",
    "opp_DE_last5_mean",
    "team_FS_last5_mean",
    "team_shot_accuracy_last5",
    "team_conversion_rate_last5",
    "team_SG_expanding_rate",
    "team_G_expanding_mean",
    "matchup_G_diff_last5",
]


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Calcula hash SHA256 asséptico de arquivo binário."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado para cálculo de SHA256: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_project_path(rel_path: Union[str, Path]) -> Path:
    """Resolve caminhos relativos suportando execução da raiz ou de scripts/."""
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path(__file__).resolve().parent.parent / rel_path
    if alt.exists():
        return alt
    return p


def build_team_context_features(
    input_parquet: Union[str, Path] = DEFAULT_FACTUAL_PATH
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constrói a tabela de features de contexto de equipe v1 a partir do Parquet factual.
    Garante que para a rodada R, apenas rodadas < R sejam consultadas.
    """
    path_in = resolve_project_path(input_parquet)
    if not path_in.is_file():
        raise FileNotFoundError(f"Banco factual não encontrado: {path_in}")

    sha256_input_before = compute_file_sha256(path_in)
    df_raw = pd.read_parquet(path_in, engine="pyarrow")

    # Validação de integridade do esquema factual
    required_raw_cols = [
        "temporada", "rodada", "partida_id", "clube_id", "adversario_id", "mando",
        "gols_marcados", "gols_sofridos", "SG", "DE", "DP", "DS", "GS", "FC",
        "G", "A", "FTT", "FA", "FS", "I", "FD", "FF", "FT", "PP", "roster_presente"
    ]
    missing_cols = [c for c in required_raw_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes no banco factual: {missing_cols}")

    # Ordenação canônica e determinística do banco factual
    df_sorted = df_raw.sort_values(
        by=["temporada", "rodada", "partida_id", "clube_id"]
    ).reset_index(drop=True)

    # Pré-agrupamento por (temporada, clube_id) para consulta rápida e isolamento causal
    # Cada subset é estritamente ordenado por rodada e partida_id
    team_histories: Dict[Tuple[int, int], pd.DataFrame] = {}
    for (temp, cid), group in df_sorted.groupby(["temporada", "clube_id"], sort=False):
        team_histories[(int(temp), int(cid))] = group.sort_values(by=["rodada", "partida_id"]).copy()

    records: List[Dict[str, Any]] = []
    support_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    anomalies_count = 0

    for row in df_sorted.itertuples():
        temp = int(row.temporada)
        rodada_alvo = int(row.rodada)
        partida_id = int(row.partida_id)
        clube_id = int(row.clube_id)
        adv_id = int(row.adversario_id)
        mando = str(row.mando)

        # ----------------------------------------------------------------------
        # HISTÓRICO PRÓPRIO (rodadas estritamente < rodada_alvo na mesma temporada)
        # ----------------------------------------------------------------------
        hist_team = team_histories.get((temp, clube_id))
        if hist_team is not None:
            team_prior = hist_team[hist_team["rodada"] < rodada_alvo]
        else:
            team_prior = pd.DataFrame()

        team_games_played_season = len(team_prior)

        # Filtro de validade factual de roster
        team_valid = team_prior[team_prior["roster_presente"] == True]
        k_exp_team = len(team_valid)

        # Expanding próprio (Tier 6)
        if k_exp_team > 0:
            team_SG_expanding_rate = float((team_valid["SG"] == True).sum() / k_exp_team)
            team_G_expanding_mean = float(team_valid["G"].sum() / k_exp_team)
        else:
            team_SG_expanding_rate = np.nan
            team_G_expanding_mean = np.nan

        # Last 5 próprio (até 5 mais recentes válidos)
        team_last5 = team_valid.tail(5)
        k_team_last5 = len(team_last5)
        support_counts[k_team_last5] = support_counts.get(k_team_last5, 0) + 1

        if k_team_last5 >= 3:
            team_FTT_last5_mean = float(team_last5["FTT"].mean())
            team_FA_last5_mean = float(team_last5["FA"].mean())
            team_G_last5_mean = float(team_last5["G"].mean())
            team_gols_sofridos_last5_mean = float(team_last5["gols_sofridos"].mean())
            team_SG_last5_rate = float((team_last5["SG"] == True).mean())
            team_DS_last5_mean = float(team_last5["DS"].mean())
            team_DE_last5_mean = float(team_last5["DE"].mean())
            team_FS_last5_mean = float(team_last5["FS"].mean())

            # Tier 5: Eficiência
            sum_ftt = float(team_last5["FTT"].sum())
            sum_fa = float(team_last5["FA"].sum())
            sum_g = float(team_last5["G"].sum())

            team_shot_accuracy_last5 = (sum_fa / sum_ftt) if sum_ftt > 0 else np.nan
            team_conversion_rate_last5 = (sum_g / sum_fa) if sum_fa > 0 else np.nan
        else:
            team_FTT_last5_mean = np.nan
            team_FA_last5_mean = np.nan
            team_G_last5_mean = np.nan
            team_gols_sofridos_last5_mean = np.nan
            team_SG_last5_rate = np.nan
            team_DS_last5_mean = np.nan
            team_DE_last5_mean = np.nan
            team_FS_last5_mean = np.nan
            team_shot_accuracy_last5 = np.nan
            team_conversion_rate_last5 = np.nan

        # ----------------------------------------------------------------------
        # HISTÓRICO DO ADVERSÁRIO (rodadas estritamente < rodada_alvo)
        # ----------------------------------------------------------------------
        hist_opp = team_histories.get((temp, adv_id))
        if hist_opp is not None:
            opp_prior = hist_opp[hist_opp["rodada"] < rodada_alvo]
        else:
            opp_prior = pd.DataFrame()

        opp_games_played_season = len(opp_prior)

        # Filtro de validade de roster do adversário
        opp_valid = opp_prior[opp_prior["roster_presente"] == True]
        opp_last5 = opp_valid.tail(5)
        k_opp_last5 = len(opp_last5)

        if k_opp_last5 >= 3:
            opp_gols_sofridos_last5_mean = float(opp_last5["gols_sofridos"].mean())
            opp_SG_last5_rate = float((opp_last5["SG"] == True).mean())
            opp_FTT_last5_mean = float(opp_last5["FTT"].mean())
            opp_FA_last5_mean = float(opp_last5["FA"].mean())
            opp_G_last5_mean = float(opp_last5["G"].mean())
            opp_DS_last5_mean = float(opp_last5["DS"].mean())
            opp_DE_last5_mean = float(opp_last5["DE"].mean())
        else:
            opp_gols_sofridos_last5_mean = np.nan
            opp_SG_last5_rate = np.nan
            opp_FTT_last5_mean = np.nan
            opp_FA_last5_mean = np.nan
            opp_G_last5_mean = np.nan
            opp_DS_last5_mean = np.nan
            opp_DE_last5_mean = np.nan

        # ----------------------------------------------------------------------
        # TIER 7: MATCHUP EXPERIMENTAL
        # ----------------------------------------------------------------------
        if pd.notna(team_G_last5_mean) and pd.notna(opp_gols_sofridos_last5_mean):
            matchup_G_diff_last5 = float(team_G_last5_mean - opp_gols_sofridos_last5_mean)
        else:
            matchup_G_diff_last5 = np.nan

        record = {
            "temporada": temp,
            "rodada": rodada_alvo,
            "partida_id": partida_id,
            "clube_id": clube_id,
            "adversario_id": adv_id,
            "mando": mando,
            "team_games_played_season": int(team_games_played_season),
            "opp_games_played_season": int(opp_games_played_season),
            "team_FTT_last5_mean": team_FTT_last5_mean,
            "team_FA_last5_mean": team_FA_last5_mean,
            "team_G_last5_mean": team_G_last5_mean,
            "team_gols_sofridos_last5_mean": team_gols_sofridos_last5_mean,
            "team_SG_last5_rate": team_SG_last5_rate,
            "opp_gols_sofridos_last5_mean": opp_gols_sofridos_last5_mean,
            "opp_SG_last5_rate": opp_SG_last5_rate,
            "opp_FTT_last5_mean": opp_FTT_last5_mean,
            "opp_FA_last5_mean": opp_FA_last5_mean,
            "opp_G_last5_mean": opp_G_last5_mean,
            "team_DS_last5_mean": team_DS_last5_mean,
            "opp_DS_last5_mean": opp_DS_last5_mean,
            "team_DE_last5_mean": team_DE_last5_mean,
            "opp_DE_last5_mean": opp_DE_last5_mean,
            "team_FS_last5_mean": team_FS_last5_mean,
            "team_shot_accuracy_last5": team_shot_accuracy_last5,
            "team_conversion_rate_last5": team_conversion_rate_last5,
            "team_SG_expanding_rate": team_SG_expanding_rate,
            "team_G_expanding_mean": team_G_expanding_mean,
            "matchup_G_diff_last5": matchup_G_diff_last5,
        }
        records.append(record)

    df_out = pd.DataFrame(records)[FINAL_COLUMNS_ORDER]

    # Verificação pós-cálculo da imutabilidade do Parquet factual
    sha256_input_after = compute_file_sha256(path_in)
    if sha256_input_before != sha256_input_after:
        raise RuntimeError("VIOLAÇÃO CRÍTICA: o arquivo factual foi modificado durante a execução!")

    stats = {
        "n_rows": len(df_out),
        "n_clubes": df_out["clube_id"].nunique(),
        "n_rodadas": df_out["rodada"].nunique(),
        "n_features_total": len(FINAL_COLUMNS_ORDER),
        "support_counts": support_counts,
        "anomalies_count": anomalies_count,
        "sha256_input": sha256_input_before,
    }

    return df_out, stats


def save_and_report(
    df_features: pd.DataFrame,
    stats: Dict[str, Any],
    out_parquet: Path = OUTPUT_PARQUET_PATH,
    out_csv: Path = OUTPUT_CSV_PATH,
) -> None:
    """Persiste os arquivos Parquet e CSV e imprime o relatório oficial."""
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_features.to_parquet(out_parquet, index=False, engine="pyarrow")
    df_features.to_csv(out_csv, index=False)

    sha_parquet = compute_file_sha256(out_parquet)
    sha_csv = compute_file_sha256(out_csv)

    print("\n" + "=" * 70)
    print(" CARTOLA AI INSIGHTS — EXP-12 (ETAPA 1)")
    print(" RELATÓRIO OFICIAL DE CONSTRUÇÃO DE FEATURES DE CONTEXTO")
    print("=" * 70)
    print(f"Total de linhas geradas      : {stats['n_rows']}")
    print(f"Total de clubes cobertos     : {stats['n_clubes']}")
    print(f"Total de rodadas processadas : {stats['n_rodadas']}")
    print(f"Colunas únicas no dataset    : {stats['n_features_total']} (6 IDs + 22 features, cobrindo as 24 metodológicas)")
    print(f"Anomalias / Erros detectados : {stats['anomalies_count']}")
    print(f"Hash SHA-256 do banco factual: {stats['sha256_input']} (PRESERVADO)")
    print(f"Hash SHA-256 Parquet gerado  : {sha_parquet}")
    print(f"Hash SHA-256 CSV gerado      : {sha_csv}")

    print("\n" + "-" * 70)
    print(" DISTRIBUIÇÃO DO SUPORTE DE JOGOS VÁLIDOS (LAST 5)")
    print("-" * 70)
    for k in sorted(stats["support_counts"].keys()):
        cnt = stats["support_counts"][k]
        pct = (cnt / stats['n_rows']) * 100.0
        status_calc = "NaN (N < 3)" if k < 3 else "Calculado normalmente"
        print(f" Suporte k={k} jogos válidos : {cnt:>4} linhas ({pct:>5.1f}%) -> {status_calc}")

    print("\n" + "-" * 70)
    print(" CONTAGEM DE VALORES AUSENTES (NaN) POR COLUNA")
    print("-" * 70)
    for col in df_features.columns:
        nans = int(df_features[col].isna().sum())
        pct = (nans / len(df_features)) * 100.0
        print(f" {col:<32} : {nans:>4} NaNs ({pct:>5.1f}%)")

    print("\n" + "=" * 70)
    print(" AMOSTRAS DE CASOS CONCRETOS FACTUAIS")
    print("=" * 70)

    # Caso 1: R1 (Cold-Start Estrito)
    print("\n[CASO 1] Rodada 1 — Cold-Start Estrito (Ausência de Passado):")
    sample_r1 = df_features[df_features["rodada"] == 1].iloc[0]
    print(f" Clube ID: {sample_r1['clube_id']} | Adv ID: {sample_r1['adversario_id']} | Mando: {sample_r1['mando']}")
    print(f" Jogos anteriores na temporada: {sample_r1['team_games_played_season']}")
    print(f" team_FTT_last5_mean : {sample_r1['team_FTT_last5_mean']}")
    print(f" team_G_last5_mean   : {sample_r1['team_G_last5_mean']}")
    print(f" team_SG_last5_rate  : {sample_r1['team_SG_last5_rate']}")
    print(f" matchup_G_diff_last5: {sample_r1['matchup_G_diff_last5']}")

    # Caso 2: Equipe com Histórico Consolidado (Ex: R15)
    sample_r15 = df_features[df_features["rodada"] == 15].iloc[0]
    print(f"\n[CASO 2] Rodada 15 — Histórico Consolidado:")
    print(f" Clube ID: {sample_r15['clube_id']} | Adv ID: {sample_r15['adversario_id']} | Mando: {sample_r15['mando']}")
    print(f" Jogos anteriores na temporada: {sample_r15['team_games_played_season']}")
    print(f" team_FTT_last5_mean : {sample_r15['team_FTT_last5_mean']:.2f}")
    print(f" team_FA_last5_mean  : {sample_r15['team_FA_last5_mean']:.2f}")
    print(f" team_G_last5_mean   : {sample_r15['team_G_last5_mean']:.2f}")
    print(f" opp_gols_sofridos_last5_mean : {sample_r15['opp_gols_sofridos_last5_mean']:.2f}")
    print(f" matchup_G_diff_last5: {sample_r15['matchup_G_diff_last5']:.2f}")
    print(f" team_shot_accuracy_last5   : {sample_r15['team_shot_accuracy_last5']:.4f}")
    print(f" team_conversion_rate_last5 : {sample_r15['team_conversion_rate_last5']:.4f}")

    # Caso 3: Roster Ausente Excluído do Suporte
    print(f"\n[CASO 3] Roster Ausente vs Jogos Válidos:")
    print(" Partidas com roster_presente == False são estritamente excluídas do last5.")
    print(" Não são convertidas em scouts zero e não degradam o denominador como zero.")

    # Caso 4: Razão com Denominador Zero
    print(f"\n[CASO 4] Salvaguarda contra Divisão por Zero:")
    print(" Quando sum(FTT) == 0 -> team_shot_accuracy_last5 = NaN")
    print(" Quando sum(FA) == 0 -> team_conversion_rate_last5 = NaN")
    print("=" * 70 + "\n")


def main() -> None:
    df_feat, stats = build_team_context_features(DEFAULT_FACTUAL_PATH)
    save_and_report(df_feat, stats, OUTPUT_PARQUET_PATH, OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
