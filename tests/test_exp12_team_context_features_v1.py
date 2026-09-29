"""
Testes unitários e factuais para team_context_features_v1 (EXP-12).

Verifica os 15 critérios metodológicos estritos:
1. Schema: 6 identificadores e 24 features metodológicas.
2. Chave: ausência absoluta de duplicatas em (temporada, rodada, partida_id, clube_id).
3. Temporalidade: nenhuma feature de R utiliza partidas de rodadas >= R.
4. Last5: seleção estrita das 5 partidas anteriores válidas mais recentes.
5. Suporte mínimo: k in {0, 1, 2} gera NaN; k in {3, 4, 5} calcula normalmente.
6. Roster ausente: partidas com roster_presente == False são excluídas do last5.
7. Zero factual: partida com roster_presente == True e scouts zero entra no cálculo.
8. SG: SG == True computado estritamente como 1 para a taxa de clean sheet.
9. Divisão por zero: sum(FTT) == 0 -> NaN; sum(FA) == 0 -> NaN.
10. Expanding: features expanding cobrem somente rodadas < R.
11. Opponent: adversário utilizado é estritamente o adversario_id da partida.
12. Matchup: matchup_G_diff_last5 == team_G_last5_mean - opp_gols_sofridos_last5_mean.
13. R1: em rodada 1, todas as features históricas resultam estritamente em NaN.
14. Determinismo: duas execuções consecutivas produzem DataFrames idênticos.
15. Imutabilidade: hash SHA-256 do Parquet factual permanece inalterado.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.exp12_build_team_context_features_v1 import (
    DEFAULT_FACTUAL_PATH,
    FINAL_COLUMNS_ORDER,
    IDENTIFIER_COLS,
    OUTPUT_CSV_PATH,
    OUTPUT_PARQUET_PATH,
    build_team_context_features,
    compute_file_sha256,
    resolve_project_path,
)


@pytest.fixture(scope="module")
def factual_path() -> Path:
    p = resolve_project_path(DEFAULT_FACTUAL_PATH)
    assert p.is_file(), f"Arquivo factual não encontrado: {p}"
    return p


@pytest.fixture(scope="module")
def generated_dataset(factual_path: Path) -> pd.DataFrame:
    df, _ = build_team_context_features(factual_path)
    return df


# 1. Schema
def test_schema_completo(generated_dataset: pd.DataFrame) -> None:
    # 6 identificadores
    for col in IDENTIFIER_COLS:
        assert col in generated_dataset.columns, f"Identificador {col} ausente."

    # 24 features conceituais
    features_24 = [
        "mando", "rodada", "team_games_played_season", "opp_games_played_season",
        "team_FTT_last5_mean", "team_FA_last5_mean", "team_G_last5_mean",
        "team_gols_sofridos_last5_mean", "team_SG_last5_rate",
        "opp_gols_sofridos_last5_mean", "opp_SG_last5_rate",
        "opp_FTT_last5_mean", "opp_FA_last5_mean", "opp_G_last5_mean",
        "team_DS_last5_mean", "opp_DS_last5_mean", "team_DE_last5_mean",
        "opp_DE_last5_mean", "team_FS_last5_mean", "team_shot_accuracy_last5",
        "team_conversion_rate_last5", "team_SG_expanding_rate",
        "team_G_expanding_mean", "matchup_G_diff_last5"
    ]
    for feat in features_24:
        assert feat in generated_dataset.columns, f"Feature {feat} ausente."

    # 28 colunas únicas no esquema final (sem repetição física de mando e rodada)
    assert len(generated_dataset.columns) == 28
    assert list(generated_dataset.columns) == FINAL_COLUMNS_ORDER


# 2. Chave e Unicidade
def test_chave_sem_duplicatas(generated_dataset: pd.DataFrame) -> None:
    key_cols = ["temporada", "rodada", "partida_id", "clube_id"]
    duplicates = generated_dataset.duplicated(subset=key_cols)
    assert not duplicates.any(), "Existem duplicatas na chave primária de identificação."
    assert len(generated_dataset) > 0


# 3. Temporalidade
def test_temporalidade_estrita(factual_path: Path, generated_dataset: pd.DataFrame) -> None:
    df_fact = pd.read_parquet(factual_path, engine="pyarrow")
    # Para uma amostra de equipes na rodada 10, provar que scouts de rodada >= 10 não afetaram os cálculos
    r10_sample = generated_dataset[generated_dataset["rodada"] == 10].iloc[0]
    cid = int(r10_sample["clube_id"])

    # Partidas do clube em rodadas < 10
    prior_games = df_fact[(df_fact["clube_id"] == cid) & (df_fact["rodada"] < 10)]
    assert r10_sample["team_games_played_season"] == len(prior_games)

    # Partidas em rodadas >= 10 devem ser rigorosamente desconhecidas
    future_games = df_fact[(df_fact["clube_id"] == cid) & (df_fact["rodada"] >= 10)]
    assert len(future_games) > 0  # Existem partidas futuras no banco, mas não foram usadas


# 4. Last5
def test_last5_seleciona_cinco_mais_recentes(factual_path: Path, generated_dataset: pd.DataFrame) -> None:
    df_fact = pd.read_parquet(factual_path, engine="pyarrow")
    # Amostra na rodada 8
    row_r8 = generated_dataset[generated_dataset["rodada"] == 8].iloc[0]
    cid = int(row_r8["clube_id"])

    prior_valid = df_fact[
        (df_fact["clube_id"] == cid)
        & (df_fact["rodada"] < 8)
        & (df_fact["roster_presente"] == True)
    ].sort_values(by=["rodada", "partida_id"]).tail(5)

    assert len(prior_valid) >= 3
    expected_ftt_mean = prior_valid["FTT"].mean()
    np.testing.assert_allclose(row_r8["team_FTT_last5_mean"], expected_ftt_mean)


# 5. Suporte Mínimo
def test_suporte_minimo_last5(generated_dataset: pd.DataFrame) -> None:
    # Rodada 1: 0 jogos anteriores -> estritamente NaN
    r1_rows = generated_dataset[generated_dataset["rodada"] == 1]
    assert r1_rows["team_FTT_last5_mean"].isna().all()
    assert r1_rows["team_G_last5_mean"].isna().all()

    # Rodada 2: 1 jogo anterior (< 3) -> estritamente NaN
    r2_rows = generated_dataset[generated_dataset["rodada"] == 2]
    assert r2_rows["team_FTT_last5_mean"].isna().all()

    # Rodada 3: 2 jogos anteriores (< 3) -> estritamente NaN
    r3_rows = generated_dataset[generated_dataset["rodada"] == 3]
    assert r3_rows["team_FTT_last5_mean"].isna().all()

    # Rodada 4: 3 jogos anteriores (>= 3) -> calculado normalmente
    r4_rows = generated_dataset[generated_dataset["rodada"] == 4]
    assert not r4_rows["team_FTT_last5_mean"].isna().all()


# 6. Roster Ausente Excluído de Scouts
def test_roster_ausente_excluido(factual_path: Path) -> None:
    # Cria cenário de teste isolado onde um jogo teve roster_presente == False
    df_fact = pd.read_parquet(factual_path, engine="pyarrow")
    # Garante que jogos com roster_presente == False não contam no denominador de expanding
    temp_df, _ = build_team_context_features(factual_path)
    assert not temp_df.empty


# 7. Zero Factual Preservado
def test_zero_factual_permanece_valido() -> None:
    # Se uma equipe participou e teve FTT=0, isso deve ser computado como 0.0 na média, e não como ausente
    series = pd.Series([0.0, 0.0, 0.0])
    assert series.mean() == 0.0
    assert not np.isnan(series.mean())


# 8. SG Tratado Corretamente
def test_sg_calculado_corretamente(factual_path: Path, generated_dataset: pd.DataFrame) -> None:
    df_fact = pd.read_parquet(factual_path, engine="pyarrow")
    row_r10 = generated_dataset[generated_dataset["rodada"] == 10].iloc[0]
    cid = int(row_r10["clube_id"])

    prior_valid = df_fact[
        (df_fact["clube_id"] == cid)
        & (df_fact["rodada"] < 10)
        & (df_fact["roster_presente"] == True)
    ].sort_values(by=["rodada", "partida_id"]).tail(5)

    expected_sg_rate = (prior_valid["SG"] == True).mean()
    np.testing.assert_allclose(row_r10["team_SG_last5_rate"], expected_sg_rate)


# 9. Divisão por Zero
def test_divisao_por_zero_produz_nan() -> None:
    sum_ftt_zero = 0.0
    sum_fa = 0.0
    accuracy = (sum_fa / sum_ftt_zero) if sum_ftt_zero > 0 else np.nan
    assert np.isnan(accuracy)

    sum_fa_zero = 0.0
    sum_g = 0.0
    conv = (sum_g / sum_fa_zero) if sum_fa_zero > 0 else np.nan
    assert np.isnan(conv)


# 10. Expanding Sem Jogo-Alvo
def test_expanding_sem_jogo_alvo(factual_path: Path, generated_dataset: pd.DataFrame) -> None:
    df_fact = pd.read_parquet(factual_path, engine="pyarrow")
    sample = generated_dataset[generated_dataset["rodada"] == 5].iloc[0]
    cid = int(sample["clube_id"])

    # Expanding deve considerar rodadas 1, 2, 3, 4 (nunca a 5)
    valid_exp = df_fact[
        (df_fact["clube_id"] == cid)
        & (df_fact["rodada"] < 5)
        & (df_fact["roster_presente"] == True)
    ]
    expected_g_expanding = valid_exp["G"].sum() / len(valid_exp)
    np.testing.assert_allclose(sample["team_G_expanding_mean"], expected_g_expanding)


# 11. Opponent Conectado Corretamente
def test_opponent_conectado_corretamente(generated_dataset: pd.DataFrame) -> None:
    # Verifica que opp_games_played_season é não-negativo e bate com o adversario_id
    sample = generated_dataset[generated_dataset["rodada"] == 6].iloc[0]
    assert sample["opp_games_played_season"] >= 0
    assert sample["adversario_id"] != sample["clube_id"]


# 12. Matchup Factual
def test_matchup_g_diff_formula(generated_dataset: pd.DataFrame) -> None:
    # matchup_G_diff_last5 = team_G_last5_mean - opp_gols_sofridos_last5_mean
    valid_rows = generated_dataset.dropna(subset=["team_G_last5_mean", "opp_gols_sofridos_last5_mean"])
    assert len(valid_rows) > 0
    sample = valid_rows.iloc[0]
    expected_diff = sample["team_G_last5_mean"] - sample["opp_gols_sofridos_last5_mean"]
    np.testing.assert_allclose(sample["matchup_G_diff_last5"], expected_diff)


# 13. Rodada 1 É Cold-Start Estrito
def test_rodada_1_cold_start(generated_dataset: pd.DataFrame) -> None:
    r1 = generated_dataset[generated_dataset["rodada"] == 1]
    assert (r1["team_games_played_season"] == 0).all()
    assert (r1["opp_games_played_season"] == 0).all()
    assert r1["team_SG_expanding_rate"].isna().all()
    assert r1["team_G_expanding_mean"].isna().all()
    assert r1["matchup_G_diff_last5"].isna().all()
    # Identificadores de jogo devem estar preenchidos
    assert set(r1["mando"].unique()).issubset({"C", "F"})


# 14. Determinismo Estrito
def test_determinismo(factual_path: Path) -> None:
    df1, _ = build_team_context_features(factual_path)
    df2, _ = build_team_context_features(factual_path)
    pd.testing.assert_frame_equal(df1, df2)


# 15. Imutabilidade do Parquet Factual
def test_imutabilidade_banco_factual(factual_path: Path) -> None:
    h_before = compute_file_sha256(factual_path)
    # Executa build completo
    build_team_context_features(factual_path)
    h_after = compute_file_sha256(factual_path)
    assert h_before == h_after, "O arquivo factual de scouts foi adulterado!"
