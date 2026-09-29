"""
Testes unitários e factuais do explorer da EXP-12.

Verifica:
1. Carregamento do Parquet
2. Presença das 33 colunas esperadas
3. Seleção de time (ID e nome)
4. Seleção de rodada
5. Histórico até rodada X
6. Histórico completo
7. Ordenação por rodada (temporada, rodada, partida_id)
8. Comparação entre duas rodadas
9. Preservação de NaN
10. Nenhum acesso à internet
11. Nenhum dado alterado no Parquet
12. Cálculo correto da diferença entre duas rodadas
13. Descoberta dinâmica de temporadas
14. Descoberta dinâmica de clubes
15. Descoberta dinâmica de rodadas
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.exp12_explorer_team_scout_history import (
    DEFAULT_PARQUET_PATH,
    EXPECTED_COLUMNS,
    TeamScoutHistoryExplorer,
    format_diff,
    format_val,
)


@pytest.fixture(scope="module")
def parquet_path() -> Path:
    path = DEFAULT_PARQUET_PATH
    if not path.is_file():
        alt = Path(__file__).resolve().parent.parent / DEFAULT_PARQUET_PATH
        if alt.is_file():
            return alt
    return path


@pytest.fixture(scope="module")
def explorer(parquet_path: Path) -> TeamScoutHistoryExplorer:
    return TeamScoutHistoryExplorer(parquet_path=parquet_path)


# 1. Carregamento do Parquet
def test_carregamento_parquet(explorer: TeamScoutHistoryExplorer) -> None:
    assert explorer.df is not None
    assert not explorer.df.empty
    assert len(explorer.df) > 0


# 2. Presença das 33 colunas esperadas
def test_presenca_colunas_esperadas(explorer: TeamScoutHistoryExplorer) -> None:
    cols = explorer.df.columns.tolist()
    for col in EXPECTED_COLUMNS:
        assert col in cols, f"Coluna obrigatória ausente: {col}"
    assert len(EXPECTED_COLUMNS) == 33


# 3. Seleção de time (por ID e por nome)
def test_selecao_time(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    teams = explorer.get_teams(temporada=seasons[0])
    assert len(teams) > 0

    first_id, first_name = teams[0]
    # Resolução por ID numérico
    resolved_by_id = explorer.resolve_team(first_id, temporada=seasons[0])
    assert resolved_by_id is not None
    assert resolved_by_id[0] == first_id

    # Resolução por nome
    resolved_by_name = explorer.resolve_team(first_name, temporada=seasons[0])
    assert resolved_by_name is not None
    assert resolved_by_name[0] == first_id

    # Resolução case-insensitive
    resolved_ci = explorer.resolve_team(first_name.lower(), temporada=seasons[0])
    assert resolved_ci is not None
    assert resolved_ci[0] == first_id


# 4. Seleção de rodada
def test_selecao_rodada(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    season = seasons[0]
    rounds = explorer.get_rounds(temporada=season)
    assert len(rounds) > 0

    teams = explorer.get_teams(temporada=season)
    tid = teams[0][0]
    target_round = rounds[0]

    rec = explorer.get_match_record(tid, target_round, season)
    assert rec is not None
    assert rec["rodada"] == target_round
    assert rec["clube_id"] == tid
    assert rec["temporada"] == season


# 5. Histórico até rodada X
def test_historico_ate_rodada_x(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    season = seasons[0]
    rounds = explorer.get_rounds(temporada=season)
    assert len(rounds) >= 2

    limit_round = rounds[min(5, len(rounds) - 1)]
    teams = explorer.get_teams(temporada=season)
    tid = teams[0][0]

    history = explorer.get_team_history(tid, max_rodada=limit_round, temporada=season)
    assert not history.empty
    assert (history["rodada"] <= limit_round).all()


# 6. Histórico completo
def test_historico_completo(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    season = seasons[0]
    teams = explorer.get_teams(temporada=season)
    tid = teams[0][0]

    full_history = explorer.get_team_history(
        tid,
        max_rodada=None,
        temporada=season,
    )

    assert not full_history.empty

    keys = list(
        zip(
            full_history["temporada"],
            full_history["rodada"],
            full_history["partida_id"],
        )
    )

    assert keys == sorted(keys)


# 7. Ordenação por rodada (temporada, rodada, partida_id)
def test_ordenacao_por_rodada(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    season = seasons[0]
    teams = explorer.get_teams(temporada=season)
    tid = teams[0][0]

    history = explorer.get_team_history(
        tid,
        max_rodada=None,
        temporada=season,
    )

    keys = list(
        zip(
            history["temporada"],
            history["rodada"],
            history["partida_id"],
        )
    )

    assert keys == sorted(keys), (
        "O histórico deve estar ordenado por "
        "(temporada, rodada, partida_id)."
    )


# 8. Comparação entre duas rodadas
def test_comparacao_duas_rodadas(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    season = seasons[0]
    rounds = explorer.get_rounds(temporada=season)
    assert len(rounds) >= 2

    teams = explorer.get_teams(temporada=season)
    tid = teams[0][0]

    ra, rb = rounds[0], rounds[1]
    comp = explorer.compare_team_rounds(tid, ra, rb, temporada=season)
    assert comp is not None
    assert comp["record_a"]["rodada"] == ra
    assert comp["record_b"]["rodada"] == rb
    assert len(comp["diffs"]) > 0


# 9. Preservação de NaN
def test_preservacao_nan() -> None:
    # Verifica que format_val preserva NaN sem convertê-lo para 0 ou 0.0
    assert format_val(np.nan) == "-"
    assert format_val(None) == "-"
    assert format_val(pd.NA) == "-"

    # Verifica que inteiros não ganham sufixo .0
    assert format_val(12) == "12"
    assert format_val(12.0) == "12"
    assert format_val(0) == "0"


# 10. Nenhum acesso à internet
def test_nenhum_acesso_internet(monkeypatch: pytest.MonkeyPatch, parquet_path: Path) -> None:
    def fake_connect(*args, **kwargs):
        raise RuntimeError("Acesso à internet não é permitido.")

    monkeypatch.setattr(socket, "socket", fake_connect)

    # Inicializa e executa operações garantindo que nada chame rede
    exp = TeamScoutHistoryExplorer(parquet_path=parquet_path)
    seasons = exp.get_seasons()
    assert len(seasons) > 0


# 11. Nenhum dado alterado no Parquet
def test_nenhum_dado_alterado_no_parquet(parquet_path: Path) -> None:
    def compute_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    hash_before = compute_hash(parquet_path)

    # Executa múltiplas operações de leitura e agregação
    exp = TeamScoutHistoryExplorer(parquet_path=parquet_path)
    s = exp.get_seasons()[0]
    teams = exp.get_teams(temporada=s)
    exp.get_team_history(teams[0][0], max_rodada=10, temporada=s)
    exp.get_round_all_teams(1, temporada=s)

    hash_after = compute_hash(parquet_path)
    assert hash_before == hash_after, "O arquivo Parquet foi modificado!"


# 12. Cálculo correto da diferença entre duas rodadas
def test_calculo_correto_diferenca_duas_rodadas() -> None:
    # Diferença positiva
    assert format_diff(6, 11) == "+5"
    # Diferença negativa
    assert format_diff(4, 2) == "-2"
    # Diferença nula
    assert format_diff(10, 10) == "0"
    # Preservação de ausência com NaN
    assert format_diff(np.nan, 5) == "-"
    assert format_diff(5, np.nan) == "-"
    assert format_diff(np.nan, np.nan) == "-"


# 13. Descoberta dinâmica de temporadas
def test_descoberta_dinamica_temporadas(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    assert isinstance(seasons, list)
    assert len(seasons) >= 1
    # Verifica que vem do próprio DataFrame
    expected = sorted(explorer.df["temporada"].dropna().unique().astype(int).tolist())
    assert seasons == expected


# 14. Descoberta dinâmica de clubes
def test_descoberta_dinamica_clubes(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    teams = explorer.get_teams(temporada=seasons[0])
    assert isinstance(teams, list)
    assert len(teams) > 0
    # Clubes devem estar ordenados alfabeticamente
    nomes = [t[1] for t in teams]
    assert nomes == sorted(nomes)


# 15. Descoberta dinâmica de rodadas
def test_descoberta_dinamica_rodadas(explorer: TeamScoutHistoryExplorer) -> None:
    seasons = explorer.get_seasons()
    rounds = explorer.get_rounds(temporada=seasons[0])
    assert isinstance(rounds, list)
    assert len(rounds) > 0
    expected = sorted(explorer.df[explorer.df["temporada"] == seasons[0]]["rodada"].dropna().unique().astype(int).tolist())
    assert rounds == expected
