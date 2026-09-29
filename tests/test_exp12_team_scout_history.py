"""
Testes de integridade factual para data/interim/exp12/team_scout_history.parquet.
Garante validação por chave composta (temporada, rodada, partida_id, clube_id),
critérios estritos de participação, exclusão de técnicos, ausência de posições artificiais,
consistência factual de mando ('C' e 'F') e determinismo de conteúdo e schema.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.exp12_build_team_scout_history import (
    build_team_scout_history,
    CartolaRawExtractor,
)

PARQUET_PATH = Path("data/interim/exp12/team_scout_history.parquet")
CSV_PATH = Path("reports/exp12/team_scout_history.csv")
F1_PARQUET_PATH = Path("data/interim/exp12/team_match_history.parquet")
RAW_API_DIR = Path("experiments/data/cartola_api_2026")


@pytest.fixture(scope="session")
def loaded_dataset():
    if not PARQUET_PATH.exists():
        if F1_PARQUET_PATH.exists() and RAW_API_DIR.exists():
            df, _ = build_team_scout_history(
                input_parquet=F1_PARQUET_PATH,
                raw_api_dir=RAW_API_DIR,
                output_parquet=PARQUET_PATH,
                output_csv=CSV_PATH,
            )
            return df
        else:
            pytest.skip("Arquivos de entrada reais não disponíveis.")
    return pd.read_parquet(PARQUET_PATH)


def test_no_duplicate_team_match(loaded_dataset):
    """Garante chave primária única: (temporada, rodada, partida_id, clube_id)."""
    dups = loaded_dataset.duplicated(subset=["temporada", "rodada", "partida_id", "clube_id"])
    assert dups.sum() == 0, f"Existem {dups.sum()} chaves primárias duplicadas."


def test_two_teams_per_match(loaded_dataset):
    """Garante exatamente 2 equipes por chave de partida: (temporada, rodada, partida_id)."""
    counts = loaded_dataset.groupby(["temporada", "rodada", "partida_id"]).size()
    anomalies = counts[counts != 2]
    assert len(anomalies) == 0, f"Partidas com contagem divergente de 2 equipes: {anomalies.to_dict()}"


def test_opponent_symmetry(loaded_dataset):
    """Garante a simetria de adversário na chave composta da partida."""
    for (temp, rod, pid), group in loaded_dataset.groupby(["temporada", "rodada", "partida_id"]):
        assert len(group) == 2, f"Partida {(temp, rod, pid)} não possui 2 linhas."
        team_a = group.iloc[0]
        team_b = group.iloc[1]

        assert team_a["adversario_id"] == team_b["clube_id"], (
            f"Simetria falhou na partida {(temp, rod, pid)}: A.adversario={team_a['adversario_id']} != B.clube={team_b['clube_id']}"
        )
        assert team_b["adversario_id"] == team_a["clube_id"], (
            f"Simetria falhou na partida {(temp, rod, pid)}: B.adversario={team_b['adversario_id']} != A.clube={team_a['clube_id']}"
        )


def test_home_away_consistency(loaded_dataset):
    """Garante exatamente 1 mandante ('C') e 1 visitante ('F') por partida."""
    for (temp, rod, pid), group in loaded_dataset.groupby(["temporada", "rodada", "partida_id"]):
        mandos = sorted(list(group["mando"]))
        assert mandos == ["C", "F"], (
            f"Configuração de mando inválida na partida {(temp, rod, pid)}: {mandos}. "
            "Esperado exatamente um 'C' e um 'F'."
        )


def test_participants_only_players(loaded_dataset):
    """Garante que nenhum técnico apareça na lista de participantes nem em participantes_por_posicao."""
    for _, row in loaded_dataset.iterrows():
        participantes = row["participantes"]
        pos_dict = json.loads(row["participantes_por_posicao"])

        assert "6" not in pos_dict, f"Posição de técnico ('6') encontrada em participantes_por_posicao na linha {row['partida_id']}"

        for p in participantes:
            assert p["posicao_id"] != 6, f"Técnico (posicao_id=6) em participantes: {p}"
            if p.get("posicao_nome"):
                pnome = p["posicao_nome"].lower()
                assert "técnico" not in pnome and "tecnico" not in pnome, f"Técnico por nome em participantes: {p}"


def test_no_duplicate_participants(loaded_dataset):
    """Garante que nenhum atleta apareça repetido dentro da mesma equipe/partida."""
    for _, row in loaded_dataset.iterrows():
        participantes = row["participantes"]
        ids = [p["atleta_id"] for p in participantes]
        assert len(ids) == len(set(ids)), f"Atleta duplicado na partida {row['partida_id']}, time {row['clube_id']}"


def test_no_artificial_positions(loaded_dataset):
    """Garante ausência de valores fabricados (-1, 'DESCONHECIDA') nas posições."""
    for _, row in loaded_dataset.iterrows():
        for p in row["participantes"]:
            assert p["posicao_id"] != -1, f"posicao_id artificial -1 detectado: {p}"
            assert p.get("posicao_nome") != "DESCONHECIDA", f"posicao_nome artificial 'DESCONHECIDA' detectado: {p}"


def test_participants_count_and_positions_alignment(loaded_dataset):
    """Garante alinhamento estrito entre n_participantes, len(participantes) e participantes_por_posicao."""
    for _, row in loaded_dataset.iterrows():
        participantes = row["participantes"]
        assert row["n_participantes"] == len(participantes)
        pos_dict = json.loads(row["participantes_por_posicao"])
        total_pos = sum(pos_dict.values())
        with_pos = sum(1 for p in participantes if p.get("posicao_id") is not None)
        assert total_pos == with_pos


def test_missing_roster_preserves_facts(loaded_dataset):
    """Garante que ausência de elenco registre roster_presente=False sem zerar scouts factuais."""
    ausentes = loaded_dataset[~loaded_dataset["roster_presente"]]
    assert len(ausentes) == 24, f"Esperado exatamente 24 registros com roster ausente, encontrado: {len(ausentes)}"
    for _, row in ausentes.iterrows():
        assert row["n_participantes"] == 0
        assert len(row["participantes"]) == 0
        assert row["roster_jogadores_raw"] == 0


def test_deterministic_content_and_hash(tmp_path):
    """Valida determinismo estrito de dados, ordenação e integridade."""
    if not (F1_PARQUET_PATH.exists() and RAW_API_DIR.exists()):
        pytest.skip("Arquivos de entrada reais não disponíveis.")

    out1_p = tmp_path / "run1.parquet"
    out1_c = tmp_path / "run1.csv"
    out2_p = tmp_path / "run2.parquet"
    out2_c = tmp_path / "run2.csv"

    df1, stats1 = build_team_scout_history(F1_PARQUET_PATH, RAW_API_DIR, out1_p, out1_c)
    df2, stats2 = build_team_scout_history(F1_PARQUET_PATH, RAW_API_DIR, out2_p, out2_c)

    pd.testing.assert_frame_equal(df1, df2, check_exact=True)
    assert stats1["sha256_parquet"] == stats2["sha256_parquet"]
    assert stats1["sha256_csv"] == stats2["sha256_csv"]


def test_parquet_roundtrip(loaded_dataset):
    """Valida releitura com schema PyArrow estruturado."""
    table = pq.read_table(PARQUET_PATH)
    col = table.column("participantes")
    assert pa.types.is_list(col.type)
    assert pa.types.is_struct(col.type.value_type)
    assert len(table) == len(loaded_dataset)


def test_csv_roundtrip(loaded_dataset):
    """Valida releitura do CSV espelho e integridade JSON em cada linha."""
    df_csv = pd.read_csv(CSV_PATH)
    assert len(df_csv) == len(loaded_dataset)
    for cell in df_csv["participantes"].dropna():
        parsed = json.loads(cell)
        assert isinstance(parsed, list)
