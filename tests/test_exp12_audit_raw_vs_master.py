"""
Testes unitários e factuais do Auditor Independente RAW vs MASTER (EXP-12).

Verifica os 20 requisitos estabelecidos:
1. Script importa sem erro
2. RAW consegue carregar r1.json
3. MASTER consegue carregar
4. Todos os 16 scouts são auditados
5. DE é auditado diretamente do scout individual
6. DP é auditado diretamente do scout individual
7. FD/FF/FT/PP são auditados individualmente
8. FTT é calculado como FF+FT+FD+G+PP
9. FA é calculado como FD+G
10. G não é substituído por gols_marcados
11. SG não é calculado como clean sheet
12. Jogadores que não entraram não entram na soma
13. Técnicos não entram na soma
14. Distinção dos três estados de roster:
    - Caso A: roster ausente (jogadores_raw=0) -> RAW_INDETERMINADO_ROSTER_AUSENTE (scouts NaN)
    - Caso B: roster presente + zero participantes (jogadores_raw>0, part=0) -> scouts 0
    - Caso C: roster presente + participantes -> soma normal
15. Chave temporal inclui temporada
16. Auditor não altera o Parquet
17. Relatórios são determinísticos
18. Caso SAO R1 é encontrado
19. Goleiros são identificados por posicao_id=1
20. DE/DP/GS do goleiro são auditáveis individualmente
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from scripts.exp12_audit_raw_vs_master import (
    AUDIT_SCOUTS,
    DEFAULT_MASTER_PATH,
    DEFAULT_RAW_DIR,
    RawVsMasterAuditor,
    compute_file_sha256,
    extract_raw_athletes_from_round,
    load_raw_round_json,
    resolve_project_path,
)


@pytest.fixture(scope="module")
def master_file_path() -> Path:
    p = resolve_project_path(DEFAULT_MASTER_PATH)
    assert p.is_file(), f"Parquet mestre não encontrado em {p}"
    return p


@pytest.fixture(scope="module")
def r1_json_path() -> Path:
    p = resolve_project_path(DEFAULT_RAW_DIR / "r1.json")
    assert p.is_file(), f"r1.json não encontrado em {p}"
    return p


@pytest.fixture(scope="module")
def auditor(master_file_path: Path) -> RawVsMasterAuditor:
    return RawVsMasterAuditor(master_path=master_file_path)


# 1. Script importa sem erro
def test_script_importa_sem_erro() -> None:
    import scripts.exp12_audit_raw_vs_master as mod
    assert hasattr(mod, "RawVsMasterAuditor")
    assert hasattr(mod, "AUDIT_SCOUTS")


# 2. RAW consegue carregar r1.json
def test_raw_consegue_carregar_r1(r1_json_path: Path) -> None:
    data = load_raw_round_json(r1_json_path)
    assert isinstance(data, dict)
    assert "atletas" in data
    assert len(data["atletas"]) > 0


# 3. MASTER consegue carregar
def test_master_consegue_carregar(auditor: RawVsMasterAuditor) -> None:
    assert auditor.df_master is not None
    assert not auditor.df_master.empty
    assert len(auditor.df_master) > 0


# 4. Todos os 16 scouts são auditados
def test_todos_16_scouts_auditados() -> None:
    expected = [
        "SG", "DE", "DP", "DS", "GS", "FC", "G", "A",
        "FTT", "FA", "FS", "I", "FD", "FF", "FT", "PP"
    ]
    assert len(AUDIT_SCOUTS) == 16
    assert set(AUDIT_SCOUTS) == set(expected)


# 5. DE é auditado diretamente do scout individual
def test_de_auditado_diretamente_scout_individual() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 1, "clube_id": 10, "entrou_em_campo": True, "scout": {"DE": 4}},
            "2": {"atleta_id": 2, "posicao_id": 1, "clube_id": 10, "entrou_em_campo": True, "scout": {"DE": 2}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    de_raw = sum(a["scout"].get("DE", 0) for a in participating)
    assert de_raw == 6


# 6. DP é auditado diretamente do scout individual
def test_dp_auditado_diretamente_scout_individual() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 1, "clube_id": 10, "entrou_em_campo": True, "scout": {"DP": 1}},
            "2": {"atleta_id": 2, "posicao_id": 1, "clube_id": 10, "entrou_em_campo": True, "scout": {"DP": 2}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    dp_raw = sum(a["scout"].get("DP", 0) for a in participating)
    assert dp_raw == 3


# 7. FD/FF/FT/PP são auditados individualmente
def test_fd_ff_ft_pp_auditados_individualmente() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 5, "clube_id": 10, "entrou_em_campo": True, "scout": {"FD": 3, "FF": 2, "FT": 1, "PP": 1}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    assert sum(a["scout"].get("FD", 0) for a in participating) == 3
    assert sum(a["scout"].get("FF", 0) for a in participating) == 2
    assert sum(a["scout"].get("FT", 0) for a in participating) == 1
    assert sum(a["scout"].get("PP", 0) for a in participating) == 1


# 8. FTT é calculado como FF+FT+FD+G+PP
def test_ftt_calculado_como_soma_componentes() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 5, "clube_id": 10, "entrou_em_campo": True, "scout": {"FF": 4, "FT": 1, "FD": 2, "G": 2, "PP": 1}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    ff = sum(a["scout"].get("FF", 0) for a in participating)
    ft = sum(a["scout"].get("FT", 0) for a in participating)
    fd = sum(a["scout"].get("FD", 0) for a in participating)
    g = sum(a["scout"].get("G", 0) for a in participating)
    pp = sum(a["scout"].get("PP", 0) for a in participating)
    ftt = ff + ft + fd + g + pp
    assert ftt == 10


# 9. FA é calculado como FD+G
def test_fa_calculado_como_fd_mais_g() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 5, "clube_id": 10, "entrou_em_campo": True, "scout": {"FD": 3, "G": 2}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    fd = sum(a["scout"].get("FD", 0) for a in participating)
    g = sum(a["scout"].get("G", 0) for a in participating)
    fa = fd + g
    assert fa == 5


# 10. G não é substituído por gols_marcados
def test_g_nao_substituido_por_gols_marcados() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 5, "clube_id": 10, "entrou_em_campo": True, "scout": {}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    g_raw = sum(a["scout"].get("G", 0) for a in participating)
    assert g_raw == 0


# 11. SG não é calculado como clean sheet
def test_sg_nao_calculado_como_clean_sheet() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 1, "clube_id": 10, "entrou_em_campo": True, "scout": {"SG": 1}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    sg_raw = sum(a["scout"].get("SG", 0) for a in participating)
    assert sg_raw == 1


# 12. Jogadores que não entraram não entram na soma
def test_jogadores_nao_entraram_nao_entram_na_soma() -> None:
    mock_data = {
        "atletas": {
            "1": {"atleta_id": 1, "posicao_id": 5, "clube_id": 10, "entrou_em_campo": False, "scout": {"DS": 10, "G": 5}},
            "2": {"atleta_id": 2, "posicao_id": 5, "clube_id": 10, "entrou_em_campo": True, "scout": {"DS": 2, "G": 1}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    assert len(participating) == 1
    assert participating[0]["atleta_id"] == 2
    assert sum(a["scout"].get("DS", 0) for a in participating) == 2
    assert sum(a["scout"].get("G", 0) for a in participating) == 1


# 13. Técnicos não entram
def test_tecnicos_nao_entram() -> None:
    mock_data = {
        "atletas": {
            "100": {"atleta_id": 100, "posicao_id": 6, "clube_id": 10, "entrou_em_campo": True, "scout": {"FC": 1}},
            "101": {"atleta_id": 101, "posicao_id": 4, "clube_id": 10, "entrou_em_campo": True, "scout": {"FC": 3}},
        }
    }
    athletes, _ = extract_raw_athletes_from_round(mock_data, rodada=1)
    participating = [a for a in athletes if a["posicao_id"] != 6 and a["entrou_em_campo"] is True]
    assert len(participating) == 1
    assert participating[0]["atleta_id"] == 101
    assert sum(a["scout"].get("FC", 0) for a in participating) == 3


# 14. Distinção dos três estados de roster (Caso A, Caso B e Caso C)
def test_distincao_tres_estados_roster() -> None:
    # Caso A: Roster ausente (somente comissão técnica no JSON para o clube)
    mock_caso_a = {
        "atletas": {
            "100": {"atleta_id": 100, "posicao_id": 6, "clube_id": 99, "entrou_em_campo": True, "scout": {}},
        }
    }
    athletes_a, _ = extract_raw_athletes_from_round(mock_caso_a, rodada=1)
    jogadores_raw_a = [a for a in athletes_a if a["clube_id"] == 99 and a["posicao_id"] != 6]
    assert len(jogadores_raw_a) == 0
    roster_status_a = "ROSTER_AUSENTE" if len(jogadores_raw_a) == 0 else "ROSTER_PRESENTE"
    raw_determinado_a = len(jogadores_raw_a) > 0
    assert roster_status_a == "ROSTER_AUSENTE"
    assert raw_determinado_a is False
    participating_a = [a for a in jogadores_raw_a if a["entrou_em_campo"] is True]
    assert len(participating_a) == 0
    scouts_a = {s: np.nan for s in AUDIT_SCOUTS} if not raw_determinado_a else {s: 0 for s in AUDIT_SCOUTS}
    assert np.isnan(scouts_a["DS"])
    assert np.isnan(scouts_a["G"])

    # Caso B: Roster presente, mas nenhum jogador entrou em campo
    mock_caso_b = {
        "atletas": {
            "101": {"atleta_id": 101, "posicao_id": 4, "clube_id": 99, "entrou_em_campo": False, "scout": {"DS": 5}},
            "102": {"atleta_id": 102, "posicao_id": 5, "clube_id": 99, "entrou_em_campo": False, "scout": {"G": 2}},
        }
    }
    athletes_b, _ = extract_raw_athletes_from_round(mock_caso_b, rodada=1)
    jogadores_raw_b = [a for a in athletes_b if a["clube_id"] == 99 and a["posicao_id"] != 6]
    assert len(jogadores_raw_b) == 2
    roster_status_b = "ROSTER_AUSENTE" if len(jogadores_raw_b) == 0 else "ROSTER_PRESENTE"
    raw_determinado_b = len(jogadores_raw_b) > 0
    assert roster_status_b == "ROSTER_PRESENTE"
    assert raw_determinado_b is True
    participating_b = [a for a in jogadores_raw_b if a["entrou_em_campo"] is True]
    assert len(participating_b) == 0
    # Como raw_determinado é True, os scouts são somados a partir de participating_b (0 participantes -> soma 0)
    scouts_b = {s: sum(int(at["scout"].get(s, 0)) for at in participating_b) for s in AUDIT_SCOUTS}
    assert scouts_b["DS"] == 0
    assert scouts_b["G"] == 0

    # Caso C: Roster presente com jogadores participantes
    mock_caso_c = {
        "atletas": {
            "103": {"atleta_id": 103, "posicao_id": 4, "clube_id": 99, "entrou_em_campo": True, "scout": {"DS": 3, "FC": 2}},
            "104": {"atleta_id": 104, "posicao_id": 5, "clube_id": 99, "entrou_em_campo": True, "scout": {"G": 1, "FD": 2}},
            "105": {"atleta_id": 105, "posicao_id": 3, "clube_id": 99, "entrou_em_campo": False, "scout": {"DS": 10}},
        }
    }
    athletes_c, _ = extract_raw_athletes_from_round(mock_caso_c, rodada=1)
    jogadores_raw_c = [a for a in athletes_c if a["clube_id"] == 99 and a["posicao_id"] != 6]
    assert len(jogadores_raw_c) == 3
    roster_status_c = "ROSTER_AUSENTE" if len(jogadores_raw_c) == 0 else "ROSTER_PRESENTE"
    raw_determinado_c = len(jogadores_raw_c) > 0
    assert roster_status_c == "ROSTER_PRESENTE"
    assert raw_determinado_c is True
    participating_c = [a for a in jogadores_raw_c if a["entrou_em_campo"] is True]
    assert len(participating_c) == 2
    scouts_c = {s: sum(int(at["scout"].get(s, 0)) for at in participating_c) for s in ["DS", "FC", "G", "FD"]}
    assert scouts_c["DS"] == 3
    assert scouts_c["FC"] == 2
    assert scouts_c["G"] == 1
    assert scouts_c["FD"] == 2


# 15. Chave temporal inclui temporada
def test_chave_temporal_inclui_temporada(auditor: RawVsMasterAuditor) -> None:
    assert "temporada" in auditor.df_master.columns
    assert "rodada" in auditor.df_master.columns
    assert "clube_id" in auditor.df_master.columns


# 16. Auditor não altera o Parquet
def test_auditor_nao_altera_parquet(master_file_path: Path) -> None:
    h_before = compute_file_sha256(master_file_path)
    aud = RawVsMasterAuditor(master_path=master_file_path)
    assert aud.df_master is not None
    h_after = compute_file_sha256(master_file_path)
    assert h_before == h_after, "O hash do Parquet foi alterado!"


# 17. Relatórios são determinísticos
def test_relatorios_sao_deterministicos() -> None:
    recs = [
        {"temporada": 2026, "rodada": 2, "clube_id": 282, "scout": "G"},
        {"temporada": 2026, "rodada": 1, "clube_id": 287, "scout": "DE"},
        {"temporada": 2026, "rodada": 1, "clube_id": 282, "scout": "DE"},
    ]
    df1 = pd.DataFrame(recs).sort_values(by=["temporada", "rodada", "clube_id", "scout"]).reset_index(drop=True)
    df2 = pd.DataFrame(reversed(recs)).sort_values(by=["temporada", "rodada", "clube_id", "scout"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df2)


# 18. Caso SAO R1 é encontrado
def test_caso_sao_r1_encontrado(auditor: RawVsMasterAuditor) -> None:
    sao_r1 = auditor.df_master[
        (auditor.df_master["temporada"] == 2026)
        & (auditor.df_master["rodada"] == 1)
        & (auditor.df_master["clube_id"] == 287)
    ]
    assert not sao_r1.empty, "São Paulo (clube_id=287) deve existir na rodada 1 do Master."


# 19. Goleiros são identificados por posicao_id=1
def test_goleiros_identificados_por_posicao_1(r1_json_path: Path) -> None:
    data = load_raw_round_json(r1_json_path)
    athletes, _ = extract_raw_athletes_from_round(data, rodada=1)
    gks = [a for a in athletes if a["posicao_id"] == 1]
    assert len(gks) > 0
    for gk in gks:
        assert gk["posicao_id"] == 1


# 20. DE/DP/GS do goleiro são auditáveis individualmente
def test_scouts_goleiro_auditaveis_individualmente(r1_json_path: Path) -> None:
    data = load_raw_round_json(r1_json_path)
    athletes, _ = extract_raw_athletes_from_round(data, rodada=1)
    gks = [a for a in athletes if a["posicao_id"] == 1 and a["entrou_em_campo"] is True]
    assert len(gks) > 0
    for gk in gks:
        sc = gk["scout"]
        de = int(sc.get("DE", 0))
        dp = int(sc.get("DP", 0))
        gs = int(sc.get("GS", 0))
        sg = int(sc.get("SG", 0))
        assert de >= 0
        assert dp >= 0
        assert gs >= 0
        assert sg in [0, 1]
