#!/usr/bin/env python3
"""
Modulo de elegibilidade pre-rodada via status_id.

Regra congelada:
    7  Provavel    -> ELEGIVEL
    2  Duvida      -> ELEGIVEL
    5  Contundido  -> NAO ELEGIVEL
    3  Suspenso    -> NAO ELEGIVEL
    6  Nulo        -> NAO ELEGIVEL

IMPORTANTE
- status_id e' snapshot de mercado PRE-rodada. Nao e' feature de regressao.
- Nao inferir semantica a partir de entrou_em_campo/pontos_num.
- Filtro aplica-se SOMENTE ao universo da rodada-alvo (nao ao historico global).
"""
from __future__ import annotations
from typing import Tuple
import pandas as pd


STATUS_MAP: dict[int, Tuple[str, bool]] = {
    7: ("Provavel",   True),
    2: ("Duvida",     True),
    5: ("Contundido", False),
    3: ("Suspenso",   False),
    6: ("Nulo",       False),
}

STATUS_ELEGIVEIS     = frozenset({7, 2})
STATUS_NAO_ELEGIVEIS = frozenset({5, 3, 6})
STATUS_CONHECIDOS    = frozenset(STATUS_MAP.keys())


def classificar_elegibilidade(status_id) -> Tuple[str, bool]:
    """
    Classifica um unico status_id.
    Retorna (nome, elegivel). Levanta ValueError para NaN/None/desconhecido.
    """
    if status_id is None:
        raise ValueError("status_id None")
    if isinstance(status_id, float) and pd.isna(status_id):
        raise ValueError("status_id NaN")
    try:
        sid = int(status_id)
    except (TypeError, ValueError):
        raise ValueError(f"status_id invalido: {status_id!r}")
    if sid not in STATUS_MAP:
        raise ValueError(f"status_id desconhecido: {sid}")
    return STATUS_MAP[sid]


def classificar_elegibilidade_series(s: pd.Series):
    """
    Versao vetorizada.
    Retorna (nomes: pd.Series[str], elegivel: pd.Series[bool]).
    Levanta se houver NaN/None/desconhecido.
    """
    nomes_map = {k: v[0] for k, v in STATUS_MAP.items()}
    eleg_map  = {k: v[1] for k, v in STATUS_MAP.items()}
    if s.isna().any():
        raise ValueError("status_id contem NaN")
    try:
        sid_int = s.astype("int64")
    except (TypeError, ValueError):
        raise ValueError("status_id nao e' inteiro")
    desconhecidos = set(sid_int.unique()) - STATUS_CONHECIDOS
    if desconhecidos:
        raise ValueError(f"status_id desconhecidos: {sorted(desconhecidos)}")
    return sid_int.map(nomes_map), sid_int.map(eleg_map)


def filtrar_universo_alvo(df_alvo: pd.DataFrame,
                          col_status: str = "status_id"
                          ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filtra o universo da rodada-alvo para jogadores elegiveis.
    Retorna (df_elegiveis, mask_elegivel).

    Nao toca nenhuma outra coluna. Nao usa informacao pos-rodada.
    """
    if col_status not in df_alvo.columns:
        raise ValueError(f"coluna {col_status} ausente")
    _, mask = classificar_elegibilidade_series(df_alvo[col_status])
    return df_alvo[mask.values].copy(), mask


def universo_elegivel_para_rodada(df: pd.DataFrame,
                                  temporada: int,
                                  rodada: int,
                                  col_status: str = "status_id"
                                  ) -> pd.DataFrame:
    """
    Conveniencia: retorna o universo elegivel para (temporada, rodada).
    NAO altera df. NAO constroi features. NAO acessa rodadas futuras.
    """
    mask = (df["temporada"] == temporada) & (df["rodada"] == rodada)
    return filtrar_universo_alvo(df[mask].copy(), col_status=col_status)[0]


def regra_resumida() -> dict:
    """Retorna a regra em formato serializavel para auditoria."""
    return {
        "elegiveis":     sorted(STATUS_ELEGIVEIS),
        "nao_elegiveis": sorted(STATUS_NAO_ELEGIVEIS),
        "mapa": {int(k): {"nome": v[0], "elegivel": bool(v[1])}
                 for k, v in STATUS_MAP.items()},
    }
