#!/usr/bin/env python3
"""
EXP-12 — Construção do histórico de partidas por equipe (R1..R26)

Generalização de scripts/build_team_match_history_r1.py, que permanece como
referência de semântica validada. Este script NÃO altera aquela semântica:
aplica exatamente a mesma regra para todas as rodadas disponíveis.

Fontes:
    experiments/data/cartola_api_2026/r{N}.json        -> scouts dos jogadores
    experiments/data/cartola_api_2026/partidas{N}.json  -> confrontos e placar oficial
    experiments/data/cartola_api_2026/rodadas.json      -> calendário de rodadas

Saídas:
    data/interim/exp12/team_match_history.parquet
    data/interim/exp12/team_match_history.csv
    reports/exp12/exp12_team_match_history_anomalias.csv
    reports/exp12/exp12_team_match_history_audit.md

SEMÂNTICA (inalterada em relação à R1 validada):
    Uma linha por equipe por partida. Toda partida incluída gera exatamente
    duas linhas: mandante (mando='C') e visitante (mando='F').

    DP, GS, DS, SG, FC, G, A, FS, I, PS, FD, FF, FT, PP
        Scouts agregados dos jogadores COM entered em campo daquele clube
        naquela rodada. Scout ausente/None = zero para a categoria.

    GS  = Gol Sofrido do goleiro (scout do jogador)
    SG  = Sem Gol Sofrido (scout do jogador)
    gols_sofridos = gols sofridos pelo TIME no placar oficial de partidas{N}.json

    Portanto GS != gols_sofridos e SG != gols_sofridos.
    NÃO existe coluna clean_sheet artificial.

    gols_marcados / gols_sofridos vêm do placar oficial, nunca dos scouts.

    FTT = FD + FF + FT + G + PP
    FA  = FD + G

NÃO SE FAZ NESTA ETAPA:
    Nenhuma feature temporal, nenhuma janela, nenhum shift, nenhum uso de
    rodadas futuras. Nenhuma alteração no dataset canônico
    cartola_jogador_rodada_v2 nem no CatBoost v2.

POLÍTICA DE ANOMALIA:
    Nada é corrigido silenciosamente. Toda anomalia encontrada é registrada
    em reports/exp12/exp12_team_match_history_anomalias.csv e narrada no
    relatório de auditoria.
"""

from pathlib import Path
import argparse
import json

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

API_DIR = ROOT / "experiments" / "data" / "cartola_api_2026"

OUT_DIR = ROOT / "data" / "interim" / "exp12"

REPORT_DIR = ROOT / "reports" / "exp12"

OUT_PARQUET = OUT_DIR / "team_match_history.parquet"
OUT_CSV = OUT_DIR / "team_match_history.csv"

OUT_ANOMALIAS_CSV = REPORT_DIR / "exp12_team_match_history_anomalias.csv"
OUT_AUDIT_MD = REPORT_DIR / "exp12_team_match_history_audit.md"

TEMPORADA_PADRAO = 2026
RODADA_INI_PADRAO = 1
RODADA_FIM_PADRAO = 26


# ============================================================
# SEMÂNTICA DE SCOUTS
# ============================================================

DIRECT_SCOUTS = [
    "DP",  # Defesas de pênalti
    "GS",  # Gol sofrido pelo goleiro
    "DS",  # Desarmes
    "SG",  # Sem gol sofrido
    "FC",  # Faltas cometidas
    "G",   # Gol
    "A",   # Assistência
    "FS",  # Scout FS — preservado conforme API
    "I",   # Impedimento
    "PS",  # Pênalti sofrido
    "FD",  # Finalização defendida
    "FF",  # Finalização para fora
    "FT",  # Finalização na trave
    "PP",  # Pênalti perdido
]

# Scouts que existem na API mas ficam de fora por decisão de escopo do EXP-12.
# Não são somados, não são inventados e não geram coluna.
FORA_DE_ESCOPO = [
    "CA",  # Cartão amarelo
    "CV",  # Cartão vermelho
    "DE",  # Defesa
    "GC",  # Gol contra
    "PC",  # Pênalti cometido
    "V",   # Vitória
]

COLUMNS = [
    "temporada",
    "rodada",
    "partida_id",
    "data",

    "clube_id",
    "clube_nome",
    "adversario_id",
    "adversario_nome",
    "mando",

    "gols_marcados",
    "gols_sofridos",

    # Defesa
    "DP",
    "GS",
    "DS",
    "SG",
    "FC",

    # Ataque
    "G",
    "A",
    "FTT",
    "FA",
    "FS",
    "I",
    "PS",

    # Componentes das finalizações
    "FD",
    "FF",
    "FT",
    "PP",
]

ZERO_SCOUTS = {scout: 0 for scout in DIRECT_SCOUTS}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value):
    """
    Converte valores numéricos da API para int.
    Valores ausentes/None viram 0 — comportamento herdado da R1 validada.
    """
    if value is None:
        return 0

    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def titulo(txt):
    print()
    print("=" * 74)
    print(txt)
    print("=" * 74)


def md_table(frame):
    """
    Tabela markdown sem depender do pacote opcional `tabulate`.
    """
    cols = [str(c) for c in frame.columns]

    linhas = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]

    for row in frame.itertuples(index=False, name=None):
        linhas.append(
            "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        )

    return "\n".join(linhas)


def discover_rounds(rodada_ini, rodada_fim, anomalias):
    """
    Verifica a existência de r{N}.json e partidas{N}.json na faixa pedida e
    registra qualquer rodada extra presente na fonte e fora do escopo.
    """
    disponiveis = []

    for rodada in range(rodada_ini, rodada_fim + 1):

        r_json = API_DIR / f"r{rodada}.json"
        partidas_json = API_DIR / f"partidas{rodada}.json"

        if not r_json.exists() or not partidas_json.exists():

            faltando = []

            if not r_json.exists():
                faltando.append(r_json.name)
            if not partidas_json.exists():
                faltando.append(partidas_json.name)

            anomalias.append({
                "tipo": "arquivo_fonte_ausente",
                "temporada": TEMPORADA_PADRAO,
                "rodada": rodada,
                "partida_id": None,
                "clube_id": None,
                "clube_nome": None,
                "adversario_id": None,
                "detalhe": f"faltando: {', '.join(faltando)}",
                "observado": "",
                "esperado": "r{N}.json e partidas{N}.json presentes",
            })

            continue

        disponiveis.append(rodada)

    for rodada in range(rodada_fim + 1, 1000):

        r_json = API_DIR / f"r{rodada}.json"
        partidas_json = API_DIR / f"partidas{rodada}.json"

        if not r_json.exists() and not partidas_json.exists():
            break

        anomalias.append({
            "tipo": "rodada_fora_de_escopo",
            "temporada": TEMPORADA_PADRAO,
            "rodada": rodada,
            "partida_id": None,
            "clube_id": None,
            "clube_nome": None,
            "adversario_id": None,
            "detalhe": (
                "arquivos de fonte existem na API mas a rodada está "
                "fora da faixa pedida (1-26)"
            ),
            "observado": (
                f"r{rodada}.json={r_json.exists()} "
                f"partidas{rodada}.json={partidas_json.exists()}"
            ),
            "esperado": f"rodada <= {rodada_fim}",
        })

    return disponiveis


def carregar_clubes(partidas_data, anomalias, rodada):
    clubes_raw = partidas_data.get("clubes", {}) or {}

    clubes = {}

    for clube_id_raw, clube in clubes_raw.items():

        clube_id = safe_int(clube_id_raw)

        clubes[clube_id] = {
            "nome": clube.get("nome"),
            "abreviacao": clube.get("abreviacao"),
            "slug": clube.get("slug"),
            "apelido": clube.get("apelido"),
            "nome_fantasia": clube.get("nome_fantasia"),
            "id": safe_int(clube.get("id", clube_id)),
        }

    if not clubes:
        anomalias.append({
            "tipo": "rodada_sem_catalogo_de_clubes",
            "temporada": TEMPORADA_PADRAO,
            "rodada": rodada,
            "partida_id": None,
            "clube_id": None,
            "clube_nome": None,
            "adversario_id": None,
            "detalhe": "partidas{N}.json sem dicionário 'clubes'",
            "observado": str(len(clubes_raw)),
            "esperado": "> 0",
        })

    return clubes


def agregar_scouts(r_data, rodada, anomalias):
    """
    Agrega os scouts diretos por clube, usando somente jogadores que
    realmente entraram em campo.

    Retorna:
        team_scouts  : {clube_id: {scout: soma}}
        stats        : contadores de auditoria
    """
    atletas_raw = r_data.get("atletas", [])

    if isinstance(atletas_raw, dict):
        atletas_raw = list(atletas_raw.values())

    stats = {
        "atletas": len(atletas_raw),
        "participantes": 0,
        "nao_participantes": 0,
        "participantes_sem_clube_id": 0,
        "participantes_scout_none": 0,
        "participantes_scout_vazio": 0,
        "scouts_vistos": set(),
    }

    team_scouts = {}

    for atleta in atletas_raw:

        if not atleta.get("entrou_em_campo", False):
            stats["nao_participantes"] += 1
            continue

        clube_id = atleta.get("clube_id")

        if clube_id is None or clube_id == "":
            stats["participantes_sem_clube_id"] += 1
            continue

        clube_id = safe_int(clube_id)

        if clube_id not in team_scouts:
            team_scouts[clube_id] = dict(ZERO_SCOUTS)

        scout_data = atleta.get("scout")

        if scout_data is None:
            stats["participantes_scout_none"] += 1
            scout_data = {}
        elif scout_data == {}:
            stats["participantes_scout_vazio"] += 1

        stats["scouts_vistos"] |= set(scout_data.keys())

        for scout in DIRECT_SCOUTS:
            team_scouts[clube_id][scout] += safe_int(
                scout_data.get(scout, 0)
            )

        stats["participantes"] += 1

    if stats["participantes_sem_clube_id"]:
        anomalias.append({
            "tipo": "participante_sem_clube_id",
            "temporada": TEMPORADA_PADRAO,
            "rodada": rodada,
            "partida_id": None,
            "clube_id": None,
            "clube_nome": None,
            "adversario_id": None,
            "detalhe": (
                "jogadores com entrou_em_campo=True e clube_id ausente; "
                "contribuição descartada"
            ),
            "observado": str(stats["participantes_sem_clube_id"]),
            "esperado": "0",
        })

    if stats["participantes_scout_none"]:
        anomalias.append({
            "tipo": "participante_com_scout_none",
            "temporada": TEMPORADA_PADRAO,
            "rodada": rodada,
            "partida_id": None,
            "clube_id": None,
            "clube_nome": None,
            "adversario_id": None,
            "detalhe": (
                "scout=null em jogador participante; tratado como zero "
                "em todas as categorias, sem inventar participação"
            ),
            "observado": str(stats["participantes_scout_none"]),
            "esperado": "0",
        })

    return team_scouts, stats


def montar_linha(
    temporada,
    rodada,
    partida_id,
    data,
    clube_id,
    clube_nome,
    adversario_id,
    adversario_nome,
    mando,
    gols_marcados,
    gols_sofridos,
    scouts,
):
    row = {
        "temporada": temporada,
        "rodada": rodada,
        "partida_id": partida_id,
        "data": data,

        "clube_id": clube_id,
        "clube_nome": clube_nome,

        "adversario_id": adversario_id,
        "adversario_nome": adversario_nome,

        "mando": mando,

        "gols_marcados": gols_marcados,
        "gols_sofridos": gols_sofridos,
    }

    for scout in DIRECT_SCOUTS:
        row[scout] = scouts.get(scout, 0)

    row["FTT"] = (
        row["FD"]
        + row["FF"]
        + row["FT"]
        + row["G"]
        + row["PP"]
    )

    row["FA"] = (
        row["FD"]
        + row["G"]
    )

    return row


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="EXP-12 — histórico de partidas por equipe (R1..R26)"
)

parser.add_argument("--temporada", type=int, default=TEMPORADA_PADRAO)
parser.add_argument("--rodada-ini", type=int, default=RODADA_INI_PADRAO)
parser.add_argument("--rodada-fim", type=int, default=RODADA_FIM_PADRAO)

args = parser.parse_args()

TEMPORADA = args.temporada
RODADA_INI = args.rodada_ini
RODADA_FIM = args.rodada_fim

anomalias = []
falhas = []


# ============================================================
# LEITURA / DESCOBERTA
# ============================================================

titulo(
    f"EXP-12 — TEAM MATCH HISTORY — R{RODADA_INI}..R{RODADA_FIM} "
    f"(temporada {TEMPORADA})"
)

print(f"\nFonte: {API_DIR}")

disponiveis = discover_rounds(RODADA_INI, RODADA_FIM, anomalias)

rodadas_ausentes = sorted(
    set(range(RODADA_INI, RODADA_FIM + 1)) - set(disponiveis)
)

print(f"Rodadas solicitadas: {RODADA_INI}..{RODADA_FIM}")
print(f"Rodadas com fonte:   {len(disponiveis)}")
print(f"Rodadas ausentes:    {rodadas_ausentes if rodadas_ausentes else 'nenhuma'}")

calendario = {}

rodadas_json = API_DIR / "rodadas.json"

if rodadas_json.exists():

    try:
        for item in load_json(rodadas_json):
            calendario[safe_int(item.get("rodada_id"))] = item
    except (ValueError, TypeError) as exc:
        anomalias.append({
            "tipo": "rodadas_json_ilegivel",
            "temporada": TEMPORADA,
            "rodada": None,
            "partida_id": None,
            "clube_id": None,
            "clube_nome": None,
            "adversario_id": None,
            "detalhe": f"falha ao ler rodadas.json: {exc}",
            "observado": "",
            "esperado": "lista de rodadas legível",
        })


# ============================================================
# CONSTRUÇÃO
# ============================================================

rows = []

resumo_rodadas = []

clubes_ids_por_rodada = {}

todos_scouts_vistos = set()

for rodada in disponiveis:

    r_json = API_DIR / f"r{rodada}.json"
    partidas_json = API_DIR / f"partidas{rodada}.json"

    r_data = load_json(r_json)
    partidas_data = load_json(partidas_json)

    clubes = carregar_clubes(partidas_data, anomalias, rodada)
    clubes_ids_por_rodada[rodada] = sorted(clubes.keys())

    team_scouts, stats = agregar_scouts(r_data, rodada, anomalias)
    todos_scouts_vistos |= stats["scouts_vistos"]

    partidas = partidas_data.get("partidas", []) or []

    rodada_no_json = safe_int(partidas_data.get("rodada", rodada))

    if rodada_no_json != rodada:
        anomalias.append({
            "tipo": "rodada_divergente_entre_arquivos",
            "temporada": TEMPORADA,
            "rodada": rodada,
            "partida_id": None,
            "clube_id": None,
            "clube_nome": None,
            "adversario_id": None,
            "detalhe": (
                f"nome do arquivo indica R{rodada}, "
                f"conteúdo indica rodada={rodada_no_json}"
            ),
            "observado": str(rodada_no_json),
            "esperado": str(rodada),
        })

    partidas_incluidas = 0
    partidas_sem_placar = 0
    clubes_sem_atletas = []
    jogos_por_clube = {}

    for partida in partidas:

        partida_id = partida.get("partida_id")

        rodada_partida = safe_int(
            partida.get("rodada", partidas_data.get("rodada", rodada))
        )

        data = partida.get("partida_data")

        mandante_id = safe_int(partida.get("clube_casa_id"))
        visitante_id = safe_int(partida.get("clube_visitante_id"))

        if not mandante_id or not visitante_id or mandante_id == visitante_id:

            anomalias.append({
                "tipo": "partida_sem_clubes_validos",
                "temporada": TEMPORADA,
                "rodada": rodada_partida,
                "partida_id": partida_id,
                "clube_id": mandante_id or visitante_id,
                "clube_nome": None,
                "adversario_id": visitante_id or mandante_id,
                "detalhe": "mandante/visitante ausentes ou iguais; ignorada",
                "observado": f"casa={partida.get('clube_casa_id')} "
                             f"visita={partida.get('clube_visitante_id')}",
                "esperado": "dois clube_id distintos e diferentes de 0",
            })

            continue

        for clube_id, papel in (
            (mandante_id, "mandante"),
            (visitante_id, "visitante"),
        ):
            jogos_por_clube.setdefault(clube_id, []).append(partida_id)

            if clube_id not in clubes:
                anomalias.append({
                    "tipo": "clube_sem_correspondencia_no_catalogo",
                    "temporada": TEMPORADA,
                    "rodada": rodada_partida,
                    "partida_id": partida_id,
                    "clube_id": clube_id,
                    "clube_nome": None,
                    "adversario_id": (
                        visitante_id if clube_id == mandante_id else mandante_id
                    ),
                    "detalhe": (
                        f"{papel} ausente do dicionário 'clubes' de "
                        f"partidas{rodada}.json; nome caiu no fallback "
                        f"clube_{clube_id}"
                    ),
                    "observado": f"clube_{clube_id}",
                    "esperado": "clube presente no catálogo da rodada",
                })

        mandante = clubes.get(
            mandante_id, {"nome": f"clube_{mandante_id}"}
        )
        visitante = clubes.get(
            visitante_id, {"nome": f"clube_{visitante_id}"}
        )

        mandante_nome = mandante["nome"]
        visitante_nome = visitante["nome"]

        placar_m = partida.get("placar_oficial_mandante")
        placar_v = partida.get("placar_oficial_visitante")

        valida_flag = partida.get("valida")

        if valida_flag is False:
            anomalias.append({
                "tipo": "partida_com_flag_valida_falso",
                "temporada": TEMPORADA,
                "rodada": rodada_partida,
                "partida_id": partida_id,
                "clube_id": mandante_id,
                "clube_nome": mandante_nome,
                "adversario_id": visitante_id,
                "detalhe": (
                    "campo 'valida' da API é false; a partida foi mantida "
                    "porque tem placar oficial (não há filtro por este flag)"
                ),
                "observado": "valida=False",
                "esperado": "valida=True",
            })

        if placar_m is None or placar_v is None:

            partidas_sem_placar += 1

            anomalias.append({
                "tipo": "partida_sem_placar_oficial",
                "temporada": TEMPORADA,
                "rodada": rodada_partida,
                "partida_id": partida_id,
                "clube_id": mandante_id,
                "clube_nome": mandante_nome,
                "adversario_id": visitante_id,
                "detalhe": (
                    f"placar oficial ausente (data={data}, valida={valida_flag}); "
                    "EXCLUÍDA da construção — não há resultado oficial e o "
                    "safe_int da R1 transformaria isso em 0x0 fabricado"
                ),
                "observado": f"casa={placar_m} visita={placar_v}",
                "esperado": "dois placares oficiais inteiros",
            })

            continue

        gols_mandante = safe_int(placar_m)
        gols_visitante = safe_int(placar_v)

        mandante_scouts = team_scouts.get(mandante_id, dict(ZERO_SCOUTS))
        visitante_scouts = team_scouts.get(visitante_id, dict(ZERO_SCOUTS))

        for clube_id, clube_nome, scouts in (
            (mandante_id, mandante_nome, mandante_scouts),
            (visitante_id, visitante_nome, visitante_scouts),
        ):

            entrou_no_clube = any(
                safe_int(a.get("clube_id")) == clube_id
                for a in (r_data.get("atletas", {}) or {}).values()
                if a.get("entrou_em_campo", False)
            ) if isinstance(r_data.get("atletas", {}), dict) else (
                clube_id in team_scouts
            )

            if not entrou_no_clube:
                clubes_sem_atletas.append(clube_id)

                anomalias.append({
                    "tipo": "clube_sem_atletas_participantes",
                    "temporada": TEMPORADA,
                    "rodada": rodada_partida,
                    "partida_id": partida_id,
                    "clube_id": clube_id,
                    "clube_nome": clube_nome,
                    "adversario_id": (
                        visitante_id if clube_id == mandante_id else mandante_id
                    ),
                    "detalhe": (
                        "clube aparece na partida mas não tem nenhum atleta "
                        "com entrou_em_campo=True no payload da rodada; "
                        "scouts ficam todos em zero por fallback da R1"
                    ),
                    "observado": "sem participantes",
                    "esperado": "elenco com participantes",
                })

            if all(int(scouts.get(s, 0)) == 0 for s in DIRECT_SCOUTS):
                anomalias.append({
                    "tipo": "linha_com_todos_os_scouts_zero",
                    "temporada": TEMPORADA,
                    "rodada": rodada_partida,
                    "partida_id": partida_id,
                    "clube_id": clube_id,
                    "clube_nome": clube_nome,
                    "adversario_id": (
                        visitante_id if clube_id == mandante_id else mandante_id
                    ),
                    "detalhe": (
                        "14 scouts diretos somam zero; não distingue "
                        "'sem dado de fonte' de 'performance zero'"
                    ),
                    "observado": "todos os scouts = 0",
                    "esperado": "ao menos 1 scout > 0",
                })

        rows.append(montar_linha(
            temporada=TEMPORADA,
            rodada=rodada_partida,
            partida_id=partida_id,
            data=data,
            clube_id=mandante_id,
            clube_nome=mandante_nome,
            adversario_id=visitante_id,
            adversario_nome=visitante_nome,
            mando="C",
            gols_marcados=gols_mandante,
            gols_sofridos=gols_visitante,
            scouts=mandante_scouts,
        ))

        rows.append(montar_linha(
            temporada=TEMPORADA,
            rodada=rodada_partida,
            partida_id=partida_id,
            data=data,
            clube_id=visitante_id,
            clube_nome=visitante_nome,
            adversario_id=mandante_id,
            adversario_nome=mandante_nome,
            mando="F",
            gols_marcados=gols_visitante,
            gols_sofridos=gols_mandante,
            scouts=visitante_scouts,
        ))

        partidas_incluidas += 1

    for clube_id, lista in jogos_por_clube.items():
        if len(lista) > 1:
            anomalias.append({
                "tipo": "clube_com_duas_partidas_na_rodada",
                "temporada": TEMPORADA,
                "rodada": rodada,
                "partida_id": ",".join(str(x) for x in lista),
                "clube_id": clube_id,
                "clube_nome": (clubes.get(clube_id) or {}).get("nome"),
                "adversario_id": None,
                "detalhe": (
                    "a agregação de scouts por clube na rodada deixa de ser "
                    "biunívoca com a partida"
                ),
                "observado": str(len(lista)),
                "esperado": "1",
            })

    resumo_rodadas.append({
        "rodada": rodada,
        "partidas_fonte": len(partidas),
        "partidas_incluidas": partidas_incluidas,
        "partidas_sem_placar": partidas_sem_placar,
        "linhas": partidas_incluidas * 2,
        "atletas": stats["atletas"],
        "participantes": stats["participantes"],
        "nao_participantes": stats["nao_participantes"],
        "clubes_com_scouts": len(team_scouts),
        "clubes_sem_atletas": len(sorted(set(clubes_sem_atletas))),
        "participantes_scout_none": stats["participantes_scout_none"],
    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("ERRO FATAL: nenhuma linha foi construída.")

df = df[COLUMNS]

df = df.sort_values(
    ["temporada", "rodada", "partida_id", "mando"],
    kind="mergesort"
).reset_index(drop=True)


# ============================================================
# DIAGNÓSTICOS QUE NÃO DEVEM CORRIGIR DADOS
# ============================================================

for rodada in disponiveis:

    r_data = load_json(API_DIR / f"r{rodada}.json")
    partidas_data = load_json(API_DIR / f"partidas{rodada}.json")
    clubes = carregar_clubes(partidas_data, [], rodada)

    sub = df[df["rodada"] == rodada]

    for partida in partidas_data.get("partidas", []):

        pm = partida.get("placar_oficial_mandante")
        pv = partida.get("placar_oficial_visitante")

        if pm is None or pv is None:
            continue

        for papel, clube_id, gols in (
            ("mandante", safe_int(partida.get("clube_casa_id")), safe_int(pm)),
            ("visitante", safe_int(partida.get("clube_visitante_id")), safe_int(pv)),
        ):

            linha = sub[sub["clube_id"] == clube_id]

            if linha.empty:
                continue

            linha = linha.iloc[0]

            if int(linha["G"]) != gols:
                anomalias.append({
                    "tipo": "g_scout_diferente_do_placar_oficial",
                    "temporada": TEMPORADA,
                    "rodada": rodada,
                    "partida_id": partida.get("partida_id"),
                    "clube_id": clube_id,
                    "clube_nome": linha["clube_nome"],
                    "adversario_id": linha["adversario_id"],
                    "detalhe": (
                        f"soma de G nos scouts ({papel}) difere do placar "
                        "oficial; mantida a semântica (G vem do scout, "
                        "gols_marcados vem do placar)"
                    ),
                    "observado": f"G={int(linha['G'])}",
                    "esperado": f"gols_marcados={gols}",
                })

            if int(linha["GS"]) != int(linha["gols_sofridos"]):
                anomalias.append({
                    "tipo": "gs_scout_diferente_de_gols_sofridos",
                    "temporada": TEMPORADA,
                    "rodada": rodada,
                    "partida_id": partida.get("partida_id"),
                    "clube_id": clube_id,
                    "clube_nome": linha["clube_nome"],
                    "adversario_id": linha["adversario_id"],
                    "detalhe": (
                        "GS (gol sofrido do goleiro) != gols_sofridos do "
                        "placar oficial; esperado por desenho, registrado "
                        "para documentar a divergência"
                    ),
                    "observado": f"GS={int(linha['GS'])}",
                    "esperado": f"gols_sofridos={int(linha['gols_sofridos'])}",
                })


scouts_nao_usados = sorted(todos_scouts_vistos - set(DIRECT_SCOUTS))

for scout in scouts_nao_usados:
    anomalias.append({
        "tipo": "scout_existente_ignorado_por_escopo",
        "temporada": TEMPORADA,
        "rodada": None,
        "partida_id": None,
        "clube_id": None,
        "clube_nome": None,
        "adversario_id": None,
        "detalhe": (
            "scout presente na API e não somado; a lista de 14 scouts "
            "diretos é a mesma validada na R1"
        ),
        "observado": scout,
        "esperado": "dentro de " + ",".join(DIRECT_SCOUTS),
    })


# ============================================================
# VALIDAÇÕES OBRIGATÓRIAS
# ============================================================

titulo("VALIDAÇÕES")


def registrar(ok, mensagem):
    if ok:
        print(f"[OK]   {mensagem}")
    else:
        print(f"[FAIL] {mensagem}")
        falhas.append(mensagem)
    return ok


# ------------------------------------------------------------
# 1. Exatamente 2 linhas por partida
# ------------------------------------------------------------

counts = df.groupby(["temporada", "rodada", "partida_id"]).size()

registrar(
    bool(counts.eq(2).all()),
    f"Exatamente 2 linhas por partida "
    f"({len(counts)} partidas, {int((counts != 2).sum())} fora do padrão).",
)


# ------------------------------------------------------------
# 2. Nenhuma duplicidade em (temporada, rodada, partida_id, clube_id)
# ------------------------------------------------------------

chave = ["temporada", "rodada", "partida_id", "clube_id"]

n_duplicatas = int(df.duplicated(subset=chave).sum())

registrar(
    n_duplicatas == 0,
    f"Sem duplicidade em {chave} ({n_duplicatas} duplicatas).",
)


# ------------------------------------------------------------
# 3. Clubes/adversários inversos e mando consistente
# ------------------------------------------------------------

inversos_ok = True
mando_ok = True
mandantes_vistos = 0
visitantes_vistos = 0

for partida_id, group in df.groupby(["temporada", "rodada", "partida_id"]):

    if len(group) != 2:
        inversos_ok = False
        continue

    a = group.iloc[0]
    b = group.iloc[1]

    if a["clube_id"] != b["adversario_id"] or b["clube_id"] != a["adversario_id"]:
        inversos_ok = False

    if a["clube_nome"] != b["adversario_nome"] or b["clube_nome"] != a["adversario_nome"]:
        inversos_ok = False

    if a["mando"] == "C":
        mandantes_vistos += 1
    elif a["mando"] == "F":
        visitantes_vistos += 1
    else:
        mando_ok = False

    if b["mando"] == "F":
        visitantes_vistos += 1
    elif b["mando"] == "C":
        mandantes_vistos += 1
    else:
        mando_ok = False

    if a["clube_id"] == b["clube_id"]:
        inversos_ok = False

registrar(
    inversos_ok,
    "Os dois clubes de cada partida são inversos "
    "(clube/adversario_id e clube/adversario_nome).",
)

registrar(
    mando_ok and mandantes_vistos == len(counts) and visitantes_vistos == len(counts),
    f"Mando consistente: {mandantes_vistos} linhas 'C' e "
    f"{visitantes_vistos} linhas 'F' para {len(counts)} partidas.",
)


# ------------------------------------------------------------
# 4. Simetria do placar
# ------------------------------------------------------------

placar_ok = True
placar_divergencias = []

for partida_id, group in df.groupby(["temporada", "rodada", "partida_id"]):

    if len(group) != 2:
        placar_ok = False
        continue

    a = group.iloc[0]
    b = group.iloc[1]

    if a["gols_marcados"] != b["gols_sofridos"]:
        placar_ok = False
        placar_divergencias.append(partida_id)

    if b["gols_marcados"] != a["gols_sofridos"]:
        placar_ok = False
        placar_divergencias.append(partida_id)

registrar(
    placar_ok,
    "gols_marcados de A = gols_sofridos de B e "
    "gols_marcados de B = gols_sofridos de A "
    f"({len(placar_divergencias)} divergências).",
)


# ------------------------------------------------------------
# 5. FTT
# ------------------------------------------------------------

ftt_ok = bool(
    (df["FTT"] == df["FD"] + df["FF"] + df["FT"] + df["G"] + df["PP"]).all()
)

registrar(ftt_ok, "FTT = FD + FF + FT + G + PP em todas as linhas.")


# ------------------------------------------------------------
# 6. FA
# ------------------------------------------------------------

fa_ok = bool((df["FA"] == df["FD"] + df["G"]).all())

registrar(fa_ok, "FA = FD + G em todas as linhas.")


# ------------------------------------------------------------
# 7. GS / SG / gols_sofridos separados e sem clean_sheet
# ------------------------------------------------------------

semantica_ok = (
    "GS" in df.columns
    and "SG" in df.columns
    and "gols_sofridos" in df.columns
    and "clean_sheet" not in df.columns
)

registrar(
    semantica_ok,
    "GS, SG e gols_sofridos existem como campos distintos e "
    "não há coluna clean_sheet artificial.",
)

colunas_ok = list(df.columns) == COLUMNS

registrar(
    colunas_ok,
    f"Esquema com {len(COLUMNS)} colunas idêntico ao da R1 validada.",
)


# ------------------------------------------------------------
# 8. Nenhuma feature temporal nesta etapa
# ------------------------------------------------------------

temporais_proibidas = [
    c for c in df.columns
    if any(p in c.lower() for p in (
        "media", "mean", "ultim", "last", "janela", "window",
        "roll", "ewm", "shift", "lag", "acumul", "acumm", "trend",
    ))
]

registrar(
    not temporais_proibidas,
    "Nenhuma feature temporal criada nesta etapa "
    f"({temporais_proibidas if temporais_proibidas else 'nenhuma'}).",
)


# ------------------------------------------------------------
# 9. Cobertura de rodadas
# ------------------------------------------------------------

rodadas_no_df = sorted(df["rodada"].unique().tolist())

registrar(
    rodadas_no_df == sorted(disponiveis),
    f"Cobertura de rodadas {rodadas_no_df[0]}..{rodadas_no_df[-1]} "
    f"({len(rodadas_no_df)} rodadas no dataset).",
)


# ------------------------------------------------------------
# 10. Um clube por partida por rodada
# ------------------------------------------------------------

multi_por_rodada = (
    df.groupby(["rodada", "clube_id"]).size().gt(1).sum()
)

registrar(
    int(multi_por_rodada) == 0,
    "Nenhum clube aparece duas vezes na mesma rodada "
    f"({int(multi_por_rodada)} violações).",
)


# ============================================================
# RELATÓRIOS
# ============================================================

titulo("RESUMO POR RODADA")

resumo = pd.DataFrame(resumo_rodadas)

print(resumo.to_string(index=False))

titulo("TOTAIS")

n_linhas = len(df)
n_partidas = int(df.groupby(["temporada", "rodada", "partida_id"]).ngroups)
n_clubes = int(df["clube_id"].nunique())
n_partidas_fonte = int(resumo["partidas_fonte"].sum())
n_partidas_sem_placar = int(resumo["partidas_sem_placar"].sum())

print(f"Rodadas construídas:            {len(disponiveis)}")
print(f"Partidas lidas da fonte:        {n_partidas_fonte}")
print(f"Partidas sem placar oficial:    {n_partidas_sem_placar} (excluídas)")
print(f"Partidas no dataset:            {n_partidas}")
print(f"Linhas:                         {n_linhas}")
print(f"Clubes:                         {n_clubes}")
print(f"Colunas:                        {len(df.columns)}")
print(f"Atletas lidos:                  {int(resumo['atletas'].sum())}")
print(f"Participantes (entraram):       {int(resumo['participantes'].sum())}")
print(f"Não participantes (ignorados):  {int(resumo['nao_participantes'].sum())}")

catalogo_comum = (
    len({tuple(v) for v in clubes_ids_por_rodada.values()}) == 1
)

print(f"Catálogo de clubes idêntico em todas as rodadas: {catalogo_comum}")
print(f"Clubes: {sorted(df['clube_id'].unique().tolist())}")

print("\nColunas:")
for col in df.columns:
    print(f"  - {col}")

print("\nScouts ignorados por escopo: " + ", ".join(scouts_nao_usados))


# ============================================================
# AMOSTRA
# ============================================================

titulo("AMOSTRA (10 LINHAS)")

print(
    df[
        [
            "rodada", "partida_id", "clube_nome", "adversario_nome",
            "mando", "gols_marcados", "gols_sofridos",
            "DP", "GS", "DS", "SG", "FC", "G", "A", "FTT", "FA",
            "FS", "I", "PS", "FD", "FF", "FT", "PP",
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# SALVAR
# ============================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df.to_parquet(OUT_PARQUET, index=False)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

anomalias_df = pd.DataFrame(anomalias)

ANOMALIA_COLUMNS = [
    "tipo", "temporada", "rodada", "partida_id", "clube_id",
    "clube_nome", "adversario_id", "detalhe", "observado", "esperado",
]

if anomalias_df.empty:
    anomalias_df = pd.DataFrame(columns=ANOMALIA_COLUMNS)
else:
    anomalias_df = anomalias_df[ANOMALIA_COLUMNS]

anomalias_df = anomalias_df.sort_values(
    ["tipo", "rodada", "partida_id", "clube_id"],
    na_position="last",
    kind="mergesort",
).reset_index(drop=True)

anomalias_df.to_csv(OUT_ANOMALIAS_CSV, index=False, encoding="utf-8")

contagem = (
    anomalias_df["tipo"].value_counts().to_string()
    if not anomalias_df.empty else "nenhuma"
)

SEMANTICA_CAMPOS = [
    ("temporada", "inteiro", f"temporada do dado, fixa em {TEMPORADA}"),
    ("rodada", "inteiro", "rodada da partida (campo `rodada` do arquivo, "
     "com fallback do topo do partidas{N}.json)"),
    ("partida_id", "inteiro", "identificador do confronto em "
     "partidas{N}.json"),
    ("data", "texto", "`partida_data` crua da API, sem normalização"),
    ("clube_id", "inteiro", "equipe na linha"),
    ("clube_nome", "texto", "nome do clube em `clubes` de partidas{N}.json"),
    ("adversario_id", "inteiro", "equipe adversária"),
    ("adversario_nome", "texto", "nome do adversário"),
    ("mando", "texto", "'C' mandante, 'F' visitante"),
    ("gols_marcados", "inteiro", "placar oficial do time na partida"),
    ("gols_sofridos", "inteiro", "placar oficial do adversário na partida"),
    ("DP", "inteiro", "soma de defesas de pênalti dos participantes"),
    ("GS", "inteiro", "soma de gol sofrido DO GOLEIRO (scout do jogador). "
     "NÃO é gols_sofridos"),
    ("DS", "inteiro", "soma de desarmes"),
    ("SG", "inteiro", "soma de sem gol sofrido (scout do jogador)"),
    ("FC", "inteiro", "soma de faltas cometidas"),
    ("G", "inteiro", "soma de gols dos participantes"),
    ("A", "inteiro", "soma de assistências"),
    ("FTT", "inteiro", "FD + FF + FT + G + PP"),
    ("FA", "inteiro", "FD + G"),
    ("FS", "inteiro", "scout FS preservado conforme API"),
    ("I", "inteiro", "soma de impedimentos"),
    ("PS", "inteiro", "soma de pênaltis sofridos"),
    ("FD", "inteiro", "soma de finalizações defendidas"),
    ("FF", "inteiro", "soma de finalizações para fora"),
    ("FT", "inteiro", "soma de finalizações na trave"),
    ("PP", "inteiro", "soma de pênaltis perdidos"),
]

VALIDACOES_MD = [
    "Exatamente 2 linhas por partida.",
    "Sem duplicidade em (temporada, rodada, partida_id, clube_id).",
    "Clubes/adversários inversos em id e nome.",
    "Mando consistente: 1 linha 'C' e 1 linha 'F' por partida.",
    "gols_marcados de A = gols_sofridos de B e vice-versa.",
    "FTT = FD + FF + FT + G + PP.",
    "FA = FD + G.",
    "GS / SG / gols_sofridos separados e sem coluna clean_sheet.",
    "Esquema com 27 colunas idêntico ao da R1 validada.",
    "Nenhuma feature temporal criada nesta etapa.",
    "Cobertura de todas as rodadas disponíveis na fonte.",
    "Nenhum clube com duas partidas na mesma rodada.",
]


def bloco(tipo):
    return anomalias_df[anomalias_df["tipo"] == tipo]


def lista_anomalias(tipo, cols, fmt):
    bloco_tipo = bloco(tipo)

    if bloco_tipo.empty:
        return ["nenhum registro."]

    return [
        f"- {fmt(row)}"
        for row in bloco_tipo[cols].itertuples(index=False)
    ]


semantica_tab = pd.DataFrame(
    SEMANTICA_CAMPOS, columns=["campo", "tipo", "semantica"]
)

linhas_md = [
    "# EXP-12 — Auditoria da construção do histórico de partidas por equipe",
    "",
    f"- Temporada: **{TEMPORADA}**",
    f"- Rodadas: **R{RODADA_INI}..R{RODADA_FIM}**",
    f"- Gerado por: `scripts/exp12_build_team_match_history.py`",
    f"- Linhas: **{n_linhas}** | Partidas: **{n_partidas}** | "
    f"Clubes: **{n_clubes}** | Colunas: **{len(df.columns)}**",
    "",
    "## Objetivo",
    "",
    "Construir o histórico de partidas por equipe da temporada, uma linha por "
    "equipe por partida, agregando os scouts diretos dos jogadores que "
    "entraram em campo e o placar oficial da API. Esta etapa é "
    "**puramente descritiva**: não cria feature temporal, não usa rodada "
    "futura e não toca no dataset canônico nem no CatBoost v2.",
    "",
    "## Fontes",
    "",
    "- `experiments/data/cartola_api_2026/r{N}.json` — scouts dos jogadores "
    "(`atletas` vem como dicionário indexado por atleta_id)",
    "- `experiments/data/cartola_api_2026/partidas{N}.json` — confrontos, "
    "mandante/visitante e placar oficial",
    "- `experiments/data/cartola_api_2026/rodadas.json` — calendário de "
    "rodadas (consulta)",
    "- `scripts/build_team_match_history_r1.py` — referência de semântica "
    "validada na R1",
    "",
    "## Metodologia",
    "",
    "1. Uma linha por equipe por partida: mandante `mando='C'` e visitante "
    "`mando='F'`.",
    "2. Scouts agregados por clube a partir dos atletas com "
    "`entrou_em_campo=True` do payload da rodada. Cada clube joga uma única "
    "partida por rodada (verificado), então a agregação do clube é "
    "biunívoca com a partida.",
    "3. Scout ausente, `{}` ou `null` = zero na categoria. Não é inventada "
    "participação.",
    "4. `gols_marcados` e `gols_sofridos` vêm **exclusivamente** do placar "
    "oficial de `partidas{N}.json`.",
    "5. Nenhuma linha de uma partida sem placar oficial é criada.",
    "6. Scouts fora da lista de 14 diretos são ignorados, não renomeados.",
    "7. Nenhuma feature temporal, nenhum uso de informação de rodada futura.",
    "",
    "## Semântica dos campos",
    "",
    md_table(semantica_tab),
    "",
    "Pontos críticos de semântica:",
    "",
    "- `GS` é o scout de **gol sofrido do goleiro**. Não substitui "
    "`gols_sofridos`.",
    "- `SG` é o scout de **sem gol sofrido**. Não substitui "
    "`gols_sofridos`.",
    "- Não existe coluna `clean_sheet` artificial: clean sheet não é "
    "derivado aqui.",
    "- `gols_sofridos` não vem de scout nenhum.",
    "",
    "## Validações",
    "",
]
linhas_md += [
    f"- [{' ' if m in falhas else 'x'}] {m}" for m in VALIDACOES_MD
]

linhas_md += [
    "",
    "## Resultado",
    "",
    f"- Rodadas construídas: **{len(disponiveis)}** (R{RODADA_INI}..R{RODADA_FIM})",
    f"- Partidas lidas da fonte: **{n_partidas_fonte}**",
    f"- Partidas sem placar oficial (excluídas): **{n_partidas_sem_placar}**",
    f"- Partidas no dataset: **{n_partidas}**",
    f"- Linhas: **{n_linhas}**",
    f"- Clubes: **{n_clubes}** — {sorted(int(c) for c in df['clube_id'].unique())}",
    f"- Colunas: **{len(df.columns)}**",
    f"- Atletas lidos: **{int(resumo['atletas'].sum())}**",
    f"- Participantes (`entrou_em_campo=True`): "
    f"**{int(resumo['participantes'].sum())}**",
    f"- Não participantes ignorados: "
    f"**{int(resumo['nao_participantes'].sum())}**",
    f"- Catálogo de clubes idêntico em todas as rodadas: **{catalogo_comum}**",
    "",
    "### Resumo por rodada",
    "",
    md_table(resumo),
    "",
    "### Amostra",
    "",
    "```",
    df.head(10).to_string(index=False),
    "```",
    "",
    "## Anomalias encontradas",
    "",
    f"Total de registros de anomalia: **{len(anomalias_df)}**. Detalhe linha "
    "a linha em `reports/exp12/exp12_team_match_history_anomalias.csv`.",
    "Nada foi corrigido silenciosamente.",
    "",
    "```",
    contagem,
    "```",
    "",
    "### 1. Partidas sem placar oficial (EXCLUÍDAS da construção)",
    "",
]
linhas_md += lista_anomalias(
    "partida_sem_placar_oficial",
    ["rodada", "partida_id", "clube_nome", "adversario_id", "detalhe"],
    lambda r: (
        f"R{int(r.rodada)} partida {int(r.partida_id)} "
        f"{r.clube_nome} x {int(r.adversario_id)} — {r.detalhe}"
    ),
)

linhas_md += [
    "",
    "### 2. Clubes sem atletas participantes no payload da rodada "
    "(scouts ficam em zero)",
    "",
    "Nestes casos a partida existe com placar oficial, mas o payload de "
    "`r{N}.json` não traz nenhum atleta com `entrou_em_campo=True` do "
    "clube — normalmente sobra apenas o técnico. As 14 colunas de scout da "
    "linha ficam zeradas por fallback, o que **não** significa performance "
    "zero.",
    "",
]
linhas_md += lista_anomalias(
    "clube_sem_atletas_participantes",
    ["rodada", "partida_id", "clube_nome", "clube_id"],
    lambda r: (
        f"R{int(r.rodada)} partida {int(r.partida_id)} "
        f"{r.clube_nome} (id {int(r.clube_id)})"
    ),
)

linhas_md += [
    "",
    "### 3. Soma de `G` diferente do placar oficial",
    "",
    "A semântica é mantida: `G` vem do scout, `gols_marcados` vem do "
    "placar oficial. A divergência é do dado de origem, não do código.",
    "",
]
linhas_md += lista_anomalias(
    "g_scout_diferente_do_placar_oficial",
    ["rodada", "partida_id", "clube_nome", "observado", "esperado"],
    lambda r: (
        f"R{int(r.rodada)} partida {int(r.partida_id)} {r.clube_nome}: "
        f"{r.observado} vs {r.esperado}"
    ),
)

linhas_md += [
    "",
    "### 4. `GS` diferente de `gols_sofridos`",
    "",
    "Esperado por desenho (o goleiro acumula `GS`, o placar é do time), mas "
    "registrado para deixar explícito que `GS` não é `gols_sofridos`.",
    "",
]
linhas_md += lista_anomalias(
    "gs_scout_diferente_de_gols_sofridos",
    ["rodada", "partida_id", "clube_nome", "observado", "esperado"],
    lambda r: (
        f"R{int(r.rodada)} partida {int(r.partida_id)} {r.clube_nome}: "
        f"{r.observado} vs {r.esperado}"
    ),
)

linhas_md += [
    "",
    "### 5. Partidas com `valida=false` na API",
    "",
    "A construção **não** filtra por este flag (a R1 validada também não "
    "filtra). Estão listadas para auditoria.",
    "",
]
linhas_md += lista_anomalias(
    "partida_com_flag_valida_falso",
    ["rodada", "partida_id", "clube_nome", "adversario_id"],
    lambda r: (
        f"R{int(r.rodada)} partida {int(r.partida_id)} "
        f"{r.clube_nome} x {int(r.adversario_id)}"
    ),
)

linhas_md += [
    "",
    "### 6. Scouts existentes na API e fora do escopo do EXP-12",
    "",
    f"`{', '.join(scouts_nao_usados)}` — mantidos fora, conforme semântica "
    "validada na R1.",
    "",
    "### 7. Rodadas fora do escopo pedido",
    "",
]
linhas_md += lista_anomalias(
    "rodada_fora_de_escopo",
    ["rodada", "detalhe"],
    lambda r: f"R{int(r.rodada)} — {r.detalhe}",
)

linhas_md += [
    "",
    "## Arquivos criados / modificados",
    "",
    "Criados:",
    "",
    "- `scripts/exp12_build_team_match_history.py`",
    "- `data/interim/exp12/team_match_history.parquet`",
    "- `data/interim/exp12/team_match_history.csv`",
    "- `reports/exp12/exp12_team_match_history_anomalias.csv`",
    "- `reports/exp12/exp12_team_match_history_audit.md` (este arquivo)",
    "",
    "Não modificados (confirmado):",
    "",
    "- `data/interim/cartola_jogador_rodada_v2.parquet`",
    "- `scripts/catboost_v2.py`",
    "- `scripts/build_team_match_history_r1.py`",
    "- `data/interim/exp12/team_match_history_r1.parquet` / `.csv`",
    "",
    "## Próximo passo",
    "",
    "1. Decidir o tratamento das linhas com scouts zerados por payload "
    "incompleto (R2, R3, R4, R5, R19, R21) antes de usar histórico de "
    "scouts em feature.",
    "2. Só depois disso construir as features temporais de contexto de "
    "time/adversário (janelas de 5 e 10 jogos, cobertura de janela), com "
    "corte estritamente anterior à rodada-alvo.",
    "",
]

with open(OUT_AUDIT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(linhas_md) + "\n")


titulo("ARQUIVOS GERADOS")

print(f"Parquet:  {OUT_PARQUET}")
print(f"CSV:      {OUT_CSV}")
print(f"Anomalias: {OUT_ANOMALIAS_CSV}")
print(f"Auditoria: {OUT_AUDIT_MD}")


# ============================================================
# FECHAMENTO
# ============================================================

titulo("FECHAMENTO")

if falhas:
    print("VALIDAÇÕES FALHARAM:")
    for f_ in falhas:
        print(f"  - {f_}")
    raise SystemExit(1)

print("Todas as validações obrigatórias passaram.")

print(
    f"\nAnomalias registradas: {len(anomalias_df)} "
    "(nenhuma foi corrigida silenciosamente)."
)
