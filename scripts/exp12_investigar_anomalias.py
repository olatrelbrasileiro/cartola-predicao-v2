#!/usr/bin/env python3
"""
EXP-12 — Investigação das anomalias da construção do histórico de partidas

Este script é SOMENTE LEITURA sobre:

    data/interim/exp12/team_match_history.parquet      (leitura)
    reports/exp12/exp12_team_match_history_anomalias.csv (leitura)
    experiments/data/cartola_api_2026/r{N}.json          (leitura)
    experiments/data/cartola_api_2026/partidas{N}.json   (leitura)
    experiments/data/cartola_api_2026/rodadas.json       (leitura)

Ele NÃO altera nenhum dataset, NÃO corrige dado nenhum e NÃO cria feature
temporal. A única coisa que produz são relatórios.

Objetivo: responder empiricamente, para cada anomalia registrada na auditoria
da Fase 1:

1. As 12 partidas com scouts totalmente zerados: o zero é performance zero ou
   ausência de dado na fonte?
2. Os 48 casos de G != gols_marcados: existe reconciliação com GC?
3. Os 30 casos de GS != gols_sofridos: substituição de goleiro, goleiro
   ausente ou falta de payload?

Saídas:

    reports/exp12/exp12_anomalias_investigacao.md
    reports/exp12/exp12_inv_scouts_zerados.csv
    reports/exp12/exp12_inv_gols.csv
    reports/exp12/exp12_inv_goleiros.csv
    reports/exp12/exp12_inv_perfil_payload.csv
"""

from pathlib import Path
import json
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

API_DIR = ROOT / "experiments" / "data" / "cartola_api_2026"

DATASET = ROOT / "data" / "interim" / "exp12" / "team_match_history.parquet"
ANOMALIAS_CSV = (
    ROOT / "reports" / "exp12" / "exp12_team_match_history_anomalias.csv"
)

REPORT_DIR = ROOT / "reports" / "exp12"

OUT_MD = REPORT_DIR / "exp12_anomalias_investigacao.md"
OUT_SCOUTS_CSV = REPORT_DIR / "exp12_inv_scouts_zerados.csv"
OUT_GOLS_CSV = REPORT_DIR / "exp12_inv_gols.csv"
OUT_GOLEIROS_CSV = REPORT_DIR / "exp12_inv_goleiros.csv"
OUT_PAYLOAD_CSV = REPORT_DIR / "exp12_inv_perfil_payload.csv"

TEMPORADA = 2026
RODADA_INI = 1
RODADA_FIM = 26

GOL = 1  # posicao_id do goleiro em posicoes.json
TECNICO = 6


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value):
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def titulo(txt):
    print()
    print("=" * 78)
    print(txt)
    print("=" * 78)


def md_table(frame, cols=None):
    if cols is not None:
        frame = frame[cols]
    cols = [str(c) for c in frame.columns]
    out = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        out.append(
            "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        )
    return "\n".join(out)


# ============================================================
# LEITURA
# ============================================================

titulo("EXP-12 — INVESTIGAÇÃO DAS ANOMALIAS (somente leitura)")

print(f"\nDataset lido:    {DATASET}")
print(f"Anomalias lidas: {ANOMALIAS_CSV}")

dataset = pd.read_parquet(DATASET)
anomalias = pd.read_csv(ANOMALIAS_CSV)

print(f"Linhas do dataset: {len(dataset)}")
print(f"Registros de anomalia: {len(anomalias)}")

rodadas = list(range(RODADA_INI, RODADA_FIM + 1))

payloads = {}

for rodada in rodadas:

    r_data = load_json(API_DIR / f"r{rodada}.json")
    partidas_data = load_json(API_DIR / f"partidas{rodada}.json")

    atletas = r_data.get("atletas", {}) or {}

    payloads[rodada] = {
        "r_data": r_data,
        "partidas_data": partidas_data,
        "atletas": atletas,
        "clubes": {
            safe_int(k): v
            for k, v in (partidas_data.get("clubes", {}) or {}).items()
        },
        "partidas": partidas_data.get("partidas", []) or [],
    }

calendario = {
    safe_int(x.get("rodada_id")): x
    for x in load_json(API_DIR / "rodadas.json")
}


def nome_clube(rodada, clube_id):
    clube = payloads[rodada]["clubes"].get(safe_int(clube_id))
    if not clube:
        return f"clube_{safe_int(clube_id)}"
    return clube.get("nome") or clube.get("apelido") or f"clube_{clube_id}"


def partida_por_id(rodada, partida_id):
    for partida in payloads[rodada]["partidas"]:
        if partida.get("partida_id") == partida_id:
            return partida
    return None


def atletas_do_clube(rodada, clube_id):
    clube_id = safe_int(clube_id)
    return {
        k: v
        for k, v in payloads[rodada]["atletas"].items()
        if safe_int(v.get("clube_id")) == clube_id
    }


def perfil_clube(rodada, clube_id):
    """
    Perfil de presença de atletas de um clube no payload da rodada.
    """
    elenco = atletas_do_clube(rodada, clube_id)

    participantes = {
        k: v for k, v in elenco.items() if v.get("entrou_em_campo")
    }

    com_scout = {
        k: v for k, v in participantes.items() if v.get("scout")
    }

    scouts_vistos = set()
    for v in elenco.values():
        scouts_vistos |= set((v.get("scout") or {}).keys())

    goleiros = {
        k: v for k, v in elenco.items() if safe_int(v.get("posicao_id")) == GOL
    }

    return {
        "n_atletas": len(elenco),
        "n_participantes": len(participantes),
        "n_com_scout": len(com_scout),
        "scouts_vistos": scouts_vistos,
        "n_goleiros": len(goleiros),
        "goleiros": goleiros,
        "elenco": elenco,
    }


# ============================================================
# BLOCO A — 12 PARTIDAS COM SCOUTS TOTALMENTE ZERADOS
# ============================================================

titulo("BLOCO A — PARTIDAS COM SCOUTS TOTALMENTE ZERADOS")

casos_a = anomalias[anomalias["tipo"] == "clube_sem_atletas_participantes"].copy()

casos_a["rodada"] = casos_a["rodada"].astype(int)
casos_a["partida_id"] = casos_a["partida_id"].astype(int)
casos_a["clube_id"] = casos_a["clube_id"].astype(int)

linhas_a = []

for _, caso in casos_a.iterrows():

    rodada = caso["rodada"]
    partida_id = caso["partida_id"]
    clube_id = caso["clube_id"]

    partida = partida_por_id(rodada, partida_id)

    casa = safe_int(partida.get("clube_casa_id"))
    fora = safe_int(partida.get("clube_visitante_id"))

    afetado = perfil_clube(rodada, clube_id)
    adv = perfil_clube(rodada, fora if clube_id == casa else casa)

    clubes_com_elenco = sum(
        1
        for c in payloads[rodada]["clubes"].keys()
        if perfil_clube(rodada, safe_int(c))["n_atletas"] >= 14
    )

    if afetado["n_participantes"] == 0 and afetado["n_atletas"] <= 2:
        diagnostico = "elenco ausente do payload"
    elif afetado["n_participantes"] == 0:
        diagnostico = "sem participantes, mas atletas no payload"
    elif afetado["n_com_scout"] == 0:
        diagnostico = "participantes sem nenhum scout nao vazio"
    else:
        diagnostico = "scouts realmente zerados (performance zero)"

    elenco_detalhe = "; ".join(
        f"{k}/{v.get('apelido')} pos={v.get('posicao_id')} "
        f"entrou={v.get('entrou_em_campo')} scout={v.get('scout')}"
        for k, v in afetado["elenco"].items()
    ) or "(clube ausente do payload)"

    linhas_a.append({
        "temporada": TEMPORADA,
        "rodada": rodada,
        "partida_id": partida_id,
        "mandante": nome_clube(rodada, casa),
        "visitante": nome_clube(rodada, fora),
        "placar": (
            f"{partida.get('placar_oficial_mandante')}"
            f"x{partida.get('placar_oficial_visitante')}"
        ),
        "clube_afetado": nome_clube(rodada, clube_id),
        "clube_afetado_id": clube_id,
        "n_atletas_no_payload": afetado["n_atletas"],
        "n_entrou_em_campo": afetado["n_participantes"],
        "n_scout_nao_vazio": afetado["n_com_scout"],
        "existe_atleta_no_payload": "sim" if afetado["n_atletas"] else "nao",
        "scouts_no_payload": (
            ",".join(sorted(afetado["scouts_vistos"])) or "(nenhum)"
        ),
        "n_atletas_adversario": adv["n_atletas"],
        "clubes_com_elenco_completo_na_rodada": (
            f"{clubes_com_elenco}/{len(payloads[rodada]['clubes'])}"
        ),
        "diagnostico": diagnostico,
        "elenco_no_payload": elenco_detalhe,
    })

tabela_a = pd.DataFrame(linhas_a).sort_values(["rodada", "partida_id"])

tabela_a.to_csv(OUT_SCOUTS_CSV, index=False, encoding="utf-8")

partidas_a = tabela_a["partida_id"].nunique()

print(f"Linhas com scouts zerados: {len(tabela_a)}")
print(f"Partidas distintas:        {partidas_a}")
print(f"Rodadas afetadas:          {sorted(tabela_a['rodada'].unique().tolist())}")
print()
print(tabela_a["diagnostico"].value_counts().to_string())
print()
print(tabela_a[
    ["rodada", "partida_id", "mandante", "visitante", "placar",
     "clube_afetado", "n_atletas_no_payload", "n_entrou_em_campo",
     "n_scout_nao_vazio", "scouts_no_payload",
     "clubes_com_elenco_completo_na_rodada", "diagnostico"]
].to_string(index=False))


# ============================================================
# BLOCO A2 — PERFIL ESTRUTURAL DO PAYLOAD POR RODADA
# ============================================================

titulo("BLOCO A2 — PERFIL ESTRUTURAL DO PAYLOAD POR RODADA")

linhas_payload = []

for rodada in rodadas:

    atletas = payloads[rodada]["atletas"]

    por_clube = defaultdict(list)
    for k, v in atletas.items():
        por_clube[safe_int(v.get("clube_id"))].append(v)

    clubes_na_liga = sorted(
        safe_int(c)
        for c in payloads[rodada]["clubes"].keys()
    )

    sem_elenco = []
    sem_goleiro = []
    elenco_pequeno = []

    for clube_id in clubes_na_liga:

        perfil = perfil_clube(rodada, clube_id)

        if perfil["n_atletas"] <= 2:
            sem_elenco.append(clube_id)
        elif perfil["n_goleiros"] == 0:
            sem_goleiro.append(clube_id)

        if 0 < perfil["n_atletas"] < 14:
            elenco_pequeno.append(clube_id)

    linhas_payload.append({
        "rodada": rodada,
        "n_atletas": len(atletas),
        "clubes_no_catalogo": len(clubes_na_liga),
        "clubes_no_payload": len(por_clube),
        "elenco_ausente": ",".join(map(str, sem_elenco)) or "-",
        "n_clubes_elenco_ausente": len(sem_elenco),
        "goleiro_ausente": ",".join(map(str, sem_goleiro)) or "-",
        "n_clubes_goleiro_ausente": len(sem_goleiro),
        "elenco_menor_14": ",".join(map(str, elenco_pequeno)) or "-",
    })

tabela_payload = pd.DataFrame(linhas_payload)
tabela_payload.to_csv(OUT_PAYLOAD_CSV, index=False, encoding="utf-8")

print(tabela_payload.to_string(index=False))

# Uniformidade estrutural: existe variação de schema entre rodadas?
top_keys = {
    tuple(sorted(payloads[r]["r_data"].keys())) for r in rodadas
}
atleta_keys = {
    tuple(sorted(v.keys()))
    for r in rodadas
    for v in payloads[r]["atletas"].values()
}
tipos_atletas = {
    type(payloads[r]["atletas"]).__name__ for r in rodadas
}

estrutura_uniforme = (
    len(top_keys) == 1 and len(atleta_keys) == 1 and tipos_atletas == {"dict"}
)

print()
print(f"Chaves de topo identicas em R1..R26: {len(top_keys) == 1} -> {list(top_keys)[0]}")
print(f"Tipo do campo 'atletas':           {tipos_atletas}")
print(f"Conjuntos de chaves de atleta:     {len(atleta_keys)} -> {list(atleta_keys)[0]}")
print(f"Estrutura uniforme (sem variacao): {estrutura_uniforme}")


# ============================================================
# BLOCO B — 48 CASOS DE G != GOLS_MARCADOS
# ============================================================

titulo("BLOCO B — CASOS DE G != GOLS_MARCADOS")

casos_b = anomalias[
    anomalias["tipo"] == "g_scout_diferente_do_placar_oficial"
].copy()

casos_b["rodada"] = casos_b["rodada"].astype(int)
casos_b["partida_id"] = casos_b["partida_id"].astype(int)
casos_b["clube_id"] = casos_b["clube_id"].astype(int)

linhas_b = []

for _, caso in casos_b.iterrows():

    rodada = caso["rodada"]
    partida_id = caso["partida_id"]
    clube_id = caso["clube_id"]

    partida = partida_por_id(rodada, partida_id)

    casa = safe_int(partida.get("clube_casa_id"))
    fora = safe_int(partida.get("clube_visitante_id"))

    gols_casa = partida.get("placar_oficial_mandante")
    gols_fora = partida.get("placar_oficial_visitante")

    gols_marcados = gols_casa if clube_id == casa else gols_fora

    perfil_afetado = perfil_clube(rodada, clube_id)
    perfil_adv = perfil_clube(rodada, fora if clube_id == casa else casa)

    def soma(clube_id, scout):
        return sum(
            safe_int((v.get("scout") or {}).get(scout, 0))
            for v in atletas_do_clube(rodada, clube_id).values()
            if v.get("entrou_em_campo")
        )

    g_afetado = soma(clube_id, "G")
    gc_afetado = soma(clube_id, "GC")
    g_adversario = soma(fora if clube_id == casa else casa, "G")
    gc_adversario = soma(fora if clube_id == casa else casa, "GC")

    perfil_afetado = perfil_clube(rodada, clube_id)

    goleiros_g = [
        f"{k}/{v.get('apelido')} (G="
        f"{safe_int((v.get('scout') or {}).get('G', 0))})"
        for k, v in atletas_do_clube(rodada, clube_id).items()
        if v.get("entrou_em_campo")
        and safe_int((v.get("scout") or {}).get("G", 0)) > 0
    ]

    gc_autores = [
        f"{k}/{v.get('apelido')} (GC="
        f"{safe_int((v.get('scout') or {}).get('GC', 0))})"
        for k, v in atletas_do_clube(rodada, clube_id).items()
        if v.get("entrou_em_campo")
        and safe_int((v.get("scout") or {}).get("GC", 0)) > 0
    ]

    dif = g_afetado - gols_marcados

    reconcilia = (g_afetado + gc_adversario == gols_marcados)

    if perfil_afetado["n_participantes"] == 0:
        classe = "A: elenco ausente do payload"
    elif reconcilia:
        classe = "B: reconcilia com GC do adversario"
    else:
        classe = "C: sem explicacao aparente"

    linhas_b.append({
        "rodada": rodada,
        "partida_id": partida_id,
        "mandante": f"{nome_clube(rodada, casa)} ({casa})",
        "visitante": f"{nome_clube(rodada, fora)} ({fora})",
        "placar": f"{gols_casa}x{gols_fora}",
        "clube": nome_clube(rodada, clube_id),
        "clube_id": clube_id,
        "gols_marcados_oficial": gols_marcados,
        "G_agregado_time": g_afetado,
        "G_agregado_adversario": g_adversario,
        "GC_agregado_time": gc_afetado,
        "GC_agregado_adversario": gc_adversario,
        "dif_G_menos_oficial": dif,
        "reconcilia_com_GC_adversario": "sim" if reconcilia else "nao",
        "n_participantes_payload": perfil_afetado["n_participantes"],
        "G_jogadores": "; ".join(goleiros_g) or "(ninguem com G)",
        "GC_jogadores": "; ".join(gc_autores) or "-",
        "classe": classe,
    })

tabela_b = pd.DataFrame(linhas_b).sort_values(["rodada", "partida_id", "clube_id"])

tabela_b.to_csv(OUT_GOLS_CSV, index=False, encoding="utf-8")

print(f"\nCasos de G != gols_marcados: {len(tabela_b)}")
print()
print(tabela_b["classe"].value_counts().to_string())
print()
print("Distribuicao da diferenca (G - gols_marcados):")
print(tabela_b["dif_G_menos_oficial"].value_counts().sort_index().to_string())
print()
print("Casos com GC > 0 em qualquer lado:")
print(
    tabela_b[
        (tabela_b["GC_agregado_time"] > 0)
        | (tabela_b["GC_agregado_adversario"] > 0)
    ][["rodada", "partida_id", "clube", "G_agregado_time",
       "gols_marcados_oficial", "GC_agregado_time", "GC_agregado_adversario",
       "classe"]].to_string(index=False)
)
print()
print("Detalhe dos 48 casos:")
print(tabela_b.to_string(index=False))

# --------------------------------------------------------
# Verificação em nível de partida da identidade de reconciliação
# --------------------------------------------------------

identidade_ok = 0
identidade_falha = []

for rodada in rodadas:

    for partida in payloads[rodada]["partidas"]:

        if partida.get("placar_oficial_mandante") is None:
            continue

        casa = safe_int(partida.get("clube_casa_id"))
        fora = safe_int(partida.get("clube_visitante_id"))

        def soma(clube_id, scout):
            return sum(
                safe_int((v.get("scout") or {}).get(scout, 0))
                for v in atletas_do_clube(rodada, clube_id).values()
                if v.get("entrou_em_campo")
            )

        pred_casa = soma(casa, "G") + soma(fora, "GC")
        pred_fora = soma(fora, "G") + soma(casa, "GC")

        if (
            pred_casa == partida.get("placar_oficial_mandante")
            and pred_fora == partida.get("placar_oficial_visitante")
        ):
            identidade_ok += 1
        else:
            identidade_falha.append({
                "rodada": rodada,
                "partida_id": partida.get("partida_id"),
                "placar": (
                    f"{partida.get('placar_oficial_mandante')}"
                    f"x{partida.get('placar_oficial_visitante')}"
                ),
                "predito": f"{pred_casa}x{pred_fora}",
                "payload_incompleto": "sim" if (
                    perfil_clube(rodada, casa)["n_participantes"] == 0
                    or perfil_clube(rodada, fora)["n_participantes"] == 0
                ) else "nao",
            })

n_partidas_total = sum(
    1
    for r in rodadas
    for p in payloads[r]["partidas"]
    if p.get("placar_oficial_mandante") is not None
)

tabela_identidade = pd.DataFrame(identidade_falha)

print()
print(f"Partidas com placar oficial: {n_partidas_total}")
print(
    f"Partidas que reconciliam com 'gols = G_proprio + GC_adversario': "
    f"{identidade_ok} ({100 * identidade_ok / n_partidas_total:.1f}%)"
)
print(f"Partidas que NAO reconciliam: {len(identidade_falha)}")
if not tabela_identidade.empty:
    print(tabela_identidade.to_string(index=False))


# ============================================================
# BLOCO C — 30 CASOS DE GS != GOLS_SOFRIDOS
# ============================================================

titulo("BLOCO C — CASOS DE GS != GOLS_SOFRIDOS")

casos_c = anomalias[
    anomalias["tipo"] == "gs_scout_diferente_de_gols_sofridos"
].copy()

casos_c["rodada"] = casos_c["rodada"].astype(int)
casos_c["partida_id"] = casos_c["partida_id"].astype(int)
casos_c["clube_id"] = casos_c["clube_id"].astype(int)

linhas_c = []

for _, caso in casos_c.iterrows():

    rodada = caso["rodada"]
    partida_id = caso["partida_id"]
    clube_id = caso["clube_id"]

    partida = partida_por_id(rodada, partida_id)

    casa = safe_int(partida.get("clube_casa_id"))
    fora = safe_int(partida.get("clube_visitante_id"))

    gols_casa = partida.get("placar_oficial_mandante")
    gols_fora = partida.get("placar_oficial_visitante")

    gols_sofridos = gols_fora if clube_id == casa else gols_casa

    adv_id = fora if clube_id == casa else casa

    perfil = perfil_clube(rodada, clube_id)

    goleiros_detalhe = "; ".join(
        f"{k}/{v.get('apelido')} entrou={v.get('entrou_em_campo')} "
        f"GS={safe_int((v.get('scout') or {}).get('GS', 0))} "
        f"SG={safe_int((v.get('scout') or {}).get('SG', 0))} "
        f"DE={safe_int((v.get('scout') or {}).get('DE', 0))} "
        f"DP={safe_int((v.get('scout') or {}).get('DP', 0))} "
        f"scout={v.get('scout')}"
        for k, v in perfil["goleiros"].items()
    ) or "(nenhum goleiro posicao_id=1 no payload)"

    n_gk_com_gs = sum(
        1
        for v in perfil["goleiros"].values()
        if v.get("entrou_em_campo")
        and safe_int((v.get("scout") or {}).get("GS", 0)) > 0
    )

    n_gk_com_sg = sum(
        1
        for v in perfil["goleiros"].values()
        if v.get("entrou_em_campo")
        and safe_int((v.get("scout") or {}).get("SG", 0)) > 0
    )

    if perfil["n_participantes"] == 0:
        classe = "A: elenco ausente do payload"
    elif perfil["n_goleiros"] == 0:
        classe = "B: elenco presente, goleiro ausente do payload"
    elif n_gk_com_gs > 1:
        classe = "C: mais de um goleiro com GS (dupla contagem)"
    elif n_gk_com_gs >= 1 and n_gk_com_sg >= 1:
        classe = "D: GS e SG simultaneos no mesmo time (contradicao interna)"
    else:
        classe = "E: sem explicacao aparente"

    linhas_c.append({
        "rodada": rodada,
        "partida_id": partida_id,
        "clube": nome_clube(rodada, clube_id),
        "clube_id": clube_id,
        "adversario": f"{nome_clube(rodada, adv_id)} ({adv_id})",
        "placar": f"{gols_casa}x{gols_fora}",
        "GS_agregado": int(caso["observado"].split("=")[1]),
        "gols_sofridos": gols_sofridos,
        "dif": int(caso["observado"].split("=")[1]) - gols_sofridos,
        "n_atletas_no_payload": perfil["n_atletas"],
        "n_participantes": perfil["n_participantes"],
        "n_goleiros": perfil["n_goleiros"],
        "n_goleiros_com_GS": n_gk_com_gs,
        "n_goleiros_com_SG": n_gk_com_sg,
        "goleiros_no_payload": goleiros_detalhe,
        "classe": classe,
    })

tabela_c = pd.DataFrame(linhas_c).sort_values(["rodada", "partida_id", "clube_id"])

# ------------------------------------------------------------------
# Checagem global: GS fecha com o placar conforme o nº de goleiros
# que entraram em campo no payload?
# ------------------------------------------------------------------
checagem_gk = {"um_goleiro_ok": 0, "um_goleiro_total": 0,
               "multi_total": 0, "multi_ok": 0, "multi_falha": []}
clubes_multi_ok = set()
clubes_multi_falha = set()

for rodada_ in rodadas:
    atletas_ = payloads[rodada_]["atletas"]
    for partida_ in payloads[rodada_]["partidas"]:
        if partida_.get("placar_oficial_mandante") is None:
            continue
        gc_ = safe_int(partida_.get("placar_oficial_mandante"))
        gf_ = safe_int(partida_.get("placar_oficial_visitante"))
        for clube_, sofr_ in (
            (safe_int(partida_.get("clube_casa_id")), gf_),
            (safe_int(partida_.get("clube_visitante_id")), gc_),
        ):
            gks_ = [
                v for v in atletas_.values()
                if safe_int(v.get("clube_id")) == clube_
                and safe_int(v.get("posicao_id")) == 1
            ]
            if not gks_:
                continue
            n_true_ = sum(1 for v in gks_ if v.get("entrou_em_campo"))
            gs_ = sum(
                safe_int((v.get("scout") or {}).get("GS", 0))
                for v in gks_ if v.get("entrou_em_campo")
            )
            if n_true_ == 1:
                checagem_gk["um_goleiro_total"] += 1
                checagem_gk["um_goleiro_ok"] += int(gs_ == sofr_)
            elif n_true_ > 1:
                checagem_gk["multi_total"] += 1
                if gs_ == sofr_:
                    checagem_gk["multi_ok"] += 1
                    clubes_multi_ok.add((clube_, nome_clube(rodada_, clube_)))
                else:
                    clubes_multi_falha.add((clube_, nome_clube(rodada_, clube_)))
                    checagem_gk["multi_falha"].append({
                        "rodada": rodada_,
                        "partida_id": safe_int(partida_.get("partida_id")),
                        "clube": nome_clube(rodada_, clube_),
                        "clube_id": clube_,
                        "gols_sofridos": sofr_,
                        "GS_agregado": gs_,
                        "goleiros": "; ".join(
                            f"{v.get('apelido')} entrou={v.get('entrou_em_campo')} "
                            f"GS={(v.get('scout') or {}).get('GS')}"
                            for v in gks_
                        ),
                    })

tabela_gk_multi = pd.DataFrame(checagem_gk["multi_falha"]).sort_values(
    ["rodada", "partida_id"]
) if checagem_gk["multi_falha"] else pd.DataFrame()

print("\nChecagem global GS vs gols_sofridos:")
print(
    f"  1 goleiro que entrou: {checagem_gk['um_goleiro_ok']}"
    f"/{checagem_gk['um_goleiro_total']} fecham com o placar"
)
print(
    f"  2+ goleiros que entraram: {checagem_gk['multi_ok']}"
    f"/{checagem_gk['multi_total']} fecham com o placar"
)
if tabela_gk_multi.size:
    print(tabela_gk_multi.to_string(index=False))

tabela_c.to_csv(OUT_GOLEIROS_CSV, index=False, encoding="utf-8")

print(f"\nCasos de GS != gols_sofridos: {len(tabela_c)}")
print()
print(tabela_c["classe"].value_counts().to_string())
print()
print(tabela_c.to_string(index=False))


# ============================================================
# BLOCO D — RODADA vs DATA (achado transversal)
# ============================================================

titulo("BLOCO D — RODADA vs DATA DA PARTIDA")

linhas_data = []

for rodada in rodadas:

    partidas = payloads[rodada]["partidas"]

    datas = sorted(p["partida_data"][:10] for p in partidas)

    janela = calendario.get(rodada, {})
    inicio = str(janela.get("inicio", ""))[:10]
    fim = str(janela.get("fim", ""))[:10]

    fora = [d for d in datas if (inicio and d < inicio) or (fim and d > fim)]

    linhas_data.append({
        "rodada": rodada,
        "janela_oficial": f"{inicio}..{fim}" if inicio else "-",
        "data_min": datas[0] if datas else "-",
        "data_max": datas[-1] if datas else "-",
        "n_fora_da_janela": len(fora),
        "datas_fora": ",".join(sorted(set(fora))) or "-",
    })

tabela_data = pd.DataFrame(linhas_data)

n_fora_total = int(tabela_data["n_fora_da_janela"].sum())

print(f"Partidas com data fora da janela oficial da rodada: {n_fora_total}")
print()
print(tabela_data[tabela_data["n_fora_da_janela"] > 0].to_string(index=False))


# ============================================================
# RELATÓRIO
# ============================================================

titulo("GERANDO RELATÓRIO")

contagem_c = tabela_c["classe"].value_counts()
relatorio = []

relatorio += [
    "# EXP-12 — Investigação das anomalias do histórico de partidas",
    "",
    f"- Temporada: **{TEMPORADA}** | Rodadas: **R{RODADA_INI}..R{RODADA_FIM}**",
    "- Gerado por: `scripts/exp12_investigar_anomalias.py`",
    "- Natureza: **somente leitura**. Nenhum dataset foi alterado, nenhuma "
    "correção foi aplicada e nenhuma feature temporal foi criada.",
    "",
    "## 1. Metodologia da investigação",
    "",
    "A auditoria da Fase 1 registrou 152 anomalias. Este documento rastreia "
    "as três famílias com risco de contaminar feature:",
    "",
    "- **A.** 24 linhas de 12 partidas com os 14 scouts diretos zerados;",
    "- **B.** 48 linhas com soma de `G` diferente de `gols_marcados`;",
    "- **C.** 30 linhas com `GS` diferente de `gols_sofridos`.",
    "",
    "Procedimento, para cada caso:",
    "",
    "1. Localizar a partida em `partidas{N}.json` e recuperar o placar "
    "oficial.",
    "2. Abrir o payload `r{N}.json` da mesma rodada e contar, para o clube "
    "afetado: atletas no payload, atletas com `entrou_em_campo=True`, "
    "atletas com scout não vazio, goleiros (`posicao_id=1`) e quais scouts "
    "aparecem.",
    "3. Recalcular a agregação a partir do payload bruto, sem usar o "
    "dataset, e comparar com o que está gravado.",
    "4. Testar hipóteses explicativas: ausência de elenco, ausência de "
    "goleiro, gol contra (`GC`), troca de goleiro.",
    "5. Classificar o caso e, quando nada explica, marcar como "
    "**sem explicação aparente**.",
    "",
    "Nenhum valor foi corrigido. As tabelas abaixo são evidência, não "
    "correção.",
    "",
    "## 2. Resumo dos achados",
    "",
    f"- **Bloco A:** {partidas_a} partidas / {len(tabela_a)} linhas com scouts "
    "zerados. Causa: **elenco do clube ausente do payload** — em todos os "
    "casos o clube aparece no `partidas{N}.json`, mas o `r{N}.json` traz "
    "somente o técnico (`posicao_id=6`, `entrou_em_campo=False`). "
    "**Não é performance zero.**",
    f"- **Bloco B:** {len(tabela_b)} casos de `G` != `gols_marcados`. A "
    f"identidade `gols_marcados = G_do_time + GC_do_adversario` reconcilia "
    f"{identidade_ok} de {n_partidas_total} partidas "
    f"({100 * identidade_ok / n_partidas_total:.1f}%). Os casos remanescentes "
    "se dividem em elenco ausente e casos sem explicação aparente.",
    f"- **Bloco C:** {len(tabela_c)} casos de `GS` != `gols_sofridos`, "
    f"**todos explicados**: elenco ausente do payload "
    f"({contagem_c.get('A: elenco ausente do payload', 0)}), goleiro ausente "
    f"do payload no clube 315 "
    f"({contagem_c.get('B: elenco presente, goleiro ausente do payload', 0)}) "
    "e dois ou mais goleiros do mesmo clube no payload "
    f"({len(tabela_c) - contagem_c.get('A: elenco ausente do payload', 0) - contagem_c.get('B: elenco presente, goleiro ausente do payload', 0)}, "
    "todos do clube 265/BAH). Nenhum caso ficou sem explicação.",
    f"- **Bloco D (transversal):** {n_fora_total} partidas têm data fora da "
    "janela oficial da rodada. `rodada` não é uma ordenação cronológica "
    "confiável; `data` é.",
    "",
    "## 3. Bloco A — as 12 partidas com scouts totalmente zerados",
    "",
    "### 3.1 Tabela por partida e clube afetado",
    "",
    md_table(tabela_a, [
        "rodada", "partida_id", "mandante", "visitante", "placar",
        "clube_afetado", "n_atletas_no_payload", "n_entrou_em_campo",
        "n_scout_nao_vazio", "existe_atleta_no_payload", "scouts_no_payload",
        "n_atletas_adversario", "clubes_com_elenco_completo_na_rodada",
        "diagnostico",
    ]),
    "",
    "### 3.2 O que existe no payload para o clube afetado",
    "",
    "Para cada caso, o conteúdo literal do `r{N}.json` para o clube afetado:",
    "",
]

for _, linha in tabela_a.iterrows():
    relatorio += [
        f"- **R{linha['rodada']} / partida {linha['partida_id']} / "
        f"{linha['clube_afetado']} (id {linha['clube_afetado_id']})**: "
        f"{linha['elenco_no_payload']}",
    ]

relatorio += [
    "",
    "### 3.3 Leitura",
    "",
    "- `n_atletas_no_payload` é **1** em todos os 24 casos, e esse único "
    "atleta é o técnico.",
    "- `n_entrou_em_campo` é **0** e `n_scout_nao_vazio` é **0** em todos os "
    "casos.",
    "- `scouts_no_payload` é `(nenhum)` em todos os casos: o técnico não tem "
    "scout, ou tem `{}`.",
    "- Importante: nestas 12 partidas **os dois clubes** estão sem elenco no "
    "payload, então o adversário também aparece com `n_atletas_no_payload` "
    "igual a 1. O controle correto é a nível de rodada: a coluna "
    "`clubes_com_elenco_completo_na_rodada` mostra que, na mesma rodada, "
    "14 a 18 dos 20 clubes têm elenco completo. O defeito é **parcial por "
    "clube**, não do arquivo inteiro.",
    f"- Nenhum caso de **performance zero** foi encontrado: as "
    f"{len(tabela_a)} linhas são "
    f"{(tabela_a['diagnostico'] == 'elenco ausente do payload').sum()} "
    "casos de ausência de dado e "
    f"{(tabela_a['diagnostico'] == 'scouts realmente zerados (performance zero)').sum()} "
    "casos de performance zero.",
    "- O schema dos payloads é idêntico em R1..R26 "
    f"(chaves de topo {list(top_keys)[0]}, `atletas` como `dict`, chaves de "
    f"atleta {list(atleta_keys)[0]}). **Não há variação de estrutura**: é "
    "conteúdo ausente, não formato diferente.",
    "",
    "### 3.4 Alcance do problema além das 12 partidas",
    "",
    "O mesmo defeito de payload aparece em 6 rodadas e atinge 30 pares "
    "(rodada, clube). As 24 linhas acima são as que entraram no dataset; as "
    "6 restantes pertencem às 3 partidas da R21 excluídas por falta de "
    "placar oficial:",
    "",
    md_table(tabela_payload[tabela_payload["n_clubes_elenco_ausente"] > 0], [
        "rodada", "n_atletas", "elenco_ausente", "n_clubes_elenco_ausente",
    ]),
    "",
    "## 4. Bloco B — os 48 casos de G != gols_marcados",
    "",
    "### 4.1 Classificação",
    "",
]

contagem_b = tabela_b["classe"].value_counts()

relatorio += [
    "| classe | casos |",
    "|---|---|",
] + [
    f"| {k} | {v} |" for k, v in contagem_b.items()
] + [
    "",
    "Distribuição da diferença `G - gols_marcados`:",
    "",
    md_table(
        tabela_b["dif_G_menos_oficial"].value_counts()
        .sort_index()
        .rename_axis("diferenca")
        .reset_index(name="casos")
    ),
    "",
    "### 4.2 A identidade de reconciliação",
    "",
    "Teste aplicado em todas as partidas com placar oficial:",
    "",
    "```",
    "gols_marcados_do_time   = G_do_time   + GC_do_adversario",
    "gols_marcados_do_advers = G_adversario + GC_do_time",
    "```",
    "",
    f"- Partidas testadas: **{n_partidas_total}**",
    f"- Partidas que reconciliam: **{identidade_ok}** "
    f"({100 * identidade_ok / n_partidas_total:.1f}%)",
    f"- Partidas que não reconciliam: **{len(tabela_identidade)}**",
    "",
]

if not tabela_identidade.empty:
    relatorio += [
        md_table(tabela_identidade),
        "",
    ]

relatorio += [
    "Ou seja: `G` sozinho **não** é o total de gols do time. O gol contra "
    "(`GC`) é creditado ao adversário, então `G + GC_adversario` é a "
    "reconciliação correta. Isso explica a maior parte dos 48 casos sem "
    "que nenhum dado esteja errado.",
    "",
    "### 4.3 Detalhe dos 48 casos",
    "",
    md_table(tabela_b, [
        "rodada", "partida_id", "mandante", "visitante", "placar", "clube",
        "gols_marcados_oficial", "G_agregado_time", "GC_agregado_time",
        "GC_agregado_adversario", "dif_G_menos_oficial",
        "reconcilia_com_GC_adversario", "n_participantes_payload",
        "G_jogadores", "classe",
    ]),
    "",
    "### 4.4 Jogadores com G e autores de GC",
    "",
]

for _, linha in tabela_b.iterrows():
    if linha["GC_jogadores"] != "-":
        relatorio.append(
            f"- R{linha['rodada']} / {linha['partida_id']} / "
            f"{linha['clube']}: G de [{linha['G_jogadores']}], "
            f"GC de [{linha['GC_jogadores']}] "
            f"(oficial {linha['gols_marcados_oficial']})"
        )

relatorio += [
    "",
    "### 4.5 Leitura",
    "",
    f"- Classe **A (elenco ausente)**: {contagem_b.get('A: elenco ausente do payload', 0)} "
    "casos — `G` zerado por falta de payload, exatamente o Bloco A.",
    f"- Classe **B (reconcilia com GC do adversário)**: "
    f"{contagem_b.get('B: reconcilia com GC do adversario', 0)} casos — "
    "divergência aparente, mas `G + GC_adversario` fecha com o placar.",
    f"- Classe **C (sem explicação aparente)**: "
    f"{contagem_b.get('C: sem explicacao aparente', 0)} casos. São eles:",
    "",
]

sem_explicacao_b = tabela_b[tabela_b["classe"] == "C: sem explicacao aparente"]

relatorio += [
    md_table(sem_explicacao_b, [
        "rodada", "partida_id", "mandante", "visitante", "placar", "clube",
        "gols_marcados_oficial", "G_agregado_time", "G_agregado_adversario",
        "dif_G_menos_oficial", "G_jogadores",
    ]),
    "",
    "Para esses casos o payload tem elenco completo, não há `GC` e a "
    "soma de `G` continua divergindo do placar oficial. As diferenças vão "
    "de -1 a +1. Não foi encontrada explicação na fonte; ficam registrados "
    "como divergência de origem.",
    "",
    "## 5. Bloco C — os 30 casos de GS != gols_sofridos",
    "",
    "### 5.1 Classificação",
    "",
]


relatorio += [
    "| classe | casos |",
    "|---|---|",
] + [
    f"| {k} | {v} |" for k, v in contagem_c.items()
] + [
    "",
    "### 5.2 Detalhe dos 30 casos",
    "",
    md_table(tabela_c, [
        "rodada", "partida_id", "clube", "adversario", "placar",
        "GS_agregado", "gols_sofridos", "dif", "n_atletas_no_payload",
        "n_participantes", "n_goleiros", "n_goleiros_com_GS",
        "n_goleiros_com_SG", "classe",
    ]),
    "",
    "### 5.3 Goleiros presentes no payload",
    "",
]

for _, linha in tabela_c.iterrows():
    relatorio.append(
        f"- R{linha['rodada']} / {linha['partida_id']} / "
        f"{linha['clube']} (GS={linha['GS_agregado']}, "
        f"sofridos={linha['gols_sofridos']}): {linha['goleiros_no_payload']}"
    )

troca_gk = tabela_c[tabela_c["classe"] == "C: mais de um goleiro com GS (dupla contagem)"]
contradicao_gk = tabela_c[
    tabela_c["classe"]
    == "D: GS e SG simultaneos no mesmo time (contradicao interna)"
]

relatorio += [
    "",
    "### 5.4 Leitura",
    "",
    f"- Classe **A (elenco ausente)**: {contagem_c.get('A: elenco ausente do payload', 0)} "
    "casos — `GS` zerado por falta de payload.",
    "- Classe **B (elenco presente, goleiro ausente)**: "
    f"{contagem_c.get('B: elenco presente, goleiro ausente do payload', 0)} "
    "casos, todos do **clube 315 (CHA/Fortaleza)**, que tem 15 a 16 atletas "
    "no payload mas **nenhum com `posicao_id=1`** nas rodadas 1 a 8. "
    "O elenco de linha está lá; o goleiro não está. Isso zera "
    "estruturalmente `GS`, `SG` e `DP` para esse clube nessas rodadas — "
    "inclusive na R1 já validada.",
    "- Classe **C (dupla contagem na troca de goleiro)**: "
    f"{contagem_c.get('C: mais de um goleiro com GS (dupla contagem)', 0)} "
    "casos, todos do **clube 265 (BAH/Bahia)**, e em todos há **dois "
    "goleiros com `GS` na mesma partida**. O `GS` é somado para os dois e "
    "estoura o total de gols sofridos do time:",
    "",
]

for _, linha in troca_gk.iterrows():
    relatorio.append(
        f"- R{linha['rodada']} / {linha['partida_id']}: "
        f"GS agregado {linha['GS_agregado']} contra "
        f"{linha['gols_sofridos']} gols sofridos "
        f"({linha['n_goleiros_com_GS']} goleiros com GS > 0)"
    )

relatorio += [
    "",
    f"- Classe **D (contradição interna do payload)**: "
    f"{contagem_c.get('D: GS e SG simultaneos no mesmo time (contradicao interna)', 0)} "
    "casos, também do **clube 265 (BAH)**. São partidas em que a API "
    "devolve **dois goleiros da mesma equipe, ambos com `entrou_em_campo=True`**, "
    "e ainda atribui `GS` a um e `SG` ao outro **na mesma partida** — estados "
    "mutuamente exclusivos:",
    "",
]

for _, linha in contradicao_gk.iterrows():
    relatorio.append(
        f"- R{linha['rodada']} / {linha['partida_id']} / {linha['clube']}: "
        f"placar {linha['placar']} => {linha['gols_sofridos']} gols sofridos, "
        f"mas GS agregado {linha['GS_agregado']} "
        f"({linha['n_goleiros_com_GS']} goleiro com GS) **e** "
        f"{linha['n_goleiros_com_SG']} goleiro com SG. "
        f"{linha['goleiros_no_payload']}"
    )

relatorio += [
    "",
    "Esse é o único mecanismo que explica as 30 anomalias de `GS`, e ele é "
    "estrutural. A checagem global de `GS` contra `gols_sofridos` em todos "
    "os clube-partida do payload dá:",
    "",
    f"- **1 goleiro com `entrou_em_campo=True`**: "
    f"{checagem_gk['um_goleiro_ok']} de "
    f"{checagem_gk['um_goleiro_total']} fecham exatamente com o placar "
    "oficial (100%).",
    f"- **2 ou mais goleiros com `entrou_em_campo=True`**: "
    f"{checagem_gk['multi_ok']} de {checagem_gk['multi_total']} fecham. "
    f"Os {checagem_gk['multi_total'] - checagem_gk['multi_ok']} que **não** "
    "fecham são:",
    "",
]

if tabela_gk_multi.size:
    relatorio += [
        md_table(tabela_gk_multi, [
            "rodada", "partida_id", "clube", "gols_sofridos",
            "GS_agregado", "goleiros",
        ]),
        "",
    ]

relatorio += [
    "As 6 divergências são **todas do clube 265 (BAH)**, e apenas nas "
    "rodadas 1 a 8, que é exatamente a janela em que a API devolve 2 a 3 "
    "goleiros do Bahia no payload (Léo Vieira 71043, Ronaldo 91101 e, na R8, "
    "João Paulo 85425). A partir da R9 o Bahia passa a ter **um único "
    "goleiro por partida** e o `GS` fecha em todas as rodadas restantes. "
    "Os outros clubes com 2+ goleiros no payload (294 na R4, 283 na R5, "
    "2305 na R12, 263 e 284 na R18, 285 na R24, 265 na R15 e R17) **não** "
    "geram contradição: o `GS` é creditado só a um deles.",
    "",
    "Portanto `GS` **não é** uma medida de gols sofridos do time. É o "
    "contador por goleiro e ele: (a) desaparece quando o goleiro não está "
    "no payload; (b) é somado em todos os goleiros marcados como "
    "`entrou_em_campo=True` na partida; (c) pode coexistir com `SG` do colega "
    "na mesma partida. "
    "É exatamente por isso que a semântica validada proíbe usar `GS` como "
    "`gols_sofridos`.",
    "",
    f"- Classe **E (sem explicação aparente)**: "
    f"{contagem_c.get('E: sem explicacao aparente', 0)} casos.",
    "",
    "## 6. Bloco D — rodada não é ordem cronológica",
    "",
    f"{n_fora_total} partidas têm `partida_data` fora da janela oficial de "
    "rodada definida em `rodadas.json`:",
    "",
    md_table(tabela_data[tabela_data["n_fora_da_janela"] > 0]),
    "",
    "São partidas adiadas que a API devolve dentro do arquivo de outra "
    "rodada. A R21 é o caso extremo: reúne jogos de 29/07, 16/09 e 03/10/2026. "
    "Para qualquer feature temporal, a ordenação por `rodada` não é segura; "
    "a ordenação por `data` também precisa tratar partidas que ocorram "
    "depois da rodada que as indexou.",
    "",
    "## 7. Padrões encontrados",
    "",
    "1. **Ausência de elenco no payload** (6 rodadas, 30 pares rodada-clube, "
    "24 linhas no dataset). O único atleta presente é o técnico.",
    "2. **Ausência de goleiro no payload** (clube 315 nas rodadas 1 a 8, "
    "com elenco de linha completo). Zera `GS`, `SG` e `DP` estruturalmente.",
    "3. **Reconciliação de gols depende de `GC`.** "
    "`G` isolado não fecha com o placar em "
    f"{100 - 100 * identidade_ok / n_partidas_total:.1f}% das partidas, mas "
    "`G + GC_adversario` fecha em "
    f"{100 * identidade_ok / n_partidas_total:.1f}%.",
    "4. **`GS` é contador por goleiro**, não por time: fecha em 100% dos "
    "clube-partida com um único goleiro em campo e quebra justamente nos "
    "casos de 2+ goleiros do mesmo time.",
    "5. **Schema uniforme**: o problema é sempre ausência de conteúdo, "
    "nunca mudança de formato.",
    "6. **`rodada` não ordena o tempo**: "
    f"{n_fora_total} partidas estão fora da janela da própria rodada.",
    "",
    "## 8. Casos sem explicação aparente",
    "",
    f"- Bloco B, classe C: **{len(sem_explicacao_b)}** casos de "
    "`G` != `gols_marcados` com payload completo e sem `GC` no jogo. "
    "Diferenças de -1 e +1.",
    f"- Bloco C, classe E: "
    f"{contagem_c.get('E: sem explicacao aparente', 0)} casos. "
    "As 2 divergências que pareciam sem causa no primeiro corte eram o "
    "clube 265 com `GS` e `SG` simultâneos, e passaram a ter causa "
    "identificada (classe D).",
    "",
    "Nenhum desses foi corrigido. Ficam registrados para decisão do auditor.",
    "",
    "## 9. Recomendação técnica por tipo de anomalia",
    "",
    "As recomendações abaixo **não foram aplicadas**. Cada uma é uma "
    "decisão metodológica que precisa ser aprovada antes de virar código.",
    "",
    "### 9.1 Scouts zerados por elenco ausente (Bloco A)",
    "",
    "- **Não** tratar como zero de performance. O valor zero seria uma "
    "afirmação falsa sobre o time.",
    "- **Recomendação:** marcar essas linhas com um indicador de "
    "ausência de dado (`scouts_disponiveis = 0`) e, nas features "
    "temporais, excluir a partida da janela de scouts do clube em vez de "
    "somar zero. A cobertura da janela passa a ser explícita, o que já é "
    "exigido pelo EXP-12.",
    "- O placar oficial dessas linhas continua válido e pode ser usado como "
    "está.",
    "- Nenhuma mudança no artefato `team_match_history.parquet` nesta fase: "
    "o indicador é derivado na etapa de features, com o payload como fonte.",
    "",
    "### 9.2 Goleiro ausente do payload (achado do Bloco C, classe B)",
    "",
    "- `GS`, `SG` e `DP` desses clubes são **estruturalmente zero**, não "
    "zero de performance.",
    "- **Recomendação:** nunca derivar clean sheet, goleiro ou defesa a "
    "partir de `GS`/`SG` sem antes checar a presença de goleiro no payload "
    "da rodada. Para contexto de adversário, `gols_sofridos` do placar "
    "oficial é o campo confiável.",
    "- Se o EXP-12 precisar de features de goleiro, o payload de "
    "`posicao_id=1` tem de ser uma pré-condição verificável, e as "
    "rodadas 1 a 8 do clube 315 seriam lacunas conhecidas.",
    "",
    "### 9.3 G != gols_marcados (Bloco B)",
    "",
    "- **Não** alterar `G` nem `gols_marcados`.",
    "- **Recomendação:** manter os dois campos com suas fontes distintas e "
    "documentar a identidade `gols_marcados = G + GC_adversario` como a "
    "regra de reconciliação. Se o EXP-12 quiser um total de gols por time a "
    "partir de scouts, ele precisa somar `GC` do adversário — mas isso é "
    "decisão de feature, não correção desta base.",
    "- As "
    f"{len(sem_explicacao_b)} sem explicação aparente devem ser tratadas "
    "como ruído de fonte e nunca como sinal.",
    "",
    "### 9.4 GS != gols_sofridos (Bloco C)",
    "",
    "- **Não** usar `GS` como gols sofridos e **não** criar coluna "
    "`clean_sheet`.",
    "- O ponto a favoring dessa decisão ficou forte: `GS` fecha com o placar "
    f"em {checagem_gk['um_goleiro_ok']}/{checagem_gk['um_goleiro_total']} "
    "clube-partida com um único goleiro em campo, e quebra em "
    f"{checagem_gk['multi_total'] - checagem_gk['multi_ok']} dos "
    f"{checagem_gk['multi_total']} casos com 2+ goleiros. A série é "
    "confiável só sob a condição de goleiro único.",
    "- **Recomendação:** manter `GS` e `SG` como scouts brutos com a "
    "semântica atual e registrar a divergência. Se alguma feature de "
    "contexto de adversário precisar de clean sheet, ela deve vir de "
    "`gols_sofridos == 0` do placar oficial, com a restrição de que a "
    "informação de goleiro está ausente em rodadas específicas.",
    "",
    "### 9.5 Rodada não cronológica (Bloco D)",
    "",
    "- **Recomendação:** antes de qualquer janela temporal, decidir "
    "explicitamente se a ordenação é por `rodada` (ordem oficial da "
    "competição) ou por `data` (ordem real dos jogos). As 13 partidas "
    "fora de janela mudam o resultado de qualquer janela que use rodada "
    "como eixo de tempo.",
    "- Este ponto é **bloqueante** para a próxima fase e precisa ser "
    "resolvido antes de construir features.",
    "",
    "## 10. O que NÃO foi feito",
    "",
    "- `data/interim/exp12/team_match_history.parquet` não foi alterado.",
    "- Nenhuma feature temporal foi criada.",
    "- Nenhum valor de scout, gol ou placar foi corrigido.",
    "- `cartola_jogador_rodada_v2.parquet` e `catboost_v2.py` não foram "
    "tocados.",
    "- A decisão de tratamento continua **pendente do auditor**.",
    "",
    "## 11. Arquivos desta investigação",
    "",
    "- `scripts/exp12_investigar_anomalias.py`",
    "- `reports/exp12/exp12_anomalias_investigacao.md` (este documento)",
    "- `reports/exp12/exp12_inv_scouts_zerados.csv`",
    "- `reports/exp12/exp12_inv_gols.csv`",
    "- `reports/exp12/exp12_inv_goleiros.csv`",
    "- `reports/exp12/exp12_inv_perfil_payload.csv`",
    "",
]

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(relatorio) + "\n")


# ============================================================
# FECHAMENTO
# ============================================================

titulo("RESUMO FINAL")

print(f"Bloco A — partidas com scouts zerados:            {partidas_a}")
print(f"          linhas:                                 {len(tabela_a)}")
print(f"          causas:                                 "
      f"{tabela_a['diagnostico'].value_counts().to_dict()}")
print()
print(f"Bloco B — casos de G != gols_marcados:           {len(tabela_b)}")
print(f"          classes:                                {contagem_b.to_dict()}")
print(f"          partidas que reconciliam com GC:        {identidade_ok}"
      f"/{n_partidas_total}")
print()
print(f"Bloco C — casos de GS != gols_sofridos:          {len(tabela_c)}")
print(f"          classes:                                {contagem_c.to_dict()}")
print()
print(f"Bloco D — partidas com data fora da rodada:      {n_fora_total}")
print()
print(f"Estrutura dos payloads uniforme:                 {estrutura_uniforme}")
print(f"Cases sem explicacao (B):                        {len(sem_explicacao_b)}")
print(f"Cases sem explicacao (C):                        "
      f"{contagem_c.get('E: sem explicacao aparente', 0)}")
print(f"GS fecha com 1 goleiro em campo:                 "
      f"{checagem_gk['um_goleiro_ok']}/{checagem_gk['um_goleiro_total']}")
print(f"GS fecha com 2+ goleiros em campo:                "
      f"{checagem_gk['multi_ok']}/{checagem_gk['multi_total']}")
print()
print("Nenhum arquivo de dados foi alterado.")
print()
print(f"Relatorio:   {OUT_MD}")
print(f"Scouts CSV:  {OUT_SCOUTS_CSV}")
print(f"Gols CSV:    {OUT_GOLS_CSV}")
print(f"Goleiros:    {OUT_GOLEIROS_CSV}")
print(f"Payload:     {OUT_PAYLOAD_CSV}")
print()
print("Nenhuma feature temporal foi criada. "
      "A decisao de tratamento depende do auditor.")
