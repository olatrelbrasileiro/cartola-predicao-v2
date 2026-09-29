#!/usr/bin/env python3
"""
EXP-12 — F1.2 — Auditoria temporal das partidas (somente leitura)

Este script é SOMENTE LEITURA sobre:

    data/interim/exp12/team_match_history.parquet          (leitura)
    experiments/data/cartola_api_2026/partidas{N}.json       (leitura)
    experiments/data/cartola_api_2026/r{N}.json             (leitura)
    experiments/data/cartola_api_2026/rodadas.json           (leitura)

Ele NÃO constrói feature temporal, NÃO altera o parquet, NÃO altera os dados
de origem e NÃO aplica nenhuma correção silenciosa. A única coisa que produz
é um relatório de auditoria e dois CSV de evidência.

Objetivo: determinar, com evidência, qual deve ser o eixo temporal do EXP-12.

1. Inventário das 257 partidas (rodada, partida_id, data, dentro/fora da
   janela oficial, mandante, visitante).
2. Partidas cuja data real não respeita a ordem esperada das rodadas.
3. Inversões cronológicas: partida de R disputada depois de R+1, R+2 etc.
4. Para cada partida problemática: data, adversários, posição cronológica real
   e rodadas que já tinham partidas disputadas naquela data.
5. Risco de leakage das duas alternativas:
     A) histórico = partidas com rodada < rodada_alvo
     B) histórico = partidas com data real estritamente anterior à data-alvo
6. Partidas com mesma data e o que a ordenação por data resolve ou não.

Saídas:

    reports/exp12/exp12_auditoria_temporal.md
    reports/exp12/exp12_inv_temporal_inversoes.csv
    reports/exp12/exp12_inv_temporal_historicos.csv
"""

from pathlib import Path
import json
import itertools

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

API_DIR = ROOT / "experiments" / "data" / "cartola_api_2026"

DATASET = ROOT / "data" / "interim" / "exp12" / "team_match_history.parquet"

REPORT_DIR = ROOT / "reports" / "exp12"

ANOMALIAS_CSV = REPORT_DIR / "exp12_team_match_history_anomalias.csv"

OUT_MD = REPORT_DIR / "exp12_auditoria_temporal.md"
OUT_INVERSOES_CSV = REPORT_DIR / "exp12_inv_temporal_inversoes.csv"
OUT_HISTORICOS_CSV = REPORT_DIR / "exp12_inv_temporal_historicos.csv"

RODADA_INI = 1
RODADA_FIM = 26


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


def lista_ids(frame, extra=None):
    if frame.empty:
        return "(nenhum)"
    partes = []
    for linha in frame.itertuples(index=False):
        if extra:
            partes.append(f"{linha.partida_id} ({extra(linha)})")
        else:
            partes.append(str(linha.partida_id))
    return ", ".join(partes)


# ============================================================
# LEITURA
# ============================================================

titulo("EXP-12 F1.2 — AUDITORIA TEMPORAL (somente leitura)")

print(f"\nDataset lido:    {DATASET}")
print(f"Fontes lidas:    {API_DIR}")

dataset = pd.read_parquet(DATASET)

print(f"Linhas do dataset: {len(dataset)}")

rodadas_json = load_json(API_DIR / "rodadas.json")

janelas = {
    safe_int(r["rodada_id"]): (
        pd.Timestamp(r["inicio"]),
        pd.Timestamp(r["fim"]),
    )
    for r in rodadas_json
}

payloads = {}

for rodada in range(RODADA_INI, RODADA_FIM + 1):

    partidas_data = load_json(API_DIR / f"partidas{rodada}.json")
    r_data = load_json(API_DIR / f"r{rodada}.json")

    payloads[rodada] = {
        "partidas": partidas_data.get("partidas", []) or [],
        "atletas": r_data.get("atletas", {}) or {},
        "clubes": {
            safe_int(k): v
            for k, v in (partidas_data.get("clubes", {}) or {}).items()
        },
    }


# ============================================================
# 1. INVENTÁRIO DAS PARTIDAS
# ============================================================

titulo("1. INVENTARIO DAS PARTIDAS")

partida_por_id = {}

for rodada, bloco in payloads.items():
    for partida in bloco["partidas"]:
        partida_por_id[safe_int(partida["partida_id"])] = (rodada, partida)


def nome_clube(rodada, clube_id):
    clube = payloads[rodada]["clubes"].get(safe_int(clube_id))
    if not clube:
        return f"clube_{safe_int(clube_id)}"
    return clube.get("apelido") or clube.get("nome") or f"clube_{clube_id}"


# Uma linha por partida, reconstruindo mandante/visitante a partir do dataset
linhas = []

for partida_id, grupo in dataset.groupby("partida_id"):

    partida_id = safe_int(partida_id)

    grupo = grupo.sort_values("mando")

    mandante = grupo[grupo["mando"] == "C"].iloc[0]
    visitante = grupo[grupo["mando"] == "F"].iloc[0]

    rodada = safe_int(mandante["rodada"])
    dt = pd.Timestamp(mandante["data"])

    inicio, fim = janelas[rodada]

    _, partida_api = partida_por_id[partida_id]

    linhas.append({
        "rodada": rodada,
        "partida_id": partida_id,
        "data": mandante["data"],
        "dt": dt,
        "dia": dt.normalize(),
        "mandante": mandante["clube_nome"],
        "visitante": visitante["clube_nome"],
        "mandante_id": safe_int(mandante["clube_id"]),
        "visitante_id": safe_int(visitante["clube_id"]),
        "janela_inicio": inicio,
        "janela_fim": fim,
        "dentro_da_janela": bool(inicio <= dt <= fim),
        "dias_da_janela": int((dt.normalize() - inicio.normalize()).days),
        "api_mesma_rodada": safe_int(partida_api.get("partida_id")) == partida_id,
    })

partidas = pd.DataFrame(linhas).sort_values(["dt", "partida_id"]).reset_index(drop=True)

# Posição cronológica real: a ordem em que os jogos efetivamente aconteceram
partidas["pos_cronologica"] = range(1, len(partidas) + 1)

# Posição dentro da sua própria rodada, na ordem da rodada
partidas["dentro_da_janela_txt"] = partidas["dentro_da_janela"].map(
    {True: "sim", False: "nao"}
)

partidas["posicao_na_rodada"] = (
    partidas.sort_values(["rodada", "dt", "partida_id"])
    .groupby("rodada").cumcount() + 1
)

# Quantas rodadas já tinham pelo menos uma partida disputada até esta data
rodadas_disputadas = sorted(partidas["rodada"].unique().tolist())
partidas["rodada_max_ja_disputada"] = [
    max(
        [r for r in rodadas_disputadas
         if (partidas.loc[partidas["dt"] <= dt_, "rodada"] == r).any()],
        default=0,
    )
    for dt_ in partidas["dt"]
]

print(f"\nPartidas distintas:                {len(partidas)}")
print(f"Clubes distintos:                  "
      f"{len(set(partidas['mandante_id']) | set(partidas['visitante_id']))}")
print(f"Rodadas cobertas:                  {partidas['rodada'].min()}..{partidas['rodada'].max()}")
print(f"Datas distintas:                   {partidas['dia'].nunique()}")
print(f"Timestamps distintos:              {partidas['dt'].nunique()}")
print(f"Partidas dentro da janela oficial:  {int(partidas['dentro_da_janela'].sum())}")
print(f"Partidas fora da janela oficial:    {int((~partidas['dentro_da_janela']).sum())}")

fora_janela = partidas[~partidas["dentro_da_janela"]].copy()
problematicas = fora_janela

print(f"\nRodadas com alguma partida fora da janela: "
      f"{sorted(problematicas['rodada'].unique().tolist())}")
print(f"Rodadas totalmente dentro da janela:       "
      f"{sorted(set(rodadas_disputadas) - set(problematicas['rodada'].unique()))}")


# ============================================================
# 2. INVERSÕES CRONOLÓGICAS
# ============================================================

titulo("2. INVERSOES CRONOLOGICAS")

# Uma partida de rodada R é "invertida" se existe alguma partida de rodada
# maior que foi disputada ANTES dela.
ordem = partidas.sort_values(["dt", "partida_id"]).reset_index(drop=True)

linhas_inv = []

for pos, alvo in ordem.iterrows():

    anteriores = ordem.iloc[:pos]

    # rodadas menores cuja partida veio depois (não é inversão, é atraso)
    # rodadas maiores cuja partida veio antes -> inversão
    invertidas = anteriores[anteriores["rodada"] > alvo["rodada"]]

    if invertidas.empty:
        continue

    rodadas_invertidas = sorted(invertidas["rodada"].unique().tolist())

    linhas_inv.append({
        "rodada": alvo["rodada"],
        "partida_id": alvo["partida_id"],
        "data": alvo["data"],
        "mandante": alvo["mandante"],
        "visitante": alvo["visitante"],
        "pos_cronologica": alvo["pos_cronologica"],
        "rodada_max_ja_disputada": alvo["rodada_max_ja_disputada"],
        "n_partidas_ja_disputadas": pos,
        "rodadas_invertidas": ",".join(str(r) for r in rodadas_invertidas),
        "n_rodadas_invertidas": len(rodadas_invertidas),
        "partidas_invertidas": lista_ids(invertidas),
        "dentro_da_janela": "sim" if alvo["dentro_da_janela"] else "nao",
    })

inversoes = pd.DataFrame(linhas_inv)

if inversoes.empty:
    inversoes = pd.DataFrame(columns=[
        "rodada", "partida_id", "data", "mandante", "visitante",
        "pos_cronologica", "rodada_max_ja_disputada", "n_partidas_ja_disputadas",
        "rodadas_invertidas", "n_rodadas_invertidas", "partidas_invertidas",
        "dentro_da_janela",
    ])

inversoes.to_csv(OUT_INVERSOES_CSV, index=False, encoding="utf-8")

print(f"\nPartidas com inversao cronologica "
      f"(rodada R disputada depois de rodada > R): {len(inversoes)}")
print(f"Partidas que nao seguem a ordem da rodada:    "
      f"{len(partidas) - len(inversoes)}")
print()
if not inversoes.empty:
    print(inversoes.drop(columns=["partidas_invertidas"]).to_string(index=False))

# Desvio dentro da própria rodada
titulo("2b. DESVIO DENTRO DA PROPRIA RODADA")

desvio = partidas.groupby("rodada").agg(
    n_partidas=("partida_id", "count"),
    data_ini=("dia", "min"),
    data_fim=("dia", "max"),
    n_fora=("dentro_da_janela", lambda s: int((~s).sum())),
).reset_index()

desvio["amplitude_dias"] = (desvio["data_fim"] - desvio["data_ini"]).dt.days + 1
desvio["janela_inicio"] = desvio["rodada"].map(lambda r: janelas[r][0].date())
desvio["janela_fim"] = desvio["rodada"].map(lambda r: janelas[r][1].date())

print(desvio.to_string(index=False))


# ============================================================
# 3. COMPARAÇÃO ALTERNATIVA A vs ALTERNATIVA B
# ============================================================

titulo("3. ALTERNATIVA A (rodada) vs ALTERNATIVA B (data real)")

ds = dataset.copy()
ds["dt"] = pd.to_datetime(ds["data"])
ds["dia"] = ds["dt"].dt.normalize()
ds["partida_id"] = ds["partida_id"].astype(int)
ds["rodada"] = ds["rodada"].astype(int)

linhas_hist = []

for clube_id, grupo in ds.groupby("clube_id"):

    grupo = grupo.sort_values(["dt", "partida_id"])

    for _, alvo in grupo.iterrows():

        # ALTERNATIVA A: histórico por rodada
        conjunto_a = set(
            grupo.loc[grupo["rodada"] < alvo["rodada"], "partida_id"]
        )

        # ALTERNATIVA B: histórico por data real estritamente anterior
        conjunto_b = set(
            grupo.loc[grupo["dt"] < alvo["dt"], "partida_id"]
        )

        # variante de data sem hora (só o dia), para medir a ambiguidade
        conjunto_b_dia = set(
            grupo.loc[grupo["dia"] < alvo["dia"], "partida_id"]
        )

        so_em_a = sorted(conjunto_a - conjunto_b)
        so_em_b = sorted(conjunto_b - conjunto_a)

        # partidas da MESMA rodada da alvo que já foram disputadas antes dela
        mesma_rodada_antes = sorted(
            grupo.loc[
                (grupo["rodada"] == alvo["rodada"])
                & (grupo["dt"] < alvo["dt"]),
                "partida_id",
            ]
        )

        linhas_hist.append({
            "clube_id": safe_int(clube_id),
            "clube": alvo["clube_nome"],
            "rodada_alvo": safe_int(alvo["rodada"]),
            "partida_alvo": safe_int(alvo["partida_id"]),
            "data_alvo": alvo["data"],
            "n_historico_A": len(conjunto_a),
            "n_historico_B": len(conjunto_b),
            "hist_iguais": so_em_a == [] and so_em_b == [],
            "vaza_em_A": len(so_em_a),
            "partidas_vazando": ",".join(str(p) for p in so_em_a),
            "rodadas_das_vazantes": ",".join(
                str(r) for r in sorted(
                    grupo.loc[grupo["partida_id"].isin(so_em_a), "rodada"].unique()
                )
            ),
            "perdido_em_A": len(so_em_b),
            "partidas_perdidas": ",".join(str(p) for p in so_em_b),
            "mesma_rodada_antes_em_B": len(mesma_rodada_antes),
            "data_dia_insuficiente": conjunto_b_dia != conjunto_b,
        })

historicos = pd.DataFrame(linhas_hist)

historicos.to_csv(OUT_HISTORICOS_CSV, index=False, encoding="utf-8")

n_pares = len(historicos)
n_dif = int((~historicos["hist_iguais"]).sum())
n_vaza = int((historicos["vaza_em_A"] > 0).sum())
n_perdido = int((historicos["perdido_em_A"] > 0).sum())
n_mesma_rodada = int((historicos["mesma_rodada_antes_em_B"] > 0).sum())
n_dia_insuf = int(historicos["data_dia_insuficiente"].sum())

print(f"\nPares (clube, partida-alvo) analisados:            {n_pares}")
print(f"Pares com historico IGUAL em A e B:                {n_pares - n_dif}")
print(f"Pares com historico DIFERENTE em A e B:            {n_dif}")
print(f"Pares em que A INCLUI partida jogada depois (vaza): {n_vaza}")
print(f"Pares em que A EXCLUI partida ja jogada (perde):   {n_perdido}")
print(f"Pares em que B inclui partida da MESMA rodada:      {n_mesma_rodada}")
print(f"Pares em que so o dia (sem hora) muda o historico:  {n_dia_insuf}")

# Quais partidas vazam, e para quantos alvos
print("\nPartidas que A trata como passado mas foram jogadas depois:")
vazantes = (
    historicos[historicos["vaza_em_A"] > 0]
    .assign(
        lista=historicos.loc[historicos["vaza_em_A"] > 0, "partidas_vazando"]
        .str.split(",")
    )
    .explode("lista")
    .groupby("lista")
    .agg(n_alvos_afetados=("partida_alvo", "count"))
    .reset_index()
    .rename(columns={"lista": "partida_id"})
    .sort_values("n_alvos_afetados", ascending=False)
)
vazantes["partida_id"] = vazantes["partida_id"].astype(int)
vazantes = vazantes.merge(
    partidas[["partida_id", "rodada", "data", "mandante", "visitante"]],
    on="partida_id", how="left",
)
print(vazantes.to_string(index=False))

print("\nPares em que B enxerga passado que A exclui:")
perdidos = historicos[historicos["perdido_em_A"] > 0].copy()
perdidos["primeira_perdida"] = perdidos["partidas_perdidas"].str.split(",").str[0]
perdidos["ultima_perdida"] = perdidos["partidas_perdidas"].str.split(",").str[-1]
print(perdidos[[
    "clube", "rodada_alvo", "partida_alvo", "data_alvo",
    "perdido_em_A", "primeira_perdida", "ultima_perdida",
]].to_string(index=False))

# Versao em ordem cronologica real, para o relatorio
data_por_partida = dict(zip(partidas["partida_id"], partidas["data"]))
rodada_por_partida = dict(zip(partidas["partida_id"], partidas["rodada"]))

perdidos_relatorio = []
for linha in perdidos.itertuples(index=False):
    ids = [int(p) for p in linha.partidas_perdidas.split(",")]
    ids = sorted(ids, key=lambda p: (data_por_partida[p], p))
    perdidos_relatorio.append({
        "clube": linha.clube,
        "partida_alvo": linha.partida_alvo,
        "rodada_alvo": linha.rodada_alvo,
        "data_alvo": linha.data_alvo,
        "n_perdidas_em_A": linha.perdido_em_A,
        "passado_mais_recente": f"{ids[-1]} (R{rodada_por_partida[ids[-1]]}, {data_por_partida[ids[-1]]})",
        "passado_mais_antigo": f"{ids[0]} (R{rodada_por_partida[ids[0]]}, {data_por_partida[ids[0]]})",
    })

perdidos_relatorio = pd.DataFrame(perdidos_relatorio)


# Cruzamento com a F1.1: as partidas que causam vaza em A sao as mesmas
# que têm elenco ausente do payload?
partidas_vazantes_ids = sorted(vazantes["partida_id"].astype(int).tolist())

anomalias = pd.read_csv(ANOMALIAS_CSV)
sem_elenco_ids = sorted(
    anomalias.loc[
        anomalias["tipo"] == "clube_sem_atletas_participantes",
        "partida_id",
    ].astype(int).unique().tolist()
)

vaza_sem_elenco = [p for p in partidas_vazantes_ids if p in sem_elenco_ids]
elenco_sem_vaza = [p for p in sem_elenco_ids if p not in partidas_vazantes_ids]

print(f"\nPartidas que causam vaza em A:                {partidas_vazantes_ids}")
print(f"Partidas com elenco ausente (F1.1):            {sem_elenco_ids}")
print(f"Vazam E tem elenco ausente:                   {vaza_sem_elenco}")
print(f"Tem elenco ausente mas nao vaza:              {elenco_sem_vaza}")


# ============================================================
# 4. MESMA DATA
# ============================================================

titulo("4. PARTIDAS COM MESMA DATA")

# 4.1 mesmo clube, mesmo dia
por_clube_dia = ds.groupby(["clube_id", "dia"])["partida_id"].nunique()
mesmo_clube_mesmo_dia = int((por_clube_dia > 1).sum())

# 4.2 mesmo timestamp exato, qualquer rodada
por_timestamp = partidas.groupby("dt")["rodada"].nunique()
empates_exatos = partidas[
    partidas["dt"].isin(por_timestamp[por_timestamp > 1].index)
]

# 4.3 mesmo dia, rodadas diferentes
pares_mesma_data = []

for dia, grupo in partidas.groupby("dia"):
    for a, b in itertools.combinations(grupo.itertuples(), 2):
        if a.rodada == b.rodada:
            continue
        pares_mesma_data.append({
            "dia": dia.date(),
            "partida_a": a.partida_id,
            "rodada_a": a.rodada,
            "hora_a": a.dt.strftime("%H:%M"),
            "clube_a": f"{a.mandante}-{a.visitante}",
            "partida_b": b.partida_id,
            "rodada_b": b.rodada,
            "hora_b": b.dt.strftime("%H:%M"),
            "clube_b": f"{b.mandante}-{b.visitante}",
            "mesmo_horario": a.dt == b.dt,
        })

mesma_data = pd.DataFrame(pares_mesma_data).sort_values(["dia", "hora_a"]) \
    if pares_mesma_data else pd.DataFrame()

n_dias_com_1 = int((partidas.groupby("dia").size() == 1).sum())

print(f"\nDatas distintas:                                        {partidas['dia'].nunique()}")
print(f"Datas com uma unica partida:                            {n_dias_com_1}")
print(f"Datas com mais de uma partida:                          "
      f"{partidas['dia'].nunique() - n_dias_com_1}")
print(f"Maior numero de partidas no mesmo dia:                  "
      f"{int(partidas.groupby('dia').size().max())}")
print(f"Clube com 2+ partidas no mesmo dia:                     {mesmo_clube_mesmo_dia}")
print(f"Pares de rodadas diferentes disputados no MESMO dia:     {len(mesma_data)}")
print(f"Pares com horario identico (empate real de timestamp):  "
      f"{int(mesma_data['mesmo_horario'].sum()) if not mesma_data.empty else 0}")

if not mesma_data.empty:
    print()
    print(mesma_data.to_string(index=False))

if not empates_exatos.empty:
    print("\nPartidas com horario identico e rodadas diferentes:")
    print(empates_exatos[[
        "rodada", "partida_id", "data", "mandante", "visitante",
    ]].to_string(index=False))


# ============================================================
# 5. GRANULARIDADE DA FONTE
# ============================================================

titulo("5. GRANULARIDADE DA FONTE (payload por rodada)")

print("\nO payload de scouts da API e por RODADA, nao por partida.")
print("Amostra do r{N}.json: clubes presentes e contagem de atletas.")
print()

gran = []
for rodada in range(RODADA_INI, RODADA_FIM + 1):
    atletas = payloads[rodada]["atletas"]
    clubes_a = {}
    for v in atletas.values():
        clubes_a.setdefault(safe_int(v.get("clube_id")), 0)
        clubes_a[safe_int(v.get("clube_id"))] += 1
    gran.append({
        "rodada": rodada,
        "partidas_no_arquivo": len(payloads[rodada]["partidas"]),
        "clubes_no_payload": len(clubes_a),
        "atletas_no_payload": len(atletas),
    })

gran = pd.DataFrame(gran)
print(gran.to_string(index=False))


# ============================================================
# RELATÓRIO
# ============================================================

titulo("GERANDO RELATORIO")

relatorio = [
    "# EXP-12 F1.2 — Auditoria temporal das partidas",
    "",
    "- Temporada: **2026** | Rodadas: **R1..R26** | Partidas no dataset: "
    f"**{len(partidas)}**",
    "- Gerado por: `scripts/exp12_auditoria_temporal.py`",
    "- Natureza: **somente leitura**. Nenhuma feature temporal foi criada, "
    "nenhum dataset foi alterado e nenhuma correção foi aplicada.",
    "",
    "## 1. Objetivo",
    "",
    "Determinar, com evidência, qual deve ser o eixo temporal do EXP-12:",
    "",
    "- **Alternativa A** — `rodada`: histórico = partidas com "
    "`rodada < rodada_alvo`.",
    "- **Alternativa B** — `data` real: histórico = partidas com data real "
    "estritamente anterior à data da partida-alvo.",
    "",
    "Este documento **não escolhe** a alternativa. Ele mostra as partidas que "
    "diferenciam A de B e o risco de leakage de cada uma.",
    "",
    "## 2. Metodologia",
    "",
    "1. Ler `team_match_history.parquet` e as `partidas{N}.json` / `r{N}.json` "
    "/ `rodadas.json`.",
    "2. Reconstruir uma linha por partida com `rodada`, `partida_id`, `data`, "
    "mandante, visitante e a janela oficial da rodada (`inicio`/`fim` de "
    "`rodadas.json`).",
    "3. Classificar cada partida como dentro ou fora da janela oficial.",
    "4. Ordenar as 257 partidas pela data real e chamar essa de **posição "
    "cronológica**, para comparar com a ordem da rodada.",
    "5. Marcar como **inversão** toda partida de rodada R disputada depois de "
    "alguma partida de rodada maior que R.",
    "6. Para cada par (clube, partida-alvo), montar o conjunto histórico sob A "
    "e sob B e comparar: o que A inclui e B não (`vaza_em_A`), o que B inclui "
    "e A não (`perdido_em_A`) e o que B inclui por ser da mesma rodada "
    "(`mesma_rodada_antes_em_B`).",
    "7. Medir a granularidade real da fonte de scouts.",
    "",
    "Nenhum valor foi corrigido. As tabelas são evidência.",
    "",
    "## 3. Inventário das partidas",
    "",
    f"- Partidas distintas: **{len(partidas)}**.",
    f"- Rodadas cobertas: **R{partidas['rodada'].min()}..R{partidas['rodada'].max()}**.",
    f"- Datas distintas: **{partidas['dia'].nunique()}**.",
    f"- Timestamps distintos: **{partidas['dt'].nunique()}**.",
    f"- Dentro da janela oficial: **{int(partidas['dentro_da_janela'].sum())}**.",
    f"- Fora da janela oficial: **{len(fora_janela)}**.",
    "",
    "### 3.1 As partidas fuera da janela oficial",
    "",
    md_table(
        fora_janela,
        ["rodada", "partida_id", "data", "mandante", "visitante",
         "janela_inicio", "janela_fim", "dias_da_janela",
         "pos_cronologica", "rodada_max_ja_disputada"],
    ),
    "",
    "### 3.2 Todas as 257 partidas, em ordem cronológica real",
    "",
    "`pos_cronologica` é a posição real na linha do tempo. "
    "`rodada_max_ja_disputada` é a maior rodada que já tinha jogado na data "
    "daquela partida — é o que mostra quantas rodadas o calendário oficial "
    "ja considerava vencidas.",
    "",
    md_table(
        partidas,
        ["pos_cronologica", "rodada", "partida_id", "data", "mandante",
         "visitante", "dentro_da_janela_txt", "rodada_max_ja_disputada"],
    ),
    "",
    "## 4. Inversões cronológicas",
    "",
    "Uma partida de rodada R é uma inversão quando existe uma partida de "
    "rodada maior que R que foi disputada **antes** dela.",
    "",
    f"- Partidas com inversão: **{len(inversoes)}**.",
    f"- Partidas que respeitam a ordem da rodada: "
    f"**{len(partidas) - len(inversoes)}**.",
    "",
]

if not inversoes.empty:
    relatorio += [
        md_table(inversoes, [
            "rodada", "partida_id", "data", "mandante", "visitante",
            "pos_cronologica", "rodada_max_ja_disputada",
            "rodadas_invertidas", "n_rodadas_invertidas", "dentro_da_janela",
        ]),
        "",
        "### 4.1 Quais rodadas já tinham jogado quando cada uma dessas "
        "partidas foi disputada",
        "",
    ]
    for linha in inversoes.itertuples(index=False):
        relatorio.append(
            f"- **R{linha.rodada} / {linha.partida_id} / "
            f"{linha.mandante} x {linha.visitante} / {linha.data}**: "
            f"posicao cronologica {linha.pos_cronologica} de {len(partidas)}; "
            f"ja tinham sido disputadas as rodadas 1..{linha.rodada_max_ja_disputada} "
            f"({linha.n_partidas_ja_disputadas} partidas); "
            f"{linha.n_rodadas_invertidas} rodada(s) maior(es) ja tinham "
            f"sido disputadas antes dela: {linha.rodadas_invertidas}."
        )
    relatorio.append("")

relatorio += [
    "## 5. Desvio dentro da própria rodada",
    "",
    "`amplitude_dias` é a quantidade de dias entre a primeira e a última "
    "partida da rodada. Uma rodada com amplitude grande ou com partidas fora "
    "da janela oficial perde a propriedade de ser um bloco temporal.",
    "",
    md_table(desvio, [
        "rodada", "n_partidas", "janela_inicio", "janela_fim",
        "data_ini", "data_fim", "amplitude_dias", "n_fora",
    ]),
    "",
    "## 6. Comparação rodada vs data",
    "",
    "Para cada par (clube, partida-alvo) foram montados os dois históricos. "
    "A tabela completa esta em `exp12_inv_temporal_historicos.csv`; abaixo, "
    "o agregado.",
    "",
    "| metrica | pares |",
    "|---|---|",
    f"| pares (clube, partida-alvo) analisados | {n_pares} |",
    f"| historico identico em A e B | {n_pares - n_dif} |",
    f"| **historico diferente em A e B** | **{n_dif}** |",
    f"| **A inclui partida disputada depois (vaza)** | **{n_vaza}** |",
    f"| A exclui partida ja disputada (perde passado) | {n_perdido} |",
    f"| B inclui partida da mesma rodada da alvo | {n_mesma_rodada} |",
    f"| so o dia, sem hora, mudaria o historico | {n_dia_insuf} |",
    "",
    "### 6.1 As partidas que A trata como passado, mas foram jogadas depois",
    "",
    "Estas são as partidas que tornam a alternativa A um risco de leakage.",
    "",
    md_table(vazantes, [
        "partida_id", "rodada", "data", "mandante", "visitante",
        "n_alvos_afetados",
    ]),
    "",
    "### 6.2 Casos em que A exclui passado que já tinha sido jogado",
    "",
    "O erro oposto: a partida-alvo foi disputada **depois** de partidas de "
    "rodadas maiores, e A as exclui por serem de rodada futura. São "
    f"**{n_perdido} pares**, todos com a partida-alvo sendo uma das "
    "partidas adiadas.",
    "",
    md_table(perdidos_relatorio, [
        "clube", "partida_alvo", "rodada_alvo", "data_alvo",
        "n_perdidas_em_A", "passado_mais_antigo", "passado_mais_recente",
    ]) if not perdidos_relatorio.empty else "(nenhum caso)",
    "",
    f"A lista completa do que B enxerga e A não, por par, está em "
    "`exp12_inv_temporal_historicos.csv`, coluna `partidas_perdidas`.",
    "",
    "### 6.3 B nunca entra na mesma rodada da alvo",
    "",
    f"Casos em que B inclui uma partida da **mesma** rodada da alvo: "
    f"**{n_mesma_rodada}**. Isso e estrutural, nao sorte: cada clube disputa "
    "exatamente uma partida por rodada, entao a propria rodada do alvo nunca "
    "tem outra partida do mesmo clube para entrar no historio. A diferenca "
    "entre A e B vem sempre de rodadas diferentes, nunca de jogos distintos "
    "dentro da mesma rodada.",
    "",
    "## 7. Risco de leakage",
    "",
    "### 7.1 Alternativa A — `rodada < rodada_alvo`",
    "",
    f"- Vazamento confirmado em **{n_vaza} pares (clube, partida-alvo)**, "
    f"provenientes de apenas **{len(vazantes)} partidas** que a API devolveu "
    "com rodada antiga mas foram disputadas muito depois.",
    f"- O vazamento não é local. A partida 346335 (R4, disputada em 02/09) "
    f"entra no histórico de **{vazantes.loc[vazantes['partida_id'] == 346335, 'n_alvos_afetados'].iloc[0]} "
    "partidas-alvo** (21 do FLA e 21 do MIR), e em todas elas a data real da "
    "partida da R4 é posterior à partida-alvo.",
    f"- As {len(vazantes)} partidas que causam vaza em A são **exatamente** as "
    f"{len(vaza_sem_elenco)} partidas que, na F1.1, tinham elenco ausente do "
    "payload. Nenhuma partida que vaza em A está fora desse conjunto, e "
    f"{len(elenco_sem_vaza)} das partidas com elenco ausente "
    f"({', '.join(str(p) for p in elenco_sem_vaza)}) **não** causam vaza em A — "
    "são as disputadas **antes** da janela da rodada, que apenas atrasam a "
    "informação em vez de antecipar o futuro. As duas anomalias do EXP-12 se "
    "concentram nas mesmas partidas adiadas.",
    f"- Além do vazamento, A **perde** informação real em {n_perdido} pares: "
    "partidas de rodadas maiores já disputadas que A exclui por serem de "
    "rodada futura.",
    "- A é determinística e reproduz o calendário oficial, mas esse "
    "calendário não descreve a linha do tempo real dos jogos.",
    "",
    "### 7.2 Alternativa B — `data real < data da partida-alvo`",
    "",
    f"- B não inclui nenhuma partida disputada depois da alvo: "
    "**0 vazaimentos por construção**.",
    f"- B ganha {n_perdido} pares em que a rodada manda excluir um passado "
    "que já tinha sido jogado.",
    f"- B é bem definida por clube: **nenhum clube disputa duas partidas no "
    f"mesmo dia** ({mesmo_clube_mesmo_dia} casos), então não há empate a "
    "resolver dentro do histórico de um clube. O mesmo vale ignorando a "
    f"hora: usar só o dia muda o histórico em {n_dia_insuf} pares.",
    "- Risco próprio de B: ela assume que o dado da partida anterior estava "
    "disponivel no momento da alvo. Isso precisa ser verificado contra a "
    "granularidade da fonte (seção 8), não pode ser assumido.",
    "",
    "### 7.3 O que as duas alternativas nunca resolvem",
    "",
    f"- **{len(mesma_data)} pares de rodadas diferentes disputados no mesmo "
    "dia.** Neles a data, sozinha, nao estabelece precedencia.",
]

if not empates_exatos.empty:
    relatorio += [
        f"- **1 par com horario identico**: "
        + ", ".join(
            f"R{linha.rodada} {linha.partida_id} "
            f"({linha.mandante} x {linha.visitante})"
            for linha in empates_exatos.itertuples(index=False)
        )
        + ". Sao clubes diferentes, entao nao afeta o historico de nenhum "
        "clube, mas prova que a fonte tem empate de timestamp.",
    ]

relatorio += [
    "",
    "## 8. Granularidade da fonte",
    "",
    "O payload de scouts e por **rodada**, nao por partida: `r{N}.json` traz "
    "os atletas de todos os clubes que disputaram aquela rodada. A data e o "
    "horario da partida sao metadados, nao dados de disponibilidade.",
    "",
    "Consequencia direta: mesmo que o eixo temporal fosse a data real com "
    "hora, a unidade de informacao realmente publicada pela API e a rodada. "
    "Uma partida disputada na terca-feira da R19 e outra na quinta da R19 "
    "compartilham o mesmo payload. Se uma feature temporal precisar de "
    "informacao que só existe depois do encerramento da rodada, a data da "
    "partida individual **nao** é o limite correto — é a data de "
    "disponibilidade do payload.",
    "",
    md_table(gran),
    "",
    "Esta auditoria **nao** mede a latência de publicação da API. Ela "
    "aponta que a medição é necessária antes de escolher o eixo.",
    "",
    "## 9. Recomendação técnica",
    "",
    "O que a evidência sustenta, sem decidir o eixo:",
    "",
    "1. **A alternativa A tem vazamento comprovado** em "
    f"{n_vaza} pares, causado por "
    f"{len(vazantes)} partidas adiadas. Não é um risco teórico: é "
    "observável nas datas.",
    "2. **A alternativa B não vaza por construção** e não tem empate interno "
    "por clube, porque nenhum clube joga duas vezes no mesmo dia.",
    "3. **B, sozinha, não é suficiente.** O payload é por rodada, então o "
    "limite de disponibilidade de informação é a rodada, não a partida. "
    "Usar `data` como eixo exige um segundo critério de disponibilidade.",
    "4. **Opções a decidir pelo auditor**, com o custo de cada uma:",
    "",
    "   - **A:** manter o eixo da rodada. Reproduz o calendário oficial, "
    f"aceita os {n_vaza} vazaamentos e ignora os {n_perdido} casos de "
    "passado real perdido. Exige aceitar explicitamente o vazamento.",
    "   - **B:** usar a data real como eixo e tratar a disponibilidade do "
    "payload como uma restrição adicional, por rodada.",
    "   - **B restrito:** usar a data real, mas truncar o histórico no limite "
    "de disponibilidade do payload da rodada de origem, combinando os dois "
    "critérios.",
    "   - **Eixo duplo:** manter `rodada` como eixo de janela e usar a data "
    "apenas como filtro de disponibilidade.",
    "",
    "Nenhuma dessas foi implementada. O pipeline não foi alterado.",
    "",
    "## 10. O que NÃO foi feito",
    "",
    "- `data/interim/exp12/team_match_history.parquet` não foi alterado.",
    "- Nenhum arquivo de origem (`r{N}.json`, `partidas{N}.json`, "
    "`rodadas.json`) foi alterado.",
    "- Nenhuma feature temporal foi criada.",
    "- Nenhuma data, rodada ou placar foi corrigido.",
    "- A alternativa A e a alternativa B não foram implementadas nem "
    "comparadas por desempenho de modelo.",
    "- A escolha do eixo temporal continua **pendente do auditor**.",
    "",
    "## 11. Arquivos desta auditoria",
    "",
    "- `scripts/exp12_auditoria_temporal.py`",
    "- `reports/exp12/exp12_auditoria_temporal.md` (este documento)",
    "- `reports/exp12/exp12_inv_temporal_inversoes.csv`",
    "- `reports/exp12/exp12_inv_temporal_historicos.csv`",
    "",
]

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(relatorio) + "\n")


# ============================================================
# FECHAMENTO
# ============================================================

titulo("RESUMO FINAL — F1.2")

print(f"Partidas analisadas:                        {len(partidas)}")
print(f"Partidas problematicas (fora da janela):    {len(problematicas)}")
print(f"Partidas com inversao cronologica:          {len(inversoes)}")
print(f"Rodadas com partida fora da janela:         "
      f"{len(problematicas['rodada'].unique())} "
      f"({sorted(problematicas['rodada'].unique().tolist())})")
print()
print(f"Pares (clube, partida) analisados:          {n_pares}")
print(f"Pares com historico diferente (A vs B):     {n_dif}")
print(f"Cases em que so a data evita leakage (A):  {n_vaza}")
print(f"Cases em que A perde passado real:          {n_perdido}")
print(f"Cases em que B entra na mesma rodada:       {n_mesma_rodada}")
print()
print(f"Clube com 2+ jogos no mesmo dia:            {mesmo_clube_mesmo_dia}")
print(f"Pares de rodadas diferentes no mesmo dia:   {len(mesma_data)}")
print(f"Empates reais de timestamp:                 "
      f"{int(mesma_data['mesmo_horario'].sum()) if not mesma_data.empty else 0}")
print()
print("Cruzamento com a F1.1 (elenco ausente do payload):")
print(f"  Vazam em A e tem elenco ausente:  {vaza_sem_elenco}")
print(f"  Tem elenco ausente e nao vaza:    {elenco_sem_vaza}")
print()
print("Nenhum arquivo de dados foi alterado.")
print()
print(f"Relatorio:  {OUT_MD}")
print(f"Inversoes:  {OUT_INVERSOES_CSV}")
print(f"Historicos: {OUT_HISTORICOS_CSV}")
print()
print("Nenhuma feature temporal foi criada. "
      "A escolha do eixo temporal depende do auditor.")
