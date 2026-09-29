#!/usr/bin/env python3
"""
EXP-12 — Construção do histórico de partidas por equipe — Rodada 1

Fonte:
    experiments/data/cartola_api_2026/r1.json
    experiments/data/cartola_api_2026/partidas1.json

r1.json:
    scouts dos jogadores

partidas1.json:
    confrontos, mandante/visitante e placar oficial

Saída:
    data/interim/exp12/team_match_history_r1.parquet
    data/interim/exp12/team_match_history_r1.csv

SEMÂNTICA IMPORTANTE:
    GS = Gol Sofrido do goleiro (scout Cartola)
    SG = Sem Gol Sofrido / clean sheet (scout Cartola)
    gols_sofridos = gols sofridos pelo TIME no placar oficial

    Portanto:
        GS != gols_sofridos
        SG != gols_sofridos
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

API_DIR = ROOT / "experiments" / "data" / "cartola_api_2026"

R_JSON = API_DIR / "r1.json"
PARTIDAS_JSON = API_DIR / "partidas1.json"

OUT_DIR = ROOT / "data" / "interim" / "exp12"

OUT_PARQUET = OUT_DIR / "team_match_history_r1.parquet"
OUT_CSV = OUT_DIR / "team_match_history_r1.csv"


# ============================================================
# SCOUTS
# ============================================================

# Scouts que queremos agregar diretamente dos jogadores.
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


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value):
    """
    Converte valores numéricos da API para int.
    Valores ausentes/None viram 0.
    """
    if value is None:
        return 0

    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


# ============================================================
# LEITURA
# ============================================================

print("=" * 70)
print("EXP-12 — TEAM MATCH HISTORY — RODADA 1")
print("=" * 70)

print(f"\nR JSON:")
print(R_JSON)

print(f"\nPARTIDAS JSON:")
print(PARTIDAS_JSON)

r_data = load_json(R_JSON)
partidas_data = load_json(PARTIDAS_JSON)


# ============================================================
# CLUBES
# ============================================================

clubes_raw = partidas_data.get("clubes", {})

clubes = {}

for clube_id, clube in clubes_raw.items():
    clube_id_int = safe_int(clube_id)

    clubes[clube_id_int] = {
        "nome": clube.get("nome"),
        "abreviacao": clube.get("abreviacao"),
        "slug": clube.get("slug"),
        "apelido": clube.get("apelido"),
        "nome_fantasia": clube.get("nome_fantasia"),
        "id": safe_int(clube.get("id", clube_id_int)),
    }


print(f"\nClubes carregados: {len(clubes)}")


# ============================================================
# JOGADORES / SCOUTS
# ============================================================

atletas_raw = r_data.get("atletas", [])
if isinstance(atletas_raw, dict):
    atletas_raw = list(atletas_raw.values())

print(f"Atletas encontrados em r1.json: {len(atletas_raw)}")


# Agregação:
#
# clube_id -> scout -> soma
#
# Somente jogadores que realmente participaram são relevantes
# para os scouts daquela partida.

team_scouts = {}

participantes = 0

for atleta in atletas_raw:

    entrou = atleta.get("entrou_em_campo", False)

    if not entrou:
        continue

    clube_id = atleta.get("clube_id")

    if clube_id is None:
        continue

    clube_id = safe_int(clube_id)

    if clube_id not in team_scouts:
        team_scouts[clube_id] = {
            scout: 0
            for scout in DIRECT_SCOUTS
        }

    scout_data = atleta.get("scout") or {}

    for scout in DIRECT_SCOUTS:
        team_scouts[clube_id][scout] += safe_int(
            scout_data.get(scout, 0)
        )

    participantes += 1


print(f"Jogadores participantes: {participantes}")
print(f"Clubes com scouts: {len(team_scouts)}")


# ============================================================
# PARTIDAS
# ============================================================

partidas = partidas_data.get("partidas", [])

print(f"Partidas encontradas: {len(partidas)}")


rows = []


for partida in partidas:

    # --------------------------------------------------------
    # Identificação da partida
    # --------------------------------------------------------

    partida_id = partida.get("partida_id")

    rodada = safe_int(
        partida.get("rodada", partidas_data.get("rodada", 1))
    )

    data = partida.get("partida_data")

    # --------------------------------------------------------
    # Clubes
    # --------------------------------------------------------

    mandante_id = safe_int(
        partida.get("clube_casa_id")
    )

    visitante_id = safe_int(
        partida.get("clube_visitante_id")
    )

    if not mandante_id or not visitante_id:
        print(
            f"[WARN] Partida {partida_id} sem clubes válidos. "
            "Ignorando."
        )
        continue

    mandante = clubes.get(
        mandante_id,
        {"nome": f"clube_{mandante_id}"}
    )

    visitante = clubes.get(
        visitante_id,
        {"nome": f"clube_{visitante_id}"}
    )

    mandante_nome = mandante["nome"]
    visitante_nome = visitante["nome"]

    # --------------------------------------------------------
    # Placar oficial
    # --------------------------------------------------------

    gols_mandante = safe_int(
        partida.get("placar_oficial_mandante")
    )

    gols_visitante = safe_int(
        partida.get("placar_oficial_visitante")
    )

    # --------------------------------------------------------
    # Scouts agregados
    # --------------------------------------------------------

    mandante_scouts = team_scouts.get(
        mandante_id,
        {scout: 0 for scout in DIRECT_SCOUTS}
    )

    visitante_scouts = team_scouts.get(
        visitante_id,
        {scout: 0 for scout in DIRECT_SCOUTS}
    )

    # ========================================================
    # LINHA DO MANDANTE
    # ========================================================

    row_home = {
        "temporada": 2026,
        "rodada": rodada,
        "partida_id": partida_id,
        "data": data,

        "clube_id": mandante_id,
        "clube_nome": mandante_nome,

        "adversario_id": visitante_id,
        "adversario_nome": visitante_nome,

        "mando": "C",

        # Resultado da partida
        "gols_marcados": gols_mandante,
        "gols_sofridos": gols_visitante,
    }

    # Scouts do mandante
    for scout in DIRECT_SCOUTS:
        row_home[scout] = mandante_scouts[scout]

    # Finalizações derivadas
    row_home["FTT"] = (
        row_home["FD"]
        + row_home["FF"]
        + row_home["FT"]
        + row_home["G"]
        + row_home["PP"]
    )

    row_home["FA"] = (
        row_home["FD"]
        + row_home["G"]
    )

    rows.append(row_home)

    # ========================================================
    # LINHA DO VISITANTE
    # ========================================================

    row_away = {
        "temporada": 2026,
        "rodada": rodada,
        "partida_id": partida_id,
        "data": data,

        "clube_id": visitante_id,
        "clube_nome": visitante_nome,

        "adversario_id": mandante_id,
        "adversario_nome": mandante_nome,

        "mando": "F",

        # Resultado da partida
        "gols_marcados": gols_visitante,
        "gols_sofridos": gols_mandante,
    }

    # Scouts do visitante
    for scout in DIRECT_SCOUTS:
        row_away[scout] = visitante_scouts[scout]

    # Finalizações derivadas
    row_away["FTT"] = (
        row_away["FD"]
        + row_away["FF"]
        + row_away["FT"]
        + row_away["G"]
        + row_away["PP"]
    )

    row_away["FA"] = (
        row_away["FD"]
        + row_away["G"]
    )

    rows.append(row_away)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# ORDEM DAS COLUNAS
# ============================================================

columns = [
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

df = df[columns]


# ============================================================
# VALIDAÇÕES
# ============================================================

print("\n" + "=" * 70)
print("VALIDAÇÕES")
print("=" * 70)


# ------------------------------------------------------------
# 1. Duas linhas por partida
# ------------------------------------------------------------

counts = (
    df.groupby("partida_id")
      .size()
)

assert counts.eq(2).all(), (
    "ERRO: alguma partida não possui exatamente 2 linhas."
)

print("[OK] Exatamente 2 linhas por partida.")


# ------------------------------------------------------------
# 2. Quantidade de partidas
# ------------------------------------------------------------

n_partidas = df["partida_id"].nunique()

assert len(df) == n_partidas * 2

print(
    f"[OK] {n_partidas} partidas -> "
    f"{len(df)} linhas."
)


# ------------------------------------------------------------
# 3. Pares inversos
# ------------------------------------------------------------

for partida_id, group in df.groupby("partida_id"):

    if len(group) != 2:
        continue

    a = group.iloc[0]
    b = group.iloc[1]

    assert a["clube_id"] == b["adversario_id"]
    assert b["clube_id"] == a["adversario_id"]

print("[OK] Pares clube/adversário são inversos.")


# ------------------------------------------------------------
# 4. Simetria do placar
# ------------------------------------------------------------

for partida_id, group in df.groupby("partida_id"):

    if len(group) != 2:
        continue

    a = group.iloc[0]
    b = group.iloc[1]

    assert a["gols_marcados"] == b["gols_sofridos"]
    assert b["gols_marcados"] == a["gols_sofridos"]

print("[OK] Placar é simétrico.")


# ------------------------------------------------------------
# 5. FTT
# ------------------------------------------------------------

ftt_expected = (
    df["FD"]
    + df["FF"]
    + df["FT"]
    + df["G"]
    + df["PP"]
)

assert (
    df["FTT"] == ftt_expected
).all()

print(
    "[OK] FTT = FD + FF + FT + G + PP."
)


# ------------------------------------------------------------
# 6. FA
# ------------------------------------------------------------

fa_expected = (
    df["FD"]
    + df["G"]
)

assert (
    df["FA"] == fa_expected
).all()

print(
    "[OK] FA = FD + G."
)


# ------------------------------------------------------------
# 7. GS NÃO É gols_sofridos
# ------------------------------------------------------------

print(
    "\n[INFO] GS = scout de gol sofrido do goleiro."
)

print(
    "[INFO] gols_sofridos = gols sofridos pelo time "
    "no placar oficial."
)

print(
    "[INFO] SG = scout de sem gol sofrido."
)

assert "GS" in df.columns
assert "SG" in df.columns
assert "gols_sofridos" in df.columns

print(
    "[OK] GS, SG e gols_sofridos estão separados."
)


# ------------------------------------------------------------
# 8. NÃO existe clean_sheet derivado
# ------------------------------------------------------------

assert "clean_sheet" not in df.columns

print(
    "[OK] Nenhuma coluna clean_sheet artificial foi criada."
)


# ============================================================
# RESUMO
# ============================================================

print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)

print(f"Linhas: {len(df)}")
print(f"Partidas: {df['partida_id'].nunique()}")
print(f"Clubes: {df['clube_id'].nunique()}")
print(f"Colunas: {len(df.columns)}")

print("\nColunas:")
for col in df.columns:
    print(f"  - {col}")


# ============================================================
# AMOSTRA
# ============================================================

print("\n" + "=" * 70)
print("AMOSTRA")
print("=" * 70)

print(
    df[
        [
            "partida_id",
            "clube_nome",
            "adversario_nome",
            "mando",
            "gols_marcados",
            "gols_sofridos",
            "GS",
            "SG",
            "DP",
            "DS",
            "G",
            "A",
            "FTT",
            "FA",
        ]
    ].to_string(index=False)
)


# ============================================================
# SALVAR
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUT_PARQUET,
    index=False
)

df.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)

print(f"Parquet: {OUT_PARQUET}")
print(f"CSV:     {OUT_CSV}")

print("\nConcluído.")
