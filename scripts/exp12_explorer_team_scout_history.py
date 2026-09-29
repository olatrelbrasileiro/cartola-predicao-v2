#!/usr/bin/env python3
"""
CARTOLA AI INSIGHTS — EXP-12
Script Interativo de Exploração do Banco Mestre Factual de Scouts de Times.

Arquivo factual: data/interim/exp12/team_scout_history.parquet

Princípios:
- Somente consulta e exibição (read-only).
- Sem rolling windows, sem médias móveis e sem rankings preditivos.
- 1 linha = 1 time × 1 partida/rodada.
- Ordem temporal oficial: temporada + rodada + partida_id.
- Coluna `data` é puramente informativa.
- Descoberta dinâmica de temporadas, rodadas e clubes.
- Preservação estrita de NaNs e valores originais de SG e scouts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Caminho padrão oficial do banco Parquet
DEFAULT_PARQUET_PATH = Path("data/interim/exp12/team_scout_history.parquet")
HISTORICAL_PARQUET_PATH = Path("data/interim/exp12/team_scout_history_2024_2025.parquet")
# 33 Colunas esperadas no Parquet factual
EXPECTED_COLUMNS = [
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
    "SG",
    "DE",
    "DP",
    "DS",
    "GS",
    "FC",
    "G",
    "A",
    "FTT",
    "FA",
    "FS",
    "I",
    "FD",
    "FF",
    "FT",
    "PP",
    "roster_presente",
    "roster_jogadores_raw",
    "roster_tecnicos_raw",
    "n_participantes",
    "participantes_por_posicao",
    "participantes",
]

SCOUTS_PRINCIPAIS_DEF = ["SG", "DE", "DP", "DS", "GS", "FC"]
SCOUTS_PRINCIPAIS_OF = ["G", "A", "FTT", "FA", "FS", "I"]
SCOUTS_COMPONENTES = ["FD", "FF", "FT", "PP"]
SCOUTS_TABELA_HISTORICO = ["SG", "DE", "DP", "DS", "GS", "FC", "G", "A", "FTT", "FA", "FS", "I"]
TODOS_SCOUTS_COMPARACAO = [
    "gols_marcados",
    "gols_sofridos",
    "SG",
    "DE",
    "DP",
    "DS",
    "GS",
    "FC",
    "G",
    "A",
    "FTT",
    "FA",
    "FS",
    "I",
    "FD",
    "FF",
    "FT",
    "PP",
]


# ==============================================================================
# FUNÇÕES DE FORMATAÇÃO E PARSE
# ==============================================================================

def format_val(val: Any) -> str:
    """Formata valor preservando inteiros sem '.0' e mantendo NaN como '-'."""
    if pd.isna(val) or val is None:
        return "-"
    if isinstance(val, (bool, np.bool_)):
        return "Sim" if val else "Não"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if val.is_integer():
            return str(int(val))
        return f"{val:.2f}"
    return str(val)


def format_diff(val_a: Any, val_b: Any) -> str:
    """Calcula e formata diferença (B - A) preservando ausência de dados."""
    if pd.isna(val_a) or pd.isna(val_b) or val_a is None or val_b is None:
        return "-"
    try:
        diff = val_b - val_a
        if isinstance(diff, (float, np.floating)) and diff.is_integer():
            diff = int(diff)
        if diff > 0:
            return f"+{diff}"
        return str(diff)
    except Exception:
        return "-"


def parse_nested_field(val: Any) -> Any:
    """Faz o parse seguro de campos aninhados como JSON ou literal python."""
    if val is None:
        return None

    # Parquet histórico 2024-2025 pode entregar list<struct>
    # diretamente como lista/array, sem precisar de JSON parsing.
    if isinstance(val, (dict, list, tuple)):
        return val

    if hasattr(val, "tolist") and not isinstance(val, str):
        converted = val.tolist()
        if isinstance(converted, (list, tuple, dict)):
            return converted

    if pd.isna(val):
        return None
    if isinstance(val, str):
        v = val.strip()
        if not v:
            return None
        try:
            return json.loads(v)
        except Exception:
            pass
        try:
            return ast.literal_eval(v)
        except Exception:
            pass
    return val


def parse_participants(val: Any) -> List[Dict[str, Any]]:
    """Extrai lista de atletas participantes do campo 'participantes'."""
    parsed = parse_nested_field(val)
    if not isinstance(parsed, list):
        return []
    
    athletes = []
    for item in parsed:
        if isinstance(item, dict):
            # Prioriza chaves padrão de identificação
            p_id = item.get("atleta_id") or item.get("id") or item.get("atleta") or "-"
            nome = item.get("nome") or item.get("apelido") or item.get("nome_cartola") or "-"
            pos = item.get("posicao") or item.get("posicao_nome") or item.get("posicao_id") or "-"
            athletes.append({
                "id": str(p_id),
                "nome": str(nome),
                "posicao": str(pos),
            })
        else:
            athletes.append({
                "id": "-",
                "nome": str(item),
                "posicao": "-",
            })
    return athletes


def parse_positions_dict(val: Any) -> Dict[str, Any]:
    """Extrai dicionário do campo 'participantes_por_posicao'."""
    parsed = parse_nested_field(val)
    if isinstance(parsed, dict):
        return parsed
    return {}


def render_table(rows: List[List[Any]], headers: List[str]) -> str:
    """Gera tabela formatada e alinhada sem depender de ANSI/cores."""
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="simple")
    
    # Fallback robusto caso tabulate não esteja instalado
    str_headers = [str(h) for h in headers]
    str_rows = [[format_val(c) for c in row] for row in rows]
    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
            else:
                col_widths.append(len(cell))
                
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(str_headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(col_widths)))
    body_lines = [
        " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        for row in str_rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


# ==============================================================================
# CLASSE EXPLORER (CORE DE DADOS)
# ==============================================================================

class TeamScoutHistoryExplorer:
    """Gerencia a consulta e visualização factual dos dados de scouts."""

    def __init__(self, parquet_path: Union[str, Path] = DEFAULT_PARQUET_PATH):
        self.parquet_path = Path(parquet_path)
        self.df: pd.DataFrame = self._load_and_validate()

    def _load_and_validate(self) -> pd.DataFrame:
        """Valida a existência dos arquivos e as colunas esperadas."""

        if not self.parquet_path.is_file():
            # Tenta resolver relativo à raiz do projeto se chamado de subpasta
            alt_path = Path(__file__).resolve().parent.parent / self.parquet_path
            if alt_path.is_file():
                self.parquet_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Erro: Arquivo Parquet não encontrado no caminho: {self.parquet_path}"
                )

        historical_path = Path(
            "data/interim/exp12/team_scout_history_2024_2025.parquet"
        )

        if not historical_path.is_file():
            historical_path = (
                Path(__file__).resolve().parent.parent / historical_path
            )

        if not historical_path.is_file():
            raise FileNotFoundError(
                f"Erro: Arquivo Parquet histórico não encontrado no caminho: "
                f"{historical_path}"
            )

        df_2024_2025 = pd.read_parquet(
            historical_path,
            engine="pyarrow",
        )

        df_2026 = pd.read_parquet(
            self.parquet_path,
            engine="pyarrow",
        )

        df = pd.concat(
            [df_2024_2025, df_2026],
            ignore_index=True,
        )

        df = df.sort_values(
            ["temporada", "rodada", "partida_id", "clube_id"]
        ).reset_index(drop=True)

        # Validação estrita das 33 colunas esperadas
        missing_cols = [
            col for col in EXPECTED_COLUMNS
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Erro de validação no Parquet. "
                f"Colunas obrigatórias ausentes: {missing_cols}"
            )

        return df

    def get_seasons(self) -> List[int]:
        """Descobre dinamicamente todas as temporadas presentes."""
        seasons = sorted(self.df["temporada"].dropna().unique().astype(int).tolist())
        return seasons

    def get_rounds(self, temporada: Optional[int] = None) -> List[int]:
        """Descobre dinamicamente as rodadas disponíveis."""
        sub = self.df if temporada is None else self.df[self.df["temporada"] == temporada]
        rounds = sorted(sub["rodada"].dropna().unique().astype(int).tolist())
        return rounds

    def get_teams(self, temporada: Optional[int] = None) -> List[Tuple[int, str]]:
        """Descobre dinamicamente clubes (clube_id, clube_nome) ordenados por nome."""
        sub = self.df if temporada is None else self.df[self.df["temporada"] == temporada]
        teams_df = (
            sub[["clube_id", "clube_nome"]]
            .dropna(subset=["clube_id"])
            .drop_duplicates()
            .sort_values(by="clube_nome")
        )
        return [(int(row.clube_id), str(row.clube_nome)) for row in teams_df.itertuples()]

    def resolve_team(self, query: Union[str, int], temporada: Optional[int] = None) -> Optional[Tuple[int, str]]:
        """Resolve um time por ID numérico ou busca por nome (case-insensitive)."""
        teams = self.get_teams(temporada=temporada)
        q_str = str(query).strip().lower()

        # Busca por ID numérico
        if q_str.isdigit():
            team_id = int(q_str)
            for tid, tname in teams:
                if tid == team_id:
                    return (tid, tname)

        # Busca por nome exato
        for tid, tname in teams:
            if tname.strip().lower() == q_str:
                return (tid, tname)

        # Busca por prefixo ou substring
        matches = [t for t in teams if q_str in t[1].lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Se houver match que inicia com o termo, prioriza
            starts = [t for t in matches if t[1].lower().startswith(q_str)]
            if len(starts) == 1:
                return starts[0]
            return matches[0]

        return None

    def get_match_record(
        self, clube_id: int, rodada: int, temporada: int
    ) -> Optional[pd.Series]:
        """Obtém exatamente 1 linha (1 time × 1 rodada) no banco."""
        mask = (
            (self.df["temporada"] == temporada)
            & (self.df["rodada"] == rodada)
            & (self.df["clube_id"] == clube_id)
        )
        sub = self.df[mask]
        if sub.empty:
            return None
        return sub.iloc[0]

    def get_team_history(
        self, clube_id: int, max_rodada: Optional[int] = None, temporada: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retorna o histórico do time ordenado estritamente por temporada, rodada e partida_id.
        Inclui rodadas onde rodada <= max_rodada.
        """
        mask = self.df["clube_id"] == clube_id
        if temporada is not None:
            mask = mask & (self.df["temporada"] == temporada)
        if max_rodada is not None:
            mask = mask & (self.df["rodada"] <= max_rodada)

        history = self.df[mask].copy()
        # Ordem temporal oficial estrita
        history = history.sort_values(by=["temporada", "rodada", "partida_id"])
        return history

    def get_round_all_teams(self, rodada: int, temporada: int) -> pd.DataFrame:
        """Retorna todos os times da rodada ordenados alfabeticamente por clube_nome."""
        mask = (self.df["temporada"] == temporada) & (self.df["rodada"] == rodada)
        sub = self.df[mask].copy()
        sub = sub.sort_values(by="clube_nome")
        return sub

    def compare_team_rounds(
        self, clube_id: int, rodada_a: int, rodada_b: int, temporada: int
    ) -> Optional[Dict[str, Any]]:
        """Compara duas rodadas do mesmo time preservando os valores originais."""
        rec_a = self.get_match_record(clube_id, rodada_a, temporada)
        rec_b = self.get_match_record(clube_id, rodada_b, temporada)

        if rec_a is None or rec_b is None:
            return None

        diffs = []
        for scout in TODOS_SCOUTS_COMPARACAO:
            val_a = rec_a.get(scout)
            val_b = rec_b.get(scout)
            diff_str = format_diff(val_a, val_b)
            diffs.append({
                "scout": scout,
                "val_a": val_a,
                "val_b": val_b,
                "diff": diff_str,
            })

        return {
            "record_a": rec_a,
            "record_b": rec_b,
            "diffs": diffs,
        }


# ==============================================================================
# FORMATADORES DE VISUALIZAÇÃO DE TERMINAL
# ==============================================================================

def format_placar_context(record: pd.Series) -> str:
    """Formata o placar factual respeitando o mando de campo."""
    clube = record["clube_nome"]
    adv = record["adversario_nome"]
    gp = format_val(record["gols_marcados"])
    gc = format_val(record["gols_sofridos"])
    mando = record["mando"]

    if mando == "C":
        return f"{clube} {gp} × {gc} {adv}"
    elif mando == "F":
        return f"{adv} {gc} × {gp} {clube}"
    return f"{clube} {gp} × {gc} {adv} (Mando: {mando})"


def display_option_1(record: pd.Series) -> None:
    """Exibe estatísticas completas de um time em uma rodada."""
    clube_nome = str(record["clube_nome"]).upper()
    rodada = record["rodada"]
    temporada = record["temporada"]

    print("\n" + "=" * 50)
    print(f" {clube_nome}")
    print(f" Rodada {rodada} — {temporada}")
    print("=" * 50)

    print(f"\nAdversário: {record['adversario_nome']}")
    print(f"Mando: {record['mando']}")
    print(f"Partida ID: {record['partida_id']}")
    print(f"Data: {record['data']}")

    print("\nPLACAR")
    print(format_placar_context(record))

    print("\n---------------- DEFENSIVOS ----------------")
    for s in SCOUTS_PRINCIPAIS_DEF:
        print(f"{s:<5}: {format_val(record[s])}")

    print("\n---------------- OFENSIVOS -----------------")
    for s in SCOUTS_PRINCIPAIS_OF:
        print(f"{s:<5}: {format_val(record[s])}")

    print("\n------------- COMPONENTES FINAIS -----------")
    for s in SCOUTS_COMPONENTES:
        print(f"{s:<5}: {format_val(record[s])}")

    print("\n----------------- ELENCO -------------------")
    roster_pres = bool(record["roster_presente"])
    print(f"Roster presente: {'Sim' if roster_pres else 'Não (ausência de roster bruto)'}")
    print(f"Jogadores no raw: {format_val(record['roster_jogadores_raw'])}")
    print(f"Técnicos no raw: {format_val(record['roster_tecnicos_raw'])}")
    print(f"Participantes: {format_val(record['n_participantes'])}")

    print("\nParticipantes por posição:")
    pos_dict = parse_positions_dict(record["participantes_por_posicao"])
    if pos_dict:
        for pos, count in pos_dict.items():
            print(f"{pos}: {format_val(count)}")
    else:
        print("Nenhum dado por posição registrado.")
    print("-" * 50)


def display_option_2_and_3(history_df: pd.DataFrame, time_nome: str, temporada: int) -> None:
    """Exibe o histórico factual e a matriz de evolução sequencial."""
    if history_df.empty:
        print("\nNenhuma partida encontrada para os critérios informados.")
        return

    print("\n" + "=" * 80)
    print(f" HISTÓRICO FACTUAL: {time_nome.upper()} ({temporada})")
    print("=" * 80)

    headers = ["Rodada", "Adversário", "Mando", "GP", "GC"] + SCOUTS_TABELA_HISTORICO
    table_rows = []

    for row in history_df.itertuples():
        r_str = f"R{int(row.rodada)}"
        vals = [
            r_str,
            str(row.adversario_nome),
            str(row.mando),
            format_val(row.gols_marcados),
            format_val(row.gols_sofridos),
        ] + [format_val(getattr(row, s)) for s in SCOUTS_TABELA_HISTORICO]
        table_rows.append(vals)

    print("\n" + render_table(table_rows, headers))

    # Seção Evolução dos Scouts (layout transposto rodada a rodada)
    print("\n" + "-" * 80)
    print("EVOLUÇÃO DOS SCOUTS")
    print("-" * 80)

    round_headers = ["Scout"] + [f"R{int(r)}" for r in history_df["rodada"]]
    scouts_evol = ["DS", "DE", "DP", "GS", "FC", "SG", "G", "A", "FTT", "FA", "FS", "I"]
    evol_rows = []

    for s in scouts_evol:
        row_vals = [s] + [format_val(val) for val in history_df[s]]
        evol_rows.append(row_vals)

    print(render_table(evol_rows, round_headers))

    print("\nGols marcados:")
    for row in history_df.itertuples():
        print(f"R{int(row.rodada)}: {format_val(row.gols_marcados)}")

    print("\nGols sofridos:")
    for row in history_df.itertuples():
        print(f"R{int(row.rodada)}: {format_val(row.gols_sofridos)}")
    print("=" * 80)


def display_option_4(round_df: pd.DataFrame, rodada: int, temporada: int) -> None:
    """Exibe todos os times em uma determinada rodada em ordem alfabética."""
    if round_df.empty:
        print(f"\nNenhum dado encontrado para a Rodada {rodada} ({temporada}).")
        return

    print("\n" + "=" * 90)
    print(f" RODADA {rodada} — TEMPORADA {temporada} (TODOS OS TIMES)")
    print("=" * 90)

    headers = ["Clube", "Adversário", "Mando", "GP", "GC"] + SCOUTS_TABELA_HISTORICO
    table_rows = []

    for row in round_df.itertuples():
        vals = [
            str(row.clube_nome),
            str(row.adversario_nome),
            str(row.mando),
            format_val(row.gols_marcados),
            format_val(row.gols_sofridos),
        ] + [format_val(getattr(row, s)) for s in SCOUTS_TABELA_HISTORICO]
        table_rows.append(vals)

    print("\n" + render_table(table_rows, headers))
    print("=" * 90)


def display_option_5(record: pd.Series) -> None:
    """Exibe os atletas presentes estritamente na coluna 'participantes'."""
    clube = str(record["clube_nome"]).upper()
    rodada = record["rodada"]
    temporada = record["temporada"]

    print("\n" + "=" * 60)
    print(f" DETALHES DO ELENCO: {clube}")
    print(f" Rodada {rodada} — {temporada}")
    print("=" * 60)
    print(f"Partida: {format_placar_context(record)}")
    print(f"Roster presente: {'Sim' if record['roster_presente'] else 'Não'}")
    print(f"Total participantes: {format_val(record['n_participantes'])}")

    athletes = parse_participants(record["participantes"])

    print("\nATLETAS PARTICIPANTES:")
    if athletes:
        p_headers = ["ID", "Nome", "Posição"]
        p_rows = [[a["id"], a["nome"], a["posicao"]] for a in athletes]
        print(render_table(p_rows, p_headers))
    else:
        print("Nenhum atleta detalhado encontrado na coluna factual 'participantes'.")

    print("\nPARTICIPANTES POR POSIÇÃO (armazenado no banco):")
    pos_stored = parse_positions_dict(record["participantes_por_posicao"])
    if pos_stored:
        for pos, count in pos_stored.items():
            print(f"  {pos}: {format_val(count)}")
    else:
        print("  Sem agregação factual de posições armazenada.")

    if athletes:
        print("\nCONTAGEM VERIFICADA NA LISTA DE PARTICIPANTES:")
        counts: Dict[str, int] = {}
        for a in athletes:
            pos = a["posicao"]
            counts[pos] = counts.get(pos, 0) + 1
        for pos, cnt in sorted(counts.items()):
            print(f"  {pos}: {cnt}")
    print("=" * 60)


def display_option_6(comp: Dict[str, Any]) -> None:
    """Exibe comparação factual entre duas rodadas do mesmo time."""
    rec_a = comp["record_a"]
    rec_b = comp["record_b"]

    clube = str(rec_a["clube_nome"]).upper()
    ra = rec_a["rodada"]
    rb = rec_b["rodada"]
    temp = rec_a["temporada"]

    print("\n" + "=" * 60)
    print(f" COMPARAÇÃO DE RODADAS: {clube} ({temp})")
    print("=" * 60)
    print(f"Rodada {ra}: Adversário: {rec_a['adversario_nome']} | Mando: {rec_a['mando']} | Placar: {format_placar_context(rec_a)}")
    print(f"Rodada {rb}: Adversário: {rec_b['adversario_nome']} | Mando: {rec_b['mando']} | Placar: {format_placar_context(rec_b)}")
    print("-" * 60)

    headers = ["Scout", f"Rodada {ra}", f"Rodada {rb}", "Diferença (B - A)"]
    rows = []
    label_map = {
        "gols_marcados": "GP (Gols Pró)",
        "gols_sofridos": "GC (Gols Contra)",
    }

    for item in comp["diffs"]:
        scout_label = label_map.get(item["scout"], item["scout"])
        rows.append([
            scout_label,
            format_val(item["val_a"]),
            format_val(item["val_b"]),
            item["diff"],
        ])

    print(render_table(rows, headers))
    print("=" * 60)


# ==============================================================================
# MENU INTERATIVO CLI
# ==============================================================================

def prompt_team_selection(explorer: TeamScoutHistoryExplorer, temporada: int) -> Optional[Tuple[int, str]]:
    """Apresenta lista numerada dinâmica de clubes e aceita seleção por número, ID ou nome."""
    teams = explorer.get_teams(temporada=temporada)
    if not teams:
        print("Nenhum clube encontrado nesta temporada.")
        return None

    print("\nClubes disponíveis:")
    for idx, (tid, tname) in enumerate(teams, start=1):
        print(f" {idx:>2}. {tname} (ID: {tid})")

    entrada = input("\nEscolha o número, nome ou ID do clube: ").strip()
    if not entrada:
        return None

    if entrada.isdigit():
        num = int(entrada)
        if 1 <= num <= len(teams):
            return teams[num - 1]

    resolved = explorer.resolve_team(entrada, temporada=temporada)
    if resolved:
        return resolved

    print(f"Clube '{entrada}' não encontrado.")
    return None


def prompt_season_selection(explorer: TeamScoutHistoryExplorer) -> int:
    """Permite selecionar temporada dinamicamente caso haja mais de uma."""
    seasons = explorer.get_seasons()
    if not seasons:
        raise RuntimeError("Nenhuma temporada encontrada no arquivo Parquet.")
    if len(seasons) == 1:
        return seasons[0]

    print("\nTemporadas disponíveis: " + ", ".join(str(s) for s in seasons))
    while True:
        choice = input(f"Selecione a temporada [{seasons[-1]}]: ").strip()
        if not choice:
            return seasons[-1]
        if choice.isdigit() and int(choice) in seasons:
            return int(choice)
        print("Temporada inválida. Tente novamente.")


def main() -> None:
    """Loop principal interativo do terminal."""
    try:
        explorer = TeamScoutHistoryExplorer()
    except Exception as e:
        print(f"\n[ERRO FATAL AO INICIAR] {e}")
        return

    seasons = explorer.get_seasons()
    temporada_atual = seasons[-1] if seasons else 2026

    while True:
        rounds = explorer.get_rounds(temporada=temporada_atual)
        min_r = rounds[0] if rounds else 0
        max_r = rounds[-1] if rounds else 0

        print("\n" + "=" * 50)
        print(" CARTOLA AI INSIGHTS")
        print(" EXP-12 — TEAM SCOUT HISTORY EXPLORER")
        print("=" * 50)
        print(f"Temporada ativa: {temporada_atual} (Disponíveis: {', '.join(str(s) for s in seasons)})")
        print(f"Rodadas disponíveis: {min_r}–{max_r}")
        print("\n1. Ver estatísticas de um time em uma rodada")
        print("2. Ver histórico de um time até uma rodada")
        print("3. Ver histórico completo de um time")
        print("4. Ver rodada de todos os times")
        print("5. Ver detalhes dos jogadores de um time em uma rodada")
        print("6. Comparar duas rodadas do mesmo time")
        print("7. Mudar temporada ativa")
        print("0. Sair")

        escolha = input("\nEscolha: ").strip()

        if escolha == "0":
            print("\nEncerrando explorer. Até logo!")
            break

        elif escolha == "7":
            temporada_atual = prompt_season_selection(explorer)

        elif escolha == "1":
            team = prompt_team_selection(explorer, temporada_atual)
            if not team:
                continue
            r_str = input(f"Informe a rodada ({min_r}–{max_r}): ").strip()
            if not r_str.isdigit():
                print("Rodada inválida.")
                continue
            rodada = int(r_str)
            rec = explorer.get_match_record(team[0], rodada, temporada_atual)
            if rec is None:
                print(f"\nNenhum registro para {team[1]} na Rodada {rodada} ({temporada_atual}).")
            else:
                display_option_1(rec)

        elif escolha == "2":
            team = prompt_team_selection(explorer, temporada_atual)
            if not team:
                continue
            r_str = input(f"Histórico até a rodada ({min_r}–{max_r}): ").strip()
            if not r_str.isdigit():
                print("Rodada inválida.")
                continue
            max_r_escolhida = int(r_str)
            history = explorer.get_team_history(team[0], max_rodada=max_r_escolhida, temporada=temporada_atual)
            display_option_2_and_3(history, team[1], temporada_atual)

        elif escolha == "3":
            team = prompt_team_selection(explorer, temporada_atual)
            if not team:
                continue
            history = explorer.get_team_history(team[0], max_rodada=None, temporada=temporada_atual)
            display_option_2_and_3(history, team[1], temporada_atual)

        elif escolha == "4":
            r_str = input(f"Informe a rodada para listar todos os times ({min_r}–{max_r}): ").strip()
            if not r_str.isdigit():
                print("Rodada inválida.")
                continue
            rodada = int(r_str)
            all_teams = explorer.get_round_all_teams(rodada, temporada_atual)
            display_option_4(all_teams, rodada, temporada_atual)

        elif escolha == "5":
            team = prompt_team_selection(explorer, temporada_atual)
            if not team:
                continue
            r_str = input(f"Informe a rodada ({min_r}–{max_r}): ").strip()
            if not r_str.isdigit():
                print("Rodada inválida.")
                continue
            rodada = int(r_str)
            rec = explorer.get_match_record(team[0], rodada, temporada_atual)
            if rec is None:
                print(f"\nNenhum registro para {team[1]} na Rodada {rodada} ({temporada_atual}).")
            else:
                display_option_5(rec)

        elif escolha == "6":
            team = prompt_team_selection(explorer, temporada_atual)
            if not team:
                continue
            ra_str = input(f"Informe a Rodada A ({min_r}–{max_r}): ").strip()
            rb_str = input(f"Informe a Rodada B ({min_r}–{max_r}): ").strip()
            if not ra_str.isdigit() or not rb_str.isdigit():
                print("Rodadas inválidas.")
                continue
            ra, rb = int(ra_str), int(rb_str)
            comp = explorer.compare_team_rounds(team[0], ra, rb, temporada_atual)
            if comp is None:
                print(f"\nNão foi possível comparar: dados ausentes para a Rodada {ra} ou {rb}.")
            else:
                display_option_6(comp)

        else:
            print("Opção inválida. Digite um número de 0 a 7.")


if __name__ == "__main__":
    main()
