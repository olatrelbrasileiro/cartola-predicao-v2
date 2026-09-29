#!/usr/bin/env python3
"""
Cartola AI Insights - Exp12: Construção do Banco Mestre Factual de Scouts de Times.

Objetivo:
- Consolidar a base factual oficial (F1) por time/partida/rodada a partir de data/interim/exp12/team_match_history.parquet.
- Extrair dos JSONs brutos (experiments/data/cartola_api_2026/r*.json) as informações factuais de elenco e participantes
  que efetivamente entraram em campo (excluindo técnicos e banco não utilizado).
- Auditar identidades de scouts (FTT, FA, G vs gols_marcados, GS vs gols_sofridos).
- Gerar:
    - data/interim/exp12/team_scout_history.parquet (schema list<struct> via PyArrow)
    - reports/exp12/team_scout_history.csv (participantes serializados como JSON)
- Não realizar qualquer pré-processamento de ML ou computação com janelas móveis/vazamento temporal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exp12_builder")

# Schema PyArrow Oficial para Participantes
PARTICIPANTE_STRUCT_TYPE = pa.struct([
    ("atleta_id", pa.int64()),
    ("atleta_nome", pa.string()),
    ("posicao_id", pa.int64()),
    ("posicao_nome", pa.string()),
])

PARTICIPANTES_LIST_TYPE = pa.list_(PARTICIPANTE_STRUCT_TYPE)

# Schema PyArrow do Parquet Mestre
OUTPUT_PYARROW_SCHEMA = pa.schema([
    ("temporada", pa.int64()),
    ("rodada", pa.int64()),
    ("partida_id", pa.int64()),
    ("data", pa.string()),
    ("clube_id", pa.int64()),
    ("clube_nome", pa.string()),
    ("adversario_id", pa.int64()),
    ("adversario_nome", pa.string()),
    ("mando", pa.string()),
    ("gols_marcados", pa.int64()),
    ("gols_sofridos", pa.int64()),
    ("SG", pa.bool_()),
    ("DE", pa.float64()),
    ("DP", pa.float64()),
    ("DS", pa.float64()),
    ("GS", pa.float64()),
    ("FC", pa.float64()),
    ("G", pa.float64()),
    ("A", pa.float64()),
    ("FTT", pa.float64()),
    ("FA", pa.float64()),
    ("FS", pa.float64()),
    ("I", pa.float64()),
    ("FD", pa.float64()),
    ("FF", pa.float64()),
    ("FT", pa.float64()),
    ("PP", pa.float64()),
    ("roster_presente", pa.bool_()),
    ("roster_jogadores_raw", pa.int64()),
    ("roster_tecnicos_raw", pa.int64()),
    ("n_participantes", pa.int64()),
    ("participantes_por_posicao", pa.string()),
    ("participantes", PARTICIPANTES_LIST_TYPE),
])


def compute_sha256(filepath: Path) -> str:
    """Calcula o hash SHA256 do arquivo em disco."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class CartolaRawExtractor:
    """
    Responsável por inspecionar e extrair de forma factual e defensiva
    os elencos e participantes reais dos JSONs da API do Cartola.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.cache: Dict[int, Dict[str, Any]] = {}

    def _load_round_json(self, rodada: int) -> Optional[Dict[str, Any]]:
        if rodada in self.cache:
            return self.cache[rodada]

        # Busca pelo arquivo r{rodada}.json ou variações de casing
        target_path = self.raw_dir / f"r{rodada}.json"
        if not target_path.exists():
            # Tenta buscar com zeros à esquerda ou maiúsculo se necessário
            alt_path = self.raw_dir / f"R{rodada}.json"
            if alt_path.exists():
                target_path = alt_path
            else:
                logger.warning(f"Arquivo bruto para rodada {rodada} não encontrado em {self.raw_dir}")
                self.cache[rodada] = None
                return None

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.cache[rodada] = data
                return data
        except Exception as e:
            logger.error(f"Erro ao abrir {target_path}: {e}")
            self.cache[rodada] = None
            return None

    @staticmethod
    def _is_coach(posicao_id: Optional[int], posicao_nome: Optional[str]) -> bool:
        """Determina com certeza se o registro se refere a técnico/comissão técnica."""
        if posicao_id == 6:
            return True
        if posicao_nome and any(term in posicao_nome.lower() for term in ["técnico", "tecnico", "treinador"]):
            return True
        return False

    @staticmethod
    def _player_entered(item: Dict[str, Any]) -> bool:
        """
        Determina se o atleta efetivamente participou da partida.
        Verifica os campos reais da API do Cartola:
        - 'entrou_em_campo': booleano explícito
        - 'jogou': booleano alternativo em certas versões da API
        - scouts presentes e não vazios
        - pontuacao diferente de zero ou scout ativo quando for endpoint de pontuados
        """
        if "entrou_em_campo" in item:
            val = item["entrou_em_campo"]
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return val > 0
            if isinstance(val, str):
                return val.lower() in ("true", "1", "sim", "yes")

        if "jogou" in item:
            val = item["jogou"]
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return val > 0

        # Se for payload de /atletas/pontuados, o fato de o atleta constar com scouts ou pontuação
        # e não ser banco não utilizado confirma entrada em campo
        scout = item.get("scout")
        if isinstance(scout, dict) and len(scout) > 0:
            return True

        pontos = item.get("pontos_num", item.get("pontuacao"))
        # Se tem pontuação registrada e diferente de None (inclusive pontuação negativa por cartões/faltas)
        if pontos is not None:
            # Jogadores de banco que não entraram ficam sem pontuação ou com pontuação nula/zero e sem scouts
            if isinstance(scout, dict) and len(scout) == 0 and float(pontos) == 0.0:
                return False
            return True

        return False

    def get_team_roster_facts(
        self, rodada: int, clube_id: int, partida_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retorna as métricas e a lista estruturada de participantes para o time na rodada.
        """
        data = self._load_round_json(rodada)
        if not data:
            return {
                "roster_presente": False,
                "roster_jogadores_raw": 0,
                "roster_tecnicos_raw": 0,
                "n_participantes": 0,
                "participantes_por_posicao": "{}",
                "participantes": [],
            }

        # Dicionário de posições da rodada (posicoes: {id: {nome, abreviacao, ...}})
        posicoes_map: Dict[int, str] = {}
        raw_posicoes = data.get("posicoes", {})
        if isinstance(raw_posicoes, dict):
            for k, v in raw_posicoes.items():
                try:
                    pid = int(k)
                    pnome = v.get("nome", v.get("abreviacao", str(pid))) if isinstance(v, dict) else str(v)
                    posicoes_map[pid] = pnome
                except (ValueError, TypeError):
                    continue

        # Extração de Atletas do JSON
        raw_athletes: List[Dict[str, Any]] = []
        raw_athletes_obj = data.get("atletas")

        if isinstance(raw_athletes_obj, dict):
            # Formato { "12345": { ... }, "67890": { ... } }
            for aid_key, item in raw_athletes_obj.items():
                if isinstance(item, dict):
                    copied = dict(item)
                    if "atleta_id" not in copied:
                        try:
                            copied["atleta_id"] = int(aid_key)
                        except ValueError:
                            pass
                    raw_athletes.append(copied)
        elif isinstance(raw_athletes_obj, list):
            # Formato [ { ... }, { ... } ]
            raw_athletes = [a for a in raw_athletes_obj if isinstance(a, dict)]

        # Filtra os atletas pertencentes ao clube_id
        clube_records: List[Dict[str, Any]] = []
        for a in raw_athletes:
            cid = a.get("clube_id")
            if cid is not None and int(cid) == int(clube_id):
                clube_records.append(a)

        raw_jogadores_count = 0
        raw_tecnicos_count = 0
        participantes: List[Dict[str, Any]] = []
        posicoes_counter: Counter[str] = Counter()
        seen_atleta_ids: Set[int] = set()

        for rec in clube_records:
            pos_id_raw = rec.get("posicao_id")
            pos_id = int(pos_id_raw) if pos_id_raw is not None else None
            pos_nome = (
                rec.get("posicao_nome")
                or posicoes_map.get(pos_id)
                or rec.get("posicao")
                or (str(pos_id) if pos_id is not None else "DESCONHECIDA")
            )

            is_coach = self._is_coach(pos_id, pos_nome)
            if is_coach:
                raw_tecnicos_count += 1
                continue

            raw_jogadores_count += 1

            if self._player_entered(rec):
                atleta_id = int(rec.get("atleta_id") or rec.get("id"))
                if atleta_id in seen_atleta_ids:
                    continue
                seen_atleta_ids.add(atleta_id)

                atleta_nome = (
                    rec.get("apelido")
                    or rec.get("nome")
                    or rec.get("nome_cartola")
                    or f"Atleta_{atleta_id}"
                )

                participantes.append({
                    "atleta_id": atleta_id,
                    "atleta_nome": str(atleta_nome),
                    "posicao_id": int(pos_id) if pos_id is not None else -1,
                    "posicao_nome": str(pos_nome),
                })
                posicoes_counter[str(pos_id)] += 1

        # Ordenar participantes determinísticamente por atleta_id
        participantes.sort(key=lambda x: x["atleta_id"])

        # Identificação de anomalia de elenco
        roster_presente = raw_jogadores_count > 0

        # DE oficial derivado diretamente do scout bruto dos goleiros
        # que efetivamente entraram em campo.
        #
        # Regra:
        # - posicao_id == 1 -> Goleiro
        # - entrou_em_campo == True
        # - scout["DE"] é somado por clube/rodada
        #
        # Não inferir DE a partir de pontuação ou de outras estatísticas.
        de_raw = 0.0

        for rec in clube_records:
            pos_id_raw = rec.get("posicao_id")
            pos_id = int(pos_id_raw) if pos_id_raw is not None else None

            if pos_id != 1:
                continue

            if not self._player_entered(rec):
                continue

            scout = rec.get("scout")
            if not isinstance(scout, dict):
                continue

            value = scout.get("DE", 0)

            try:
                de_raw += float(value or 0.0)
            except (TypeError, ValueError):
                continue

        # Formatação ordenada de participantes_por_posicao
        pos_summary = {
            k: posicoes_counter[k]
            for k in sorted(
                posicoes_counter.keys(),
                key=lambda x: int(x) if x.isdigit() else x
            )
        }

        return {
            "roster_presente": bool(roster_presente),
            "roster_jogadores_raw": int(raw_jogadores_count),
            "roster_tecnicos_raw": int(raw_tecnicos_count),
            "n_participantes": len(participantes),
            "participantes_por_posicao": json.dumps(
                pos_summary,
                separators=(",", ":")
            ),
            "participantes": participantes,
            "DE_raw": float(de_raw),
        }


def build_team_scout_history(
    input_parquet: Path,
    raw_api_dir: Path,
    output_parquet: Path,
    output_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executa a consolidação factual determinística.
    """
    if not input_parquet.exists():
        raise FileNotFoundError(f"Arquivo fonte obrigatório não encontrado: {input_parquet}")

    logger.info(f"Lendo base factual F1: {input_parquet}")
    df_f1 = pd.read_parquet(input_parquet)
    logger.info(f"Base factual F1 carregada com {len(df_f1)} linhas.")

    extractor = CartolaRawExtractor(raw_api_dir)

    rows: List[Dict[str, Any]] = []

    # Métricas para relatório de auditoria
    ftt_mismatches = []
    fa_mismatches = []
    g_vs_gols_mismatches = []
    gs_vs_gols_mismatches = []

    for _, row in df_f1.iterrows():
        temporada = int(row["temporada"])
        rodada = int(row["rodada"])
        partida_id = int(row["partida_id"])
        data_str = str(row["data"])
        clube_id = int(row["clube_id"])
        clube_nome = str(row["clube_nome"])
        adversario_id = int(row["adversario_id"])
        adversario_nome = str(row["adversario_nome"])
        mando = str(row["mando"])

        gols_marcados = int(row["gols_marcados"])
        gols_sofridos = int(row["gols_sofridos"])

        # Scouts defensivos
        sg = gols_sofridos == 0
        de = float(row.get("DE", 0.0))
        dp = float(row.get("DP", 0.0))
        ds = float(row.get("DS", 0.0))
        gs = float(row.get("GS", 0.0))
        fc = float(row.get("FC", 0.0))

        # Scouts ofensivos
        g = float(row.get("G", 0.0))
        a = float(row.get("A", 0.0))
        ftt = float(row.get("FTT", 0.0))
        fa = float(row.get("FA", 0.0))
        fs = float(row.get("FS", 0.0))
        i_scout = float(row.get("I", 0.0))

        # Componentes
        fd = float(row.get("FD", 0.0))
        ff = float(row.get("FF", 0.0))
        ft = float(row.get("FT", 0.0))
        pp = float(row.get("PP", 0.0))

        # Auditoria FTT = FF + FT + FD + G + PP
        soma_ftt = ff + ft + fd + g + pp
        if abs(ftt - soma_ftt) > 1e-4:
            ftt_mismatches.append({
                "partida_id": partida_id,
                "rodada": rodada,
                "clube": clube_nome,
                "FTT": ftt,
                "componentes_soma": soma_ftt,
            })

        # Auditoria FA = FD + G
        soma_fa = fd + g
        if abs(fa - soma_fa) > 1e-4:
            fa_mismatches.append({
                "partida_id": partida_id,
                "rodada": rodada,
                "clube": clube_nome,
                "FA": fa,
                "componentes_soma": soma_fa,
            })

        # Auditoria G vs gols_marcados
        if abs(g - gols_marcados) > 1e-4:
            g_vs_gols_mismatches.append({
                "partida_id": partida_id,
                "rodada": rodada,
                "clube": clube_nome,
                "G": g,
                "gols_marcados": gols_marcados,
            })

        # Auditoria GS vs gols_sofridos
        if abs(gs - gols_sofridos) > 1e-4:
            gs_vs_gols_mismatches.append({
                "partida_id": partida_id,
                "rodada": rodada,
                "clube": clube_nome,
                "GS": gs,
                "gols_sofridos": gols_sofridos,
            })

        # Obter fatos de elenco do JSON bruto
        roster_facts = extractor.get_team_roster_facts(rodada, clube_id, partida_id)

        # Corrigir DE usando diretamente os scouts dos goleiros
        # que efetivamente entraram em campo no JSON bruto.
        #
        # O F1 contém um DE incorreto para vários clubes/rodadas.
        # Todos os demais scouts continuam vindo do F1.
        if roster_facts.get("roster_presente", False):
            de = float(roster_facts.get("DE_raw", de))

        rows.append({
            "temporada": temporada,
            "rodada": rodada,
            "partida_id": partida_id,
            "data": data_str,
            "clube_id": clube_id,
            "clube_nome": clube_nome,
            "adversario_id": adversario_id,
            "adversario_nome": adversario_nome,
            "mando": mando,
            "gols_marcados": gols_marcados,
            "gols_sofridos": gols_sofridos,
            "SG": sg,
            "DE": de,
            "DP": dp,
            "DS": ds,
            "GS": gs,
            "FC": fc,
            "G": g,
            "A": a,
            "FTT": ftt,
            "FA": fa,
            "FS": fs,
            "I": i_scout,
            "FD": fd,
            "FF": ff,
            "FT": ft,
            "PP": pp,
            "roster_presente": roster_facts["roster_presente"],
            "roster_jogadores_raw": roster_facts["roster_jogadores_raw"],
            "roster_tecnicos_raw": roster_facts["roster_tecnicos_raw"],
            "n_participantes": roster_facts["n_participantes"],
            "participantes_por_posicao": roster_facts["participantes_por_posicao"],
            "participantes": roster_facts["participantes"],
        })

    df_out = pd.DataFrame(rows)

    # Ordenação determinística estrita
    df_out = df_out.sort_values(
        by=["temporada", "rodada", "partida_id", "clube_id"]
    ).reset_index(drop=True)

    # Criar diretórios de destino
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Gravar Parquet com PyArrow oficial
    logger.info(f"Gravando Parquet oficial em: {output_parquet}")
    arrow_table = pa.Table.from_pandas(df_out, schema=OUTPUT_PYARROW_SCHEMA, preserve_index=False)
    pq.write_table(arrow_table, output_parquet, compression="snappy")

    # 2. Gravar CSV espelho (participantes serializados como JSON em UTF-8)
    logger.info(f"Gravando CSV espelho em: {output_csv}")
    df_csv = df_out.copy()
    df_csv["participantes"] = df_csv["participantes"].apply(
        lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    )
    df_csv.to_csv(output_csv, index=False, encoding="utf-8")

    # Métricas consolidadas
    n_linhas = len(df_out)
    partidas_unicas = df_out["partida_id"].nunique()
    clubes_unicos = df_out["clube_id"].nunique()
    rodadas_unicas = df_out["rodada"].nunique()
    temporadas_unicas = df_out["temporada"].nunique()

    # Checar duplicatas de chave primária
    duplicatas = df_out.duplicated(subset=["temporada", "rodada", "partida_id", "clube_id"]).sum()

    # Validação de pares por partida
    counts_por_partida = df_out.groupby("partida_id").size()
    partidas_com_2_linhas = (counts_por_partida == 2).sum()
    partidas_com_problema = (counts_por_partida != 2).sum()

    roster_presente_count = int(df_out["roster_presente"].sum())
    roster_ausente_count = n_linhas - roster_presente_count
    total_participantes = int(df_out["n_participantes"].sum())

    sha_parquet = compute_sha256(output_parquet)
    sha_csv = compute_sha256(output_csv)

    stats = {
        "linhas": n_linhas,
        "partidas": partidas_unicas,
        "clubes": clubes_unicos,
        "rodadas": rodadas_unicas,
        "temporadas": temporadas_unicas,
        "duplicatas": int(duplicatas),
        "partidas_com_2_linhas": int(partidas_com_2_linhas),
        "partidas_com_problema": int(partidas_com_problema),
        "roster_presente": roster_presente_count,
        "roster_ausente": roster_ausente_count,
        "participantes_total": total_participantes,
        "anomalias_FTT": len(ftt_mismatches),
        "anomalias_FA": len(fa_mismatches),
        "anomalias_G_vs_gols": len(g_vs_gols_mismatches),
        "anomalias_GS_vs_gols_sofridos": len(gs_vs_gols_mismatches),
        "parquet": str(output_parquet),
        "csv": str(output_csv),
        "sha256_parquet": sha_parquet,
        "sha256_csv": sha_csv,
        "exemplos_anomalias": {
            "FTT": ftt_mismatches[:3],
            "FA": fa_mismatches[:3],
            "G_vs_gols": g_vs_gols_mismatches[:3],
            "GS_vs_gols_sofridos": gs_vs_gols_mismatches[:3],
        }
    }

    return df_out, stats


def print_report(stats: Dict[str, Any]) -> None:
    """Exibe o relatório formatado conforme o padrão exigido na especificação."""
    print("\n=== TEAM SCOUT HISTORY ===")
    print(f"linhas: {stats['linhas']}")
    print(f"partidas: {stats['partidas']}")
    print(f"clubes: {stats['clubes']}")
    print(f"rodadas: {stats['rodadas']}")
    print(f"temporadas: {stats['temporadas']}")
    print()
    print(f"duplicatas: {stats['duplicatas']}")
    print(f"partidas_com_2_linhas: {stats['partidas_com_2_linhas']}")
    print(f"partidas_com_problema: {stats['partidas_com_problema']}")
    print()
    print(f"roster_presente: {stats['roster_presente']}")
    print(f"roster_ausente: {stats['roster_ausente']}")
    print()
    print(f"participantes_total: {stats['participantes_total']}")
    print()
    print(f"anomalias_FTT: {stats['anomalias_FTT']}")
    print(f"anomalias_FA: {stats['anomalias_FA']}")
    print(f"anomalias_G_vs_gols: {stats['anomalias_G_vs_gols']}")
    print(f"anomalias_GS_vs_gols_sofridos: {stats['anomalias_GS_vs_gols_sofridos']}")
    print()
    print(f"parquet: {stats['parquet']}")
    print(f"csv: {stats['csv']}")
    print()
    print(f"sha256_parquet: {stats['sha256_parquet']}")
    print(f"sha256_csv: {stats['sha256_csv']}")
    print("==========================")


def verify_roundtrip(parquet_path: Path, csv_path: Path, original_len: int) -> None:
    """Realiza teste de releitura e integridade nos arquivos gerados em disco."""
    logger.info("Iniciando verificação de releitura (round-trip test)...")

    # 1. Teste de Releitura Parquet
    table = pq.read_table(parquet_path)
    assert len(table) == original_len, f"Tamanho divergente no Parquet: {len(table)} vs {original_len}"
    assert "participantes" in table.column_names, "Coluna 'participantes' ausente no Parquet"
    col_type = table.schema.field("participantes").type
    assert pa.types.is_list(col_type), f"Esperado list<struct> em participantes, encontrado: {col_type}"

    # 2. Teste de Releitura CSV
    df_csv_in = pd.read_csv(csv_path)
    assert len(df_csv_in) == original_len, f"Tamanho divergente no CSV: {len(df_csv_in)} vs {original_len}"

    # Validar integridade JSON de uma amostra do CSV
    sample_cell = df_csv_in["participantes"].iloc[0]
    parsed_sample = json.loads(sample_cell)
    assert isinstance(parsed_sample, list), "Célula de participantes no CSV não é uma lista JSON válida"

    logger.info("Verificação de releitura concluída com sucesso absoluto.")


def main():
    parser = argparse.ArgumentParser(description="Build Team Scout History Fact Base")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/interim/exp12/team_match_history.parquet"),
        help="Caminho do arquivo fonte F1",
    )
    parser.add_argument(
        "--raw-api-dir",
        type=Path,
        default=Path("experiments/data/cartola_api_2026"),
        help="Diretório contendo os JSONs r1.json ... r26.json",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/interim/exp12/team_scout_history.parquet"),
        help="Caminho do parquet final gerado",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/exp12/team_scout_history.csv"),
        help="Caminho do CSV espelho final gerado",
    )

    args = parser.parse_args()

    df_out, stats = build_team_scout_history(
        input_parquet=args.input_parquet,
        raw_api_dir=args.raw_api_dir,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
    )

    verify_roundtrip(args.output_parquet, args.output_csv, len(df_out))
    print_report(stats)


if __name__ == "__main__":
    main()
