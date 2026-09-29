#!/usr/bin/env python3
"""
CARTOLA AI INSIGHTS — EXP-12
Auditor Independente: RAW (JSONs Cartola API) vs MASTER (team_scout_history.parquet)

Princípios e Restrições:
- Somente leitura dos dados de entrada (read-only).
- Zero alterações no Parquet mestre (verificação estrita via hash SHA256 antes e depois).
- Zero correções heurísticas, sem ML e sem acesso à internet/API.
- A fonte RAW de scouts individuais é a referência absoluta.
- Chave de comparação: (temporada, rodada, clube_id, scout).
- 16 Scouts auditados: SG, DE, DP, DS, GS, FC, G, A, FTT, FA, FS, I, FD, FF, FT, PP.
- Regra de participação estrita: atleta com entrou_em_campo=True e posicao_id != 6 (sem comissão técnica).
- Distinção factual do roster:
    * Caso A: len(jogadores_raw) == 0 -> ROSTER_AUSENTE (RAW_INDETERMINADO_ROSTER_AUSENTE, scouts NaN)
    * Caso B: len(jogadores_raw) > 0 e participantes == 0 -> ROSTER_PRESENTE (scouts 0)
    * Caso C: len(jogadores_raw) > 0 e participantes > 0 -> ROSTER_PRESENTE (soma normal)
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Caminhos padrão do projeto
DEFAULT_MASTER_PATH = Path("data/interim/exp12/team_scout_history.parquet")
DEFAULT_RAW_DIR = Path("experiments/data/cartola_api_2026")
DEFAULT_REPORTS_DIR = Path("reports/exp12")

# 16 Scouts auditados
AUDIT_SCOUTS: List[str] = [
    "SG", "DE", "DP", "DS", "GS", "FC", "G", "A",
    "FTT", "FA", "FS", "I", "FD", "FF", "FT", "PP"
]

# 33 Colunas esperadas no Parquet mestre
EXPECTED_MASTER_COLUMNS: List[str] = [
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

FLOAT_TOLERANCE = 1e-9


# ==============================================================================
# FUNÇÕES UTILITÁRIAS E INTEGRIDADE
# ==============================================================================

def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Calcula hash SHA256 de um arquivo binário."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo para cálculo de hash não encontrado: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_project_path(rel_path: Union[str, Path]) -> Path:
    """Resolve caminho relativo considerando a raiz do projeto."""
    path = Path(rel_path)
    if path.exists():
        return path
    alt = Path(__file__).resolve().parent.parent / rel_path
    if alt.exists():
        return alt
    return path


def parse_master_participants(val):
    if val is None:
        return set()

    # Parquet: list<struct>
    if isinstance(val, (list, tuple)):
        result = set()

        for item in val:
            if item is None:
                continue

            if isinstance(item, dict):
                atleta_id = item.get("atleta_id")
            else:
                try:
                    atleta_id = item["atleta_id"]
                except Exception:
                    atleta_id = getattr(item, "atleta_id", None)

            if atleta_id is not None:
                result.add(int(atleta_id))

        return result

    # CSV / fallback: JSON string
    if isinstance(val, str):
        if not val.strip():
            return set()

        try:
            parsed = json.loads(val)
        except Exception:
            return set()

        return parse_master_participants(parsed)

    # Escalar NaN
    try:
        if pd.isna(val):
            return set()
    except (TypeError, ValueError):
        pass

    return set()

def load_raw_round_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Carrega arquivo JSON bruto de uma rodada."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_raw_athletes_from_round(
    data: Dict[str, Any], rodada: int, temporada: int = 2026
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """
    Extrai lista de atletas brutos com verificação factual de participação.
    Retorna (lista_de_atletas, mapa_nomes_clubes).
    """
    clubes_map: Dict[int, str] = {}
    clubes_data = data.get("clubes") or {}
    if isinstance(clubes_data, dict):
        for cid_str, cinfo in clubes_data.items():
            try:
                cid = int(cid_str)
                cname = cinfo.get("nome") or cinfo.get("abreviacao") or f"Clube {cid}"
                clubes_map[cid] = str(cname)
            except (ValueError, TypeError):
                pass

    posicoes_map: Dict[int, str] = {
        1: "Goleiro",
        2: "Lateral",
        3: "Zagueiro",
        4: "Meia",
        5: "Atacante",
        6: "Técnico",
    }
    pos_data = data.get("posicoes") or {}
    if isinstance(pos_data, dict):
        for pid_str, pinfo in pos_data.items():
            try:
                pid = int(pid_str)
                pname = pinfo.get("nome")
                if pname:
                    posicoes_map[pid] = str(pname)
            except (ValueError, TypeError):
                pass

    raw_athletes: List[Dict[str, Any]] = []
    atletas_data = data.get("atletas") or {}

    iterator = atletas_data.items() if isinstance(atletas_data, dict) else enumerate(atletas_data)

    for key, at_info in iterator:
        if not isinstance(at_info, dict):
            continue

        raw_aid = at_info.get("atleta_id") or key
        try:
            atleta_id = int(raw_aid)
        except (ValueError, TypeError):
            continue

        clube_id_raw = at_info.get("clube_id")
        try:
            clube_id = int(clube_id_raw) if clube_id_raw is not None else -1
        except (ValueError, TypeError):
            clube_id = -1

        posicao_id_raw = at_info.get("posicao_id")
        try:
            posicao_id = int(posicao_id_raw) if posicao_id_raw is not None else -1
        except (ValueError, TypeError):
            posicao_id = -1

        # Regra de participação fundamental
        tem_campo_entrou = "entrou_em_campo" in at_info
        entrou_em_campo = (at_info.get("entrou_em_campo") is True) if tem_campo_entrou else False

        scout_dict = at_info.get("scout")
        if not isinstance(scout_dict, dict):
            scout_dict = {}

        raw_athletes.append({
            "temporada": temporada,
            "rodada": rodada,
            "atleta_id": atleta_id,
            "atleta_nome": str(at_info.get("apelido") or at_info.get("nome") or f"Atleta {atleta_id}"),
            "clube_id": clube_id,
            "clube_nome": clubes_map.get(clube_id, f"Clube {clube_id}"),
            "posicao_id": posicao_id,
            "posicao_nome": posicoes_map.get(posicao_id, f"Posição {posicao_id}"),
            "tem_campo_entrou": tem_campo_entrou,
            "entrou_em_campo": entrou_em_campo,
            "scout": scout_dict,
        })

    return raw_athletes, clubes_map


# ==============================================================================
# AUDITOR CENTRAL
# ==============================================================================

class RawVsMasterAuditor:
    """Audita os scouts do banco mestre contra os dados brutos de jogadores."""

    def __init__(
        self,
        master_path: Union[str, Path] = DEFAULT_MASTER_PATH,
        raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
        reports_dir: Union[str, Path] = DEFAULT_REPORTS_DIR,
    ):
        self.master_path = resolve_project_path(master_path)
        self.raw_dir = resolve_project_path(raw_dir)
        self.reports_dir = resolve_project_path(reports_dir)

        # Hash inicial do Parquet
        self.sha256_before = compute_file_sha256(self.master_path)

        # Carrega e valida Parquet
        self.df_master = self._load_and_validate_master()

        # Descoberta das rodadas RAW
        self.raw_round_files: Dict[int, Path] = self._discover_raw_rounds()

        # Estruturas preenchidas durante a execução
        self.raw_athletes_all: List[Dict[str, Any]] = []
        self.clubes_map: Dict[int, str] = {}
        self.comparison_records: List[Dict[str, Any]] = []
        self.differences_records: List[Dict[str, Any]] = []
        self.participants_records: List[Dict[str, Any]] = []
        self.goalkeepers_records: List[Dict[str, Any]] = []
        self.finishing_records: List[Dict[str, Any]] = []

    def _load_and_validate_master(self) -> pd.DataFrame:
        """Carrega e valida presença das 33 colunas obrigatórias."""
        df = pd.read_parquet(self.master_path, engine="pyarrow")
        missing = [c for c in EXPECTED_MASTER_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes no Master Parquet: {missing}")
        return df

    def _discover_raw_rounds(self) -> Dict[int, Path]:
        """Descobre os arquivos r{rodada}.json existentes."""
        files = glob.glob(str(self.raw_dir / "r*.json"))
        rounds_map: Dict[int, Path] = {}
        for f in files:
            p = Path(f)
            m = re.search(r"r(\d+)\.json$", p.name)
            if m:
                rodada = int(m.group(1))
                rounds_map[rodada] = p
        if not rounds_map:
            raise FileNotFoundError(f"Nenhum arquivo r{{rodada}}.json encontrado em {self.raw_dir}")
        return dict(sorted(rounds_map.items()))

    def run_audit(self) -> None:
        """Executa auditoria completa de forma determinística."""
        temporada = 2026

        # 1. Carrega todos os atletas RAW
        for rodada, fpath in self.raw_round_files.items():
            round_data = load_raw_round_json(fpath)
            athletes, cmap = extract_raw_athletes_from_round(round_data, rodada=rodada, temporada=temporada)
            self.raw_athletes_all.extend(athletes)
            self.clubes_map.update(cmap)

        # Mapeia clubes mestre para enriquecer nomes caso necessário
        master_club_names = (
            self.df_master[["clube_id", "clube_nome"]]
            .dropna()
            .drop_duplicates()
            .set_index("clube_id")["clube_nome"]
            .to_dict()
        )
        self.clubes_map.update({int(k): str(v) for k, v in master_club_names.items()})

        # 2. Agrupa atletas RAW por (temporada, rodada, clube_id)
        raw_by_club_round: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
        for at in self.raw_athletes_all:
            key = (at["temporada"], at["rodada"], at["clube_id"])
            raw_by_club_round.setdefault(key, []).append(at)

        # 3. Auditoria clube × rodada
        master_keys = set(
            zip(
                self.df_master["temporada"].astype(int),
                self.df_master["rodada"].astype(int),
                self.df_master["clube_id"].astype(int),
            )
        )
        raw_keys = set(raw_by_club_round.keys())
        all_keys = sorted(master_keys.union(raw_keys))

        for temp, rod, cid in all_keys:
            cname = self.clubes_map.get(cid, f"Clube {cid}")
            athletes_club_round = raw_by_club_round.get((temp, rod, cid), [])

            # Determinação factual do estado do roster ANTES do filtro de participação
            # Jogadores no RAW: atletas do clube com posicao_id != 6 (sem comissão técnica)
            jogadores_raw = [
                at for at in athletes_club_round
                if at["posicao_id"] != 6
            ]
            raw_jogadores_count = len(jogadores_raw)

            if raw_jogadores_count == 0:
                roster_status = "ROSTER_AUSENTE"
                raw_determinado = False
            else:
                roster_status = "ROSTER_PRESENTE"
                raw_determinado = True

            # Participantes: jogadores presentes no roster que realmente entraram em campo
            participating_athletes = [
                at for at in jogadores_raw
                if at["entrou_em_campo"] is True
            ]
            raw_part_count = len(participating_athletes)

            # Busca linha correspondente no MASTER
            master_sub = self.df_master[
                (self.df_master["temporada"] == temp)
                & (self.df_master["rodada"] == rod)
                & (self.df_master["clube_id"] == cid)
            ]

            master_row: Optional[pd.Series] = master_sub.iloc[0] if not master_sub.empty else None
            master_present = master_row is not None
            master_n_part = int(master_row["n_participantes"]) if (master_present and pd.notna(master_row["n_participantes"])) else 0
            roster_presente = bool(master_row["roster_presente"]) if master_present else (raw_jogadores_count > 0)
            master_part_ids = parse_master_participants(master_row["participantes"]) if master_present else set()

            # 3.1 Relatório de Participantes e Verificação Cruzada
            for at in sorted(athletes_club_round, key=lambda x: x["atleta_id"]):
                self.participants_records.append({
                    "temporada": temp,
                    "rodada": rod,
                    "clube_id": cid,
                    "clube_nome": cname,
                    "atleta_id": at["atleta_id"],
                    "atleta_nome": at["atleta_nome"],
                    "posicao_id": at["posicao_id"],
                    "posicao_nome": at["posicao_nome"],
                    "entrou_em_campo": at["entrou_em_campo"],
                    "presente_no_master": at["atleta_id"] in master_part_ids,
                })

            # 3.2 Relatórios Especiais: Goleiros e Finalizações
            for at in participating_athletes:
                sc = at["scout"]
                # Goleiros (posicao_id == 1)
                if at["posicao_id"] == 1:
                    self.goalkeepers_records.append({
                        "temporada": temp,
                        "rodada": rod,
                        "clube_id": cid,
                        "clube_nome": cname,
                        "atleta_id": at["atleta_id"],
                        "atleta_nome": at["atleta_nome"],
                        "DE": int(sc.get("DE", 0)),
                        "DP": int(sc.get("DP", 0)),
                        "GS": int(sc.get("GS", 0)),
                        "SG": int(sc.get("SG", 0)),
                        "entrou_em_campo": at["entrou_em_campo"],
                    })

                # Finalizações
                g_ind = int(sc.get("G", 0))
                fd_ind = int(sc.get("FD", 0))
                ff_ind = int(sc.get("FF", 0))
                ft_ind = int(sc.get("FT", 0))
                pp_ind = int(sc.get("PP", 0))
                ftt_ind = ff_ind + ft_ind + fd_ind + g_ind + pp_ind
                fa_ind = fd_ind + g_ind

                self.finishing_records.append({
                    "temporada": temp,
                    "rodada": rod,
                    "clube_id": cid,
                    "clube_nome": cname,
                    "atleta_id": at["atleta_id"],
                    "atleta_nome": at["atleta_nome"],
                    "G": g_ind,
                    "FD": fd_ind,
                    "FF": ff_ind,
                    "FT": ft_ind,
                    "PP": pp_ind,
                    "FTT_individual": ftt_ind,
                    "FA_individual": fa_ind,
                })

            # 3.3 Cálculo dos Scouts Agregados RAW
            raw_scouts: Dict[str, Any] = {}

            if raw_determinado:
                # SG é uma propriedade do time no MASTER:
                # True se o time não sofreu gols; False caso tenha sofrido.
                # O auditor não extrai placar do RAW de atletas, portanto
                # a comparação do SG é feita diretamente contra gols_sofridos
                # da mesma linha do MASTER, somente quando ela existe.
                raw_scouts["SG"] = np.nan

                # Demais scouts: somas diretas dos participantes.
                for scout_name in [
                    "DE", "DP", "DS", "GS", "FC", "G", "A",
                    "FS", "I", "FD", "FF", "FT", "PP"
                ]:
                    raw_scouts[scout_name] = sum(
                        int(at["scout"].get(scout_name, 0))
                        for at in participating_athletes
                    )

                # Scouts derivados de finalização.
                raw_scouts["FTT"] = (
                    raw_scouts["FF"]
                    + raw_scouts["FT"]
                    + raw_scouts["FD"]
                    + raw_scouts["G"]
                    + raw_scouts["PP"]
                )
                raw_scouts["FA"] = raw_scouts["FD"] + raw_scouts["G"]

            else:
                for scout_name in AUDIT_SCOUTS:
                    raw_scouts[scout_name] = np.nan

            # 3.4 Comparação Scout por Scout
            for scout in AUDIT_SCOUTS:
                master_val = (
                    master_row[scout]
                    if (master_present and pd.notna(master_row[scout]))
                    else np.nan
                )

                if not master_present:
                    raw_val = raw_scouts[scout]
                    status = "MASTER_AUSENTE"
                    diff = np.nan

                elif not raw_determinado:
                    raw_val = np.nan
                    status = "RAW_INDETERMINADO_ROSTER_AUSENTE"
                    diff = np.nan

                elif scout == "SG":
                    # SG do time é bool: clean sheet = gols_sofridos == 0.
                    # O placar está no MASTER; o RAW de atletas apenas
                    # distribui o scout SG entre os jogadores da defesa.
                    raw_val = int(master_row["gols_sofridos"]) == 0
                    diff = int(bool(master_val)) - int(bool(raw_val))
                    status = (
                        "OK"
                        if bool(master_val) == bool(raw_val)
                        else "DIVERGENTE"
                    )

                else:
                    raw_val = raw_scouts[scout]
                    diff = master_val - raw_val
                    status = (
                        "DIVERGENTE"
                        if abs(diff) > FLOAT_TOLERANCE
                        else "OK"
                    )

                rec = {
                    "temporada": temp,
                    "rodada": rod,
                    "clube_id": cid,
                    "clube_nome": cname,
                    "scout": scout,
                    "raw_value": raw_val,
                    "master_value": master_val,
                    "difference": diff,
                    "status": status,
                    "roster_status": roster_status,
                    "raw_jogadores": raw_jogadores_count,
                    "raw_participantes": raw_part_count,
                    "roster_presente": roster_presente,
                    "master_n_participantes": master_n_part,
                }
                self.comparison_records.append(rec)
                if status == "DIVERGENTE":
                    self.differences_records.append({
                        "temporada": temp,
                        "rodada": rod,
                        "clube_id": cid,
                        "clube_nome": cname,
                        "scout": scout,
                        "raw_value": raw_val,
                        "master_value": master_val,
                        "difference": diff,
                        "roster_status": roster_status,
                        "raw_jogadores": raw_jogadores_count,
                        "raw_participantes": raw_part_count,
                        "roster_presente": roster_presente,
                        "master_n_participantes": master_n_part,
                    })

        # 4. Geração dos Relatórios CSV
        self._write_reports()

        # 5. Verificação Estrita de Imutabilidade do Parquet
        sha256_after = compute_file_sha256(self.master_path)
        if self.sha256_before != sha256_after:
            raise RuntimeError("ERRO CRÍTICO: o auditor alterou o banco mestre.")

    def _write_reports(self) -> None:
        """Salva os relatórios determinísticos em reports/exp12/."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. raw_vs_master_all.csv
        df_all = pd.DataFrame(self.comparison_records).sort_values(
            by=["temporada", "rodada", "clube_id", "scout"]
        )
        df_all.to_csv(self.reports_dir / "raw_vs_master_all.csv", index=False)

        # 2. raw_vs_master_differences.csv
        df_diff = pd.DataFrame(self.differences_records)
        if not df_diff.empty:
            df_diff = df_diff.sort_values(by=["scout", "temporada", "rodada", "clube_id"])
        else:
            df_diff = pd.DataFrame(columns=[
                "temporada", "rodada", "clube_id", "clube_nome", "scout",
                "raw_value", "master_value", "difference", "roster_status",
                "raw_jogadores", "raw_participantes", "roster_presente",
                "master_n_participantes"
            ])
        df_diff.to_csv(self.reports_dir / "raw_vs_master_differences.csv", index=False)

        # 3. raw_vs_master_participants.csv
        df_parts = pd.DataFrame(self.participants_records).sort_values(
            by=["temporada", "rodada", "clube_id", "atleta_id"]
        )
        df_parts.to_csv(self.reports_dir / "raw_vs_master_participants.csv", index=False)

        # 4. raw_goalkeeper_audit.csv
        df_gk = pd.DataFrame(self.goalkeepers_records)
        if not df_gk.empty:
            df_gk = df_gk.sort_values(by=["temporada", "rodada", "clube_id", "atleta_id"])
        else:
            df_gk = pd.DataFrame(columns=[
                "temporada", "rodada", "clube_id", "clube_nome", "atleta_id",
                "atleta_nome", "DE", "DP", "GS", "SG", "entrou_em_campo"
            ])
        df_gk.to_csv(self.reports_dir / "raw_goalkeeper_audit.csv", index=False)

        # 5. raw_finishing_audit.csv
        df_fin = pd.DataFrame(self.finishing_records)
        if not df_fin.empty:
            df_fin = df_fin.sort_values(by=["temporada", "rodada", "clube_id", "atleta_id"])
        else:
            df_fin = pd.DataFrame(columns=[
                "temporada", "rodada", "clube_id", "clube_nome", "atleta_id",
                "atleta_nome", "G", "FD", "FF", "FT", "PP", "FTT_individual", "FA_individual"
            ])
        df_fin.to_csv(self.reports_dir / "raw_finishing_audit.csv", index=False)

    def print_terminal_report(self) -> None:
        """Exibe o relatório formatado exigido no terminal."""
        df_all = pd.DataFrame(self.comparison_records)
        if df_all.empty:
            print("Nenhum registro para exibir.")
            return

        temporada = 2026
        rodadas = sorted(df_all["rodada"].unique())
        min_r = rodadas[0] if rodadas else 0
        max_r = rodadas[-1] if rodadas else 0
        clubes_count = df_all["clube_id"].nunique()
        cr_count = df_all[["rodada", "clube_id"]].drop_duplicates().shape[0]

        print("\n" + "=" * 60)
        print("AUDITORIA RAW → MASTER")
        print("=" * 60)
        print(f"temporada: {temporada}")
        print(f"rodadas auditadas: {min_r}–{max_r}")
        print(f"clubes auditados: {clubes_count}")
        print(f"clubes×rodadas: {cr_count}\n")

        print("-" * 60)
        print(f"{'SCOUT':<18} {'RAW válido':<14} {'divergências':<15} {'% divergência':<15}")
        print("-" * 60)

        divergencias_por_scout: Dict[str, int] = {}
        for s in AUDIT_SCOUTS:
            sub = df_all[df_all["scout"] == s]
            raw_valido = int((sub["status"].isin(["OK", "DIVERGENTE"])).sum())
            divs = int((sub["status"] == "DIVERGENTE").sum())
            pct = (divs / raw_valido * 100.0) if raw_valido > 0 else 0.0
            divergencias_por_scout[s] = divs
            print(f"{s:<18} {raw_valido:<14} {divs:<15} {pct:.2f}%")
        print("-" * 60)

        # Seção SAO R1
        self._print_sao_r1()

        # Top Divergências
        print("\n" + "=" * 60)
        print("TOP DIVERGÊNCIAS")
        print("=" * 60)
        for s in AUDIT_SCOUTS:
            sub_div = df_all[(df_all["scout"] == s) & (df_all["status"] == "DIVERGENTE")].copy()
            if sub_div.empty:
                print(f"\n{s}: 0 divergências")
            else:
                sub_div["abs_diff"] = sub_div["difference"].abs()
                top10 = sub_div.sort_values(by=["abs_diff", "rodada", "clube_id"], ascending=[False, True, True]).head(10)
                print(f"\nScout: {s} ({len(sub_div)} divergências)")
                print(f"{'Rodada':<8} | {'Clube':<20} | {'RAW':<8} | {'MASTER':<8} | {'Diferença':<10}")
                print("-" * 62)
                for row in top10.itertuples():
                    raw_str = f"{row.raw_value:.0f}" if pd.notna(row.raw_value) else "-"
                    mst_str = f"{row.master_value:.0f}" if pd.notna(row.master_value) else "-"
                    diff_val = row.difference
                    diff_str = f"{diff_val:+.0f}" if pd.notna(diff_val) else "-"
                    print(f"R{row.rodada:<6} | {str(row.clube_nome)[:20]:<20} | {raw_str:<8} | {mst_str:<8} | {diff_str:<10}")

        # Resumo final
        total_divergencias = len(self.differences_records)
        sha256_after = compute_file_sha256(self.master_path)
        hash_ok = self.sha256_before == sha256_after

        print("\n" + "=" * 60)
        print("AUDITORIA CONCLUÍDA")
        print("=" * 60)
        print(f"Parquet master alterado: NÃO")
        print(f"Scouts auditados: {len(AUDIT_SCOUTS)}")
        print(f"Divergências totais: {total_divergencias}\n")

        for s in ["DE", "DP", "FD", "FF", "FT", "PP"]:
            print(f"{s} divergências: {divergencias_por_scout.get(s, 0)}")

        print("\nRelatórios:")
        print(f"reports/exp12/raw_vs_master_all.csv")
        print(f"reports/exp12/raw_vs_master_differences.csv")
        print(f"reports/exp12/raw_vs_master_participants.csv")
        print(f"reports/exp12/raw_goalkeeper_audit.csv")
        print(f"reports/exp12/raw_finishing_audit.csv")
        print(f"\nSHA256 antes:  {self.sha256_before}")
        print(f"SHA256 depois: {sha256_after}")
        print(f"Hash preservado: {'SIM' if hash_ok else 'NÃO'}")
        print("=" * 60 + "\n")

    def _print_sao_r1(self) -> None:
        """Imprime a seção obrigatória específica de SAO R1."""
        temp = 2026
        rod = 1
        cid = 287

        sub_master = self.df_master[
            (self.df_master["temporada"] == temp)
            & (self.df_master["rodada"] == rod)
            & (self.df_master["clube_id"] == cid)
        ]
        if sub_master.empty:
            print("\nCASO ESPECÍFICO: SAO R1 (clube_id=287) não encontrado no Master.")
            return

        master_row = sub_master.iloc[0]
        athletes_sao = [
            at for at in self.raw_athletes_all
            if at["temporada"] == temp and at["rodada"] == rod and at["clube_id"] == cid
        ]
        jogadores_sao = [at for at in athletes_sao if at["posicao_id"] != 6]
        participants_sao = [at for at in jogadores_sao if at["entrou_em_campo"] is True]
        gks_sao = [at for at in participants_sao if at["posicao_id"] == 1]

        print("\n" + "=" * 60)
        print("CASO ESPECÍFICO: SAO R1 (clube_id=287)")
        print("=" * 60)
        print(f"SAO — R1 ({master_row['clube_nome']})")
        print(f"\nJogadores no RAW: {len(jogadores_sao)} | Participantes RAW: {len(participants_sao)}")
        for at in participants_sao:
            print(f"  [{at['atleta_id']}] {at['atleta_nome']} ({at['posicao_nome']})")

        print(f"\nGoleiros ({len(gks_sao)} participantes):")
        for gk in gks_sao:
            sc = gk["scout"]
            print(f"  [{gk['atleta_id']}] {gk['atleta_nome']} -> DE={sc.get('DE', 0)}, DP={sc.get('DP', 0)}, GS={sc.get('GS', 0)}, SG={sc.get('SG', 0)}")

        print("\n" + f"{'SCOUT':<16} {'RAW':<10} {'MASTER':<10} {'DIF':<10}")
        print("-" * 46)

        raw_scouts = {}
        for s in ["SG", "DE", "DP", "DS", "GS", "FC", "G", "A", "FS", "I", "FD", "FF", "FT", "PP"]:
            raw_scouts[s] = sum(int(at["scout"].get(s, 0)) for at in participants_sao)
        raw_scouts["FTT"] = raw_scouts["FF"] + raw_scouts["FT"] + raw_scouts["FD"] + raw_scouts["G"] + raw_scouts["PP"]
        raw_scouts["FA"] = raw_scouts["FD"] + raw_scouts["G"]

        for s in AUDIT_SCOUTS:
            r_val = raw_scouts[s]
            m_val = master_row[s]
            diff = m_val - r_val
            print(f"{s:<16} {r_val:<10.0f} {m_val:<10.0f} {diff:<+10.0f}")

        # Contexto de gols
        raw_g = raw_scouts["G"]
        master_gm = master_row["gols_marcados"]
        master_gs = master_row["gols_sofridos"]
        clean_sheet = (master_gs == 0)

        print("\nCONTEXTO DE PLACAR:")
        print(f"gols_marcados: RAW (scout G) = {raw_g} | MASTER = {master_gm:.0f}")
        print(f"gols_sofridos: RAW (scout GS) = {raw_scouts['GS']} | MASTER = {master_gs:.0f}")
        print(f"clean_sheet_pelo_placar (gols_sofridos == 0): {clean_sheet} | RAW scout SG = {raw_scouts['SG']}")
        print("=" * 60)


def main() -> None:
    """Ponto de entrada do script."""
    auditor = RawVsMasterAuditor()
    auditor.run_audit()
    auditor.print_terminal_report()


if __name__ == "__main__":
    main()
