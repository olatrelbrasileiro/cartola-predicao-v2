#!/usr/bin/env python
"""
EXP-12 F2 - GATE 2 - Auditoria RAW de roster.

NAO altera F1.
NAO executa o builder F2.
NAO usa heuristica de scout zero (FTT==0, FA==0, etc).
Inspeciona estrutura real antes de decidir qualquer coisa.

Schema confirmado (via diagnostico manual):
  atletas: dict {atleta_id: {scout, apelido, foto, pontuacao,
                             posicao_id, clube_id, entrou_em_campo}}
  posicoes: dict {id_str: {id, nome, abreviacao}}
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR   = ROOT / "experiments" / "data" / "cartola_api_2026"
HIST_PATH = ROOT / "data" / "interim" / "exp12" / "team_match_history.parquet"
OUT_CSV   = ROOT / "reports" / "exp12" / "f2_missing_roster_raw_audit.csv"
OUT_MD    = ROOT / "reports" / "exp12" / "f2_missing_roster_raw_audit.md"

SEASON = 2026
ROUNDS = list(range(1, 27))

CLUBE_FIELD_CANDIDATES = ("clube_id", "cod_time", "time_id", "id_clube")
POSICAO_FIELD_CANDIDATES = ("posicao_id", "position_id", "id_posicao")


# ----------------------- discovery -----------------------

def load_round(r: int):
    p = RAW_DIR / f"r{r}.json"
    if not p.exists():
        return None, p
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh), p


def extract_atletas(data):
    """Devolve lista de dicts de atletas + string descrevendo a fonte."""
    atl = data.get("atletas") if isinstance(data, dict) else None
    if isinstance(atl, dict):
        return list(atl.values()), "atletas(dict).values()"
    if isinstance(atl, list):
        return atl, "atletas(list)"
    raise RuntimeError(
        "Campo 'atletas' ausente ou com tipo inesperado. "
        f"top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
    )


def extract_pos_map(data) -> dict:
    """Devolve {posicao_id:int -> nome:str}."""
    pos = data.get("posicoes") if isinstance(data, dict) else None
    m = {}
    if isinstance(pos, dict):
        for k, v in pos.items():
            if isinstance(v, dict):
                pid = v.get("id", k)
                nome = v.get("nome", "")
                try:
                    pid = int(pid)
                except (TypeError, ValueError):
                    pass
                m[pid] = nome
    elif isinstance(pos, list):
        for v in pos:
            if isinstance(v, dict):
                pid = v.get("id")
                nome = v.get("nome", "")
                if pid is not None:
                    m[pid] = nome
    return m


def detect_fields(atletas):
    if not atletas or not isinstance(atletas[0], dict):
        raise RuntimeError("lista de atletas vazia ou itens nao-dict")
    kk = set(atletas[0].keys())
    clube = next((c for c in CLUBE_FIELD_CANDIDATES if c in kk), None)
    pos   = next((c for c in POSICAO_FIELD_CANDIDATES if c in kk), None)
    if clube is None:
        raise RuntimeError(f"campo de clube nao encontrado. Chaves atleta: {sorted(kk)}")
    if pos is None:
        raise RuntimeError(f"campo de posicao nao encontrado. Chaves atleta: {sorted(kk)}")
    return clube, pos


def build_tecnico_set(pos_map: dict) -> set:
    """Posicoes cujo nome contenha 'tec' (empirico via pos_map)."""
    tec = set()
    for pid, nome in pos_map.items():
        if nome and re.search(r"t[eé]c", str(nome), re.IGNORECASE):
            tec.add(pid)
    return tec


# ----------------------- main -----------------------

def main() -> int:
    if not RAW_DIR.exists():
        print(f"[ERRO] pasta raw nao encontrada: {RAW_DIR}")
        return 2
    if not HIST_PATH.exists():
        print(f"[ERRO] historico nao encontrado: {HIST_PATH}")
        return 2

    rounds_data = {}
    missing_files = []
    for r in ROUNDS:
        data, path = load_round(r)
        if data is None:
            missing_files.append(str(path))
            continue
        rounds_data[r] = data

    if not rounds_data:
        print("[ERRO] nenhum r{R}.json carregado.")
        return 2

    r0 = min(rounds_data)
    data0 = rounds_data[r0]

    try:
        atletas0, chave_atletas = extract_atletas(data0)
        clube_field, pos_field = detect_fields(atletas0)
    except Exception as e:
        print("\n=== ESTRUTURA RAW NAO RECONHECIDA ===")
        print(f"[erro] {e}")
        if isinstance(data0, dict):
            print(f"top-level keys: {list(data0.keys())}")
        print("\nPARANDO. Reveja o schema real antes de prosseguir.")
        return 3

    pos_map = extract_pos_map(data0)
    tecnico_set = build_tecnico_set(pos_map)

    # contagem por (round, clube)
    per_round_club = defaultdict(lambda: {"n_total": 0, "n_jogadores": 0, "n_tecnicos": 0})
    for r, data in rounds_data.items():
        try:
            atletas, _ = extract_atletas(data)
        except Exception as e:
            print(f"[aviso] r{r}: {e}")
            continue
        for a in atletas:
            c = a.get(clube_field)
            p = a.get(pos_field)
            if c is None:
                continue
            per_round_club[(r, c)]["n_total"] += 1
            if p in tecnico_set:
                per_round_club[(r, c)]["n_tecnicos"] += 1
            else:
                per_round_club[(r, c)]["n_jogadores"] += 1

    hist = pd.read_parquet(HIST_PATH)
    hist_season = hist[hist.temporada == SEASON].copy()

    rows = []
    for _, h in hist_season.iterrows():
        r = int(h.rodada); c = int(h.clube_id)
        cnt = per_round_club.get((r, c),
                                 {"n_total": 0, "n_jogadores": 0, "n_tecnicos": 0})
        rows.append({
            "temporada": int(h.temporada),
            "rodada": r,
            "partida_id": h.partida_id,
            "clube_id": c,
            "clube_nome": h.clube_nome,
            "n_registros_raw": cnt["n_total"],
            "n_jogadores_raw": cnt["n_jogadores"],
            "n_tecnicos_raw":  cnt["n_tecnicos"],
            "FTT": h.get("FTT"),
            "FA":  h.get("FA"),
            "G":   h.get("G"),
            "FD":  h.get("FD"),
            "FF":  h.get("FF"),
            "FT":  h.get("FT"),
            "PP":  h.get("PP"),
        })
    df = pd.DataFrame(rows)

    n_club_rounds = len(df)
    com_jog = int((df["n_jogadores_raw"] > 0).sum())
    sem_jog = int((df["n_jogadores_raw"] == 0).sum())

    casos = df[df["n_jogadores_raw"] == 0].copy()
    n_part = int(casos["partida_id"].nunique())
    n_lin  = len(casos)

    print("=== AUDITORIA RAW DE ROSTER ===")
    print(f"arquivos examinados: {len(rounds_data)}")
    print(f"rodadas: {min(rounds_data)}..{max(rounds_data)}")
    if missing_files:
        print(f"[aviso] arquivos ausentes: {missing_files}")
    print(f"club-rounds examinados: {n_club_rounds}")
    print(f"club-rounds com jogadores: {com_jog}")
    print(f"club-rounds sem jogadores: {sem_jog}")
    print(f"partidas afetadas: {n_part}")

    print("\n=== ESTRUTURA RAW CONFIRMADA ===")
    print(f"chave da lista de atletas: {chave_atletas}")
    print(f"clube_field detectado:     {clube_field}")
    print(f"pos_field detectado:       {pos_field}")
    print(f"posicoes (id->nome):       {pos_map}")
    print(f"posicoes classificadas como tecnico: {tecnico_set}")
    print("amostra de um registro de atleta:")
    print("  " + json.dumps(atletas0[0], ensure_ascii=False, default=str)[:400])

    print("\n=== CASOS DE ROSTER AUSENTE ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 220):
        print(casos.to_string(index=False))

    print("\n=== CONFRONTO COM F1 ===")
    print("esperado: 12 partidas / 24 linhas")
    print(f"obtido:   {n_part} partidas / {n_lin} linhas")
    if n_part == 12 and n_lin == 24:
        gate = "APROVADO"
        print(">>> GATE APROVADO")
    else:
        gate = "REPROVADO"
        print(">>> GATE REPROVADO - NAO prosseguir com o builder.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("# EXP-12 F2 - GATE 2 - Auditoria RAW de roster\n\n")
        fh.write(f"**Status**: {gate}\n\n")
        fh.write(f"- arquivos examinados: {len(rounds_data)}\n")
        fh.write(f"- rodadas: {min(rounds_data)}..{max(rounds_data)}\n")
        fh.write(f"- club-rounds examinados: {n_club_rounds}\n")
        fh.write(f"- club-rounds com jogadores: {com_jog}\n")
        fh.write(f"- club-rounds sem jogadores: {sem_jog}\n")
        fh.write(f"- partidas afetadas: {n_part}\n\n")
        fh.write("## Estrutura RAW confirmada\n\n")
        fh.write(f"- chave da lista de atletas: `{chave_atletas}`\n")
        fh.write(f"- clube_field: `{clube_field}`\n")
        fh.write(f"- pos_field: `{pos_field}`\n")
        fh.write(f"- pos_map: `{pos_map}`\n")
        fh.write(f"- tecnico_set: `{tecnico_set}`\n\n")
        fh.write("## Casos de roster ausente\n\n")
        fh.write(casos.to_string(index=False) if not casos.empty else "(nenhum)")
        fh.write("\n")
    print(f"\nCSV salvo: {OUT_CSV}")
    print(f"MD  salvo: {OUT_MD}")

    return 0 if gate == "APROVADO" else 1


if __name__ == "__main__":
    sys.exit(main())
