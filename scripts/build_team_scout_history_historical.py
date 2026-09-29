#!/usr/bin/env python
"""
scripts/build_team_scout_history_historical.py

Constrói o Banco Mestre factual de scouts por time para 2024 e 2025, com o
schema do banco 2026 (data/interim/exp12/team_scout_history.parquet).

Fontes:
  - data/raw/caRtola/{2024,2025}/rodada_{1..38}.csv (ou rodada-{1..38}.csv)
       · scouts ACUMULADOS por atleta por temporada
       · rodada vem do FILENAME (sem heurística de data)
  - data/raw/odds/bzzoiro_events_{2024,2025}.parquet
       · evento, mandante, visitante, home_team_id, away_team_id,
         event_date, round_number
       · NÃO possui HG/AG
  - data/raw/odds/football_data_bra.csv
       · HG, AG (placar real), odds de fechamento
       · link com o bzzoiro por (season, home_id, away_id)

Saída:
  - data/interim/exp12/team_scout_history_2024_2025.parquet
  - reports/exp12_historical_validation.csv
  - reports/exp12_historical_gs_validation.csv
  - reports/exp12_historical_gs_summary.csv
  - reports/exp12_historical_gs_divergences.csv
  - reports/exp12_historical_scout_coverage.csv
  - reports/exp12_historical_anomalies.csv
  - reports/exp12_club_mapping.csv

Diferenças inevitáveis vs. banco 2026 (documentadas, não mascaradas):
  1. partida_id: 2026 usa ID da API; histórico não tem esse ID. Geramos
     determinístico: int(f"{season}{rodada:02d}{home_id:05d}{away_id:05d}").
  2. roster_*: no histórico é um proxy baseado no CSV caRtola, não
     necessariamente idêntico à construção da API em 2026.

Regras fechadas:
  - Rodada = filename. Sem assign_rounds().
  - Placar = football_data_bra.csv (HG/AG). GS do goleiro apenas valida.
  - DE_time  = soma dos deltas de DE dos goleiros participantes.
  - GS_time  = MAX  dos deltas de GS dos goleiros participantes.
  - SG_time  = (gols_sofridos == 0) [bool].
  - FTT = FD + FF + FT + G + PP ; FA = FD + G.
  - Coluna AUSENTE != NaN != 0:
       ausente  -> delta = NaN e liga estado "unknown" do scout;
       unknown + NaN         -> delta = NaN (mantém unknown);
       unknown + numérico(v) -> delta = NaN, baseline=v, desliga unknown;
       presente + NaN        -> delta = 0, mantém acumulado;
       presente + numérico(v):
           se baseline is None -> delta = v
           senão              -> delta = v - baseline
       delta negativo -> erro fatal.
  - Qualquer inconsistência estrutural é FATAL e impede a escrita do
    parquet (nenhuma escrita parcial).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# Paths & constants
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
CARTOLA_RAW = ROOT / "data" / "raw" / "caRtola"
BZZOIRO_DIR = ROOT / "data" / "raw" / "odds"
FOOTBALL_DATA = ROOT / "data" / "raw" / "odds" / "football_data_bra.csv"
OUT_DIR = ROOT / "data" / "interim" / "exp12"
OUT_PARQUET = OUT_DIR / "team_scout_history_2024_2025.parquet"
REPORTS_DIR = ROOT / "reports"

SEASONS = (2024, 2025)
N_ROUNDS = 38
MATCHES_PER_ROUND = 10
CLUBS_PER_ROUND = 20
MATCHES_PER_SEASON = 380
ROWS_PER_SEASON = 760

GOLEIRO_POSICAO_ID = 1
TECNICO_POSICAO_ID = 6

POSICAO_NOME = {
    1: "Goleiro", 2: "Lateral", 3: "Zagueiro",
    4: "Meia", 5: "Atacante", 6: "Técnico",
}

CUMULATIVE_SCOUTS = [
    "DE", "GS", "DS", "FC", "G", "A", "FS", "I",
    "FD", "FF", "FT", "PP", "DP",
]

# Confirmado no builder 2026: scouts aditivos = soma sobre participantes.
ADDITIVE_SCOUTS = ["DS", "FC", "G", "A", "FS", "I", "FD", "FF", "FT", "PP", "DP"]

FINAL_ORDER = [
    "temporada", "rodada", "partida_id", "data",
    "clube_id", "clube_nome", "adversario_id", "adversario_nome", "mando",
    "gols_marcados", "gols_sofridos",
    "SG", "DE", "DP", "DS", "GS", "FC", "G", "A", "FTT", "FA",
    "FS", "I", "FD", "FF", "FT", "PP",
    "roster_presente", "roster_jogadores_raw", "roster_tecnicos_raw",
    "n_participantes", "participantes_por_posicao", "participantes",
]

PARQUET_SCHEMA = pa.schema([
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
    ("participantes", pa.list_(pa.struct([
        ("atleta_id", pa.int64()),
        ("atleta_nome", pa.string()),
        ("posicao_id", pa.int64()),
        ("posicao_nome", pa.string()),
    ]))),
])


# ============================================================================
# Clubes
# ============================================================================

CARTOLA_CLUBES_2024 = {
    293: "CAP", 373: "ACG", 282: "CAM", 265: "BAH", 263: "BOT",
    280: "RBB", 264: "COR", 288: "CRI", 283: "CRU", 1371: "CUI",
    262: "FLA", 266: "FLU", 356: "FOR", 284: "GRE", 285: "INT",
    286: "JUV", 275: "PAL", 276: "SAO", 267: "VAS", 287: "VIT",
}

CARTOLA_CLUBES_2025 = {
    265: "BAH", 263: "BOT", 282: "CAM", 354: "CEA", 264: "COR",
    283: "CRU", 262: "FLA", 266: "FLU", 356: "FOR", 284: "GRE",
    285: "INT", 286: "JUV", 2305: "MIR", 275: "PAL", 280: "RBB",
    277: "SAN", 276: "SAO", 292: "SPT", 267: "VAS", 287: "VIT",
}

CARTOLA_CLUBES_BY_SEASON = {2024: CARTOLA_CLUBES_2024, 2025: CARTOLA_CLUBES_2025}

ALL_CARTOLA_IDS = set(CARTOLA_CLUBES_2024) | set(CARTOLA_CLUBES_2025)

CLUBE_NOMES_POR_ID: dict[int, list[str]] = {
    293: ["Athletico", "Athletico-PR", "Atletico-PR", "Athletico Paranaense"],
    373: ["Atletico GO", "Atletico-GO", "Atletico Goianiense", "Atlético Goianiense"],
    282: ["Atletico-MG", "Atletico MG", "Atletico Mineiro", "Atlético Mineiro"],
    265: ["Bahia", "EC Bahia"],
    263: ["Botafogo", "Botafogo RJ", "Botafogo-RJ"],
    280: ["Bragantino", "Red Bull Bragantino", "RB Bragantino"],
    354: ["Ceara", "Ceará"],
    264: ["Corinthians", "SC Corinthians"],
    288: ["Criciuma", "Criciúma"],
    283: ["Cruzeiro", "Cruzeiro EC"],
    1371: ["Cuiaba", "Cuiabá"],
    262: ["Flamengo", "Flamengo RJ"],
    266: ["Fluminense", "Fluminense RJ"],
    356: ["Fortaleza", "Fortaleza EC"],
    284: ["Gremio", "Grêmio"],
    285: ["Internacional", "Inter", "SC Internacional"],
    286: ["Juventude", "EC Juventude"],
    2305: ["Mirassol", "Mirassol FC"],
    275: ["Palmeiras", "SE Palmeiras"],
    277: ["Santos", "Santos FC"],
    276: ["Sao Paulo", "São Paulo", "Sao Paulo FC"],
    292: ["Sport Recife", "Sport", "Sport Club do Recife"],
    267: ["Vasco", "Vasco da Gama", "CR Vasco"],
    287: ["Vitoria", "Vitória", "EC Vitória"],
}


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s)


def _build_clube_lookup() -> dict[str, int]:
    lookup = {}
    for cid, names in CLUBE_NOMES_POR_ID.items():
        for n in names:
            lookup[_norm(n)] = cid
    return lookup


CLUBE_LOOKUP = _build_clube_lookup()


def resolve_clube_id_by_name(name) -> int | None:
    return CLUBE_LOOKUP.get(_norm(name))


# ============================================================================
# Erros / logs
# ============================================================================

ANOMALIES: list[dict] = []
STRICT_ERRORS: list[dict] = []


def log_anomaly(tipo: str, **kw) -> None:
    ANOMALIES.append({"tipo": tipo, **kw})


def strict_error(tipo: str, **kw) -> None:
    rec = {"tipo": tipo}
    rec.update(kw)
    STRICT_ERRORS.append(rec)
    ANOMALIES.append({"tipo": tipo, "severity": "fatal", **kw})


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, float):
        try:
            return bool(int(v))
        except (ValueError, OverflowError):
            return False
    return str(v).strip().lower() in {"true", "1", "t", "yes", "sim"}


# ============================================================================
# Cobertura de scouts
# ============================================================================

SCOUT_COVERAGE: dict[tuple[int, str], dict] = defaultdict(lambda: {
    "n_atleta_rodada": 0,
    "n_col_present": 0,
    "n_col_present_numeric": 0,
    "n_col_present_nan": 0,
    "n_col_absent": 0,
    "n_delta_valid": 0,
    "n_delta_nan_schema_absence": 0,
    "n_delta_nan_after_gap": 0,
    "athletes_seen": set(),
    "athletes_with_gap": set(),
})


def _cov(season: int, scout: str) -> dict:
    return SCOUT_COVERAGE[(season, scout)]


def build_scout_coverage_report() -> pd.DataFrame:
    rows = []
    for (season, scout), d in sorted(SCOUT_COVERAGE.items()):
        n = d["n_atleta_rodada"] or 1
        rows.append({
            "temporada": season,
            "scout": scout,
            "n_atleta_rodada": d["n_atleta_rodada"],
            "pct_col_presente": d["n_col_present"] / n,
            "pct_valor_numerico": d["n_col_present_numeric"] / n,
            "pct_valor_nan_col_presente": d["n_col_present_nan"] / n,
            "pct_col_ausente": d["n_col_absent"] / n,
            "n_delta_valido": d["n_delta_valid"],
            "n_delta_nan_por_ausencia_schema": d["n_delta_nan_schema_absence"],
            "n_delta_nan_pos_gap": d["n_delta_nan_after_gap"],
            "n_atletas_com_gap_schema": len(d["athletes_with_gap"]),
            "n_atletas": len(d["athletes_seen"]),
        })
    return pd.DataFrame(rows)


# ============================================================================
# bzzoiro + football_data
# ============================================================================

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _attach_score_from_football_data(bz: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Anexa HG/AG do football_data_bra.csv ao DataFrame do bzzoiro.
    Link: (season, home_id, away_id) com ambos os ids resolvidos via
    resolve_clube_id_by_name. Não usa data, não usa round, não usa odds.
    Qualquer evento sem correspondência é erro fatal — nada de fallback.
    """
    if not FOOTBALL_DATA.exists():
        raise FileNotFoundError(f"Fonte ausente: {FOOTBALL_DATA}")

    fb = pd.read_csv(FOOTBALL_DATA)

    fb["_home_id"] = fb["Home"].map(resolve_clube_id_by_name)
    fb["_away_id"] = fb["Away"].map(resolve_clube_id_by_name)
    fb["_season"] = pd.to_numeric(fb["Season"], errors="coerce")
    fb["_HG"] = pd.to_numeric(fb["HG"], errors="coerce")
    fb["_AG"] = pd.to_numeric(fb["AG"], errors="coerce")

    fb_s = fb[fb["_season"] == season].copy()

    for _, r in fb_s[fb_s["_home_id"].isna()].iterrows():
        strict_error("unmapped_team_football_data",
                     temporada=season, nome=str(r["Home"]), lado="home")
    for _, r in fb_s[fb_s["_away_id"].isna()].iterrows():
        strict_error("unmapped_team_football_data",
                     temporada=season, nome=str(r["Away"]), lado="away")

    fb_ok = fb_s.dropna(subset=["_home_id", "_away_id", "_HG", "_AG"]).copy()
    if len(fb_ok) > 0:
        fb_ok["_home_id"] = fb_ok["_home_id"].astype(int)
        fb_ok["_away_id"] = fb_ok["_away_id"].astype(int)

    dups = fb_ok.duplicated(["_home_id", "_away_id"], keep=False)
    if dups.any():
        for _, r in fb_ok[dups].iterrows():
            strict_error("football_data_duplicate_pair",
                         temporada=season,
                         home_id=int(r["_home_id"]),
                         away_id=int(r["_away_id"]))

    lookup: dict[tuple[int, int], tuple[int, int]] = {
        (int(r["_home_id"]), int(r["_away_id"])): (int(r["_HG"]), int(r["_AG"]))
        for _, r in fb_ok.iterrows()
    }

    hg_out: list[float] = []
    ag_out: list[float] = []
    for _, r in bz.iterrows():
        hid = int(r["home_id"]) if pd.notna(r["home_id"]) else None
        aid = int(r["away_id"]) if pd.notna(r["away_id"]) else None
        if hid is None or aid is None or (hid, aid) not in lookup:
            strict_error("score_not_found_football_data",
                         temporada=season,
                         home_id=hid, away_id=aid,
                         home_raw=str(r["home_raw"]),
                         away_raw=str(r["away_raw"]))
            hg_out.append(np.nan)
            ag_out.append(np.nan)
        else:
            hg, ag = lookup[(hid, aid)]
            hg_out.append(hg)
            ag_out.append(ag)

    bz = bz.copy()
    bz["HG"] = hg_out
    bz["AG"] = ag_out
    return bz


def load_bzzoiro_events(season: int) -> pd.DataFrame:
    path = BZZOIRO_DIR / f"bzzoiro_events_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Fonte ausente: {path}")

    raw = pd.read_parquet(path)
    home_col = _pick_col(raw, ["home", "home_team", "home_name", "homeTeam", "Home"])
    away_col = _pick_col(raw, ["away", "away_team", "away_name", "awayTeam", "Away"])
    date_col = _pick_col(raw, ["date", "event_date", "kickoff", "datetime",
                                "start_time", "utcDate", "match_date"])
    round_col = _pick_col(raw, ["round_number", "round", "matchday",
                                  "rodada", "roundNumber", "round_num"])
    home_id_col = _pick_col(raw, ["home_team_id", "home_id", "homeTeamId", "homeId"])
    away_id_col = _pick_col(raw, ["away_team_id", "away_id", "awayTeamId", "awayId"])

    missing = [n for n, c in [("home", home_col), ("away", away_col),
                                ("round", round_col)] if c is None]
    if missing:
        raise RuntimeError(
            f"{path}: colunas essenciais não encontradas ({missing}). "
            f"Colunas: {list(raw.columns)}"
        )

    out = pd.DataFrame()
    out["home_raw"] = raw[home_col].astype(str)
    out["away_raw"] = raw[away_col].astype(str)
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT
    out["round_number_odds"] = pd.to_numeric(raw[round_col], errors="coerce")
    out["temporada"] = season

    if out["round_number_odds"].isna().any():
        strict_error("bzzoiro_missing_round_number",
                     temporada=season,
                     n_linhas_sem_round=int(out["round_number_odds"].isna().sum()))

    def _resolve_pair(name_series, id_series):
        by_name = name_series.map(resolve_clube_id_by_name)
        if id_series is None:
            return by_name, "name_only"
        ids = pd.to_numeric(id_series, errors="coerce")
        non_null = ids.dropna()
        id_universe_ok = len(non_null) > 0 and set(non_null.astype(int)).issubset(ALL_CARTOLA_IDS)
        if not id_universe_ok:
            return by_name, "name_only_ids_incompatible"
        by_id = ids.astype("Int64")
        for i, (n, ni) in enumerate(zip(by_name, by_id)):
            if pd.notna(ni) and pd.notna(n) and int(ni) != int(n):
                strict_error("team_id_name_conflict",
                             temporada=season,
                             nome=str(name_series.iloc[i]),
                             id_por_nome=int(n),
                             id_coluna=int(ni))
        resolved = by_id.where(by_id.notna(), by_name)
        return resolved, "id_preferred"

    home_id, mode_h = _resolve_pair(out["home_raw"],
                                    raw[home_id_col] if home_id_col else None)
    away_id, mode_a = _resolve_pair(out["away_raw"],
                                    raw[away_id_col] if away_id_col else None)
    out["home_id"] = home_id
    out["away_id"] = away_id
    out["resolution_mode"] = f"home={mode_h};away={mode_a}"

    for _, r in out[out["home_id"].isna()].iterrows():
        strict_error("unmapped_team_bzzoiro",
                     temporada=season, nome=r["home_raw"], lado="home")
    for _, r in out[out["away_id"].isna()].iterrows():
        strict_error("unmapped_team_bzzoiro",
                     temporada=season, nome=r["away_raw"], lado="away")

    out = out.dropna(subset=["home_id", "away_id"]).reset_index(drop=True)

    # --- placar real: football_data_bra.csv (link por season + home + away) ---
    out = _attach_score_from_football_data(out, season)
    return out


# ============================================================================
# caRtola
# ============================================================================

def _round_files(season_dir: Path) -> list[tuple[int, Path]]:
    seen: dict[int, Path] = {}
    for pat in ("rodada_*.csv", "rodada-*.csv"):
        for p in season_dir.glob(pat):
            m = re.search(r"rodada[_-](\d+)", p.stem, re.IGNORECASE)
            if m:
                seen.setdefault(int(m.group(1)), p)
    return sorted(seen.items())


def load_cartola_season(season: int):
    season_dir = CARTOLA_RAW / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(season_dir)

    round_dfs: dict[int, pd.DataFrame] = {}
    round_cols: dict[int, set[str]] = {}
    clube_nome_map: dict[int, str] = dict(CARTOLA_CLUBES_BY_SEASON.get(season, {}))

    for rodada, path in _round_files(season_dir):
        df = pd.read_csv(path, encoding="utf-8-sig")
        round_dfs[rodada] = df
        round_cols[rodada] = set(df.columns)

        if "atletas.clube_id" in df.columns:
            nome_col = next(
                (c for c in ("atletas.clube.id.full.name",
                             "atletas.clube_id.full.name",
                             "atletas.clube.nome")
                 if c in df.columns), None)
            if nome_col is not None:
                for _, r in df[["atletas.clube_id", nome_col]].dropna().drop_duplicates().iterrows():
                    try:
                        cid = int(r["atletas.clube_id"])
                    except (TypeError, ValueError):
                        continue
                    clube_nome_map.setdefault(cid, str(r[nome_col]))

    return round_dfs, round_cols, clube_nome_map


# ============================================================================
# Delta reconstruction
# ============================================================================

def build_player_deltas(round_dfs: dict, round_cols: dict, season: int) -> pd.DataFrame:
    per_atleta: dict[int, list[dict]] = defaultdict(list)

    for rodada in sorted(round_dfs):
        df = round_dfs[rodada]
        cols = round_cols[rodada]
        required = {"atletas.atleta_id", "atletas.clube_id", "atletas.posicao_id"}
        if not required.issubset(df.columns):
            strict_error("cartola_missing_required_cols",
                         temporada=season, rodada=rodada,
                         missing=sorted(required - set(df.columns)))
            continue

        for _, row in df.iterrows():
            try:
                aid = int(row["atletas.atleta_id"])
                cid = int(row["atletas.clube_id"])
                pid = int(row["atletas.posicao_id"])
            except (TypeError, ValueError):
                continue

            entry = {
                "rodada": rodada,
                "clube_id": cid,
                "posicao_id": pid,
                "apelido": str(row.get("atletas.apelido", "") or ""),
                "nome": str(row.get("atletas.nome", "") or ""),
                "entrou_em_campo": _parse_bool(row.get("atletas.entrou_em_campo", False)),
                "col_present": {},
                "raw": {},
            }
            for s in CUMULATIVE_SCOUTS:
                if s in cols:
                    entry["col_present"][s] = True
                    entry["raw"][s] = row.get(s, np.nan)
                else:
                    entry["col_present"][s] = False
                    entry["raw"][s] = None
            per_atleta[aid].append(entry)

    rows: list[dict] = []
    for aid, entries in per_atleta.items():
        entries.sort(key=lambda e: e["rodada"])
        last_known: dict[str, float | None] = {s: None for s in CUMULATIVE_SCOUTS}
        unknown: dict[str, bool] = {s: False for s in CUMULATIVE_SCOUTS}

        for e in entries:
            out = {
                "temporada": season,
                "atleta_id": aid,
                "rodada": e["rodada"],
                "clube_id": e["clube_id"],
                "posicao_id": e["posicao_id"],
                "apelido": e["apelido"],
                "nome": e["nome"],
                "entrou_em_campo": e["entrou_em_campo"],
            }
            for s in CUMULATIVE_SCOUTS:
                cov = _cov(season, s)
                cov["n_atleta_rodada"] += 1
                cov["athletes_seen"].add(aid)

                if not e["col_present"][s]:
                    cov["n_col_absent"] += 1
                    cov["n_delta_nan_schema_absence"] += 1
                    unknown[s] = True
                    cov["athletes_with_gap"].add(aid)
                    out[f"d_{s}"] = np.nan
                    continue

                cov["n_col_present"] += 1
                raw = e["raw"][s]

                if pd.isna(raw):
                    cov["n_col_present_nan"] += 1
                    if unknown[s]:
                        out[f"d_{s}"] = np.nan
                        cov["n_delta_nan_after_gap"] += 1
                    else:
                        if last_known[s] is None:
                            last_known[s] = 0.0
                        out[f"d_{s}"] = 0.0
                        cov["n_delta_valid"] += 1
                    continue

                cov["n_col_present_numeric"] += 1
                raw_f = float(raw)

                if unknown[s]:
                    out[f"d_{s}"] = np.nan
                    last_known[s] = raw_f
                    unknown[s] = False
                    cov["n_delta_nan_after_gap"] += 1
                    continue

                if last_known[s] is None:
                    delta = raw_f
                else:
                    delta = raw_f - last_known[s]

                if delta < -1e-9:
                    strict_error("negative_delta",
                                 temporada=season, rodada=e["rodada"],
                                 atleta_id=aid, scout=s,
                                 prev=last_known[s], curr=raw_f)
                    delta = 0.0
                out[f"d_{s}"] = delta
                last_known[s] = raw_f
                cov["n_delta_valid"] += 1

            rows.append(out)

    return pd.DataFrame(rows)


# ============================================================================
# Agregação time
# ============================================================================

def aggregate_to_team(player_deltas: pd.DataFrame) -> pd.DataFrame:
    participants = player_deltas[player_deltas["entrou_em_campo"]].copy()

    gks = participants[participants["posicao_id"] == GOLEIRO_POSICAO_ID]
    if len(gks) > 0:
        gk = gks.groupby(["rodada", "clube_id"]).agg(
            DE=("d_DE", lambda s: s.sum(min_count=1)),
            GS=("d_GS", lambda s: s.max()),
        ).reset_index()
    else:
        gk = pd.DataFrame(columns=["rodada", "clube_id", "DE", "GS"])

    if len(participants) > 0:
        cols = [f"d_{s}" for s in ADDITIVE_SCOUTS]
        sums = participants.groupby(["rodada", "clube_id"])[cols].sum(min_count=1)
        sums = sums.rename(columns={f"d_{s}": s for s in ADDITIVE_SCOUTS}).reset_index()
    else:
        sums = pd.DataFrame(columns=["rodada", "clube_id"] + ADDITIVE_SCOUTS)

    return sums.merge(gk, on=["rodada", "clube_id"], how="outer")


# ============================================================================
# Roster & participantes
# ============================================================================

def _is_tecnico_mask(grp: pd.DataFrame) -> pd.Series:
    pos_tec = grp["posicao_id"].astype(int).eq(TECNICO_POSICAO_ID)
    nome = grp["nome"].fillna("").astype(str)
    apelido = grp["apelido"].fillna("").astype(str)
    text = (nome + " " + apelido).map(_norm)
    name_tec = text.str.contains("tecnico|treinador", regex=True, na=False)
    return pos_tec | name_tec


def build_roster_and_participants(player_deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rodada, cid), grp in player_deltas.groupby(["rodada", "clube_id"], sort=False):
        is_tec = _is_tecnico_mask(grp)
        n_jogadores = int((~is_tec).sum())
        n_tecnicos = int(is_tec.sum())

        parts = grp[(grp["entrou_em_campo"]) & (~is_tec)].copy()
        parts = parts.sort_values("atleta_id")

        por_pos: dict[int, int] = defaultdict(int)
        participantes: list[dict] = []
        seen: set[int] = set()

        for _, p in parts.iterrows():
            aid = int(p["atleta_id"])
            if aid in seen:
                strict_error("duplicate_participant",
                             temporada=int(p["temporada"]), rodada=int(rodada),
                             clube_id=int(cid), atleta_id=aid)
                continue
            seen.add(aid)
            pid = int(p["posicao_id"])
            nome = (p["apelido"] or p["nome"]) or f"atleta_{aid}"
            participantes.append({
                "atleta_id": aid,
                "atleta_nome": str(nome),
                "posicao_id": pid,
                "posicao_nome": POSICAO_NOME.get(pid, "Desconhecido"),
            })
            por_pos[pid] += 1

        n_part = len(participantes)
        roster_presente = n_jogadores > 0

        if n_jogadores < n_part:
            strict_error("roster_jogadores_lt_participantes",
                         temporada=int(grp["temporada"].iloc[0]),
                         rodada=int(rodada), clube_id=int(cid),
                         roster_jogadores_raw=n_jogadores,
                         n_participantes=n_part)
        if n_tecnicos < 0:
            strict_error("roster_tecnicos_negative",
                         rodada=int(rodada), clube_id=int(cid),
                         valor=n_tecnicos)

        rows.append({
            "rodada": int(rodada),
            "clube_id": int(cid),
            "roster_presente": roster_presente,
            "roster_jogadores_raw": n_jogadores,
            "roster_tecnicos_raw": n_tecnicos,
            "n_participantes": n_part,
            "participantes_por_posicao": json.dumps(
                {str(k): int(v) for k, v in sorted(por_pos.items())},
                separators=(",", ":")),
            "participantes": participantes,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Link rodada caRtola <-> bzzoiro (estrito)
# ============================================================================

def link_round(season: int, rodada: int,
               clube_ids_cartola: set[int],
               bz: pd.DataFrame) -> pd.DataFrame:
    cand = bz[bz["round_number_odds"] == rodada].copy()

    if len(cand) != MATCHES_PER_ROUND:
        strict_error("round_match_count",
                     temporada=season, rodada=rodada,
                     encontrado=len(cand), esperado=MATCHES_PER_ROUND)
        return cand.iloc[0:0]

    clubs_bz = set(cand["home_id"].astype(int)).union(set(cand["away_id"].astype(int)))
    if clubs_bz != clube_ids_cartola:
        strict_error("round_club_mismatch",
                     temporada=season, rodada=rodada,
                     apenas_bz=sorted(clubs_bz - clube_ids_cartola),
                     apenas_cartola=sorted(clube_ids_cartola - clubs_bz))
        return cand.iloc[0:0]

    appearences = pd.concat([cand["home_id"], cand["away_id"]], ignore_index=True)
    if appearences.duplicated().any():
        dups = appearences[appearences.duplicated()].unique().tolist()
        strict_error("round_duplicate_club",
                     temporada=season, rodada=rodada, clubes=dups)
        return cand.iloc[0:0]

    return cand


# ============================================================================
# Season builder
# ============================================================================

def build_season(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[{season}] carregando caRtola ...")
    round_dfs, round_cols, clube_nome_map = load_cartola_season(season)
    print(f"[{season}] rodadas={len(round_dfs)}  clubes={len(clube_nome_map)}")

    print(f"[{season}] carregando bzzoiro + football_data ...")
    bz = load_bzzoiro_events(season)
    print(f"[{season}] bzzoiro: {len(bz)} eventos  modo={bz['resolution_mode'].iloc[0] if len(bz) else '-'}")

    print(f"[{season}] reconstruindo deltas ...")
    player_deltas = build_player_deltas(round_dfs, round_cols, season)
    team_scouts = aggregate_to_team(player_deltas)
    roster_part = build_roster_and_participants(player_deltas)

    print(f"[{season}] integrando partidas ...")
    match_rows: list[dict] = []
    for rodada in sorted(round_dfs.keys()):
        df_round = round_dfs[rodada]
        if "atletas.clube_id" not in df_round.columns:
            continue
        clube_ids_cartola = set(
            pd.to_numeric(df_round["atletas.clube_id"], errors="coerce")
            .dropna().astype(int).unique()
        )
        if len(clube_ids_cartola) != CLUBS_PER_ROUND:
            strict_error("cartola_round_club_count",
                         temporada=season, rodada=rodada,
                         encontrado=len(clube_ids_cartola),
                         esperado=CLUBS_PER_ROUND)
            continue

        matches = link_round(season, rodada, clube_ids_cartola, bz)
        for _, m in matches.iterrows():
            hid, aid = int(m["home_id"]), int(m["away_id"])
            hg, ag = m["HG"], m["AG"]
            if pd.isna(hg) or pd.isna(ag):
                strict_error("missing_score",
                             temporada=season, rodada=rodada,
                             partida=f"{m['home_raw']} x {m['away_raw']}")
                continue
            hg, ag = int(hg), int(ag)
            date_str = (m["date"].strftime("%Y-%m-%d %H:%M:%S")
                        if pd.notna(m["date"]) else "")
            pid = int(f"{season}{rodada:02d}{hid:05d}{aid:05d}")

            for cid, oid, mando, gm, gs in [
                (hid, aid, "C", hg, ag),
                (aid, hid, "F", ag, hg),
            ]:
                match_rows.append({
                    "temporada": season,
                    "rodada": rodada,
                    "partida_id": pid,
                    "data": date_str,
                    "clube_id": cid,
                    "clube_nome": clube_nome_map.get(cid, ""),
                    "adversario_id": oid,
                    "adversario_nome": clube_nome_map.get(oid, ""),
                    "mando": mando,
                    "gols_marcados": gm,
                    "gols_sofridos": gs,
                })

    matches_df = pd.DataFrame(match_rows)
    if matches_df.empty:
        return matches_df, player_deltas

    merged = matches_df.merge(team_scouts, on=["rodada", "clube_id"], how="left")
    merged = merged.merge(roster_part, on=["rodada", "clube_id"], how="left")

    merged["SG"] = (merged["gols_sofridos"] == 0)
    merged["FTT"] = merged[["FD", "FF", "FT", "G", "PP"]].sum(axis=1, min_count=1)
    merged["FA"] = merged[["FD", "G"]].sum(axis=1, min_count=1)

    mask_no_roster = merged["roster_presente"].isna()
    for _, r in merged[mask_no_roster].iterrows():
        strict_error("missing_roster_for_club_round",
                     temporada=season, rodada=int(r["rodada"]),
                     clube_id=int(r["clube_id"]))
    merged["roster_presente"] = merged["roster_presente"].fillna(False).astype(bool)
    merged["roster_jogadores_raw"] = merged["roster_jogadores_raw"].fillna(0).astype(int)
    merged["roster_tecnicos_raw"] = merged["roster_tecnicos_raw"].fillna(0).astype(int)
    merged["n_participantes"] = merged["n_participantes"].fillna(0).astype(int)
    merged["participantes_por_posicao"] = merged["participantes_por_posicao"].fillna("{}")
    merged["participantes"] = merged["participantes"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return merged, player_deltas


# ============================================================================
# Cast / write
# ============================================================================

def cast_final_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["temporada"] = df["temporada"].astype("int64")
    out["rodada"] = df["rodada"].astype("int64")
    out["partida_id"] = df["partida_id"].astype("int64")
    out["data"] = df["data"].astype(str)
    out["clube_id"] = df["clube_id"].astype("int64")
    out["clube_nome"] = df["clube_nome"].fillna("").astype(str)
    out["adversario_id"] = df["adversario_id"].astype("int64")
    out["adversario_nome"] = df["adversario_nome"].fillna("").astype(str)
    out["mando"] = df["mando"].astype(str)
    out["gols_marcados"] = df["gols_marcados"].astype("int64")
    out["gols_sofridos"] = df["gols_sofridos"].astype("int64")
    out["SG"] = df["SG"].astype(bool)
    for c in ["DE", "DP", "DS", "GS", "FC", "G", "A", "FTT", "FA",
              "FS", "I", "FD", "FF", "FT", "PP"]:
        out[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce").astype("float64")
    out["roster_presente"] = df["roster_presente"].astype(bool)
    out["roster_jogadores_raw"] = df["roster_jogadores_raw"].astype("int64")
    out["roster_tecnicos_raw"] = df["roster_tecnicos_raw"].astype("int64")
    out["n_participantes"] = df["n_participantes"].astype("int64")
    out["participantes_por_posicao"] = df["participantes_por_posicao"].astype(str)
    out["participantes"] = df["participantes"]
    return out[FINAL_ORDER]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ============================================================================
# Validações finais estritas
# ============================================================================

def check_partida_id_semantics(final: pd.DataFrame) -> None:
    for pid, grp in final.groupby("partida_id"):
        if len(grp) != 2:
            strict_error("partida_id_row_count",
                         partida_id=int(pid), n=len(grp))
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        if not (a["adversario_id"] == b["clube_id"]
                and b["adversario_id"] == a["clube_id"]):
            strict_error("partida_id_not_inverse",
                         partida_id=int(pid),
                         a=[int(a["clube_id"]), int(a["adversario_id"])],
                         b=[int(b["clube_id"]), int(b["adversario_id"])])
        mandos = sorted(grp["mando"].tolist())
        if mandos != ["C", "F"]:
            strict_error("partida_id_mando_invalid",
                         partida_id=int(pid), mandos=mandos)
        if int(a["gols_marcados"]) != int(b["gols_sofridos"]) or \
           int(a["gols_sofridos"]) != int(b["gols_marcados"]):
            strict_error("partida_id_score_not_inverse",
                         partida_id=int(pid),
                         a=[int(a["gols_marcados"]), int(a["gols_sofridos"])],
                         b=[int(b["gols_marcados"]), int(b["gols_sofridos"])])


def check_round_structure(final: pd.DataFrame) -> None:
    for (season, rodada), grp in final.groupby(["temporada", "rodada"]):
        if grp["partida_id"].nunique() != MATCHES_PER_ROUND:
            strict_error("round_match_count_final",
                         temporada=int(season), rodada=int(rodada),
                         n=int(grp["partida_id"].nunique()))
        if grp["clube_id"].nunique() != CLUBS_PER_ROUND:
            strict_error("round_club_count_final",
                         temporada=int(season), rodada=int(rodada),
                         n=int(grp["clube_id"].nunique()))
        if grp["clube_id"].duplicated().any():
            dups = grp.loc[grp["clube_id"].duplicated(), "clube_id"].unique().tolist()
            strict_error("round_club_duplicated_final",
                         temporada=int(season), rodada=int(rodada), clubes=dups)


def check_season_totals(final: pd.DataFrame) -> None:
    for season, grp in final.groupby("temporada"):
        if int(grp["partida_id"].nunique()) != MATCHES_PER_SEASON:
            strict_error("season_match_count",
                         temporada=int(season),
                         n=int(grp["partida_id"].nunique()),
                         esperado=MATCHES_PER_SEASON)
        if int(grp["rodada"].nunique()) != N_ROUNDS:
            strict_error("season_round_count",
                         temporada=int(season),
                         n=int(grp["rodada"].nunique()),
                         esperado=N_ROUNDS)
        if len(grp) != ROWS_PER_SEASON:
            strict_error("season_row_count",
                         temporada=int(season), n=len(grp),
                         esperado=ROWS_PER_SEASON)


# ============================================================================
# Reports
# ============================================================================

def run_validations(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, rodada), grp in final.groupby(["temporada", "rodada"]):
        sym_ok, placar_ok = True, True
        for _, g_p in grp.groupby("partida_id"):
            if len(g_p) != 2:
                sym_ok = False; continue
            a, b = g_p.iloc[0], g_p.iloc[1]
            if a["adversario_id"] != b["clube_id"] or b["adversario_id"] != a["clube_id"]:
                sym_ok = False
            if int(a["gols_sofridos"]) != int(b["gols_marcados"]):
                placar_ok = False
            if int(b["gols_sofridos"]) != int(a["gols_marcados"]):
                placar_ok = False
        rows.append({
            "temporada": int(season), "rodada": int(rodada),
            "n_rows": len(grp),
            "n_matches": int(grp["partida_id"].nunique()),
            "n_clubs": int(grp["clube_id"].nunique()),
            "n_C": int((grp["mando"] == "C").sum()),
            "n_F": int((grp["mando"] == "F").sum()),
            "duplicate_keys": int(grp.duplicated(
                ["temporada", "rodada", "partida_id", "clube_id"]).sum()),
            "symmetric": bool(sym_ok),
            "placar_consistente": bool(placar_ok),
        })
    return pd.DataFrame(rows)


def build_gs_validation(player_deltas: pd.DataFrame,
                        final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gks = player_deltas[(player_deltas["posicao_id"] == GOLEIRO_POSICAO_ID)
                        & player_deltas["entrou_em_campo"]]
    rows = []
    for _, m in final.iterrows():
        sub = gks[(gks["rodada"] == m["rodada"]) & (gks["clube_id"] == m["clube_id"])]
        vals = sub["d_GS"].dropna().tolist()
        if vals:
            gs_sum, gs_max, n = float(sum(vals)), float(max(vals)), len(vals)
        else:
            gs_sum = gs_max = np.nan; n = 0
        gs_odds = int(m["gols_sofridos"])
        rows.append({
            "temporada": int(m["temporada"]),
            "rodada": int(m["rodada"]),
            "clube_id": int(m["clube_id"]),
            "gols_sofridos_odds": gs_odds,
            "gs_goleiro_sum": gs_sum,
            "gs_goleiro_max": gs_max,
            "n_goleiros_participantes": n,
            "match_sum": (abs(gs_sum - gs_odds) < 1e-9) if pd.notna(gs_sum) else None,
            "match_max": (abs(gs_max - gs_odds) < 1e-9) if pd.notna(gs_max) else None,
        })
    detail = pd.DataFrame(rows)

    summary_rows = []
    for season, g in detail.groupby("temporada"):
        summary_rows.append({
            "temporada": int(season),
            "n_linhas": len(g),
            "n_match_sum": int(g["match_sum"].fillna(False).sum()),
            "n_match_max": int(g["match_max"].fillna(False).sum()),
            "n_sum_ne_max": int((g["gs_goleiro_sum"] != g["gs_goleiro_max"]).sum()),
            "n_max_ne_odds": int((~g["match_max"].fillna(False)).sum()),
            "n_sem_goleiro_participante": int((g["n_goleiros_participantes"] == 0).sum()),
        })
    summary = pd.DataFrame(summary_rows)

    div = detail.loc[detail["match_max"].fillna(False).eq(False)].copy()
    div.to_csv(REPORTS_DIR / "exp12_historical_gs_divergences.csv", index=False)
    return detail, summary


def build_club_mapping(bz_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    seen_por_season: dict[int, set[str]] = {}
    for season, bz in bz_by_season.items():
        seen_por_season[season] = (
            set(bz["home_raw"].unique()) | set(bz["away_raw"].unique())
        )
    all_seen = set().union(*seen_por_season.values()) if seen_por_season else set()

    rows = []
    for name in sorted(all_seen):
        cid = resolve_clube_id_by_name(name)
        seasons_present = [s for s, names in seen_por_season.items() if name in names]
        rows.append({
            "nome_bzzoiro": name,
            "nome_normalizado": _norm(name),
            "clube_id": int(cid) if cid is not None else None,
            "clube_nome_cartola": CARTOLA_CLUBES_BY_SEASON.get(
                seasons_present[0], {}).get(cid, "") if cid is not None else "",
            "temporadas": ",".join(str(s) for s in sorted(seasons_present)),
            "status_validacao": "ok" if cid is not None else "nao_mapeado",
        })
    return pd.DataFrame(rows)


def dump_reports(seasons_out, player_deltas_by_season, bz_by_season, final=None):
    print("\n[reports] gerando ...")
    if final is not None:
        run_validations(final).to_csv(
            REPORTS_DIR / "exp12_historical_validation.csv", index=False)
        pd_all = pd.concat(player_deltas_by_season.values(), ignore_index=True)
        detail, summary = build_gs_validation(pd_all, final)
        detail.to_csv(REPORTS_DIR / "exp12_historical_gs_validation.csv", index=False)
        summary.to_csv(REPORTS_DIR / "exp12_historical_gs_summary.csv", index=False)
    build_scout_coverage_report().to_csv(
        REPORTS_DIR / "exp12_historical_scout_coverage.csv", index=False)
    anom_df = pd.DataFrame(ANOMALIES) if ANOMALIES else pd.DataFrame(columns=["tipo"])
    anom_df.to_csv(REPORTS_DIR / "exp12_historical_anomalies.csv", index=False)
    build_club_mapping(bz_by_season).to_csv(
        REPORTS_DIR / "exp12_club_mapping.csv", index=False)


def raise_strict(msg: str) -> None:
    detail = "\n".join(
        f"  [{e['tipo']}] { {k: v for k, v in e.items() if k != 'tipo'} }"
        for e in STRICT_ERRORS[:30]
    )
    raise RuntimeError(
        f"{msg}\n{len(STRICT_ERRORS)} erro(s) fatal(is):\n{detail}"
        + (f"\n  ... (+{len(STRICT_ERRORS)-30})" if len(STRICT_ERRORS) > 30 else "")
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    seasons_out: list[pd.DataFrame] = []
    player_deltas_by_season: dict[int, pd.DataFrame] = {}
    bz_by_season: dict[int, pd.DataFrame] = {}

    for season in SEASONS:
        season_df, pdlt = build_season(season)
        seasons_out.append(season_df)
        player_deltas_by_season[season] = pdlt
        bz_by_season[season] = load_bzzoiro_events(season)

    if STRICT_ERRORS:
        dump_reports(seasons_out, player_deltas_by_season, bz_by_season)
        raise_strict("Erros fatais durante construção (nada foi escrito).")

    final = pd.concat(seasons_out, ignore_index=True)
    final = final.sort_values(
        ["temporada", "rodada", "partida_id", "mando"]
    ).reset_index(drop=True)
    final = cast_final_schema(final)

    check_partida_id_semantics(final)
    check_round_structure(final)
    check_season_totals(final)

    dump_reports(seasons_out, player_deltas_by_season, bz_by_season, final=final)

    if STRICT_ERRORS:
        raise_strict("Erros fatais de validação (parquet NÃO foi escrito).")

    write_parquet(final, OUT_PARQUET)
    print(f"[out] {OUT_PARQUET} ({len(final)} linhas, {final.shape[1]} colunas)")

    print("\n=== Resumo ===")
    for season, g in final.groupby("temporada"):
        print(f"  {season}: linhas={len(g)}  partidas={g['partida_id'].nunique()}  "
              f"clubes={g['clube_id'].nunique()}  rodadas={g['rodada'].nunique()}")
    tipos = defaultdict(int)
    for a in ANOMALIES:
        tipos[a["tipo"]] += 1
    print(f"Anomalias: {len(ANOMALIES)}  (fatais: {len(STRICT_ERRORS)})")
    for t, n in sorted(tipos.items(), key=lambda x: -x[1]):
        print(f"  - {t}: {n}")


if __name__ == "__main__":
    main()
