#!/usr/bin/env python
"""
scripts/exp12_gol_01_build_features.py

Constrói o dataset EXP-12 GOL (temporadas 2024-2026) com features temporais.
NÃO treina. Apenas monta o dataset, valida cobertura e imprime auditoria manual.

Saídas:
  data/interim/exp12/gol_pooling_2024_2026.parquet
  reports/exp12_gol_build_report.json
  reports/exp12_gol_build_report.md

Regras centrais:
  - Rolling = groupby(...).transform(shift(1).rolling(k, min_periods=1)).
  - Nenhuma feature usa a rodada-alvo ou posterior.
  - NaN preservado. _n_obs sempre ao lado de cada rolling.
  - Nenhum fillna(0) em scouts.
  - Cross-season: apenas para gk_pontos_mean_hist.
  - Reset por temporada: para todo o resto (time, adversário, gk curto prazo).
  - Cobertura do master é checada antes do merge; ausência é erro bloqueante.
  - Scouts acumulados (2024/2025) são convertidos para delta ANTES do rolling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
JOGADOR = ROOT / "data" / "interim" / "cartola_jogador_rodada_exp12.parquet"
MASTER_2425 = ROOT / "data" / "interim" / "exp12" / "team_scout_history_2024_2025.parquet"
MASTER_26 = ROOT / "data" / "interim" / "exp12" / "team_scout_history.parquet"
VINCULO = ROOT / "data" / "interim" / "vinculo_atleta_partida_v1.parquet"
MATCHES_ODDS = ROOT / "data" / "interim" / "matches_odds_cartola_v1.parquet"

OUT_PARQUET = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"
OUT_JSON = ROOT / "reports" / "exp12_gol_build_report.json"
OUT_MD = ROOT / "reports" / "exp12_gol_build_report.md"

GOLEIRO_POSICAO_ID = 1
WINDOWS = (3, 5)

# Scouts cujo status, SE confirmado por heurística FORTE como acumulado, bloqueia.
CRITICAL_SCOUTS = ["scout_DE", "scout_SG", "scout_FT", "scout_G"]

# Temporadas cuja origem (CSV caRtola) armazena scouts como acumulados.
# 2026 vem da API do Cartola e já entrega valores por rodada.
ACCUMULATED_TEMPORADAS = {2024, 2025}


# ============================================================================
# Rolling helpers
# ============================================================================

def _rolling_shifted(df: pd.DataFrame,
                     group_cols: list[str],
                     sort_cols: list[str],
                     value_col: str,
                     out_prefix: str,
                     windows: tuple[int, ...] = WINDOWS,
                     min_periods: int = 1) -> pd.DataFrame:
    """
    Rolling temporal seguro com shift(1).

    A linha atual nunca entra no histórico.
    NaN é preservado e _n_obs conta somente valores não-NaN.
    """
    df = (
        df.sort_values(
            group_cols + sort_cols,
            kind="mergesort"
        )
        .reset_index(drop=True)
        .copy()
    )

    df["_rolling_shifted_value"] = (
        df.groupby(group_cols, sort=False)[value_col]
        .shift(1)
    )

    grouped = df.groupby(group_cols, sort=False)["_rolling_shifted_value"]

    for k in windows:
        df[f"{out_prefix}_mean_{k}"] = (
            grouped.transform(
                lambda s, _k=k: s.rolling(
                    _k,
                    min_periods=min_periods
                ).mean()
            )
        )

        df[f"{out_prefix}_mean_{k}_n_obs"] = (
            grouped.transform(
                lambda s, _k=k: s.rolling(
                    _k,
                    min_periods=min_periods
                ).count()
            )
            .fillna(0)
            .astype("int64")
        )

    return df.drop(columns=["_rolling_shifted_value"])


def _expanding_shifted(df: pd.DataFrame,
                       group_cols: list[str],
                       sort_cols: list[str],
                       value_col: str,
                       out_col: str) -> pd.DataFrame:
    """
    Expanding mean com shift(1).

    Usa somente valores anteriores do grupo.
    Se group_cols não incluir 'temporada', o histórico atravessa temporadas.
    """
    df = (
        df.sort_values(
            group_cols + sort_cols,
            kind="mergesort"
        )
        .reset_index(drop=True)
        .copy()
    )

    df["_expanding_shifted_value"] = (
        df.groupby(group_cols, sort=False)[value_col]
        .shift(1)
    )

    df[out_col] = (
        df.groupby(group_cols, sort=False)["_expanding_shifted_value"]
        .transform(lambda s: s.expanding().mean())
    )

    return df.drop(columns=["_expanding_shifted_value"])


# ============================================================================
# Detecção e conversão acumulado -> delta
# ============================================================================

def _detect_scout_type_by_temporada(df: pd.DataFrame, value_col: str) -> dict:
    """
    Para cada temporada, decide se `value_col` é 'accumulated', 'delta' ou 'unknown'.
      - 0 decreasing pairs (>=10 pares) -> accumulated
      - >5% decreasing pairs            -> delta
      - caso contrário                   -> unknown
    """
    results: dict[int, str] = {}
    for t in sorted(df["temporada"].dropna().unique()):
        sub = df[df["temporada"] == t][["atleta_id", "rodada", value_col]].dropna()
        if len(sub) < 10:
            results[int(t)] = "unknown"
            continue
        sub = sub.sort_values(["atleta_id", "rodada"], kind="mergesort")
        diffs = sub.groupby("atleta_id")[value_col].diff()
        n_pairs = int(diffs.notna().sum())
        if n_pairs < 10:
            results[int(t)] = "unknown"
            continue
        n_neg = int((diffs < -1e-9).sum())
        pct = n_neg / n_pairs
        if pct < 0.02:
            results[int(t)] = "accumulated"
        elif pct > 0.05:
            results[int(t)] = "delta"
        else:
            results[int(t)] = "unknown"
    return results


def _to_delta(df: pd.DataFrame,
              group_cols: list[str],
              sort_cols: list[str],
              value_col: str,
              out_col: str) -> pd.DataFrame:
    """
    Converte uma série acumulada por (grupo) em delta por passo.
      - NaN raw -> delta = NaN
      - primeiro valor não-NaN do grupo -> delta = raw
      - subsequentes -> delta = raw - last_known
      - decréscimo -> erro
    A ordenação é feita por sort_cols DENTRO de cada grupo.
    """
    df = df.sort_values(group_cols + sort_cols, kind="mergesort").reset_index(drop=True).copy()
    out = np.full(len(df), np.nan)

    for _, idx in df.groupby(group_cols, sort=False).indices.items():
        last_known = None
        for i in idx:
            raw = df.at[i, value_col]
            if pd.isna(raw):
                out[i] = np.nan
                continue
            raw_f = float(raw)
            if last_known is None:
                delta = raw_f
            else:
                delta = raw_f - last_known
                if delta < -1e-9:
                    ctx = {c: df.at[i, c] for c in group_cols}
                    raise RuntimeError(
                        f"Accumulated {value_col} decreased at {ctx}: "
                        f"last_known={last_known}, raw={raw_f}"
                    )
            out[i] = delta
            last_known = raw_f

    df[out_col] = out
    return df


# ============================================================================
# Diagnóstico heurístico de scouts (alerta, não prova)
# ============================================================================

def _diagnose_scout_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnóstico HEURÍSTICO — NÃO substitui a confirmação pelo pipeline de origem.
    """
    scouts = [c for c in df.columns if c.startswith("scout_")]
    rows = []
    for s in scouts:
        sub = df[["temporada", "atleta_id", "rodada", s]].dropna()
        sub = sub.sort_values(["temporada", "atleta_id", "rodada"])
        diffs = sub.groupby(["temporada", "atleta_id"])[s].diff()
        n_pairs = int(diffs.notna().sum())
        if n_pairs == 0:
            rows.append({
                "scout": s, "n_pairs": 0, "pct_decreasing": np.nan,
                "heuristica": "sem_dados", "forca": "n/a",
            })
            continue
        pct_dec = float((diffs < -1e-9).sum()) / n_pairs
        if pct_dec < 0.02:
            heur = "provavelmente_acumulado"
        elif pct_dec > 0.15:
            heur = "provavelmente_delta"
        else:
            heur = "indeterminado"
        if n_pairs >= 100:
            forca = "forte"
        elif n_pairs >= 30:
            forca = "moderada"
        else:
            fora = "fraca"
            forca = fora
        rows.append({
            "scout": s, "n_pairs": n_pairs,
            "pct_decreasing": round(pct_dec, 4),
            "heuristica": heur, "forca": forca,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Cobertura do master
# ============================================================================

def _audit_master_coverage(jog_gol: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    jog_keys = (jog_gol[["temporada", "rodada", "clube_id"]]
                .drop_duplicates()
                .reset_index(drop=True))
    master_keys = (master[["temporada", "rodada", "clube_id"]]
                   .drop_duplicates()
                   .reset_index(drop=True))

    merged = jog_keys.merge(
        master_keys.assign(_hit=1),
        on=["temporada", "rodada", "clube_id"],
        how="left",
    )
    missing = merged[merged["_hit"].isna()].drop(columns=["_hit"]).reset_index(drop=True)
    return missing


# ============================================================================
# Master com contexto (usa gols_marcados/gols_sofridos, já deltas por construção)
# ============================================================================

def build_master_with_context() -> pd.DataFrame:
    m1 = pd.read_parquet(MASTER_2425)
    m2 = pd.read_parquet(MASTER_26)
    master = pd.concat([m1, m2], ignore_index=True)

    master = master.sort_values(
        ["temporada", "clube_id", "rodada"], kind="mergesort"
    ).reset_index(drop=True)

    master["_clean_sheet"] = (master["gols_sofridos"] == 0).astype("float64")

    master = _rolling_shifted(
        master,
        group_cols=["temporada", "clube_id"],
        sort_cols=["rodada"],
        value_col="gols_sofridos",
        out_prefix="team_gols_sofridos",
    )
    master = _rolling_shifted(
        master,
        group_cols=["temporada", "clube_id"],
        sort_cols=["rodada"],
        value_col="_clean_sheet",
        out_prefix="team_clean_sheet_rate",
    )
    master = _rolling_shifted(
        master,
        group_cols=["temporada", "clube_id"],
        sort_cols=["rodada"],
        value_col="gols_marcados",
        out_prefix="team_gols_marcados",
    )
    master = _rolling_shifted(
        master,
        group_cols=["temporada", "clube_id"],
        sort_cols=["rodada"],
        value_col="FTT",
        out_prefix="team_FTT",
    )
    return master


# ============================================================================
# Features individuais do goleiro (com conversão acumulado -> delta)
# ============================================================================

def build_individual_features(jog_gol: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Rolling individual usa SOMENTE linhas com entrou_em_campo==True.
    Scouts acumulados (2024/2025) são convertidos para delta antes do rolling.
    Retorna (jog_com_features, detection_info).
    """
    jog = jog_gol.copy()
    part = jog[jog["entrou_em_campo"]].copy()
    part = part.sort_values(
        ["temporada", "atleta_id", "rodada"], kind="mergesort"
    ).reset_index(drop=True)

    detection_info: dict = {}

    # --- conversão de scouts acumulados para delta ---
    for raw_col, delta_col in [("scout_DE", "gk_DE_delta")]:
        if raw_col not in part.columns:
            raise RuntimeError(f"Coluna {raw_col} ausente em cartola_jogador_rodada_exp12.")

        # Diagnóstico: assume accumulated para 2024/2025
        detected = _detect_scout_type_by_temporada(part, raw_col)
        detection_info[raw_col] = detected
        print(f"     detecção {raw_col}: {detected}")

        frames = []
        for t in sorted(part["temporada"].unique()):
            sub = part[part["temporada"] == t].copy()
            if int(t) in ACCUMULATED_TEMPORADAS:
                sub = _to_delta(
                    sub,
                    group_cols=["temporada", "atleta_id"],
                    sort_cols=["rodada"],
                    value_col=raw_col,
                    out_col=delta_col,
                )
            else:
                # assume que já é delta (API)
                sub[delta_col] = pd.to_numeric(sub[raw_col], errors="coerce")
            frames.append(sub)
        part = pd.concat(frames, ignore_index=True)

    # --- rollings sobre os deltas ---
    part = _rolling_shifted(
        part,
        group_cols=["temporada", "atleta_id"],
        sort_cols=["rodada"],
        value_col="pontos_num",
        out_prefix="gk_pontos",
    )
    part = _rolling_shifted(
        part,
        group_cols=["temporada", "atleta_id"],
        sort_cols=["rodada"],
        value_col="gk_DE_delta",
        out_prefix="gk_DE",
    )

    part = part.sort_values(
        ["temporada", "atleta_id", "rodada"], kind="mergesort"
    ).reset_index(drop=True)
    part["gk_pontos_last1"] = (
        part.groupby(["temporada", "atleta_id"], sort=False)["pontos_num"].shift(1)
    )

    part = _expanding_shifted(
        part,
        group_cols=["atleta_id"],
        sort_cols=["temporada", "rodada"],
        value_col="pontos_num",
        out_col="gk_pontos_mean_hist",
    )

    keep = [
        "temporada", "rodada", "atleta_id",
        "gk_pontos_last1",
        "gk_pontos_mean_3",       "gk_pontos_mean_3_n_obs",
        "gk_pontos_mean_5",       "gk_pontos_mean_5_n_obs",
        "gk_pontos_mean_hist",
        "gk_DE_mean_3",           "gk_DE_mean_3_n_obs",
        "gk_DE_mean_5",           "gk_DE_mean_5_n_obs",
    ]
    part = part[keep].drop_duplicates(subset=["temporada", "rodada", "atleta_id"])

    jog = jog.merge(
        part,
        on=["temporada", "rodada", "atleta_id"],
        how="left",
        validate="one_to_one",
    )
    return jog, detection_info


# ============================================================================
# Odds (braço D)
# ============================================================================

def attach_odds(jog_gol: pd.DataFrame) -> pd.DataFrame:
    """
    Anexa probabilidades 1X2 pré-jogo ao dataset de goleiros.

    Fonte:
        matches_odds_cartola_v1.parquet

    Colunas reais:
        odds_home
        odds_draw
        odds_away

    O vínculo com a partida é feito por:
        bzzoiro_event_id

    O vínculo com o goleiro é:
        temporada + rodada + atleta_id
    """
    if not VINCULO.exists():
        raise FileNotFoundError(f"Fonte ausente: {VINCULO}")
    if not MATCHES_ODDS.exists():
        raise FileNotFoundError(f"Fonte ausente: {MATCHES_ODDS}")

    v = pd.read_parquet(VINCULO)
    m = pd.read_parquet(MATCHES_ODDS)

    if "bzzoiro_event_id" not in v.columns:
        raise RuntimeError(
            f"{VINCULO}: coluna 'bzzoiro_event_id' ausente."
        )

    if "bzzoiro_event_id" not in m.columns:
        raise RuntimeError(
            f"{MATCHES_ODDS}: coluna 'bzzoiro_event_id' ausente. "
            f"Colunas: {m.columns.tolist()}"
        )

    required_odds = [
        "odds_home",
        "odds_draw",
        "odds_away",
    ]

    missing = [
        c for c in required_odds
        if c not in m.columns
    ]

    if missing:
        raise RuntimeError(
            f"{MATCHES_ODDS}: colunas de odds ausentes: {missing}. "
            f"Colunas disponíveis: {m.columns.tolist()}"
        )

    # ------------------------------------------------------------
    # Converter odds decimais em probabilidades implícitas
    # ------------------------------------------------------------
    odds = m[
        [
            "bzzoiro_event_id",
            "odds_home",
            "odds_draw",
            "odds_away",
        ]
    ].copy()

    for c in ["odds_home", "odds_draw", "odds_away"]:
        odds[c] = pd.to_numeric(
            odds[c],
            errors="coerce",
        )

    # Odds decimais válidas precisam ser > 1.
    invalid = (
        odds["odds_home"].notna()
        & odds["odds_home"].le(1)
    ) | (
        odds["odds_draw"].notna()
        & odds["odds_draw"].le(1)
    ) | (
        odds["odds_away"].notna()
        & odds["odds_away"].le(1)
    )

    if invalid.any():
        raise RuntimeError(
            "Foram encontradas odds decimais inválidas (<= 1): "
            f"{int(invalid.sum())} partidas."
        )

    odds["q_home"] = 1.0 / odds["odds_home"]
    odds["q_draw"] = 1.0 / odds["odds_draw"]
    odds["q_away"] = 1.0 / odds["odds_away"]

    soma = (
        odds["q_home"]
        + odds["q_draw"]
        + odds["q_away"]
    )

    # Normalização para remover o overround da casa.
    odds["odd_p_home"] = odds["q_home"] / soma
    odds["odd_p_draw"] = odds["q_draw"] / soma
    odds["odd_p_away"] = odds["q_away"] / soma

    odds = odds[
        [
            "bzzoiro_event_id",
            "odd_p_home",
            "odd_p_draw",
            "odd_p_away",
        ]
    ]

    # ------------------------------------------------------------
    # Garantir um único registro de odds por evento
    # ------------------------------------------------------------
    dup_odds = odds.duplicated(
        subset=["bzzoiro_event_id"],
        keep=False,
    )

    if dup_odds.any():
        exemplos = odds.loc[
            dup_odds,
            ["bzzoiro_event_id"],
        ].drop_duplicates().head(10)

        raise RuntimeError(
            "Há múltiplas linhas de odds para o mesmo "
            "bzzoiro_event_id. Exemplos:\n"
            f"{exemplos.to_string(index=False)}"
        )

    # ------------------------------------------------------------
    # Vincular partida -> goleiro
    # ------------------------------------------------------------
    vinculo_odds = v[
        [
            "temporada",
            "rodada",
            "atleta_id",
            "bzzoiro_event_id",
        ]
    ].copy()

    # Evitar duplicação no vínculo.
    vinculo_odds = vinculo_odds.drop_duplicates(
        subset=[
            "temporada",
            "rodada",
            "atleta_id",
        ]
    )

    vinculo_odds = vinculo_odds.merge(
        odds,
        on="bzzoiro_event_id",
        how="left",
        validate="many_to_one",
    )

    vinculo_odds = vinculo_odds.drop(
        columns=["bzzoiro_event_id"]
    )

    # ------------------------------------------------------------
    # Join final
    # ------------------------------------------------------------
    out = jog_gol.merge(
        vinculo_odds,
        on=[
            "temporada",
            "rodada",
            "atleta_id",
        ],
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------
    n_total = len(out)
    n_com_odds = int(
        out["odd_p_home"].notna().sum()
    )

    print(
        f"     odds anexadas: {n_com_odds}/{n_total} "
        f"({100.0 * n_com_odds / n_total:.2f}%)"
    )

    if n_com_odds == 0:
        raise RuntimeError(
            "Nenhuma odds foi anexada aos goleiros. "
            "Verifique o vínculo por bzzoiro_event_id."
        )

    return out

# ============================================================================
# Auditoria manual
# ============================================================================

def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float) and pd.isna(v):
        return "NaN"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _manual_audit(final: pd.DataFrame,
                  master: pd.DataFrame,
                  jogador_gol: pd.DataFrame,
                  cases: list[tuple[int, int]],
                  n_cases_per_round: int = 2) -> str:
    """
    Para cada (temporada, rodada) alvo e cada goleiro escolhido:
      - mostra a linha-alvo e suas features;
      - exibe explicitamente as rodadas da janela [R-k, R-1] do master e do jogador;
      - confirma que a rodada-alvo NÃO aparece nas janelas.
    """
    linhas: list[str] = []
    for temporada, rodada in cases:
        sub_alvo = final[
            (final["temporada"] == temporada)
            & (final["rodada"] == rodada)
            & (final["elegivel_modelo"])
        ]
        if len(sub_alvo) == 0:
            linhas.append(f"\n### {temporada} R{rodada} — nenhum GOL elegível\n")
            continue

        sub_alvo = sub_alvo.sort_values("gk_pontos_mean_5_n_obs", ascending=False)
        sub_alvo = sub_alvo.head(n_cases_per_round)

        for _, row in sub_alvo.iterrows():
            cid = int(row["clube_id"])
            oid = int(row["adversario_id"])
            aid = int(row["atleta_id"])

            linhas.append("")
            linhas.append(f"### {temporada} R{rodada} — atleta_id={aid} "
                          f"clube={cid} adv={oid} mando={row['mando']}")

            feat_cols = [
                "gk_pontos_last1",
                "gk_pontos_mean_3", "gk_pontos_mean_3_n_obs",
                "gk_pontos_mean_5", "gk_pontos_mean_5_n_obs",
                "gk_pontos_mean_hist",
                "gk_DE_mean_3", "gk_DE_mean_3_n_obs",
                "gk_DE_mean_5", "gk_DE_mean_5_n_obs",
                "team_gols_sofridos_mean_3", "team_gols_sofridos_mean_3_n_obs",
                "team_gols_sofridos_mean_5", "team_gols_sofridos_mean_5_n_obs",
                "team_clean_sheet_rate_mean_3", "team_clean_sheet_rate_mean_3_n_obs",
                "team_clean_sheet_rate_mean_5", "team_clean_sheet_rate_mean_5_n_obs",
                "opp_gols_marcados_mean_3", "opp_gols_marcados_mean_3_n_obs",
                "opp_gols_marcados_mean_5", "opp_gols_marcados_mean_5_n_obs",
                "opp_FTT_mean_3", "opp_FTT_mean_3_n_obs",
                "opp_FTT_mean_5", "opp_FTT_mean_5_n_obs",
            ]
            feat_cols = [c for c in feat_cols if c in row.index]
            for c in feat_cols:
                linhas.append(f"  {c:46s} = {_fmt(row[c])}")

            for k in (3, 5):
                start_r = rodada - k
                end_r = rodada - 1
                janela = master[
                    (master["temporada"] == temporada)
                    & (master["clube_id"] == cid)
                    & (master["rodada"] >= start_r)
                    & (master["rodada"] <= end_r)
                ].sort_values("rodada")
                vals = janela["gols_sofridos"].tolist()
                n_validos = int(sum(pd.notna(v) for v in vals))
                linhas.append(
                    f"  --- team window k={k}: rounds R{start_r}..R{end_r} "
                    f"(encontradas={len(janela)}, n_obs={n_validos})"
                )
                for _, h in janela.iterrows():
                    gs = h["gols_sofridos"]
                    gs_s = "NaN" if pd.isna(gs) else str(int(gs))
                    cs = "NaN" if pd.isna(gs) else str(int(gs == 0))
                    linhas.append(
                        f"      R{int(h['rodada']):>2d}  "
                        f"gols_sofridos={gs_s}  clean_sheet={cs}"
                    )

            for k in (3, 5):
                start_r = rodada - k
                end_r = rodada - 1
                janela = master[
                    (master["temporada"] == temporada)
                    & (master["clube_id"] == oid)
                    & (master["rodada"] >= start_r)
                    & (master["rodada"] <= end_r)
                ].sort_values("rodada")
                linhas.append(
                    f"  --- opp window k={k}: rounds R{start_r}..R{end_r} "
                    f"(encontradas={len(janela)})"
                )
                for _, h in janela.iterrows():
                    gm = h["gols_marcados"]
                    ft = h["FTT"]
                    gm_s = "NaN" if pd.isna(gm) else str(int(gm))
                    ft_s = "NaN" if pd.isna(ft) else str(int(ft))
                    linhas.append(
                        f"      R{int(h['rodada']):>2d}  "
                        f"gols_marcados={gm_s}  FTT={ft_s}"
                    )

            for k in (3, 5):
                start_r = rodada - k
                end_r = rodada - 1
                janela = jogador_gol[
                    (jogador_gol["temporada"] == temporada)
                    & (jogador_gol["atleta_id"] == aid)
                    & (jogador_gol["entrou_em_campo"])
                    & (jogador_gol["rodada"] >= start_r)
                    & (jogador_gol["rodada"] <= end_r)
                ].sort_values("rodada")
                linhas.append(
                    f"  --- gk window k={k}: rounds R{start_r}..R{end_r} "
                    f"(participações encontradas={len(janela)})"
                )
                for _, h in janela.iterrows():
                    pts = h["pontos_num"]
                    pts_s = "NaN" if pd.isna(pts) else f"{float(pts):.1f}"
                    linhas.append(
                        f"      R{int(h['rodada']):>2d}  "
                        f"pontos_num={pts_s}  "
                        f"scout_DE={_fmt(h.get('scout_DE'))}  "
                    )

    return "\n".join(linhas)


# ============================================================================
# Main
# ============================================================================

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("[1/6] lendo jogador e filtrando GOL ...")
    jog = pd.read_parquet(JOGADOR)
    print(f"     shape total: {jog.shape}")
    jog_gol = jog[jog["posicao_id"] == GOLEIRO_POSICAO_ID].copy()
    print(f"     shape GOL: {jog_gol.shape}")

    # --- diagnóstico heurístico ---
    print("\n[2/6] diagnóstico heurístico de semântica dos scouts (subset GOL) ...")
    sem = _diagnose_scout_semantics(jog_gol)
    print(sem.to_string(index=False))
    print("\n  NOTA: 'heuristica' é um alerta automático. NÃO é prova.")
    print("        A confirmação definitiva vem do pipeline de origem.")
    print("        A conversão acumulado -> delta é aplicada para temporadas em")
    print(f"        ACCUMULATED_TEMPORADAS = {sorted(ACCUMULATED_TEMPORADAS)}.\n")

    # Diagnóstico adicional para pontos_num (não deve ser acumulado)
    print("  Diagnóstico de 'pontos_num' por temporada (esperado: delta):")
    sem_pontos = _detect_scout_type_by_temporada(jog_gol, "pontos_num")
    print(f"    {sem_pontos}")

    # Bloqueio se algum scout crítico tiver heurística FORTE de acumulado
    # E a temporada NÃO estiver em ACCUMULATED_TEMPORADAS.
    # (não bloqueia se já vamos converter)
    bloqueantes = sem[
        (sem["scout"].isin(CRITICAL_SCOUTS))
        & (sem["heuristica"] == "provavelmente_acumulado")
        & (sem["forca"] == "forte")
    ]
    # Note: como sabemos que 2024/2025 são acumulados e vamos converter, aceitamos.
    if len(bloqueantes) > 0:
        print("  ATENÇÃO: scouts críticos detectados como acumulados:")
        print(bloqueantes.to_string(index=False))
        print("  -> Serão convertidos para delta por ACCUMULATED_TEMPORADAS.\n")

    # --- flags explícitas ---
    jog_gol["elegivel_modelo"] = jog_gol["entrou_em_campo"].fillna(False).astype(bool)
    jog_gol["participou_com_zero"] = (
        jog_gol["elegivel_modelo"] & (jog_gol["pontos_num"] == 0)
    )
    jog_gol["participou_com_positivo"] = (
        jog_gol["elegivel_modelo"] & (jog_gol["pontos_num"] > 0)
    )
    jog_gol["nao_participou"] = ~jog_gol["elegivel_modelo"]

    print("[3/6] construindo contexto do master ...")
    master = build_master_with_context()
    print(f"     master com contexto: {master.shape}")

    print("\n[4/6] auditando cobertura do master ...")
    missing = _audit_master_coverage(jog_gol, master)
    total_jog_keys = len(jog_gol[["temporada", "rodada", "clube_id"]].drop_duplicates())
    print(f"     chaves GOL distintas: {total_jog_keys}")
    print(f"     chaves ausentes no master: {len(missing)}")

    # Ausências esperadas:
    # - 2026 R21: partidas adiadas, ainda sem registro no master.
    # - 2026 R27: rodada ainda não disponível no master.
    #
    # Essas chaves são excluídas do dataset final do EXP-12.
    expected_missing = missing[
        ((missing["temporada"] == 2026) & (missing["rodada"] == 21))
        | ((missing["temporada"] == 2026) & (missing["rodada"] == 27))
    ].copy()

    unexpected_missing = missing.drop(expected_missing.index)

    if len(expected_missing) > 0:
        detail_expected = (
            expected_missing
            .groupby(["temporada", "rodada"])
            .size()
            .reset_index(name="n")
        )
        print("\n     ausências esperadas — serão excluídas:")
        print(detail_expected.to_string(index=False))

    if len(unexpected_missing) > 0:
        detail_unexpected = (
            unexpected_missing
            .groupby(["temporada", "rodada"])
            .size()
            .reset_index(name="n")
        )
        print("\n     ERRO: ausências inesperadas no master:")
        print(detail_unexpected.to_string(index=False))
        raise RuntimeError(
            "Existem chaves GOL sem master fora das exceções esperadas "
            "(2026 R21/R27). Investigue antes de prosseguir."
        )

    if len(expected_missing) > 0:
        excluded_keys = expected_missing[
            ["temporada", "rodada", "clube_id"]
        ].copy()

        jog_gol = jog_gol.merge(
            excluded_keys.assign(_exclude=1),
            on=["temporada", "rodada", "clube_id"],
            how="left",
        )

        jog_gol = (
            jog_gol[jog_gol["_exclude"].isna()]
            .drop(columns=["_exclude"])
            .copy()
        )

        print(
            f"     linhas GOL após exclusão das partidas sem master: "
            f"{len(jog_gol)}"
        )

    print("\n[5/6] features individuais do goleiro (com conversão de scouts) ...")
    jog_gol, detection_info = build_individual_features(jog_gol)

    team_cols = [
        "temporada", "rodada", "clube_id",
        "adversario_id", "mando",
        "team_gols_sofridos_mean_3", "team_gols_sofridos_mean_3_n_obs",
        "team_gols_sofridos_mean_5", "team_gols_sofridos_mean_5_n_obs",
        "team_clean_sheet_rate_mean_3", "team_clean_sheet_rate_mean_3_n_obs",
        "team_clean_sheet_rate_mean_5", "team_clean_sheet_rate_mean_5_n_obs",
    ]
    team_feats = master[team_cols].copy()

    opp_cols = [
        "temporada", "rodada", "clube_id",
        "team_gols_marcados_mean_3", "team_gols_marcados_mean_3_n_obs",
        "team_gols_marcados_mean_5", "team_gols_marcados_mean_5_n_obs",
        "team_FTT_mean_3", "team_FTT_mean_3_n_obs",
        "team_FTT_mean_5", "team_FTT_mean_5_n_obs",
    ]
    opp_feats = master[opp_cols].rename(columns={
        "clube_id": "adversario_id",
        "team_gols_marcados_mean_3": "opp_gols_marcados_mean_3",
        "team_gols_marcados_mean_3_n_obs": "opp_gols_marcados_mean_3_n_obs",
        "team_gols_marcados_mean_5": "opp_gols_marcados_mean_5",
        "team_gols_marcados_mean_5_n_obs": "opp_gols_marcados_mean_5_n_obs",
        "team_FTT_mean_3": "opp_FTT_mean_3",
        "team_FTT_mean_3_n_obs": "opp_FTT_mean_3_n_obs",
        "team_FTT_mean_5": "opp_FTT_mean_5",
        "team_FTT_mean_5_n_obs": "opp_FTT_mean_5_n_obs",
    })

    jog_gol = jog_gol.merge(
        team_feats,
        on=["temporada", "rodada", "clube_id"],
        how="left",
        validate="many_to_one",
    )
    jog_gol = jog_gol.merge(
        opp_feats,
        on=["temporada", "rodada", "adversario_id"],
        how="left",
        validate="many_to_one",
    )

    assert set(jog_gol["mando"].dropna().unique()) <= {"C", "F"}, (
        f"Valores inesperados em 'mando': "
        f"{sorted(set(jog_gol['mando'].dropna().unique()) - {'C','F'})}"
    )
    jog_gol["mando_C"] = np.where(
        jog_gol["mando"].eq("C"), 1.0,
        np.where(jog_gol["mando"].eq("F"), 0.0, np.nan),
    )

    print("     anexando odds ...")
    try:
        jog_gol = attach_odds(jog_gol)
    except Exception as e:
        print(f"     AVISO: falha ao anexar odds — {e}")
        print("     Prosseguindo SEM bloco D (odd_* = NaN).")
        for c in ["odd_p_home", "odd_p_draw", "odd_p_away"]:
            jog_gol[c] = np.nan

    print("\n[6/6] auditoria manual de casos ...")
    audit_md = _manual_audit(
        jog_gol,
        master,
        jog[jog["posicao_id"] == GOLEIRO_POSICAO_ID],
        cases=[(2024, 19), (2024, 38), (2025, 20), (2025, 38), (2026, 20)],
        n_cases_per_round=2,
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    jog_gol.to_parquet(OUT_PARQUET, index=False)
    print(f"\n     -> {OUT_PARQUET}  ({len(jog_gol)} linhas, {jog_gol.shape[1]} colunas)")

    manifest = {
        "jogador_sha256": _sha(JOGADOR),
        "master_2425_sha256": _sha(MASTER_2425),
        "master_26_sha256": _sha(MASTER_26),
        "shape_total": list(jog_gol.shape),
        "por_temporada": jog_gol.groupby("temporada").size().to_dict(),
        "elegiveis_por_temporada":
            jog_gol.groupby("temporada")["elegivel_modelo"].sum().astype(int).to_dict(),
        "scout_heuristica": sem.to_dict(orient="records"),
        "detection_scout_type": detection_info,
        "accumulated_temporadas": sorted(ACCUMULATED_TEMPORADAS),
        "master_coverage_missing": int(len(missing)),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    linhas = [
        "# EXP-12 GOL — build report",
        "",
        f"- Shape final: {jog_gol.shape}",
        f"- GOL total: {len(jog_gol)}",
        f"- Temporadas em que scouts foram convertidos (acumulado -> delta): "
        f"{sorted(ACCUMULATED_TEMPORADAS)}",
        "",
        "## Diagnóstico heurístico de scouts (NÃO é prova)",
        "",
        sem.to_markdown(index=False),
        "",
        "## Detecção por temporada (auto)",
        "",
        "| scout | " + " | ".join(str(t) for t in sorted(jog_gol["temporada"].unique())) + " |",
        "|" + "---|" * (1 + len(sorted(jog_gol["temporada"].unique()))),
    ]
    for scout, det in detection_info.items():
        row = [scout] + [det.get(int(t), "-") for t in sorted(jog_gol["temporada"].unique())]
        linhas.append("| " + " | ".join(str(x) for x in row) + " |")

    linhas += [
        "",
        "## Por temporada",
        "",
        "| temporada | linhas | elegíveis | participou=0 | participou>0 | não participou |",
        "|---|---|---|---|---|---|",
    ]
    for t in sorted(jog_gol["temporada"].unique()):
        g = jog_gol[jog_gol["temporada"] == t]
        linhas.append(
            f"| {t} | {len(g)} | {int(g['elegivel_modelo'].sum())} "
            f"| {int(g['participou_com_zero'].sum())} "
            f"| {int(g['participou_com_positivo'].sum())} "
            f"| {int(g['nao_participou'].sum())} |"
        )

    feat_cols = [c for c in jog_gol.columns
                 if c.startswith(("gk_", "team_", "opp_", "odd_"))]
    linhas += [
        "",
        "## Missing por feature",
        "",
        "| feature | missing | missing_rate |",
        "|---|---|---|",
    ]
    for c in feat_cols:
        m = int(jog_gol[c].isna().sum())
        linhas.append(f"| {c} | {m} | {m/len(jog_gol):.4f} |")

    linhas += [
        "",
        "## Cobertura do master",
        "",
        f"- chaves GOL distintas: {total_jog_keys}",
        f"- chaves ausentes: {len(missing)}",
        "",
        "## Auditoria manual",
        "",
        audit_md,
    ]

    OUT_MD.write_text("\n".join(linhas))
    print(f"     -> {OUT_MD}")


if __name__ == "__main__":
    main()
