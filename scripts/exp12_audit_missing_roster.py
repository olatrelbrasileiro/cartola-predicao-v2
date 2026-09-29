#!/usr/bin/env python
"""
EXP-12 F2 - Pre-voo: identificar indicador de roster ausente.
Nao altera F1. Se nao houver indicador confiavel -> exit != 0 -> F2 bloqueado.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "interim" / "exp12" / "team_match_history.parquet"

ROSTER_INDICATOR_CANDIDATES = (
    "n_atletas", "atletas_count", "n_jogadores", "jogadores_count",
    "roster_size", "roster_count", "num_atletas",
    "payload_vazio", "roster_ausente", "sem_roster", "roster_missing",
)
SCOUT_COLS = ["FD", "FF", "FT", "G", "PP", "FTT", "FA"]
RESULT_COLS = ["gols_marcados", "gols_sofridos"]


def main() -> int:
    if not HIST.exists():
        print(f"[ERRO] nao encontrado: {HIST}")
        return 2

    h = pd.read_parquet(HIST)
    print("=== SCHEMA REAL ===")
    print(f"shape: {h.shape}")
    print(f"colunas: {list(h.columns)}")
    print("dtypes:")
    for c, t in h.dtypes.items():
        print(f"  - {c}: {t}")
    print(f"temporadas: {sorted(h['temporada'].unique().tolist())}")
    print(f"rodadas: {h['rodada'].nunique()} (min={h['rodada'].min()}, max={h['rodada'].max()})")
    print(f"clubes: {h['clube_id'].nunique()}")
    print(f"duplicatas (temporada,rodada,clube_id): "
          f"{int(h.duplicated(subset=['temporada','rodada','clube_id']).sum())}")

    present = [c for c in ROSTER_INDICATOR_CANDIDATES if c in h.columns]
    print(f"\ncandidatos a indicador presentes: {present}")

    if not present:
        print("\n[ERRO] Nenhum indicador de roster ausente encontrado.")
        print("Colunas disponiveis:")
        for c in h.columns:
            print(f"  - {c}: {h[c].nunique(dropna=True)} unicos, {int(h[c].isna().sum())} NaN")
        print("\nF2 NAO PODE PROSSEGUIR sem decisao metodologica.")
        return 3

    indicator = present[0]
    print(f"\n[OK] usando indicador: {indicator}")

    ind = h[indicator]
    if ind.dtype == bool or set(ind.dropna().unique()).issubset({0, 1}):
        mask_absent = ind.fillna(0).astype(bool)
    else:
        mask_absent = ind.fillna(0) == 0

    n_rows = int(mask_absent.sum())
    print(f"linhas com roster ausente segundo '{indicator}': {n_rows}")

    key = ["temporada", "rodada", "clube_id"]
    pcol = next((c for c in ("partida_id", "match_id", "jogo_id") if c in h.columns), None)
    cols_to_show = key + ([pcol] if pcol else []) + [indicator] + \
                   [c for c in SCOUT_COLS if c in h.columns] + \
                   [c for c in RESULT_COLS if c in h.columns]

    absent = h.loc[mask_absent, cols_to_show].copy()
    print("\n=== LINHAS AFETADAS (roster ausente) ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(absent.to_string(index=False))

    n_matches = (absent[pcol].nunique() if pcol
                 else absent[["temporada", "rodada"]].drop_duplicates().shape[0])
    print(f"\npartidas distintas afetadas: {n_matches}")
    print(f"linhas afetadas (clubes): {len(absent)}")

    if "FTT" in h.columns:
        ftt = h.loc[mask_absent, "FTT"]
        n0 = int((ftt == 0).sum()); nn = int(ftt.isna().sum())
        print(f"\nFTT nas linhas afetadas: {n0} zeros, {nn} NaN, {len(ftt)-n0-nn} outros")
        if (len(ftt) - n0 - nn) > 0:
            print("[aviso] FTT nao e 0/NaN nessas linhas - verificar com F1.")

    print("\n=== ESPERADO (auditoria F1) ===")
    print("12 partidas / 24 linhas com roster ausente.")
    print(f"OBTIDO: {n_matches} partidas / {len(absent)} linhas")
    if n_matches != 12 or len(absent) != 24:
        print("[AVISO] contagem difere do esperado - investigar antes de prosseguir.")

    diag = ROOT / "reports" / "exp12" / "f2_missing_roster_cases.csv"
    diag.parent.mkdir(parents=True, exist_ok=True)
    absent.to_csv(diag, index=False)
    print(f"\ndiagnostico salvo em {diag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
