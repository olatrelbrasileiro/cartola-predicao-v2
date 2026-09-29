#!/usr/bin/env python
"""
EXP-12 F2.5 - Merge player x round com contexto F2 (20 features).
Leitura read-only nos canonicos. Nao altera F1 nem F2.
Output: data/interim/cartola_jogador_rodada_exp12.parquet
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT/"data/interim/cartola_jogador_rodada_v2.parquet"
F2     = ROOT/"data/interim/exp12/team_context_features.parquet"
OUT    = ROOT/"data/interim/cartola_jogador_rodada_exp12.parquet"

KEYS = ["temporada","rodada","clube_id"]

F2_FEATURES = [
    "team_goals_scored_last3","team_goals_scored_last5",
    "team_goals_conceded_last3","team_goals_conceded_last5",
    "team_clean_sheet_rate_last3","team_clean_sheet_rate_last5",
    "team_shots_last3","team_shots_last5",
    "team_shots_on_target_last3","team_shots_on_target_last5",
    "opp_goals_scored_last3","opp_goals_scored_last5",
    "opp_goals_conceded_last3","opp_goals_conceded_last5",
    "opp_clean_sheet_rate_last3","opp_clean_sheet_rate_last5",
    "opp_shots_last3","opp_shots_last5",
    "opp_shots_on_target_last3","opp_shots_on_target_last5",
]
assert len(F2_FEATURES) == 20


def _verify_dtype(df: pd.DataFrame, col: str, src: str) -> None:
    if str(df[col].dtype) != "int64":
        raise RuntimeError(
            f"[merge] {src}.{col} dtype={df[col].dtype}, esperado int64"
        )


def main() -> None:
    if not PLAYER.exists():
        raise SystemExit(f"[merge] ausente: {PLAYER}")
    if not F2.exists():
        raise SystemExit(f"[merge] ausente: {F2}")

    p = pd.read_parquet(PLAYER)
    f = pd.read_parquet(F2)

    # ---- 1. chaves ----
    for c in KEYS:
        if c not in p.columns:
            raise RuntimeError(f"[merge] player sem coluna {c!r}")
        if c not in f.columns:
            raise RuntimeError(f"[merge] F2 sem coluna {c!r}")
        _verify_dtype(p, c, "player")
        _verify_dtype(f, c, "F2")

    # ---- 2. colisao de nomes ----
    collide = [c for c in F2_FEATURES if c in p.columns]
    if collide:
        raise RuntimeError(
            f"[merge] colisao: colunas ja existentes no player: {collide}"
        )

    # ---- 3. unicidade F2 ----
    n_dup = int(f.duplicated(subset=KEYS).sum())
    if n_dup:
        raise RuntimeError(f"[merge] F2 com {n_dup} duplicatas em {KEYS}")

    # ---- 4. snapshot do player ----
    p_orig = p.copy(deep=True)
    orig_cols = list(p.columns)
    n_before = len(p)

    # ---- 5. merge ----
    f_small = f[KEYS + F2_FEATURES].copy()
    out = p.merge(f_small, on=KEYS, how="left", validate="many_to_one")

    n_after = len(out)
    if n_after != n_before:
        raise RuntimeError(f"[merge] row count mudou: {n_before} -> {n_after}")

    # ---- 6. colunas originais intactas ----
    if list(out.columns)[:len(orig_cols)] != orig_cols:
        raise RuntimeError("[merge] ordem/colunas originais mudaram")
    for c in orig_cols:
        a, b = p_orig[c], out[c]
        if a.dtype.kind == "f":
            if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True):
                raise RuntimeError(f"[merge] coluna {c} divergiu (float)")
        else:
            if not a.equals(b):
                raise RuntimeError(f"[merge] coluna {c} divergiu")

    # ---- 7. salvar ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    # ---- 8. relatorio ----
    print("=== EXP-12 F2.5 - MERGE PLAYER x CONTEXT ===")
    print(f"player v2 : {n_before} linhas, {len(orig_cols)} colunas")
    print(f"F2        : {len(f)} chaves unicas, {len(F2_FEATURES)} features")
    print(f"output    : {n_after} linhas, {len(out.columns)} colunas")
    print(f"sha256    : {hashlib.sha256(OUT.read_bytes()).hexdigest()}")

    m2026 = out["temporada"] == 2026
    n2026 = int(m2026.sum())
    if n2026:
        any_f2 = out.loc[m2026, F2_FEATURES].notna().any(axis=1)
        cov = int(any_f2.sum())
        print(f"\n2026      : {n2026} linhas")
        print(f"  com pelo menos 1 F2 : {cov} ({cov/n2026:.1%})")
        print(f"  sem nenhuma F2      : {n2026 - cov}")

    any_f2_all = out[F2_FEATURES].notna().any(axis=1)
    print(f"\nlinhas sem nenhuma F2 no dataset total: {int((~any_f2_all).sum())}")
    print(f"linhas com pelo menos 1 F2            : {int(any_f2_all.sum())}")

    print("\nNaN por feature (dataset integrado):")
    for c in F2_FEATURES:
        print(f"  {c}: {int(out[c].isna().sum())}")


if __name__ == "__main__":
    main()
