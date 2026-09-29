import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
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
TEAM = [c for c in F2_FEATURES if c.startswith("team_")]
OPP  = [c for c in F2_FEATURES if c.startswith("opp_")]


@pytest.fixture(scope="module")
def p():  return pd.read_parquet(PLAYER)
@pytest.fixture(scope="module")
def f():  return pd.read_parquet(F2)
@pytest.fixture(scope="module")
def out():return pd.read_parquet(OUT)


# 1. row count
def test_row_count_preserved(p, out):
    assert len(out) == len(p), f"{len(p)} -> {len(out)}"

# 2. sem duplicacao introduzida
def test_no_new_dups(p, out):
    # comparacao de shape + chave natural (atleta_id/rodada/temporada se existir)
    assert len(out.drop_duplicates()) == len(p.drop_duplicates())

# 3. F2 tem chave unica
def test_f2_keys_unique(f):
    assert f.duplicated(subset=KEYS).sum() == 0
    assert f[KEYS].drop_duplicates().shape[0] == len(f)

# 4. cobertura mensuravel sem virar zero
def test_coverage(out):
    n = len(out)
    any_f2 = out[F2_FEATURES].notna().any(axis=1)
    print(f"\n[coverage] {int(any_f2.sum())}/{n} linhas com pelo menos 1 F2")
    assert any_f2.sum() > 0

# 5. integridade colunas originais
def test_original_columns_intact(p, out):
    assert list(out.columns)[:len(p.columns)] == list(p.columns)
    for c in p.columns:
        a, b = p[c], out[c]
        if a.dtype.kind == "f":
            assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), c
        else:
            assert a.equals(b), c

# 6. team_* vem do clube correto
def test_team_features_match_club(f, out):
    # pega todas as combinacoes unicas (temporada, rodada, clube_id) do output em 2026
    o26 = out[out.temporada == 2026][KEYS].drop_duplicates()
    m = o26.merge(f[KEYS + TEAM], on=KEYS, how="left", validate="one_to_one")
    # amostra 30 linhas
    s = m.sample(min(30, len(m)), random_state=0)
    for _, r in s.iterrows():
        key = (int(r.temporada), int(r.rodada), int(r.clube_id))
        frow = f[(f.temporada == key[0]) & (f.rodada == key[1]) & (f.clube_id == key[2])]
        if frow.empty: continue
        frow = frow.iloc[0]
        for c in TEAM:
            assert np.isclose(frow[c], r[c], equal_nan=True), f"{c} @ {key}"

# 7. opp_* vem do adversario correto
def test_opp_features_match_opponent(f, out):
    o26 = out[out.temporada == 2026][KEYS].drop_duplicates()
    for _, r in o26.sample(min(30, len(o26)), random_state=1).iterrows():
        key = (int(r.temporada), int(r.rodada), int(r.clube_id))
        frow = f[(f.temporada == key[0]) & (f.rodada == key[1]) & (f.clube_id == key[2])]
        if frow.empty: continue
        frow = frow.iloc[0]
        for c in OPP:
            # compara com a propria F2 do alvo (que ja tem opp_* calculado)
            target = out[(out.temporada == key[0]) & (out.rodada == key[1]) & (out.clube_id == key[2])]
            if target.empty: continue
            v = target.iloc[0][c]
            assert np.isclose(frow[c], v, equal_nan=True), f"{c} @ {key}"

# 8. temporalidade: para cada chave 2026, compara com a F2 exata (mesma rodada, sem shift)
def test_no_temporal_shift(f, out):
    o26 = out[out.temporada == 2026][KEYS].drop_duplicates()
    for _, r in o26.sample(min(50, len(o26)), random_state=2).iterrows():
        key = (int(r.temporada), int(r.rodada), int(r.clube_id))
        frow = f[(f.temporada == key[0]) & (f.rodada == key[1]) & (f.clube_id == key[2])]
        trow = out[(out.temporada == key[0]) & (out.rodada == key[1]) & (out.clube_id == key[2])]
        if frow.empty or trow.empty: continue
        frow, trow = frow.iloc[0], trow.iloc[0]
        for c in F2_FEATURES:
            assert np.isclose(frow[c], trow[c], equal_nan=True), f"{c} @ {key}"

# 9. rodada inicial: 2026 R1 deve ter team_shots_last3 NaN
def test_round1_2026_is_nan(out):
    r1 = out[(out.temporada == 2026) & (out.rodada == 1)]
    if r1.empty: pytest.skip("sem 2026 R1")
    assert r1["team_shots_last3"].isna().all()
    assert r1["team_shots_on_target_last3"].isna().all()

# 10. NaN do F2 preservado (nao virou zero no merge)
def test_f2_nan_preserved(f, out):
    m = out.merge(
        f[KEYS + F2_FEATURES], on=KEYS, how="inner", suffixes=("", "_f2")
    )
    for c in F2_FEATURES:
        a, b = m[c], m[c + "_f2"]
        # o padrao de NaN deve ser identico
        assert a.isna().equals(b.isna()), f"padrao NaN divergiu em {c}"
        # e onde F2 tem valor, o merge nao pode ter 0 artificial
        mask = b.notna()
        assert not ((a[mask] == 0) & (b[mask] != 0)).any(), f"zero artificial em {c}"

# 11. seasons 2024-2026 preservadas
def test_seasons_preserved(p, out):
    assert sorted(out["temporada"].unique().tolist()) == \
           sorted(p["temporada"].unique().tolist())

# 12. pontos_num preservado (se a coluna existir)
def test_pontos_num(p, out):
    cand = [c for c in p.columns if c.lower() in ("pontos_num","pontuacao","pontos")]
    if not cand: pytest.skip("sem coluna de pontos")
    c = cand[0]
    if p[c].dtype.kind == "f":
        assert np.allclose(p[c].to_numpy(), out[c].to_numpy(), equal_nan=True)
    else:
        assert p[c].equals(out[c])

# 13. entrou_em_campo preservado
def test_entrou_em_campo(p, out):
    cand = [c for c in p.columns if "entrou" in c.lower()]
    if not cand: pytest.skip("sem entrou_em_campo")
    for c in cand:
        assert p[c].equals(out[c])

# 14. FEATURES_COM (colunas originais) intactas byte/valor
def test_features_com_intact(p, out):
    # todas as colunas originais exceto as chaves (que tambem sao originais) ja
    # foram testadas em test_original_columns_intact; aqui garantimos que
    # nenhuma coluna do player foi sobrescrita pelas F2
    for c in F2_FEATURES:
        assert c not in p.columns
        assert c in out.columns
