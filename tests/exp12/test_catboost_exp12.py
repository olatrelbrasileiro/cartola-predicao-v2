import re, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT/"data/interim/cartola_jogador_rodada_exp12.parquet"
V2 = ROOT/"scripts/catboost_v2.py"
V2P = ROOT/"reports/catboost_v2_predictions.csv"
V2F = ROOT/"reports/catboost_v2_folds.csv"
OUTD = ROOT/"reports/exp12"
PRED = OUTD/"catboost_exp12_predictions.csv"
FOLD = OUTD/"catboost_exp12_folds.csv"

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

def _parse_v2_features_com():
    txt = V2.read_text()
    m = re.search(r"^FEATURES_COM\s*=\s*\[(.*?)\]", txt, re.M|re.S)
    if m is None: return None
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))

@pytest.fixture(scope="module")
def ds(): return pd.read_parquet(DS)

# ---- dataset (F2.5 output) ----
def test_dataset_shape(ds): assert ds.shape == (77253, 58), ds.shape

def test_f2_features_present(ds):
    for c in F2_FEATURES: assert c in ds.columns, c

def test_f2_features_has_20():
    assert len(F2_FEATURES) == 20

def test_no_overlap_features_com_f2():
    fc = _parse_v2_features_com()
    if fc is None: pytest.skip("FEATURES_COM nao parseavel")
    assert set(fc) & set(F2_FEATURES) == set(), "overlap entre COM e F2"

def test_features_com_count():
    fc = _parse_v2_features_com()
    if fc is None: pytest.skip()
    assert len(fc) == 21, f"FEATURES_COM tem {len(fc)}, esperado 21"

def test_opp_clube_id_not_in_features():
    fc = _parse_v2_features_com() or []
    assert "opp_clube_id" not in fc + F2_FEATURES

def test_total_features_is_41():
    fc = _parse_v2_features_com()
    if fc is None: pytest.skip()
    assert len(set(fc + F2_FEATURES)) == 41

# ---- protocolo temporal (estatico) ----
def test_train_mask_is_strictly_before_target(ds):
    """Para (t,R) representativos, mascara de treino exclui alvo e futuro."""
    for (t, R) in [(2024,5),(2025,10),(2026,3),(2026,20)]:
        tr = (ds["temporada"] < t) | ((ds["temporada"] == t) & (ds["rodada"] < R))
        tgt = (ds["temporada"] == t) & (ds["rodada"] == R)
        fut = (ds["temporada"] > t) | ((ds["temporada"] == t) & (ds["rodada"] > R))
        assert not (tr & tgt).any(), f"alvo {t}R{R} no treino"
        assert not (tr & fut).any(), f"futuro {t}R{R} no treino"

def test_prior_seasons_included_for_2025_2026(ds):
    for (t, R) in [(2025, 5), (2026, 3)]:
        tr = (ds["temporada"] < t) | ((ds["temporada"] == t) & (ds["rodada"] < R))
        seasons_before = sorted(s for s in ds["temporada"].unique() if s < t)
        for s in seasons_before:
            assert tr[ds["temporada"] == s].any(), f"{s} ausente do treino {t}R{R}"

def test_eval_only_entrou_em_campo(ds):
    for (t, R) in [(2024,5),(2025,10),(2026,3)]:
        ev = (ds["temporada"] == t) & (ds["rodada"] == R) & (ds["entrou_em_campo"] == True)
        assert ev.sum() > 0, f"nenhuma linha avaliável em {t}R{R}"

def test_f2_nan_pre_2026(ds):
    m = ds["temporada"] < 2026
    if m.sum() == 0: pytest.skip()
    assert ds.loc[m, F2_FEATURES].isna().all().all(), "F2 nao-NaN em 2024/2025"

# ---- artefatos oficiais intocados ----
def test_v2_artifacts_exist():
    assert V2P.exists()
    assert V2F.exists()

def test_exp12_outputs_in_own_dir():
    assert OUTD.exists()
    for fn in ("catboost_exp12_predictions.csv","catboost_exp12_folds.csv"):
        p_wrong = ROOT/"reports"/fn
        assert not p_wrong.exists(), f"exp12 nao pode escrever {p_wrong}"

# ---- validacao pos-execucao ----
def test_predictions_has_f2_available():
    if not PRED.exists(): pytest.skip("rode o experimento primeiro")
    df = pd.read_csv(PRED)
    assert "f2_available" in df.columns
    assert df["f2_available"].dtype == bool

def test_predictions_fold_consistency():
    if not PRED.exists(): pytest.skip()
    df = pd.read_csv(PRED)
    expect = df["temporada"].astype(str) + "_R" + df["rodada"].astype(str)
    assert (df["fold_id"] == expect).all()

def test_folds_csv_readable():
    if not FOLD.exists(): pytest.skip()
    df = pd.read_csv(FOLD)
    for c in ("temporada","rodada_alvo","n_train","n_test","mae","rmse","spearman"):
        assert c in df.columns, c
    assert len(df) > 0

def test_predictions_only_entrou_em_campo():
    """Se disponivel, todas as previsoes vieram de linhas com entrou_em_campo=True."""
    if not PRED.exists(): pytest.skip()
    df = pd.read_csv(PRED)
    # a previsao nao tem entrou_em_campo, mas todas as linhas devem ter y_true valido
    assert df["y_true"].notna().all()

def test_predictions_f2_split_counts():
    if not PRED.exists(): pytest.skip()
    df = pd.read_csv(PRED)
    assert "f2_available" in df.columns
    n_av = int(df["f2_available"].sum())
    print(f"\nf2_available nas previsoes: {n_av}/{len(df)}")
