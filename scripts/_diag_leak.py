import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from exp12_build_team_context_features import (
    build_team_features, apply_roster_absence_mask, load_roster_absence_keys,
    detect_opp_col, ALL_FEATURES,
)

hist = pd.read_parquet("data/interim/exp12/team_match_history.parquet")
hist_masked = apply_roster_absence_mask(hist, load_roster_absence_keys())
out = build_team_features(hist_masked)
opp_col = detect_opp_col(hist_masked)

def mf(hist, T, R, club, opp_col):
    a = (hist.temporada < T) | ((hist.temporada == T) & (hist.rodada < R))
    def blk(ids):
        h = hist[a & hist.clube_id.isin(ids)].sort_values(
            ["clube_id", "temporada", "rodada"])
        d = {}
        for k in (3, 5):
            def agg(col, cs=False):
                r = {}
                for cid, g in h.groupby("clube_id"):
                    t = g.tail(k)
                    if cs:
                        v = t.gols_sofridos.dropna()
                        r[cid] = ((v == 0).astype(float).mean() if len(v) else np.nan)
                    else:
                        r[cid] = t[col].mean()
                return r
            d[f"gs{k}"] = agg("gols_marcados")
            d[f"gc{k}"] = agg("gols_sofridos")
            d[f"cs{k}"] = agg(None, cs=True)
            d[f"sh{k}"] = agg("FTT")
            d[f"st{k}"] = agg("FA")
        return d
    m = hist[(hist.temporada == T) & (hist.rodada == R) & (hist.clube_id == club)]
    oid = int(m[opp_col].iloc[0]) if len(m) else None
    tm = blk([club]); op = blk([oid]) if oid else None
    f = {}
    for k in (3, 5):
        f[f"team_goals_scored_last{k}"]     = tm[f"gs{k}"].get(club, np.nan)
        f[f"team_goals_conceded_last{k}"]   = tm[f"gc{k}"].get(club, np.nan)
        f[f"team_clean_sheet_rate_last{k}"] = tm[f"cs{k}"].get(club, np.nan)
        f[f"team_shots_last{k}"]            = tm[f"sh{k}"].get(club, np.nan)
        f[f"team_shots_on_target_last{k}"]  = tm[f"st{k}"].get(club, np.nan)
        if op:
            f[f"opp_goals_scored_last{k}"]     = op[f"gs{k}"].get(oid, np.nan)
            f[f"opp_goals_conceded_last{k}"]   = op[f"gc{k}"].get(oid, np.nan)
            f[f"opp_clean_sheet_rate_last{k}"] = op[f"cs{k}"].get(oid, np.nan)
            f[f"opp_shots_last{k}"]            = op[f"sh{k}"].get(oid, np.nan)
            f[f"opp_shots_on_target_last{k}"]  = op[f"st{k}"].get(oid, np.nan)
        else:
            for c in ALL_FEATURES:
                if c.startswith("opp_") and c.endswith(f"last{k}"):
                    f[c] = np.nan
    return f

sample = out.sample(min(100, len(out)), random_state=7)
mis = []
for _, r in sample.iterrows():
    T, R, c = int(r.temporada), int(r.rodada), int(r.clube_id)
    man = mf(hist_masked, T, R, c, opp_col)
    for feat in ALL_FEATURES:
        e, g = man[feat], r[feat]
        if pd.isna(e) and pd.isna(g):
            continue
        if not np.isclose(e, g, equal_nan=True):
            mis.append((T, R, c, feat, e, g))

print(f"\nMismatches: {len(mis)}")
for m in mis[:40]:
    print(f"  T={m[0]} R={m[1]} c={m[2]} f={m[3]} exp={m[4]} got={m[5]}")
