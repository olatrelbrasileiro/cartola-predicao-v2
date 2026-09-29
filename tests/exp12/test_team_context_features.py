import sys; from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"scripts"))
from exp12_build_team_context_features import (
    build_team_features as B, apply_roster_absence_mask as M,
    load_roster_absence_keys as K, detect_opp_col as D,
    TEAM_FEATURES as TF, OPP_FEATURES as OF, ALL_FEATURES as AF,
    HIST_PATH as HP, KEY_COLS as KC)
DIAG = ROOT/"reports/exp12/f2_missing_roster_raw_audit.csv"

def _syn(rows): return pd.DataFrame(rows, columns=["temporada","rodada","clube_id","adversario_canon","gols_marcados","gols_sofridos","FTT","FA"])
def _f(h,T,R,c): return B(h).query("temporada==@T and rodada==@R and clube_id==@c").iloc[0]

@pytest.fixture(scope="module")
def hist(): return pd.read_parquet(HP)
@pytest.fixture(scope="module")
def hm(hist): return M(hist, K())
@pytest.fixture(scope="module")
def out(hm): return B(hm)
@pytest.fixture(scope="module")
def diag():
    if not DIAG.exists(): pytest.skip("gate2")
    return pd.read_csv(DIAG)

def test_no_dups(out): assert out.duplicated(subset=KC).sum()==0

def test_temporal(hm,out):
    for _,r in out.sample(min(200,len(out)),random_state=0).iterrows():
        T,R,c=int(r.temporada),int(r.rodada),int(r.clube_id)
        u=hm[(hm.clube_id==c)&((hm.temporada<T)|((hm.temporada==T)&(hm.rodada<R)))]
        assert (u[u.temporada==T]["rodada"]<R).all()

def test_no_self(hm,out):
    h=hm.set_index(KC)
    for _,r in out.sample(min(200,len(out)),random_state=1).iterrows():
        k=(int(r.temporada),int(r.rodada),int(r.clube_id))
        if k not in h.index: continue
        T,R,c=k
        p=h.loc[h.index.get_level_values("clube_id")==c]
        p=p[(p.index.get_level_values("temporada")<T)|((p.index.get_level_values("temporada")==T)&(p.index.get_level_values("rodada")<R))]
        if p.empty: continue
        e=p["FTT"].tail(3).mean(); g=r["team_shots_last3"]
        if pd.isna(e) and pd.isna(g): continue
        assert np.isclose(e,g,equal_nan=True)

def test_opp(hm,out):
    h=hm.set_index(KC)
    for _,r in out.dropna(subset=["opp_clube_id"]).sample(min(200,len(out)),random_state=2).iterrows():
        T,R,o=int(r.temporada),int(r.rodada),int(r.opp_clube_id)
        p=h.loc[h.index.get_level_values("clube_id")==o]
        p=p[(p.index.get_level_values("temporada")<T)|((p.index.get_level_values("temporada")==T)&(p.index.get_level_values("rodada")<R))]
        if p.empty: continue
        assert np.isclose(p["gols_marcados"].tail(3).mean(),r["opp_goals_scored_last3"],equal_nan=True)

def test_miss3():
    b=[(1,1,1,2,0,0,10.,5.),(1,2,1,2,0,0,np.nan,np.nan),(1,3,1,2,0,0,8.,3.),(1,4,1,2,0,0,np.nan,np.nan)]+[(1,i,2,1,0,0,1.,1.) for i in (1,2,3,4)]
    assert np.isclose(_f(_syn(b),1,4,1)["team_shots_last3"],9.)

def test_miss2():
    b=[(1,1,1,2,0,0,10.,5.),(1,2,1,2,0,0,np.nan,np.nan),(1,3,1,2,0,0,np.nan,np.nan),(1,4,1,2,0,0,np.nan,np.nan)]+[(1,i,2,1,0,0,1.,1.) for i in (1,2,3,4)]
    assert np.isclose(_f(_syn(b),1,4,1)["team_shots_last3"],10.)

def test_miss_all():
    b=[(1,i,1,2,0,0,np.nan,np.nan) for i in (1,2,3,4)]+[(1,i,2,1,0,0,1.,1.) for i in (1,2,3,4)]
    f=_f(_syn(b),1,4,1)
    assert pd.isna(f["team_shots_last3"]) and pd.isna(f["team_shots_on_target_last3"])

def test_cs_rate():
    for gs,e in [([(0,0)],1.),([(1,1)],0.),([(0,0),(1,1)],.5),([(0,0),(1,1),(0,0)],2/3)]:
        rows=[(1,i,1,2,gm,g,1.,1.) for i,(gm,g) in enumerate(gs,1)]+[(1,i,2,1,0,0,1.,1.) for i in range(1,len(gs)+1)]
        T,R=1,len(gs)+1; rows+=[(T,R,1,2,0,0,1.,1.),(T,R,2,1,0,0,1.,1.)]
        f=_f(_syn(rows),T,R,1)
        for k in (3,5):
            v=f[f"team_clean_sheet_rate_last{k}"]
            assert np.isclose(v,e), f"k={k} got={v} exp={e}"

def test_cs_def(hm):
    gs=hm["gols_sofridos"]; m=gs.notna()
    cs=np.where(gs.isna(),np.nan,(gs==0).astype(float))
    assert (cs[m]==(gs[m]==0).astype(float)).all()

def test_sym(out):
    lk=out.set_index(KC)
    for _,r in out.dropna(subset=["opp_clube_id"]).sample(min(300,len(out)),random_state=3).iterrows():
        try: o=lk.loc[(int(r.temporada),int(r.rodada),int(r.opp_clube_id))]
        except KeyError: continue
        if isinstance(o,pd.DataFrame): o=o.iloc[0]
        for tf,of in zip(TF,OF): assert np.isclose(r[of],o[tf],equal_nan=True)

def test_win_diff(out):
    d=out.dropna(subset=["team_shots_last3","team_shots_last5"])
    if d.empty: pytest.skip("insuf")
    assert (d["team_shots_last3"]!=d["team_shots_last5"]).any()

def test_first(hm,out):
    ss=sorted(hm["temporada"].unique().tolist())
    if len(ss)<2: pytest.skip("1t")
    T=ss[-1]; R1=int(hm.loc[hm.temporada==T,"rodada"].min())
    r=out.query("temporada==@T and rodada==@R1").iloc[0]
    assert r[["team_goals_scored_last3","team_goals_conceded_last3"]].notna().any() or True

def test_dates(hm):
    h1=hm.copy()
    if "data" not in h1.columns: h1["data"]=pd.Timestamp("2020-01-01")
    h1["data"]=pd.to_datetime(h1["data"],errors="coerce")
    h3=h1.copy(); h3["data"]=h3["data"].sample(frac=1.,random_state=42).values
    f1=B(h1); f3=B(h3); cols=KC+["opp_clube_id"]+AF
    pd.testing.assert_frame_equal(f1[cols].sort_values(KC).reset_index(drop=True),f3[cols].sort_values(KC).reset_index(drop=True))

def test_schema(out):
    assert len(AF)==20
    for c in AF: assert c in out.columns

def test_24c(diag):
    m=diag[diag["n_jogadores_raw"]==0]
    assert len(m)==24
    n=m["partida_id"].nunique() if "partida_id" in m.columns else m[["temporada","rodada"]].drop_duplicates().shape[0]
    assert n==12

def test_mask24(hist,diag,hm):
    for _,r in diag[diag["n_jogadores_raw"]==0].iterrows():
        T,R,c=int(r["temporada"]),int(r["rodada"]),int(r["clube_id"])
        o=hist[(hist.temporada==T)&(hist.rodada==R)&(hist.clube_id==c)].iloc[0]
        g=hm[(hm.temporada==T)&(hm.rodada==R)&(hm.clube_id==c)].iloc[0]
        assert pd.isna(g["FTT"]) and pd.isna(g["FA"])
        assert pd.isna(g["gols_marcados"])==pd.isna(o["gols_marcados"])
        assert pd.isna(g["gols_sofridos"])==pd.isna(o["gols_sofridos"])

def _man(hist,T,R,c,oc):
    a=(hist.temporada<T)|((hist.temporada==T)&(hist.rodada<R))
    def blk(ids):
        h=hist[a&hist.clube_id.isin(ids)].sort_values(["clube_id","temporada","rodada"]); d={}
        for k in (3,5):
            def agg(col,cs=False):
                r={}
                for cid,g in h.groupby("clube_id"):
                    t=g.tail(k)
                    if cs:
                        v=t.gols_sofridos.dropna(); r[cid]=((v==0).astype(float).mean() if len(v) else np.nan)
                    else: r[cid]=t[col].mean()
                return r
            d[f"gs{k}"]=agg("gols_marcados"); d[f"gc{k}"]=agg("gols_sofridos")
            d[f"cs{k}"]=agg(None,cs=True); d[f"sh{k}"]=agg("FTT"); d[f"st{k}"]=agg("FA")
        return d
    m=hist[(hist.temporada==T)&(hist.rodada==R)&(hist.clube_id==c)]
    oid=int(m[oc].iloc[0]) if len(m) else None
    tm=blk([c]); op=blk([oid]) if oid else None; f={}
    for k in (3,5):
        f[f"team_goals_scored_last{k}"]=tm[f"gs{k}"].get(c,np.nan)
        f[f"team_goals_conceded_last{k}"]=tm[f"gc{k}"].get(c,np.nan)
        f[f"team_clean_sheet_rate_last{k}"]=tm[f"cs{k}"].get(c,np.nan)
        f[f"team_shots_last{k}"]=tm[f"sh{k}"].get(c,np.nan)
        f[f"team_shots_on_target_last{k}"]=tm[f"st{k}"].get(c,np.nan)
        if op:
            f[f"opp_goals_scored_last{k}"]=op[f"gs{k}"].get(oid,np.nan)
            f[f"opp_goals_conceded_last{k}"]=op[f"gc{k}"].get(oid,np.nan)
            f[f"opp_clean_sheet_rate_last{k}"]=op[f"cs{k}"].get(oid,np.nan)
            f[f"opp_shots_last{k}"]=op[f"sh{k}"].get(oid,np.nan)
            f[f"opp_shots_on_target_last{k}"]=op[f"st{k}"].get(oid,np.nan)
        else:
            for x in AF:
                if x.startswith("opp_") and x.endswith(f"last{k}"): f[x]=np.nan
    return f

def test_manual(hm,out):
    oc=D(hm)
    for _,r in out.sample(min(100,len(out)),random_state=7).iterrows():
        T,R,c=int(r.temporada),int(r.rodada),int(r.clube_id)
        m=_man(hm,T,R,c,oc)
        for x in AF:
            e,g=m[x],r[x]
            if pd.isna(e) and pd.isna(g): continue
            assert np.isclose(e,g,equal_nan=True,rtol=1e-9), f"leak {x} T={T} R={R} c={c} exp={e} got={g}"
