"""Inspecao read-only dos insumos do merge exp12. Nao escreve nada."""
from pathlib import Path
import pandas as pd

P = Path("data/interim/cartola_jogador_rodada_v2.parquet")
F = Path("data/interim/exp12/team_context_features.parquet")

for label, p in [("PLAYER v2", P), ("F2 CONTEXT", F)]:
    if not p.exists():
        print(f"[ERRO] {label}: {p} NAO EXISTE"); raise SystemExit(2)

p = pd.read_parquet(P)
f = pd.read_parquet(F)

def dump(name, df):
    print(f"\n=== {name} ===")
    print(f"shape: {df.shape}")
    print("dtypes:")
    for c, t in df.dtypes.items():
        print(f"  {c}: {t}")

dump("PLAYER v2", p)
dump("F2 CONTEXT", f)

F2_KEYS = ["temporada","rodada","clube_id"]
print("\n=== F2 KEYS - unicidade ===")
for c in F2_KEYS:
    print(f"  F2 tem coluna {c!r}? {c in f.columns}  dtype={f[c].dtype if c in f.columns else '-'}")
if all(c in f.columns for c in F2_KEYS):
    print(f"  duplicatas em {F2_KEYS}: {int(f.duplicated(subset=F2_KEYS).sum())}")
    print(f"  chaves unicas: {f[F2_KEYS].drop_duplicates().shape[0]}")
    print(f"  temporadas F2: {sorted(f['temporada'].unique().tolist())}")
    print(f"  rodadas F2:    {sorted(f['rodada'].unique().tolist())}")
    print(f"  clubes F2:     {sorted(f['clube_id'].unique().tolist())}")

print("\n=== PLAYER - candidatos a chave ===")
def cands(df, toks, excl=()):
    return [c for c in df.columns if any(t in c.lower() for t in toks)
            and not any(e in c.lower() for e in excl)]
season_c = cands(p, ("temporada","season","ano"))
round_c  = cands(p, ("rodada","round"))
club_c   = cands(p, ("clube","club","time","team"), excl=("nome","nome_fantasia"))
player_c = cands(p, ("atleta","player","jogador"))
print(f"  temporada: {season_c}")
print(f"  rodada:    {round_c}")
print(f"  clube:     {club_c}")
print(f"  jogador:   {player_c}")

print("\n=== dtype compatibility (por chave F2) ===")
for c in F2_KEYS:
    if c in p.columns:
        print(f"  {c}: F2={f[c].dtype}  player={p[c].dtype}  "
              f"iguais={f[c].dtype == p[c].dtype}")
    else:
        print(f"  {c}: F2={f[c].dtype}  player=<AUSENTE>")

print("\n=== temporadas em cada lado ===")
if season_c:
    sc = season_c[0]
    ps = sorted(p[sc].dropna().unique().tolist())
    fs = sorted(f["temporada"].dropna().unique().tolist()) if "temporada" in f.columns else []
    print(f"  player[{sc}]: {ps}")
    print(f"  F2[temporada]: {fs}")
    print(f"  intersecao: {sorted(set(ps) & set(fs))}")

print("\n=== F2 - NaN por coluna ===")
for c in f.columns:
    n = int(f[c].isna().sum())
    if n:
        print(f"  {c}: {n}")

print("\n=== F2 - opp_clube_id dtype/NaN ===")
if "opp_clube_id" in f.columns:
    print(f"  dtype={f['opp_clube_id'].dtype}  NaN={int(f['opp_clube_id'].isna().sum())}")

print("\n=== PLAYER - possiveis colunas de pontos/entrou ===")
for c in p.columns:
    lc = c.lower()
    if any(t in lc for t in ("ponto","pontuacao","entrou","status","jogou")):
        print(f"  {c}: dtype={p[c].dtype}  n_unicos={p[c].nunique(dropna=True)}")

print("\n=== FIM DA INSPECAO ===")
