#!/usr/bin/env python3
"""
ETAPA 3 — Auditoria da base canonica (cartola_jogador_rodada_v2.parquet).
Somente leitura. Nao cria features, nao altera dados, nao decide treino.
"""
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "cartola-predicao-v2"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
REP = ROOT / "reports"
TMP = ROOT / ".tmp"
REP.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

RELATORIO = REP / "auditoria_base_canonica_v2.md"
CSV_ALVO = REP / "auditoria_alvo.csv"
CSV_TEMPORAL = REP / "auditoria_temporal.csv"
CSV_SCOUTS = REP / "auditoria_scouts.csv"
CSV_OUTLIERS = REP / "auditoria_outliers.csv"
CSV_EVIDENCIAS = REP / "auditoria_evidencias.csv"

SCOUTS = ["A","CA","CV","DE","DS","FC","FD","FF","FS","FT","G","I",
          "PC","PS","PP","SG","V","DP","GC"]

# ---------------------------------------------------------------- util
def descr(x):
    s = pd.Series(x).dropna()
    if len(s) == 0:
        return dict(count=0, mean=np.nan, median=np.nan, std=np.nan,
                    min=np.nan, max=np.nan, p01=np.nan, p05=np.nan,
                    p25=np.nan, p50=np.nan, p75=np.nan, p95=np.nan,
                    p99=np.nan, n_zeros=0, n_neg=0, n_pos=0)
    return dict(
        count=int(len(s)),
        mean=float(s.mean()),
        median=float(s.median()),
        std=float(s.std()) if len(s) > 1 else 0.0,
        min=float(s.min()),
        max=float(s.max()),
        p01=float(s.quantile(0.01)),
        p05=float(s.quantile(0.05)),
        p25=float(s.quantile(0.25)),
        p50=float(s.quantile(0.50)),
        p75=float(s.quantile(0.75)),
        p95=float(s.quantile(0.95)),
        p99=float(s.quantile(0.99)),
        n_zeros=int((s == 0).sum()),
        n_neg=int((s < 0).sum()),
        n_pos=int((s > 0).sum()),
    )

def adicionar_grupo(df, keys, d, label_key, label_val):
    for k, sub in df.groupby(keys, dropna=False):
        if not isinstance(k, tuple):
            k = (k,)
        row = {label_key: label_val}
        for kk, vv in zip(keys, k):
            row[kk] = vv
        row.update(descr(sub["pontos_num"]))
        d.append(row)

# ---------------------------------------------------------------- carregar
df = pd.read_parquet(PARQUET)
inicio = datetime.now()

# ---------------------------------------------------------------- 3.1 ALVO
print("== 3.1 ALVO ==")
linhas_alvo = []
adicionar_grupo(df, ["temporada"], linhas_alvo, "grupo", "temporada")
adicionar_grupo(df, ["rodada"], linhas_alvo, "grupo", "rodada")
adicionar_grupo(df, ["posicao_id"], linhas_alvo, "grupo", "posicao")
adicionar_grupo(df, ["temporada","posicao_id"], linhas_alvo, "grupo", "temporada_posicao")

# Separacao por participacao (A, B, C)
linhas_alvo.append({"grupo": "A_nao_participou", **descr(df.loc[df["entrou_em_campo"] == False, "pontos_num"])})
linhas_alvo.append({"grupo": "B_entrou_e_zero", **descr(df.loc[(df["entrou_em_campo"] == True) & (df["pontos_num"] == 0), "pontos_num"])})
linhas_alvo.append({"grupo": "C_entrou_e_pontuou", **descr(df.loc[(df["entrou_em_campo"] == True) & (df["pontos_num"] != 0), "pontos_num"])})

# Separacao por temporada
for a in sorted(df["temporada"].unique()):
    sub = df[df["temporada"] == a]
    linhas_alvo.append({"grupo": f"A_nao_participou_{a}", **descr(sub.loc[sub["entrou_em_campo"] == False, "pontos_num"])})
    linhas_alvo.append({"grupo": f"B_entrou_e_zero_{a}", **descr(sub.loc[(sub["entrou_em_campo"] == True) & (sub["pontos_num"] == 0), "pontos_num"])})
    linhas_alvo.append({"grupo": f"C_entrou_e_pontuou_{a}", **descr(sub.loc[(sub["entrou_em_campo"] == True) & (sub["pontos_num"] != 0), "pontos_num"])})

df_alvo = pd.DataFrame(linhas_alvo)
df_alvo.to_csv(CSV_ALVO, index=False)

# ---------------------------------------------------------------- 3.2 TEMPORAL
print("== 3.2 TEMPORAL ==")
linhas_temp = []
for (a, r), sub in df.groupby(["temporada","rodada"]):
    p = sub["pontos_num"]
    entrou = sub["entrou_em_campo"].fillna(False).astype(bool)
    linhas_temp.append({
        "temporada": int(a), "rodada": int(r),
        "n_jogadores": int(len(sub)),
        "n_entrou": int(entrou.sum()),
        "n_nao_entrou": int((~entrou).sum()),
        "prop_entrou": float(entrou.mean()),
        "pts_mean": float(p.mean()),
        "pts_median": float(p.median()),
        "pts_zeros": int((p == 0).sum()),
        "pts_min": float(p.min()),
        "pts_max": float(p.max()),
        "prop_zeros": float((p == 0).mean()),
    })
df_temp = pd.DataFrame(linhas_temp).sort_values(["temporada","rodada"])

# classificacao heuristica
def classificar(row):
    # snapshot: quase todos sem entrar, pts_mean == 0, prop_zeros == 1
    if row["prop_entrou"] < 0.05 and row["prop_zeros"] > 0.99:
        return "POSSIVEL_SNAPSHOT"
    # anomalia mais suave: prop_entrou bem abaixo da mediana da temporada
    med = df_temp[df_temp["temporada"] == row["temporada"]]["prop_entrou"].median()
    if row["prop_entrou"] < med * 0.5:
        return "ANOMALA_INVESTIGAR"
    return "NORMAL"

df_temp["classificacao"] = df_temp.apply(classificar, axis=1)
df_temp.to_csv(CSV_TEMPORAL, index=False)

# ---------------------------------------------------------------- 3.3 jogos_num vs entrou
print("== 3.3 jogos_num vs entrou ==")
d = df.sort_values(["atleta_id","temporada","rodada"]).copy()
d["jogos_prev"] = d.groupby(["atleta_id","temporada"])["jogos_num"].shift(1)
d["rodada_prev"] = d.groupby(["atleta_id","temporada"])["rodada"].shift(1)
d = d[(d["rodada_prev"].notna()) & (d["rodada"] - d["rodada_prev"] == 1)].copy()
d["delta_jogos"] = d["jogos_num"] - d["jogos_prev"]
d["entrou_bool"] = d["entrou_em_campo"].fillna(False).astype(bool)

caso1 = d[(d["entrou_bool"] == True) & (d["delta_jogos"] <= 0)]        # entrou mas nao incrementou
caso2 = d[(d["entrou_bool"] == False) & (d["delta_jogos"] > 0)]        # nao entrou mas incrementou
caso3 = d[(d["delta_jogos"] < 0)]                                      # jogos_num retrocedeu

def exemplos(dd, n=5):
    cols = ["temporada","rodada","atleta_id","apelido","jogos_prev","jogos_num",
            "delta_jogos","entrou_em_campo","pontos_num"]
    return dd[cols].head(n).to_dict(orient="records")

res33 = {
    "n_linhas_analisadas": int(len(d)),
    "caso_entrou_e_nao_incrementou": int(len(caso1)),
    "caso_nao_entrou_e_incrementou": int(len(caso2)),
    "caso_jogos_num_retrocedeu": int(len(caso3)),
    "exemplos_caso1": exemplos(caso1),
    "exemplos_caso2": exemplos(caso2),
    "exemplos_caso3": exemplos(caso3),
}

# ---------------------------------------------------------------- 3.4 SCOUTS
print("== 3.4 SCOUTS ==")
linhas_sc = []
# reconstruir rodadas em que DP/GC nao existiam no RAW
RAW = ROOT / "data" / "raw" / "caRtola"
sem_col = {"DP": {}, "GC": {}}
for s in ["DP","GC"]:
    for a in ["2024","2025","2026"]:
        rodadas = set()
        for f in sorted((RAW / a).glob("*.csv")):
            cols = pd.read_csv(f, nrows=0, low_memory=False).columns.tolist()
            if s not in cols:
                import re
                m = re.search(r"rodada-(\d+)", f.name)
                if m:
                    rodadas.add(int(m.group(1)))
        sem_col[s][a] = rodadas

for s in SCOUTS:
    col = f"scout_{s}"
    for (a, r), sub in df.groupby(["temporada","rodada"]):
        vals = sub[col]
        nn = vals.notna()
        # estrutural?
        estrutural = (s in sem_col) and (r in sem_col[s].get(str(a), set()))
        linhas_sc.append({
            "scout": s, "temporada": int(a), "rodada": int(r),
            "n_linhas": int(len(vals)),
            "n_nan": int(vals.isna().sum()),
            "prop_nan": float(vals.isna().mean()),
            "n_zero": int((vals == 0).sum()),
            "n_positivo": int((vals > 0).sum()),
            "min": float(vals.min()) if nn.any() else np.nan,
            "max": float(vals.max()) if nn.any() else np.nan,
            "ausencia_estrutural": bool(estrutural),
        })
df_sc = pd.DataFrame(linhas_sc)
df_sc.to_csv(CSV_SCOUTS, index=False)

# resumo de NaN estrutural vs nao estrutural
res_scout_resumo = {}
for s in ["DP","GC"]:
    sub = df_sc[df_sc["scout"] == s]
    n_estrut = int(sub.loc[sub["ausencia_estrutural"], "n_linhas"].sum())
    n_nan_total = int(sub["n_nan"].sum())
    res_scout_resumo[s] = {
        "linhas_ausencia_estrutural": n_estrut,
        "linhas_nan_total": n_nan_total,
        "nan_nao_estrutural": n_nan_total - n_estrut,
    }

# ---------------------------------------------------------------- 3.5 OUTLIERS
print("== 3.5 OUTLIERS ==")
cols_out = ["temporada","rodada","atleta_id","apelido","posicao_id","clube_id",
            "clube_nome","pontos_num","entrou_em_campo","jogos_num",
            "scout_G","scout_A","scout_SG","scout_DS","scout_FC","scout_FS",
            "scout_CA","scout_CV","scout_GC","scout_PC","scout_PS","scout_PP",
            "scout_FT","scout_FD","scout_FF","scout_I","scout_V","scout_DE","scout_GS"
            if "scout_GS" in df.columns else "scout_G"]
cols_out = [c for c in cols_out if c in df.columns]

extremos = pd.concat([
    df.nsmallest(20, "pontos_num"),
    df.nlargest(20, "pontos_num"),
]).drop_duplicates(subset=["temporada","rodada","atleta_id"])
extremos[cols_out].to_csv(CSV_OUTLIERS, index=False)

res_out = {
    "min_global": float(df["pontos_num"].min()),
    "max_global": float(df["pontos_num"].max()),
    "n_lt_menos5": int((df["pontos_num"] < -5).sum()),
    "n_gt_20": int((df["pontos_num"] > 20).sum()),
    "n_gt_30": int((df["pontos_num"] > 30).sum()),
    "n_gt_40": int((df["pontos_num"] > 40).sum()),
}

# ---------------------------------------------------------------- 3.6 CLUBES / TRANSFERENCIAS
print("== 3.6 CLUBES ==")
grupo = df.groupby(["temporada","atleta_id"])["clube_id"].nunique()
trocas = grupo[grupo > 1]
res_clubes = {
    "n_pares_temporada_atleta": int(len(grupo)),
    "n_pares_com_troca_de_clube": int(len(trocas)),
    "prop_trocas": float(len(trocas) / len(grupo)),
}
exemplos_trocas = []
for (a, aid), _ in trocas.head(10).items():
    sub = df[(df["temporada"] == a) & (df["atleta_id"] == aid)].sort_values("rodada")
    registros = sub[["rodada","clube_id","clube_nome"]].to_dict(orient="records")
    exemplos_trocas.append({"temporada": int(a), "atleta_id": int(aid), "registros": registros})
res_clubes["exemplos"] = exemplos_trocas

# ---------------------------------------------------------------- 3.7 IDENTIDADE E POSICAO
print("== 3.7 IDENTIDADE/POSICAO ==")
# posicao_id unica dentro da temporada?
pos_dentro = df.groupby(["temporada","atleta_id"])["posicao_id"].nunique()
pos_dentro_mult = pos_dentro[pos_dentro > 1]
res_pos = {
    "n_pares_temporada_atleta": int(len(pos_dentro)),
    "n_pares_com_posicao_variavel_na_mesma_temporada": int(len(pos_dentro_mult)),
}
# posicao_id distinta entre temporadas para o mesmo atleta_id
pos_entre = df.groupby("atleta_id")["posicao_id"].nunique()
pos_entre_mult = pos_entre[pos_entre > 1]
res_pos["n_atletas_com_posicao_distinta_entre_temporadas"] = int(len(pos_entre_mult))

# clubes com nomes diferentes para mesmo clube_id
nomes_clube = df.groupby("clube_id")["clube_nome"].nunique()
clubes_mult_nome = nomes_clube[nomes_clube > 1]
res_pos["n_clubes_com_multiplos_nomes"] = int(len(clubes_mult_nome))
exemplos_clubes_mult = []
for cid, _ in clubes_mult_nome.head(10).items():
    sub = df[df["clube_id"] == cid]
    nomes = sorted(sub["clube_nome"].dropna().unique().tolist())
    exemplos_clubes_mult.append({"clube_id": int(cid), "nomes": nomes})
res_pos["exemplos_clubes_multiplos_nomes"] = exemplos_clubes_mult

# ---------------------------------------------------------------- 3.8 EVIDENCIAS
print("== 3.8 EVIDENCIAS ==")
evid = []
for _, row in df_temp.iterrows():
    cls = row["classificacao"]
    if cls == "NORMAL":
        continue
    impacto = "investigar no harness (possivel snapshot pre-rodada)"
    if cls == "ANOMALA_INVESTIGAR":
        impacto = "verificar se e snapshot ou rodada com poucos jogos"
    evid.append({
        "temporada": int(row["temporada"]),
        "rodada": int(row["rodada"]),
        "classificacao": cls,
        "evidencia": f"prop_entrou={row['prop_entrou']:.4f}; prop_zeros={row['prop_zeros']:.4f}; pts_mean={row['pts_mean']:.4f}",
        "impacto_potencial": impacto,
    })
# tambem registra 2025/R1 se nao caiu na heuristica
if not any((e["temporada"] == 2025 and e["rodada"] == 1) for e in evid):
    sub = df[(df["temporada"] == 2025) & (df["rodada"] == 1)]
    evid.append({
        "temporada": 2025, "rodada": 1,
        "classificacao": "POSSIVEL_SNAPSHOT",
        "evidencia": f"prop_entrou={sub['entrou_em_campo'].mean():.4f}; prop_zeros={(sub['pontos_num']==0).mean():.4f}",
        "impacto_potencial": "investigar no harness (possivel snapshot pre-rodada)",
    })
df_evid = pd.DataFrame(evid)
df_evid.to_csv(CSV_EVIDENCIAS, index=False)

# ---------------------------------------------------------------- RELATORIO MD
fim = datetime.now()
with open(RELATORIO, "w", encoding="utf-8") as fh:
    w = fh.write
    w("# Auditoria da base canonica v2 — caRtola\n\n")
    w(f"- **Inicio:** {inicio.isoformat(timespec='seconds')}\n")
    w(f"- **Fim:** {fim.isoformat(timespec='seconds')}\n")
    w(f"- **Duracao:** {(fim-inicio).total_seconds():.1f}s\n\n")

    w("## 3.1 Alvo (`pontos_num`)\n\n")
    w("Ver `reports/auditoria_alvo.csv`.\n\n")
    w("Amostra (temporada):\n\n")
    amostra = df_alvo[df_alvo["grupo"].isin(["temporada","A_nao_participou","B_entrou_e_zero","C_entrou_e_pontuou"])]
    w("```\n" + amostra.to_string(index=False) + "\n```\n\n")

    w("## 3.2 Auditoria temporal\n\n")
    w("Ver `reports/auditoria_temporal.csv`.\n\n")
    w("Contagem de rodadas por classificacao:\n\n")
    w(df_temp.groupby(["temporada","classificacao"]).size().to_string() + "\n\n")
    w("Rodadas nao-NORMAL:\n\n")
    nn = df_temp[df_temp["classificacao"] != "NORMAL"]
    if len(nn):
        w("```\n" + nn.to_string(index=False) + "\n```\n\n")
    else:
        w("_Nenhuma._\n\n")

    w("## 3.3 `entrou_em_campo` vs `jogos_num`\n\n")
    for k in ["n_linhas_analisadas","caso_entrou_e_nao_incrementou",
              "caso_nao_entrou_e_incrementou","caso_jogos_num_retrocedeu"]:
        w(f"- {k}: {res33[k]}\n")
    w("\nExemplos de 'entrou e nao incrementou' (ate 5):\n\n")
    w("```\n" + pd.DataFrame(res33["exemplos_caso1"]).to_string(index=False) + "\n```\n\n")
    w("Exemplos de 'nao entrou e incrementou' (ate 5):\n\n")
    w("```\n" + pd.DataFrame(res33["exemplos_caso2"]).to_string(index=False) + "\n```\n\n")
    w("Exemplos de 'jogos_num retrocedeu' (ate 5):\n\n")
    w("```\n" + pd.DataFrame(res33["exemplos_caso3"]).to_string(index=False) + "\n```\n\n")

    w("## 3.4 Scouts\n\n")
    w("Ver `reports/auditoria_scouts.csv`.\n\n")
    w("Resumo de NaN estrutural vs nao-estrutural:\n\n")
    for s, info in res_scout_resumo.items():
        w(f"- `scout_{s}`: {info}\n")
    w("\nNaN por scout (agregado por temporada):\n\n")
    agg = df_sc.groupby(["scout","temporada"]).agg(
        n_linhas=("n_linhas","sum"),
        n_nan=("n_nan","sum"),
        n_zero=("n_zero","sum"),
        n_positivo=("n_positivo","sum"),
    ).reset_index()
    agg["prop_nan"] = agg["n_nan"] / agg["n_linhas"]
    w("```\n" + agg.to_string(index=False) + "\n```\n\n")

    w("## 3.5 Outliers de pontos\n\n")
    for k, v in res_out.items():
        w(f"- {k}: {v}\n")
    w("\nTop 20 maiores e menores (ver CSV completo em `reports/auditoria_outliers.csv`):\n\n")
    w("```\n" + extremos[cols_out].to_string(index=False) + "\n```\n\n")

    w("## 3.6 Clubes / transferencias\n\n")
    for k, v in res_clubes.items():
        if k == "exemplos":
            continue
        w(f"- {k}: {v}\n")
    w("\nExemplos (ate 10):\n\n")
    for e in res_clubes["exemplos"]:
        w(f"- temporada={e['temporada']} atleta_id={e['atleta_id']}: {e['registros']}\n")
    w("\n")

    w("## 3.7 Identidade e posicao\n\n")
    for k, v in res_pos.items():
        if k == "exemplos_clubes_multiplos_nomes":
            continue
        w(f"- {k}: {v}\n")
    w("\nClubes com multiplos nomes (ate 10):\n\n")
    for e in res_pos["exemplos_clubes_multiplos_nomes"]:
        w(f"- clube_id={e['clube_id']}: {e['nomes']}\n")
    w("\n")

    w("## 3.8 Evidencias de elegibilidade (descritivas)\n\n")
    w("Ver `reports/auditoria_evidencias.csv`.\n\n")
    if len(df_evid):
        w("```\n" + df_evid.to_string(index=False) + "\n```\n\n")
    else:
        w("_Nenhuma rodada sinalizada._\n\n")

    w("## Artefatos gerados\n\n")
    for p in [RELATORIO, CSV_ALVO, CSV_TEMPORAL, CSV_SCOUTS,
              CSV_OUTLIERS, CSV_EVIDENCIAS]:
        w(f"- `{p.relative_to(ROOT)}`\n")

# ---------------------------------------------------------------- resumo terminal
print()
print("=== ETAPA 3 — RESUMO ===")
print(f"linhas analisadas: {len(df)}")
print(f"temporadas: {sorted(df['temporada'].unique().tolist())}")
print(f"rodadas: {df['rodada'].min()}..{df['rodada'].max()}")
print(f"jogadores distintos (atleta_id): {df['atleta_id'].nunique()}")
n_norm = int((df_temp['classificacao']=='NORMAL').sum())
n_anom = int((df_temp['classificacao']=='ANOMALA_INVESTIGAR').sum())
n_snap = int((df_temp['classificacao']=='POSSIVEL_SNAPSHOT').sum())
print(f"rodadas normais: {n_norm}")
print(f"rodadas anomalas: {n_anom}")
print(f"possiveis snapshots: {n_snap}")
print(f"top outliers: {res_out}")
print(f"scouts auditados: {len(SCOUTS)}")
print(f"transferencias detectadas (pares temporada/atleta com >1 clube): {res_clubes['n_pares_com_troca_de_clube']}")
print(f"conflitos de posicao (mesma temporada): {res_pos['n_pares_com_posicao_variavel_na_mesma_temporada']}")
print(f"conflitos de posicao (entre temporadas): {res_pos['n_atletas_com_posicao_distinta_entre_temporadas']}")
print(f"clubes com multiplos nomes: {res_pos['n_clubes_com_multiplos_nomes']}")
print()
print("artefatos:")
for p in [RELATORIO, CSV_ALVO, CSV_TEMPORAL, CSV_SCOUTS,
          CSV_OUTLIERS, CSV_EVIDENCIAS]:
    print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")
