#!/usr/bin/env python3
"""Auditoria do football_data_bra.csv."""
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
RAW = ROOT/"data"/"raw"/"odds"/"football_data_bra.csv"
REP = ROOT/"reports"
REP.mkdir(parents=True, exist_ok=True)

OUT_MD = REP/"auditoria_odds_historicas.md"
OUT_CSV = REP/"auditoria_odds_historicas.csv"

COLS = ["Season","Date","Time","Home","Away","HG","AG","Res",
        "AvgCH","AvgCD","AvgCA","MaxCH","MaxCD","MaxCA"]

df = pd.read_csv(RAW, encoding="utf-8")
print(f"linhas: {len(df)}  colunas: {len(df.columns)}")

df = df[COLS].copy()
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

rows = []

def bucket(dfx, nome):
    return {
        "grupo": nome,
        "n_partidas": int(len(dfx)),
        "n_temp": int(dfx["Season"].nunique()) if "Season" in dfx else None,
        "avgch_nulos": int(dfx["AvgCH"].isna().sum()),
        "avgcd_nulos": int(dfx["AvgCD"].isna().sum()),
        "avgca_nulos": int(dfx["AvgCA"].isna().sum()),
        "avgch_leq1": int((dfx["AvgCH"] <= 1).sum()),
        "avgcd_leq1": int((dfx["AvgCD"] <= 1).sum()),
        "avgca_leq1": int((dfx["AvgCA"] <= 1).sum()),
        "avgch_max": float(dfx["AvgCH"].max()) if dfx["AvgCH"].notna().any() else None,
        "avgca_max": float(dfx["AvgCA"].max()) if dfx["AvgCA"].notna().any() else None,
    }

rows.append(bucket(df, "global"))

for ano in sorted(df["Season"].unique()):
    rows.append(bucket(df[df["Season"] == ano], f"ano_{ano}"))

# duplicatas
dup_home_away_date = df.duplicated(subset=["Season","Date","Home","Away"]).sum()

# cobertura por temporada
cobertura = []
for ano in sorted(df["Season"].unique()):
    sub = df[df["Season"] == ano]
    completos = sub.dropna(subset=["AvgCH","AvgCD","AvgCA"])
    cobertura.append({
        "Season": int(ano),
        "n_partidas": int(len(sub)),
        "n_completas": int(len(completos)),
        "cobertura": round(len(completos)/len(sub), 4) if len(sub) else None,
    })
df_cob = pd.DataFrame(cobertura)

# clubes únicos
clubes = sorted(set(df["Home"].dropna()) | set(df["Away"].dropna()))

# relatório md
fim = datetime.now()
L = []
L.append("# Auditoria das odds históricas (Football-Data)\n\n")
L.append(f"- Arquivo: `{RAW.relative_to(ROOT)}`\n")
L.append(f"- Linhas: {len(df)}\n")
L.append(f"- Temporadas: {sorted(df['Season'].unique().tolist())}\n")
L.append(f"- Data: {fim.isoformat(timespec='seconds')}\n\n")

L.append("## Fonte\n\n")
L.append("Football-Data.co.uk — `BRA.csv` (Brasileirão Série A, 2012–2026).\n")
L.append("Colunas usadas: `AvgCH`, `AvgCD`, `AvgCA` = **odds de fechamento (closing)**.\n\n")

L.append("## Duplicatas (Season, Date, Home, Away)\n\n")
L.append(f"- `{int(dup_home_away_date)}`\n\n")

L.append("## Cobertura por temporada\n\n")
L.append("```\n")
L.append(df_cob.to_string(index=False))
L.append("\n```\n\n")

L.append("## Diagnóstico global\n\n")
L.append(f"- Nulos AvgCH: {rows[0]['avgch_nulos']}\n")
L.append(f"- Nulos AvgCD: {rows[0]['avgcd_nulos']}\n")
L.append(f"- Nulos AvgCA: {rows[0]['avgca_nulos']}\n")
L.append(f"- <=1 AvgCH: {rows[0]['avgch_leq1']}\n")
L.append(f"- <=1 AvgCD: {rows[0]['avgcd_leq1']}\n")
L.append(f"- <=1 AvgCA: {rows[0]['avgca_leq1']}\n")
L.append(f"- AvgCH max: {rows[0]['avgch_max']}\n")
L.append(f"- AvgCA max: {rows[0]['avgca_max']}\n\n")

L.append("## Clubes distintos\n\n")
L.append(f"- total: {len(clubes)}\n\n")
for c in clubes:
    L.append(f"- {c}\n")

OUT_MD.write_text("".join(L), encoding="utf-8")
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print()
print("== RESUMO ==")
print(f"linhas: {len(df)}")
print(f"temporadas: {sorted(df['Season'].unique().tolist())}")
print(f"duplicatas exatas (Season/Date/Home/Away): {int(dup_home_away_date)}")
print()
print("Cobertura por temporada (2024-2026):")
print(df_cob[df_cob["Season"] >= 2024].to_string(index=False))
print()
print(f"nulos AvgCH global: {rows[0]['avgch_nulos']} / {len(df)}")
print(f"clubes distintos: {len(clubes)}")
print()
print(f"relatorio: {OUT_MD.relative_to(ROOT)}")
print(f"csv:       {OUT_CSV.relative_to(ROOT)}")
