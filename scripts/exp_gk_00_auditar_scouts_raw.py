import pandas as pd
from pathlib import Path

ROOT = Path("data/raw/caRtola")
SCOUTS = ["DE", "GS", "SG", "DS"]
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

all_rows = []
schema_rows = []

for season in [2024, 2025, 2026]:
    files = sorted((ROOT / str(season)).glob("rodada-*.csv"))

    for f in files:
        rodada = int(f.stem.split("-")[1])
        df = pd.read_csv(f, low_memory=False)

        required = ["atletas.atleta_id", "atletas.apelido",
                    "atletas.posicao_id", "atletas.entrou_em_campo"]

        if not all(c in df.columns for c in required):
            continue

        gk = df[
            (df["atletas.posicao_id"] == 1) &
            (df["atletas.entrou_em_campo"] == True)
        ].copy()

        for scout in SCOUTS:
            exists = scout in df.columns

            schema_rows.append({
                "season": season,
                "rodada": rodada,
                "scout": scout,
                "coluna_existe": exists,
                "n_goleiros": len(gk),
            })

            for _, row in gk.iterrows():
                raw = (
                    pd.to_numeric(row[scout], errors="coerce")
                    if exists else
                    float("nan")
                )

                all_rows.append({
                    "season": season,
                    "rodada": rodada,
                    "atleta_id": row["atletas.atleta_id"],
                    "atleta": row["atletas.apelido"],
                    "scout": scout,
                    "coluna_existe": exists,
                    "raw": raw,
                })

d = pd.DataFrame(all_rows)

# Ordena para reconstruir o acumulado
d = d.sort_values(
    ["season", "atleta_id", "scout", "rodada"]
).reset_index(drop=True)

# Reconstrução:
# - coluna ausente = desconhecido
# - coluna existente + NaN = acumulado não mudou
# - primeiro NaN observado = 0
d["acumulado_corrigido"] = pd.NA

for (season, atleta_id, scout), idx in d.groupby(
    ["season", "atleta_id", "scout"]
).groups.items():

    prev = 0.0

    for i in idx:
        row = d.loc[i]

        if not row["coluna_existe"]:
            d.loc[i, "acumulado_corrigido"] = pd.NA
            continue

        raw = row["raw"]

        if pd.notna(raw):
            prev = float(raw)

        d.loc[i, "acumulado_corrigido"] = prev

# Delta por rodada
d["delta"] = pd.NA

for (_, _, _), idx in d.groupby(
    ["season", "atleta_id", "scout"]
).groups.items():

    prev = None

    for i in idx:
        val = d.loc[i, "acumulado_corrigido"]

        if pd.isna(val):
            prev = None
            continue

        val = float(val)

        if prev is None:
            d.loc[i, "delta"] = val
        else:
            d.loc[i, "delta"] = val - prev

        prev = val

# -------------------------
# RELATÓRIO PRINCIPAL
# -------------------------

summary = []

for season in [2024, 2025, 2026]:
    for scout in SCOUTS:
        x = d[
            (d["season"] == season) &
            (d["scout"] == scout)
        ]

        total = len(x)
        schema_present = int(x["coluna_existe"].sum())

        raw_present = int(
            x.loc[x["coluna_existe"], "raw"].notna().sum()
        )

        corrected_known = int(
            x["acumulado_corrigido"].notna().sum()
        )

        positive = int(
            (
                pd.to_numeric(
                    x["acumulado_corrigido"],
                    errors="coerce"
                ) > 0
            ).sum()
        )

        summary.append({
            "season": season,
            "scout": scout,
            "observacoes_goleiros": total,
            "observacoes_com_coluna": schema_present,
            "raw_preenchido_pct": (
                100 * raw_present / schema_present
                if schema_present else 0
            ),
            "apos_regra_conhecido_pct": (
                100 * corrected_known / total
                if total else 0
            ),
            "acumulado_maior_que_zero_pct": (
                100 * positive / schema_present
                if schema_present else 0
            ),
            "coluna_ausente_observacoes": total - schema_present,
        })

summary_df = pd.DataFrame(summary)

# -------------------------
# ANOMALIAS
# -------------------------

anomalias = d[
    pd.to_numeric(d["delta"], errors="coerce") < 0
].copy()

# -------------------------
# SALVA
# -------------------------

d.to_csv(
    OUT / "exp_gk_00_goleiros_corrigido.csv",
    index=False
)

pd.DataFrame(schema_rows).to_csv(
    OUT / "exp_gk_00_schema.csv",
    index=False
)

summary_df.to_csv(
    OUT / "exp_gk_00_resumo_corrigido.csv",
    index=False
)

anomalias.to_csv(
    OUT / "exp_gk_00_anomalias.csv",
    index=False
)

print("\n" + "=" * 80)
print("RESUMO — SCOUTS ACUMULADOS")
print("=" * 80)

print(
    summary_df[
        [
            "season",
            "scout",
            "raw_preenchido_pct",
            "apos_regra_conhecido_pct",
            "acumulado_maior_que_zero_pct",
            "coluna_ausente_observacoes",
        ]
    ].to_string(index=False)
)

print("\nAnomalias — reduções no acumulado:")
print(f"{len(anomalias)}")

if len(anomalias):
    print(anomalias.head(20).to_string(index=False))

print("\nArquivos gerados:")
print("- reports/exp_gk_00_goleiros_corrigido.csv")
print("- reports/exp_gk_00_resumo_corrigido.csv")
print("- reports/exp_gk_00_schema.csv")
print("- reports/exp_gk_00_anomalias.csv")
