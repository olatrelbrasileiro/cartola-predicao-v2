#!/usr/bin/env python3
"""
ETAPA 4 — Testes automatizados do harness.
Le reports/harness_folds_v2.csv e valida invariantes.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "cartola-predicao-v2"
CSV_FOLDS = ROOT / "reports" / "harness_folds_v2.csv"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"

ESPERADO_FOLDS_TOTAL = 76
ESPERADO_FOLDS = {2024: 29, 2025: 29, 2026: 18}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()

def main():
    falhas = []

    # carregar manifesto
    dfm = pd.read_csv(CSV_FOLDS)

    # 1. total de folds
    if len(dfm) != ESPERADO_FOLDS_TOTAL:
        falhas.append(f"total de folds: {len(dfm)} != {ESPERADO_FOLDS_TOTAL}")

    # 2. folds por temporada
    for t, esperado in ESPERADO_FOLDS.items():
        n = int((dfm["temporada"] == t).sum())
        if n != esperado:
            falhas.append(f"folds {t}: {n} != {esperado}")

    # 3. train_max < test_round em todos os folds
    bad = dfm[dfm["train_max_round"] >= dfm["test_round"]]
    if len(bad):
        falhas.append(f"{len(bad)} folds com train_max >= test_round")

    # 4. test_round == rodada_alvo
    bad = dfm[dfm["test_round"] != dfm["rodada_alvo"]]
    if len(bad):
        falhas.append(f"{len(bad)} folds com test_round != rodada_alvo")

    # 5. treino de uma unica temporada por fold (coluna test_rodadas contem so 1 rodada)
    for _, row in dfm.iterrows():
        rs = str(row["train_rodadas"]).split(",")
        ts = str(row["test_rodadas"]).split(",")
        if len(ts) != 1:
            falhas.append(f"fold {row['temporada']}/R{row['rodada_alvo']}: multiplas rodadas no teste")
        if any(int(x) >= int(row["rodada_alvo"]) for x in rs):
            falhas.append(f"fold {row['temporada']}/R{row['rodada_alvo']}: treino com rodada >= alvo")

    # 6. 2025/R1 nunca aparece
    for _, row in dfm.iterrows():
        if row["temporada"] == 2025:
            rodadas_train = set(map(int, str(row["train_rodadas"]).split(",")))
            rodadas_test = set(map(int, str(row["test_rodadas"]).split(",")))
            if 1 in rodadas_train or 1 in rodadas_test:
                falhas.append(f"fold 2025/R{row['rodada_alvo']}: 2025/R1 presente")

    # 7. nenhum B ambigua na avaliacao principal
    if (dfm["n_test_ambiguous_B"] > 0).any():
        # isso e' ok: o campo existe justamente pra rastrear. O que nao pode
        # e' B dentro de n_test_used_for_metrics. Vamos checar consistencia:
        for _, row in dfm.iterrows():
            if row["n_test_used_for_metrics"] + row["n_test_ambiguous_B"] != row["n_test_eligible"]:
                falhas.append(
                    f"fold {row['temporada']}/R{row['rodada_alvo']}: "
                    f"used_for_metrics + ambiguous_B != eligible "
                    f"({row['n_test_used_for_metrics']}+{row['n_test_ambiguous_B']}"
                    f" != {row['n_test_eligible']})"
                )

    # 8. todos os folds tem treino e teste
    for _, row in dfm.iterrows():
        if row["n_train_eligible"] <= 0:
            falhas.append(f"fold {row['temporada']}/R{row['rodada_alvo']}: treino vazio")
        if row["n_test_eligible"] <= 0:
            falhas.append(f"fold {row['temporada']}/R{row['rodada_alvo']}: teste vazio")

    # 9. Parquet continua com 77253 linhas
    df = pd.read_parquet(PARQUET)
    if len(df) != 77253:
        falhas.append(f"parquet: {len(df)} linhas != 77253")
    if list(df.columns)[:3] != ["temporada","atleta_id","apelido"]:
        falhas.append(f"parquet: colunas fora de ordem: {list(df.columns)[:3]}")

    # saida
    print("=== HARNESS TESTS ===")
    if falhas:
        print("FAIL")
        for f in falhas:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("PASS")
        print(f" folds totais: {len(dfm)}")
        print(f" folds 2024: {int((dfm['temporada']==2024).sum())}")
        print(f" folds 2025: {int((dfm['temporada']==2025).sum())}")
        print(f" folds 2026: {int((dfm['temporada']==2026).sum())}")
        print(f" parquet linhas: {len(df)}")
        print(f" parquet sha256: {sha256(PARQUET)[:16]}...")

if __name__ == "__main__":
    main()
