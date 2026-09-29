#!/usr/bin/env python3
"""
ETAPA 4 — Harness walk-forward expansivo.

Infraestrutura de validacao temporal. NAO cria features, NAO treina modelos,
NAO faz feature engineering. So constroi folds e valida leakage.
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path.home() / "cartola-predicao-v2"
PARQUET = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
POLICY = ROOT / "config" / "elegibilidade_v2.yaml"
CONFIG = ROOT / "config.yaml"
REP = ROOT / "reports"
TMP = ROOT / ".tmp"
REP.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

CSV_FOLDS = REP / "harness_folds_v2.csv"
RELATORIO = REP / "harness_walk_forward_v2.md"

TARGET_COL = "pontos_num"
KEY_COLS = ["temporada", "rodada", "atleta_id"]
MIN_TRAIN_ROUNDS = None  # carregado de config.yaml em main()
ESPERADO_FOLDS = {2024: 29, 2025: 29, 2026: 18}
ESPERADO_TOTAL_FOLDS = 76

# ---------------------------------------------------------------- util
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()

# ---------------------------------------------------------------- loaders
def load_canonical_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df.sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)
    return df

def load_policy() -> dict:
    with open(POLICY, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------- runtime flags
def derive_runtime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva flags em MEMORIA. Nao altera o Parquet."""
    out = df.copy()
    out["flag_ambigua_B"] = (
        (out["entrou_em_campo"] == False) & (out[TARGET_COL] != 0)
    )
    # elegibilidade: exclui 2025/R1
    out["elegivel"] = ~((out["temporada"] == 2025) & (out["rodada"] == 1))
    return out

# ---------------------------------------------------------------- folds
def build_folds(df: pd.DataFrame) -> list:
    folds = []
    for temporada in sorted(df["temporada"].unique()):
        sub_t = df[df["temporada"] == temporada]
        rodadas = sorted(sub_t["rodada"].unique())
        for r in rodadas:
            if r <= MIN_TRAIN_ROUNDS:
                continue
            folds.append({
                "temporada": int(temporada),
                "rodada_alvo": int(r),
                "rodada_treino_inicio": int(min(rodadas)),
                "rodada_treino_fim": int(r - 1),
            })
    return folds

# ---------------------------------------------------------------- run_fold
def run_fold(df: pd.DataFrame, fold: dict) -> dict:
    t = fold["temporada"]
    r = fold["rodada_alvo"]

    mask_temporada = df["temporada"] == t
    train_raw = df[mask_temporada & (df["rodada"] < r)]
    test_raw = df[mask_temporada & (df["rodada"] == r)]

    train_eligible = train_raw[train_raw["elegivel"]]
    test_eligible = test_raw[test_raw["elegivel"]]

    test_amb = test_eligible[test_eligible["flag_ambigua_B"]]
    test_metrics = test_eligible[~test_eligible["flag_ambigua_B"]]

    meta = {
        "temporada": int(t),
        "rodada_alvo": int(r),
        "rodada_treino_inicio": int(fold["rodada_treino_inicio"]),
        "rodada_treino_fim": int(fold["rodada_treino_fim"]),
        "n_train_raw": int(len(train_raw)),
        "n_train_eligible": int(len(train_eligible)),
        "n_test_raw": int(len(test_raw)),
        "n_test_eligible": int(len(test_eligible)),
        "n_test_ambiguous_B": int(len(test_amb)),
        "n_test_used_for_metrics": int(len(test_metrics)),
        "train_rodadas": sorted(train_eligible["rodada"].unique().tolist()),
        "test_rodadas": sorted(test_eligible["rodada"].unique().tolist()),
        "target_column": TARGET_COL,
        "eligibility_policy": str(POLICY.relative_to(ROOT)),
        "train_min_round": int(train_eligible["rodada"].min()) if len(train_eligible) else None,
        "train_max_round": int(train_eligible["rodada"].max()) if len(train_eligible) else None,
        "test_round": int(test_eligible["rodada"].min()) if len(test_eligible) else None,
    }
    return {
        "meta": meta,
        "train_eligible": train_eligible,
        "test_eligible": test_eligible,
        "test_metrics": test_metrics,
        "test_ambiguous_B": test_amb,
        "train_raw": train_raw,
        "test_raw": test_raw,
    }

# ---------------------------------------------------------------- validate_fold
def validate_fold(fold_result: dict) -> None:
    m = fold_result["meta"]
    # TESTE 1: train_max < rodada_alvo
    assert m["train_max_round"] is not None, f"[T1] fold sem treino: {m}"
    assert m["train_max_round"] < m["rodada_alvo"], \
        f"[T1] train_max_round={m['train_max_round']} >= rodada_alvo={m['rodada_alvo']}"

    # TESTE 2: test_round == rodada_alvo
    assert m["test_round"] == m["rodada_alvo"], \
        f"[T2] test_round={m['test_round']} != rodada_alvo={m['rodada_alvo']}"

    # TESTE 3: sem interseccao de chaves
    train_keys = set(map(tuple, fold_result["train_eligible"][KEY_COLS].values.tolist()))
    test_keys = set(map(tuple, fold_result["test_eligible"][KEY_COLS].values.tolist()))
    inter = train_keys & test_keys
    assert not inter, f"[T3] interseccao treino/teste: {len(inter)} chaves"

    # TESTE 4/5: 2025/R1 ausente
    df_train = fold_result["train_eligible"]
    df_test = fold_result["test_eligible"]
    bad_train = df_train[(df_train["temporada"] == 2025) & (df_train["rodada"] == 1)]
    bad_test = df_test[(df_test["temporada"] == 2025) & (df_test["rodada"] == 1)]
    assert len(bad_train) == 0, "[T4] 2025/R1 no treino"
    assert len(bad_test) == 0, "[T5] 2025/R1 no teste"

    # TESTE 6: nenhuma linha B na metrica principal
    assert fold_result["test_metrics"]["flag_ambigua_B"].sum() == 0, \
        "[T6] linha flag_ambigua_B=True na metrica principal"

    # TESTE 8: temporadas nao se misturam
    assert df_train["temporada"].nunique() == 1, "[T8] treino com mais de uma temporada"
    assert df_test["temporada"].nunique() == 1, "[T8] teste com mais de uma temporada"
    assert df_train["temporada"].iloc[0] == df_test["temporada"].iloc[0], \
        "[T8] treino e teste de temporadas diferentes"

# ---------------------------------------------------------------- run all
def validate_all_folds(results: list) -> None:
    for r in results:
        validate_fold(r)

    # TESTE 7: numero de folds por temporada
    por_temp = {}
    for r in results:
        t = r["meta"]["temporada"]
        por_temp[t] = por_temp.get(t, 0) + 1
    assert por_temp == ESPERADO_FOLDS, \
        f"[T7] contagem de folds por temporada divergente: {por_temp}"
    assert sum(por_temp.values()) == ESPERADO_TOTAL_FOLDS, \
        f"[T7] total de folds != {ESPERADO_TOTAL_FOLDS}"

def save_fold_manifest(results: list) -> pd.DataFrame:
    linhas = []
    for r in results:
        m = dict(r["meta"])
        m["train_rodadas"] = ",".join(map(str, m["train_rodadas"]))
        m["test_rodadas"] = ",".join(map(str, m["test_rodadas"]))
        linhas.append(m)
    dfm = pd.DataFrame(linhas)
    cols = ["temporada","rodada_alvo","rodada_treino_inicio","rodada_treino_fim",
            "n_train_raw","n_train_eligible","n_test_raw","n_test_eligible",
            "n_test_ambiguous_B","n_test_used_for_metrics",
            "train_min_round","train_max_round","test_round",
            "train_rodadas","test_rodadas",
            "target_column","eligibility_policy"]
    dfm = dfm[cols]
    dfm.to_csv(CSV_FOLDS, index=False)
    return dfm

# ---------------------------------------------------------------- main
def main():
    global MIN_TRAIN_ROUNDS
    inicio = datetime.now()
    print("== ETAPA 4 — HARNESS WALK-FORWARD ==")

    # imutabilidade do Parquet
    hash_antes = sha256(PARQUET)
    n_linhas_antes = len(pd.read_parquet(PARQUET))
    cols_antes = list(pd.read_parquet(PARQUET).columns)
    print(f"Parquet SHA-256 (antes): {hash_antes[:16]}...")
    print(f"Linhas antes: {n_linhas_antes}")

    # load
    policy = load_policy()  # carregado para documentar; nao define min_train_rounds
    cfg = load_config()
    mtr = cfg.get("validation", {}).get("min_train_rounds")
    if not isinstance(mtr, int) or mtr <= 0:
        print(f"ERRO: config.yaml sem validation.min_train_rounds valido "
              f"(recebido: {mtr!r})", file=sys.stderr)
        sys.exit(3)
    MIN_TRAIN_ROUNDS = mtr
    df = load_canonical_data()
    print(f"Linhas carregadas: {len(df)}")
    print(f"Colunas: {len(df.columns)}")
    print(f"min_train_rounds (config.yaml): {MIN_TRAIN_ROUNDS}")

    df = derive_runtime_flags(df)
    n_excluidas_2025R1 = int(((df["temporada"] == 2025) & (df["rodada"] == 1)).sum())
    n_ambigua_B = int(df["flag_ambigua_B"].sum())
    print(f"Linhas excluidas por 2025/R1: {n_excluidas_2025R1}")
    print(f"Linhas flag_ambigua_B=True: {n_ambigua_B}")

    # folds
    folds = build_folds(df)
    print(f"Folds construidos: {len(folds)}")

    # executar
    results = [run_fold(df, f) for f in folds]

    # validar
    validate_all_folds(results)
    print("Validacao de folds: OK")

    # salvar manifesto
    dfm = save_fold_manifest(results)
    print(f"Manifesto salvo: {CSV_FOLDS} ({len(dfm)} linhas)")

    # imutabilidade do Parquet depois
    hash_depois = sha256(PARQUET)
    cols_depois = list(pd.read_parquet(PARQUET).columns)
    n_linhas_depois = len(pd.read_parquet(PARQUET))
    imutavel = (hash_antes == hash_depois
                and n_linhas_antes == n_linhas_depois
                and cols_antes == cols_depois)
    print(f"Parquet SHA-256 (depois): {hash_depois[:16]}...")
    print(f"Parquet imutavel: {imutavel}")

    # relatorio
    fim = datetime.now()
    with open(RELATORIO, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# Harness walk-forward v2 — relatorio\n\n")
        w(f"- **Inicio:** {inicio.isoformat(timespec='seconds')}\n")
        w(f"- **Fim:** {fim.isoformat(timespec='seconds')}\n")
        w(f"- **Duracao:** {(fim-inicio).total_seconds():.2f}s\n\n")

        w("## 1. Politica carregada\n\n")
        w(f"- Arquivo: `{POLICY.relative_to(ROOT)}`\n")
        w(f"- min_train_rounds (harness): {MIN_TRAIN_ROUNDS}\n")
        w(f"- Walk-forward expansivo intra-temporada.\n\n")

        w("## 2. Numero total de linhas\n\n")
        w(f"- {len(df)}\n\n")

        w("## 3. Linhas excluidas (2025/R1)\n\n")
        w(f"- {n_excluidas_2025R1}\n\n")

        w("## 4. Linhas flag_ambigua_B\n\n")
        w(f"- {n_ambigua_B}\n")
        w("- Permanecem no dataframe; excluidas apenas da metrica principal.\n\n")

        w("## 5. Numero de folds por temporada\n\n")
        for t in sorted(dfm["temporada"].unique()):
            w(f"- {t}: {int((dfm['temporada']==t).sum())}\n")
        w("\n")

        w("## 6. Numero total de folds\n\n")
        w(f"- {len(dfm)}\n\n")

        w("## 7. Tabela-resumo dos folds\n\n")
        w("```\n")
        w(dfm[["temporada","rodada_alvo","train_min_round","train_max_round",
               "test_round","n_train_eligible","n_test_eligible",
               "n_test_ambiguous_B","n_test_used_for_metrics"]].to_string(index=False))
        w("\n```\n\n")

        w("## 8. Testes de leakage\n\n")
        w("- TESTE 1 (train_max < rodada_alvo): PASS\n")
        w("- TESTE 2 (test_round == rodada_alvo): PASS\n")
        w("- TESTE 3 (sem interseccao treino/teste): PASS\n")
        w("- TESTE 4 (2025/R1 fora do treino): PASS\n")
        w("- TESTE 5 (2025/R1 fora do teste): PASS\n")
        w("- TESTE 6 (sem B na metrica principal): PASS\n")
        w("- TESTE 7 (contagem de folds): PASS\n")
        w("- TESTE 8 (sem mistura de temporadas): PASS\n\n")

        w("## 9. Imutabilidade do Parquet\n\n")
        w(f"- SHA-256 (antes): `{hash_antes}`\n")
        w(f"- SHA-256 (depois): `{hash_depois}`\n")
        w(f"- Linhas antes/depois: {n_linhas_antes}/{n_linhas_depois}\n")
        w(f"- Colunas iguais: {cols_antes == cols_depois}\n")
        w(f"- **Imutavel: {imutavel}**\n\n")

        w("## 10. Conclusao\n\n")
        w("Harness pronto para receber baselines (B0–B5) e modelos (Ridge, CatBoost).\n")
        w("Nenhuma feature criada nesta etapa. Nenhum modelo treinado.\n")

    print()
    print("=== ETAPA 4 — RESUMO ===")
    print(f"folds criados: {len(dfm)}")
    print(f"folds 2024: {int((dfm['temporada']==2024).sum())}")
    print(f"folds 2025: {int((dfm['temporada']==2025).sum())}")
    print(f"folds 2026: {int((dfm['temporada']==2026).sum())}")
    print(f"linhas excluidas 2025/R1: {n_excluidas_2025R1}")
    print(f"linhas flag_ambigua_B: {n_ambigua_B}")
    print(f"parquet imutavel: {imutavel}")
    print(f"manifesto: {CSV_FOLDS.relative_to(ROOT)}")
    print(f"relatorio: {RELATORIO.relative_to(ROOT)}")

    if not imutavel:
        print("ERRO: parquet alterado.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
