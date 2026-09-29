#!/usr/bin/env python3
"""
Auditoria da etapa de elegibilidade pre-rodada via status_id.

Verifica:
A. Cobertura (por temporada e rodada)
B. Distribuicao de status por temporada
C. Integridade (nulos reais, desconhecidos, duplicatas, multi-status)
D. Teste de invariancia do FILTRO as variaveis de resultado
E. Integracao com o harness (76 folds) sem alterar historico

NAO treina modelo. NAO altera o Parquet canonico.
"""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def detect_root():
    for c in [Path.home() / "cartola-predicao-v2",
              Path.home() / "cartola-ai-insights",
              Path("/workspaces/cartola-predicao-v2"),
              Path("/workspaces/cartola-ai-insights"),
              Path("/data/data/com.termux/files/home/cartola-predicao-v2"),
              Path("/data/data/com.termux/files/home/cartola-ai-insights")]:
        if (c / "data" / "interim" / "cartola_jogador_rodada_v2.parquet").exists():
            return c
    raise RuntimeError("Raiz do projeto nao encontrada")


ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"
HARNESS = REP / "harness_folds_v2.csv"

OUT_COB = REP / "status_elegibilidade_v1_cobertura.csv"
OUT_FOLDS = REP / "status_elegibilidade_v1_folds.csv"
OUT_DIST = REP / "status_elegibilidade_v1_distribuicao.csv"
OUT_VALID = REP / "status_elegibilidade_v1_validation.json"
OUT_MD = REP / "status_elegibilidade_v1.md"

HASH_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"

ESPERADO_FOLDS_POR_TEMP = {2024: 29, 2025: 29, 2026: 18}


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "status_eleg", str(SCRIPTS / "status_elegibilidade.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


se = _load_module()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------- testes
def teste_integridade(df):
    problemas = []
    status = df["status_id"]

    n_nulos_reais = int(status.isna().sum())
    if n_nulos_reais > 0:
        problemas.append(f"status_id tem {n_nulos_reais} valores NaN reais")

    try:
        sid_int = status.astype("int64")
    except Exception as e:
        problemas.append(f"status_id nao inteiro: {e}")
        return {"problemas": problemas}

    presentes = sorted(sid_int.unique().tolist())
    desconhecidos = [c for c in presentes if c not in se.STATUS_CONHECIDOS]
    if desconhecidos:
        problemas.append(f"status_id desconhecidos: {desconhecidos}")

    dup = int(df.duplicated(subset=["temporada", "rodada", "atleta_id"]).sum())
    if dup > 0:
        problemas.append(f"duplicatas (T,R,A): {dup}")

    multi = (df.groupby(["temporada", "rodada", "atleta_id"])["status_id"]
               .nunique(dropna=False))
    n_multi = int((multi > 1).sum())
    if n_multi > 0:
        problemas.append(f"celulas (T,R,A) com mais de um status: {n_multi}")

    return {
        "n_linhas": int(len(df)),
        "n_nulos_reais": n_nulos_reais,
        "status_presentes": presentes,
        "status_desconhecidos": desconhecidos,
        "duplicatas_T_R_A": dup,
        "multi_status_celulas": n_multi,
        "problemas": problemas,
    }


def teste_invariancia_resultados(df, folds):
    """
    Teste de invariancia do FILTRO as variaveis de resultado.

    Prova que o conjunto elegivel depende EXCLUSIVAMENTE de status_id,
    nao de nenhuma variavel de resultado (pontos_num, entrou_em_campo,
    scout_*, media_num, variacao_num, preco_num).

    NAO prova proveniencia temporal do snapshot; essa foi auditada
    separadamente em reports/status_id_auditoria.md.
    """
    falhas = []
    for (t, R) in folds:
        sub = df[(df["temporada"] == t) & (df["rodada"] == R)].copy()
        if len(sub) == 0:
            falhas.append(f"{t}/R{R}: rodada-alvo vazia")
            continue

        df_orig, _ = se.filtrar_universo_alvo(sub)
        eleg_orig = set(df_orig["atleta_id"].astype(int).tolist())

        if len(eleg_orig) == 0:
            falhas.append(f"{t}/R{R}: universo elegivel vazio")

        sub_p = sub.copy()
        sub_p["pontos_num"] = -sub_p["pontos_num"].fillna(0)
        sub_p["entrou_em_campo"] = ~sub_p["entrou_em_campo"].fillna(False).astype(bool)
        for c in sub_p.columns:
            if c.startswith("scout_"):
                sub_p[c] = 0.0
        if "media_num" in sub_p.columns:
            sub_p["media_num"] = 123.45
        if "variacao_num" in sub_p.columns:
            sub_p["variacao_num"] = -99.0
        if "preco_num" in sub_p.columns:
            sub_p["preco_num"] = 0.01

        df_p, _ = se.filtrar_universo_alvo(sub_p)
        eleg_p = set(df_p["atleta_id"].astype(int).tolist())

        if eleg_orig != eleg_p:
            diff = eleg_orig.symmetric_difference(eleg_p)
            falhas.append(
                f"{t}/R{R}: conjunto elegivel muda com perturbacao "
                f"({len(diff)} atletas divergentes)")
    return falhas


def teste_cobertura(df):
    linhas = []
    for (t, R), sub in df.groupby(["temporada", "rodada"]):
        total = len(sub)
        try:
            df_e, _ = se.filtrar_universo_alvo(sub)
        except Exception as e:
            linhas.append({
                "temporada": int(t), "rodada": int(R),
                "total": total, "elegiveis": None, "nao_elegiveis": None,
                "pct_elegivel": None, "erro": str(e),
            })
            continue
        n_e = len(df_e)
        linhas.append({
            "temporada": int(t), "rodada": int(R),
            "total": total,
            "elegiveis": n_e,
            "nao_elegiveis": total - n_e,
            "pct_elegivel": round(n_e / total * 100, 4) if total else None,
            "erro": "",
        })
    return pd.DataFrame(linhas).sort_values(["temporada", "rodada"])


def teste_distribuicao(df):
    df2 = df.copy()
    df2["sid_int"] = df2["status_id"].astype("int64")
    tab = (df2.groupby(["temporada", "sid_int"]).size()
              .unstack(fill_value=0))
    tab.columns = [f"status_{int(c)}" for c in tab.columns]
    tab["total"] = tab.sum(axis=1)
    return tab.reset_index()


def teste_folds_harness(df):
    if not HARNESS.exists():
        return pd.DataFrame(), ["harness_folds_v2.csv ausente"]
    man = pd.read_csv(HARNESS)
    linhas = []
    falhas = []
    for _, r in man.iterrows():
        t = int(r["temporada"]); R = int(r["rodada_alvo"])
        sub = df[(df["temporada"] == t) & (df["rodada"] == R)]
        if len(sub) == 0:
            falhas.append(f"{t}/R{R}: rodada-alvo vazia")
            continue
        try:
            df_e, _ = se.filtrar_universo_alvo(sub)
        except Exception as e:
            falhas.append(f"{t}/R{R}: erro no filtro: {e}")
            continue
        linhas.append({
            "temporada": t, "rodada_alvo": R,
            "n_universo": int(len(sub)),
            "n_elegiveis": int(len(df_e)),
            "n_nao_elegiveis": int(len(sub) - len(df_e)),
            "pct_elegivel": round(len(df_e) / len(sub) * 100, 4) if len(sub) else None,
        })
    return pd.DataFrame(linhas), falhas


# ------------------------------------------------------------- main
def main():
    inicio = datetime.now()
    print("== Auditoria de elegibilidade via status_id v1 ==")

    h_antes = sha256(CANON)
    print(f"Parquet canonico SHA-256: {h_antes[:16]}...")
    if h_antes != HASH_CANON:
        print("ERRO: hash diverge do esperado. Abortando.", file=sys.stderr)
        sys.exit(2)

    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)

    print()
    print("--- [1] Integridade ---")
    integ = teste_integridade(df)
    for k, v in integ.items():
        if k == "problemas":
            continue
        print(f"  {k}: {v}")
    if integ["problemas"]:
        print("  PROBLEMAS:")
        for p in integ["problemas"]:
            print(f"    - {p}")

    print()
    print("--- [2] Distribuicao por temporada ---")
    dist = teste_distribuicao(df)
    print(dist.to_string(index=False))
    dist.to_csv(OUT_DIST, index=False)

    print()
    print("--- [3] Cobertura por (temporada, rodada) ---")
    cob = teste_cobertura(df)
    cob.to_csv(OUT_COB, index=False)
    print(f"  rodadas cobertas: {len(cob)}")
    if len(cob):
        print(f"  cobertura elegivel media: {cob['pct_elegivel'].mean():.2f}%")
        print(f"  rodadas com elegivel vazio: {int((cob['elegiveis'] == 0).sum())}")
        print(f"  rodadas com erro: {int((cob['erro'] != '').sum())}")

    print()
    print("--- [4] Folds do harness ---")
    folds_df, folds_erros = teste_folds_harness(df)
    obs_por_temp = {}
    if len(folds_df):
        folds_df.to_csv(OUT_FOLDS, index=False)
        obs_por_temp = {int(k): int(v) for k, v in
                        folds_df.groupby("temporada").size().items()}
        print(f"  folds processados: {len(folds_df)}")
        print(f"  universos vazios: {int((folds_df['n_elegiveis'] == 0).sum())}")
        print(f"  cobertura elegivel media nos folds: "
              f"{folds_df['pct_elegivel'].mean():.2f}%")
        print(f"  folds por temporada (observado): {obs_por_temp}")
        print(f"  folds por temporada (esperado):  {ESPERADO_FOLDS_POR_TEMP}")
    if folds_erros:
        print(f"  erros: {len(folds_erros)}")
        for e in folds_erros[:10]:
            print(f"    - {e}")

    print()
    print("--- [5] Teste de invariancia do filtro as variaveis de resultado (76 folds) ---")
    if HARNESS.exists():
        man = pd.read_csv(HARNESS)
        folds = [(int(t), int(R)) for t, R in
                 zip(man["temporada"], man["rodada_alvo"])]
    else:
        folds = []
    inv_falhas = teste_invariancia_resultados(df, folds)
    if inv_falhas:
        print(f"  FALHAS: {len(inv_falhas)}")
        for f in inv_falhas[:10]:
            print(f"    - {f}")
    else:
        print(f"  OK: {len(folds)} folds — conjunto elegivel invariante a "
              f"perturbacao de pontos_num/entrou_em_campo/scouts")

    print()
    print("--- [6] Integridade final ---")
    h_depois = sha256(CANON)
    integridade_ok = (h_antes == h_depois == HASH_CANON)
    print(f"  Parquet SHA-256 inalterado: {integridade_ok}")

    # ----- criterios
    folds_76_ok = (
        len(folds_df) == 76
        and folds_df[["temporada", "rodada_alvo"]].drop_duplicates().shape[0] == 76
        and obs_por_temp == ESPERADO_FOLDS_POR_TEMP
    )
    nenhum_fold_sem_elegiveis = (
        len(folds_df) == 76
        and bool((folds_df["n_elegiveis"] > 0).all())
    )

    criterios = {
        "parquet_sha_ok": bool(integridade_ok),
        "sem_nulos_reais": integ["n_nulos_reais"] == 0,
        "sem_desconhecidos": len(integ["status_desconhecidos"]) == 0,
        "sem_duplicatas": integ["duplicatas_T_R_A"] == 0,
        "sem_multi_status": integ["multi_status_celulas"] == 0,
        "folds_76": bool(folds_76_ok),
        "nenhum_fold_sem_elegiveis": bool(nenhum_fold_sem_elegiveis),
        "todos_folds_invariantes": len(inv_falhas) == 0,
        "elegiveis_apenas_7_2": set(se.STATUS_ELEGIVEIS) == {7, 2},
        "nao_elegiveis_3_5_6": set(se.STATUS_NAO_ELEGIVEIS) == {3, 5, 6},
    }
    pass_geral = all(criterios.values())

    print()
    print("=== CRITERIOS ===")
    for k, v in criterios.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    print()
    print(f"=== RESULTADO: {'PASS' if pass_geral else 'FAIL'} ===")

    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "parquet_sha256": h_antes,
        "parquet_sha_esperado": HASH_CANON,
        "integridade_ok": bool(integridade_ok),
        "regra": se.regra_resumida(),
        "integridade": integ,
        "criterios": criterios,
        "resultado": "PASS" if pass_geral else "FAIL",
        "n_folds_avaliados": len(folds_df),
        "folds_por_temporada_observado": obs_por_temp,
        "folds_por_temporada_esperado": ESPERADO_FOLDS_POR_TEMP,
        "invariância_resultados_falhas": len(inv_falhas),
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    L = []
    L.append("# Elegibilidade pre-rodada via `status_id` — v1\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Parquet: `{CANON.relative_to(ROOT)}`\n")
    L.append(f"- SHA-256: `{h_antes}`\n")
    L.append(f"- Linhas: {len(df)}\n\n")

    L.append("## 0. Regra congelada\n\n")
    L.append("| status_id | nome | elegivel |\n|---|---|---|\n")
    for k, v in se.STATUS_MAP.items():
        L.append(f"| {k} | {v[0]} | {v[1]} |\n")
    L.append("\n")
    L.append("O filtro aplica-se SOMENTE ao universo da rodada-alvo. ")
    L.append("`status_id` nao e' feature de regressao. ")
    L.append("Nao se infere semantica a partir de resultados historicos.\n\n")

    L.append("## A. Cobertura por (temporada, rodada)\n\n")
    L.append(f"- Rodadas cobertas: {len(cob)}\n")
    if len(cob):
        L.append(f"- Cobertura elegivel media: {cob['pct_elegivel'].mean():.2f}%\n")
        L.append(f"- Rodadas com elegivel vazio: {int((cob['elegiveis']==0).sum())}\n")
    L.append("\nArquivo: `reports/status_elegibilidade_v1_cobertura.csv`\n\n")

    L.append("## B. Distribuicao por temporada\n\n")
    L.append(dist.to_string(index=False))
    L.append("\n\nArquivo: `reports/status_elegibilidade_v1_distribuicao.csv`\n\n")

    L.append("## C. Integridade\n\n")
    L.append(f"- Linhas totais: {integ['n_linhas']}\n")
    L.append(f"- Nulos reais em `status_id`: {integ['n_nulos_reais']}\n")
    L.append(f"- Codigos presentes: {integ['status_presentes']}\n")
    L.append(f"- Codigos desconhecidos: {integ['status_desconhecidos']}\n")
    L.append(f"- Duplicatas (T,R,A): {integ['duplicatas_T_R_A']}\n")
    L.append(f"- Celulas (T,R,A) com mais de um status: {integ['multi_status_celulas']}\n\n")

    L.append("## D. Teste de invariancia do filtro as variaveis de resultado\n\n")
    L.append("Para cada um dos 76 folds, o conjunto elegivel foi recalculado em uma ")
    L.append("copia da rodada-alvo com `pontos_num`, `entrou_em_campo`, `scout_*`, ")
    L.append("`media_num`, `variacao_num` e `preco_num` perturbados. ")
    L.append("Isso prova que o filtro depende exclusivamente de `status_id`. ")
    L.append("NAO substitui a auditoria de proveniencia temporal do snapshot, ")
    L.append("que foi realizada separadamente (`reports/status_id_auditoria.md`).\n\n")
    if len(inv_falhas) == 0:
        L.append(f"- **PASS**: conjunto elegivel invariante em {len(folds)} folds.\n\n")
    else:
        L.append(f"- **FAIL**: {len(inv_falhas)} folds com sensibilidade indevida.\n\n")

    L.append("## E. Integracao com o harness\n\n")
    L.append(f"- Folds processados: {len(folds_df)}\n")
    L.append(f"- Folds por temporada (observado): {obs_por_temp}\n")
    L.append(f"- Folds por temporada (esperado):  {ESPERADO_FOLDS_POR_TEMP}\n")
    if len(folds_df):
        L.append(f"- Cobertura elegivel media nos folds: "
                 f"{folds_df['pct_elegivel'].mean():.2f}%\n")
    L.append(f"\nArquivo por fold: `reports/status_elegibilidade_v1_folds.csv`\n\n")

    L.append("## F. Criterios de PASS\n\n")
    for k, v in criterios.items():
        L.append(f"- {k}: {'OK' if v else 'FAIL'}\n")
    L.append(f"\n**Resultado: {('PASS' if pass_geral else 'FAIL')}**\n\n")

    L.append("## G. Integridade do Parquet canonico\n\n")
    L.append(f"- SHA-256 (antes): `{h_antes}`\n")
    L.append(f"- SHA-256 (depois): `{h_depois}`\n")
    L.append(f"- Inalterado: **{integridade_ok}**\n\n")

    L.append("---\n\n")
    L.append("## PENDENCIAS / RISCOS\n\n")
    L.append("- **Semantica dos codigos**: a atribuicao `7=Provavel`, `2=Duvida`, ")
    L.append("  `5=Contundido`, `3=Suspenso`, `6=Nulo` deve ser documentada com a ")
    L.append("  fonte (documentacao oficial ou decisao humana rastreavel). Este ")
    L.append("  relatorio apenas aplica a regra, nao a valida contra a fonte.\n")
    L.append("- **Consequencia do filtro no MAE**: NAO medida nesta etapa por ")
    L.append("  decisao metodologica. O efeito do filtro na performance do modelo ")
    L.append("  sera objeto de etapa posterior.\n")
    L.append("- **Universo elegivel e resultado historico**: o filtro pode excluir ")
    L.append("  jogadores que efetivamente entraram em campo. Isso e' esperado e ")
    L.append("  nao indica erro nos dados; reflete o snapshot de mercado.\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    print()
    print("Artefatos:")
    for p in [OUT_COB, OUT_FOLDS, OUT_DIST, OUT_VALID, OUT_MD]:
        if p.exists():
            print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")

    if not pass_geral:
        sys.exit(1)


if __name__ == "__main__":
    main()
