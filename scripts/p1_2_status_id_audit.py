#!/usr/bin/env python3
"""
P1.2 — Auditoria temporal do status_id.

Classificacao padrao: INDETERMINADO.
Correlacao com entrou_em_campo NAO prova temporalidade.
status_id NAO e' usado como feature.
"""
import hashlib, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("raiz nao encontrada")


ROOT = detect_root()
CANON = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
OUT_MD = ROOT/"reports"/"p1_2_status_id_audit.md"
OUT_CSV = ROOT/"reports"/"p1_2_status_id_por_status.csv"
OUT_TRANS = ROOT/"reports"/"p1_2_status_id_transicoes.csv"

HASH_ESPERADO = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def grep_docs_status():
    hits = []
    alvos = [ROOT/"README.md", ROOT/"ETAPAS.md", ROOT/"config.yaml",
             ROOT/"config", ROOT/"docs", ROOT/"reports", ROOT/"cartola_src"]
    for sub in alvos:
        if not sub.exists(): continue
        files = [sub] if sub.is_file() else list(sub.rglob("*"))
        for f in files:
            if not f.is_file(): continue
            if f.suffix.lower() not in {".md",".txt",".yaml",".yml",".py",".json",".csv"}: continue
            if f.stat().st_size > 5_000_000: continue
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for m in re.finditer(r"\bstatus[_\s-]?id\b", txt, re.IGNORECASE):
                linha = txt[:m.start()].count("\n") + 1
                trecho = txt.splitlines()[max(0, linha-2)][:120] if linha>0 else ""
                hits.append(f"{f.relative_to(ROOT)}:{linha}  |  {trecho}")
    return hits[:40]


def main():
    inicio = datetime.now()
    h_before = sha256(CANON)
    if h_before != HASH_ESPERADO:
        print("ERRO: hash do parquet nao bate", file=sys.stderr); sys.exit(2)

    df = pd.read_parquet(CANON)
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)

    L = []
    L.append("# P1.2 — Auditoria temporal do status_id\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Classificacao padrao: **INDETERMINADO**\n\n")

    L.append("## AVISO METODOLOGICO\n\n")
    L.append("- Correlacao entre status_id e entrou_em_campo NAO prova temporalidade.\n")
    L.append("- Persistencia entre rodadas NAO prova temporalidade.\n")
    L.append("- Sem documentacao oficial ou timestamp de coleta, classificacao = INDETERMINADO.\n\n")

    # 1
    L.append("## 1. Distribuicao\n\n")
    dist = df["status_id"].value_counts(dropna=False).sort_index()
    L.append("| status_id | n | % |\n|---|---|---|\n")
    for k, v in dist.items():
        L.append(f"| {k} | {v} | {100*v/len(df):.2f}% |\n")
    L.append("\n")

    # 2
    L.append("## 2. P(entrou | status_id) — DIAGNOSTICO, NAO PROVA\n\n")
    ct = df.groupby("status_id").agg(
        n=("entrou_bool","size"), n_part=("entrou_bool","sum"))
    ct["p_entrou"] = ct["n_part"] / ct["n"]
    ct = ct.reset_index()
    ct.to_csv(OUT_CSV, index=False)
    L.append("| status_id | n | n_part | p_entrou |\n|---|---|---|---|\n")
    for _, r in ct.iterrows():
        L.append(f"| {int(r['status_id'])} | {int(r['n'])} | {int(r['n_part'])} | "
                 f"{r['p_entrou']:.4f} |\n")
    L.append("\n**AVISO:** associacao contemporanea. Nao informa temporalidade.\n\n")

    # 3
    L.append("## 3. Transicoes entre rodadas consecutivas\n\n")
    df_s = df.sort_values(["temporada","atleta_id","rodada"]).copy()
    df_s["status_prev"] = df_s.groupby(["temporada","atleta_id"])["status_id"].shift(1)
    df_s["rodada_prev"] = df_s.groupby(["temporada","atleta_id"])["rodada"].shift(1)
    trans = df_s.dropna(subset=["status_prev","status_id"]).copy()
    trans = trans[(trans["rodada"] - trans["rodada_prev"]) == 1]
    tt = trans.groupby(["status_prev","status_id"]).size().reset_index(name="n").sort_values("n", ascending=False)
    tt.to_csv(OUT_TRANS, index=False)
    n_iguais = int((trans["status_prev"] == trans["status_id"]).sum())
    L.append(f"- transicoes: {len(trans)}\n")
    L.append(f"- permanencia: {n_iguais}/{len(trans)} ({100*n_iguais/len(trans):.1f}%)\n")
    L.append("- **AVISO:** permanencia alta nao distingue pre-rodada de pos-rodada estavel.\n\n")

    # 4 docs
    L.append("## 4. Documentacao interna\n\n")
    hits = grep_docs_status()
    if hits:
        L.append(f"- ocorrencias: {len(hits)}\n\n")
        for h in hits[:20]:
            L.append(f"  - {h}\n")
    else:
        L.append("- nenhuma.\n")
    L.append("\n")

    # 5 CAMINHOS PARA PASS
    L.append("## 5. Caminhos explicitos para PASS\n\n")
    L.append("Para transformar INDETERMINADO em PASS, e' necessario UM dos dois:\n\n")
    L.append("**(A) Documentacao oficial** que estabeleca que `status_id` representa "
             "um estado do jogador disponivel ANTES da rodada (ex.: pagina GE/Cartola, "
             "glossario da API oficial).\n\n")
    L.append("**(B) Evidencia temporal de coleta** que demonstre, com timestamp, que "
             "`status_id` era registrado antes do inicio da rodada (ex.: snapshot de "
             "coleta com hora anterior ao cutoff).\n\n")
    L.append("Sem A ou B, a classificacao permanece INDETERMINADO.\n\n")

    # 6 classificacao
    L.append("## 6. Classificacao final\n\n")
    L.append("**FATO:** status_id correlaciona com entrou_em_campo; permanencia alta.\n")
    L.append("**INFERENCIA:** padrao compativel com status de mercado pre-rodada.\n")
    L.append("**HIPOTESE:** status_id pode codificar estado pre-rodada. Nao testada.\n")
    L.append("**RECOMENDACAO:** nao usar como feature. Reabrir se A ou B for obtido.\n\n")
    L.append("## Status final: **INDETERMINADO**\n")

    h_after = sha256(CANON)
    L.append(f"\n## Integridade\n\n- Parquet inalterado: {h_before == h_after}\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")
    print("classificacao: INDETERMINADO")
    print("artefatos: reports/p1_2_status_id_audit.md, _por_status.csv, _transicoes.csv")


if __name__ == "__main__":
    main()
