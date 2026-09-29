#!/usr/bin/env python3
"""Auditoria final de consistência dos artefatos da Etapa 5."""
import hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
REP = ROOT/"reports"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
MANIFEST = REP/"harness_folds_v2.csv"

BASE = {
    "b0":  REP/"baseline_b0_folds.csv",
    "b1":  REP/"baseline_b1_folds.csv",
    "b2":  REP/"baseline_b2_folds.csv",
    "b3":  REP/"baseline_b3_folds.csv",
    "b2s": REP/"baseline_b2s_folds.csv",
}
SPOTS = {
    "b2_spot": REP/"baseline_b2_spotcheck.csv",
    "b3_spot": REP/"baseline_b3_spotcheck.csv",
}
SUMS = {
    "b0":  REP/"baseline_b0_summary.md",
    "b1":  REP/"baseline_b1_summary.md",
    "b2":  REP/"baseline_b2_summary.md",
    "b3":  REP/"baseline_b3_summary.md",
    "b2s": REP/"baseline_b2s_summary.md",
}
SPECS = {
    "b2s_spec":   ROOT/"config"/"b2s_spec.yaml",
    "b2s_specmd": REP/"baseline_b2s_spec.md",
    "scouts":     ROOT/"config"/"pontuacao_scouts_auditoria.yaml",
    "3c_md":      REP/"auditoria_representacao_scouts_3c.md",
    "3c_csv":     REP/"auditoria_representacao_scouts_3c.csv",
}
AUDITS = {
    "h1":       REP/"auditoria_h1_summary.md",
    "h2a":      REP/"auditoria_h2a_summary.md",
    "h2b":      REP/"auditoria_h2b_summary.md",
    "b2b3":     REP/"auditoria_b2_b3_summary.md",
    "b2b3_erros": REP/"auditoria_b2_b3_erros.csv",
}

OUT_CSV = REP/"auditoria_etapa5_consistencia.csv"
OUT_MD  = REP/"auditoria_etapa5_consistencia.md"


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    inicio = datetime.now()
    faltas = []
    rows = []

    # existencia
    for k, p in {**BASE, **SPOTS, **SUMS, **SPECS, **AUDITS}.items():
        ex = p.exists()
        rows.append({"artefato": k, "arquivo": str(p.relative_to(ROOT)),
                     "existe": ex, "bytes": p.stat().st_size if ex else 0})
        if not ex:
            faltas.append(f"{k}: {p.name}")

    # folds: mesmas chaves em todos
    man = pd.read_csv(MANIFEST)
    chaves = set(map(tuple, man[["temporada","rodada_alvo"]].values.tolist()))
    for k, p in BASE.items():
        d = pd.read_csv(p)
        kk = set(map(tuple, d[["temporada","rodada_alvo"]].values.tolist()))
        if kk != chaves:
            faltas.append(f"{k}: chaves divergem do manifesto")

    # metricas oficiais recalculadas
    def mae_pond(p):
        d = pd.read_csv(p)
        return float((d["mae"]*d["n_test_metric"]).sum() / d["n_test_metric"].sum())

    def sp_medio(p):
        d = pd.read_csv(p)
        return float(d["spearman"].mean())

    b0 = mae_pond(BASE["b0"]); b1 = mae_pond(BASE["b1"])
    b2 = mae_pond(BASE["b2"]); b3 = mae_pond(BASE["b3"])
    b2s_d = pd.read_csv(BASE["b2s"])
    b2s = float((b2s_d["b2s_mae"]*b2s_d["n_test"]).sum()/b2s_d["n_test"].sum())
    b2s_sp = float(b2s_d["b2s_spearman"].mean())

    # Parquet
    pq = pd.read_parquet(PARQUET)
    h_pq = sha256(PARQUET)
    pq_ok = (len(pq) == 77253 and pq.shape[1] == 38)

    # H2A α coerente com B2S α
    h2a = pd.read_csv(REP/"auditoria_h2a_folds.csv")
    h2a_alpha = float(h2a["alpha_star"].mean())
    b2s_alpha = float(b2s_d["alpha_star"].mean())

    fim = datetime.now()
    L = []
    L.append("# Auditoria final da Etapa 5\n\n")
    L.append(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fim: {fim.isoformat(timespec='seconds')}\n\n")

    L.append("## Arquivos\n\n")
    L.append("| artefato | existe | bytes |\n|---|---|---|\n")
    for r in rows:
        L.append(f"| {r['artefato']} | {'SIM' if r['existe'] else 'NAO'} | {r['bytes']} |\n")
    L.append("\n")

    L.append("## Métricas oficiais (recalculadas dos CSVs)\n\n")
    L.append(f"- B0  MAE = {b0:.4f}\n")
    L.append(f"- B1  MAE = {b1:.4f}\n")
    L.append(f"- B2  MAE = {b2:.4f}\n")
    L.append(f"- B3  MAE = {b3:.4f}  (diagnostico)\n")
    L.append(f"- B2S MAE = {b2s:.4f}  Spearman = {b2s_sp:.4f}\n\n")

    L.append("## Coerências\n\n")
    L.append(f"- 76 folds em todos os baselines: {'OK' if not faltas else 'FALHAS'}\n")
    L.append(f"- α médio H2A = {h2a_alpha:.3f}; α médio B2S = {b2s_alpha:.3f} "
             f"(diferença = {abs(h2a_alpha - b2s_alpha):.4f})\n")
    L.append(f"- Parquet: linhas={len(pq)} colunas={pq.shape[1]}  OK={pq_ok}\n")
    L.append(f"- Parquet SHA: {h_pq}\n\n")

    L.append("## Falhas\n\n")
    if faltas:
        for f in faltas: L.append(f"- {f}\n")
    else:
        L.append("- nenhuma\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"artefatos: {len(rows)}")
    print(f"faltas: {len(faltas)}")
    for f in faltas: print(f"  - {f}")
    print(f"B0={b0:.4f}  B1={b1:.4f}  B2={b2:.4f}  B3={b3:.4f}  B2S={b2s:.4f}")
    print(f"B2S Spearman={b2s_sp:.4f}")
    print(f"Parquet OK: {pq_ok}")
    print(f"sha256: {h_pq}")


if __name__ == "__main__":
    main()
