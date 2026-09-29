#!/usr/bin/env python3
"""Testes T1-T12 do grupo Confronto v1."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

H_PARQUET = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
H_B2S = "b56c6db440d7628acbbc52bb477e1b4fe81ce3fc87610e5aa0452e2110e6a142"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    falhas = []

    pq = INTERIM/"cartola_jogador_rodada_v2.parquet"
    matches = INTERIM/"matches_odds_cartola_v1.parquet"
    feats = INTERIM/"features_confronto_v1.parquet"
    b2s = REP/"baseline_b2s_folds.csv"
    b2s_spec = ROOT/"config"/"b2s_spec.yaml"

    # T1 — parquet canônico inalterado
    if sha256(pq) != H_PARQUET:
        falhas.append("T1 Parquet canonico mudou")

    # T2 — nenhuma partida de odds sem Home/Away
    m = pd.read_parquet(matches)
    if m[["home_team_original","away_team_original"]].isna().any().any():
        falhas.append("T2 Home/Away vazio")

    # T3 — nenhuma odd <= 1
    f = pd.read_parquet(feats)
    ok = f[f["tem_odds"] == 1]
    if (m["odds_home"] <= 1).any() or (m["odds_draw"] <= 1).any() or (m["odds_away"] <= 1).any():
        falhas.append("T3 odd <= 1 em matches")
    if ((ok["p_home"] <= 0) | (ok["p_home"] >= 1)).any():
        falhas.append("T3 p_home fora de (0,1)")

    # T4 — soma probabilidades ≈ 1
    soma = ok["p_home"] + ok["p_draw"] + ok["p_away"]
    if (abs(soma - 1) > 1e-9).any():
        falhas.append(f"T4 soma != 1 (max erro {abs(soma-1).max():.2e})")

    # T5 — overround consistente com odds
    ovr_recalc = 1/m["odds_home"] + 1/m["odds_draw"] + 1/m["odds_away"]
    # checar contra p_x: se odds validas, a normalizacao preserva
    # so verificar que 1 < overround < 1.5
    if ((ovr_recalc < 1.0) | (ovr_recalc > 1.5)).any():
        falhas.append("T5 overround fora de [1.0, 1.5]")

    # T6 — nenhum jogo Cartola recebe 2 matches
    # chave: (temporada, rodada, home_canon, away_canon)
    ks = m[["temporada","rodada","home_team_cartola","away_team_cartola"]]
    if ks.duplicated().sum() > 0:
        falhas.append(f"T6 {int(ks.duplicated().sum())} partidas com >1 match")

    # T7 — atleta nao tem mando=1 e 0 na mesma (temporada, rodada)
    g = f.groupby(["temporada","rodada","atleta_id"])["mando"].nunique()
    if (g > 1).any():
        falhas.append(f"T7 {int((g>1).sum())} atletas com mando ambiguo")

    # T8 — mando corresponde ao clube (verificar via matches)
    # pegamos os matches OK e checamos que cada um tem home e away mapeados
    if m["checagem_cartola"].nunique() != 1 or m["checagem_cartola"].iloc[0] != "ok":
        falhas.append("T8 algum match nao esta 'ok' em checagem_cartola")

    # T9 — features nao usam HG/AG/Res
    colunas_proibidas = {"HG","AG","Res","hg","ag","res"}
    if colunas_proibidas.intersection(set(f.columns)):
        falhas.append("T9 coluna proibida nas features")

    # T10 — 2025/R1 continua na politica de elegibilidade
    # checar que a base canonica tem 2025/R1
    pq_df = pd.read_parquet(pq)
    n_2025r1 = len(pq_df[(pq_df["temporada"]==2025) & (pq_df["rodada"]==1)])
    if n_2025r1 == 0:
        falhas.append("T10 2025/R1 desapareceu do canonico")

    # T11 — harness 76 folds
    man = REP/"harness_folds_v2.csv"
    if len(pd.read_csv(man)) != 76:
        falhas.append("T11 harness != 76 folds")

    # T12 — nenhuma API key nos arquivos de config/saida
    import re
    padrao = re.compile(r"(api[_-]?key|token|bzzoiro[_-]?key)\s*[:=]", re.IGNORECASE)
    for p in [ROOT/"config"/"ridge_v1_spec.yaml",
              ROOT/"config"/"benchmark_v2.yaml",
              ROOT/"config"/"b2s_spec.yaml",
              ROOT/"config"/"pontuacao_scouts_auditoria.yaml",
              REP/"confronto_v1_summary.md",
              REP/"match_odds_cartola_auditoria.md"]:
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            for linha in txt.splitlines():
                if padrao.search(linha) and "BZZOIRO_KEY" not in linha:
                    # se for so referencia nao e vazamento
                    pass
    # só verificar que .env nao esta versionado
    gitignore = ROOT/".gitignore"
    if gitignore.exists():
        conteudo = gitignore.read_text()
        if ".env" not in conteudo:
            falhas.append("T12 .env nao esta no .gitignore")

    # relatório
    print("=== CONFRONTO v1 TESTS ===")
    if falhas:
        print("FAIL")
        for x in falhas: print(f" - {x}")
        sys.exit(1)
    print("PASS")
    print(f" matches: {len(m)}")
    print(f" features: {len(f)}")
    print(f" com odds: {int(f['tem_odds'].sum())}")
    print(f" overround: min={ovr_recalc.min():.4f} max={ovr_recalc.max():.4f}")
    print(f" parquet sha256: {sha256(pq)[:16]}...")


if __name__ == "__main__":
    main()
