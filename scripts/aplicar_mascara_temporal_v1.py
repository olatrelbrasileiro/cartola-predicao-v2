#!/usr/bin/env python3
"""Aplica mascara temporal ao Confronto v1 (Caminho 1 — assuncao operacional).

Classifica cada partida em:
- COMPATIVEL_JANELA_FD (manter, sob assuncao)
- FD_INDETERMINADO    (mascarar)
- RISCO_CUTOFF        (mascarar)
- ADIADA              (mascarar)

Gera:
- data/interim/features_confronto_v1_temporal_safe.parquet
- reports/auditoria_temporal_matches.csv  (1027 partidas)
- reports/temporal_safe_summary.md
NAO altera features_confronto_v1.parquet.
"""
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home()/"cartola-predicao-v2"
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"
CSV_CUTOFF = REP/"auditoria_cutoff_confronto_v1.csv"
PARQUET_CANON = INTERIM/"cartola_jogador_rodada_v2.parquet"
FEATS_IN = INTERIM/"features_confronto_v1.parquet"

ESPERADO = {"COMPATIVEL_JANELA_FD": 834, "FD_INDETERMINADO": 91,
            "RISCO_CUTOFF": 55, "ADIADA": 47}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def fd_ceiling_utc(match_date):
    if pd.isna(match_date):
        return pd.NaT
    wd = match_date.weekday()
    d = match_date.normalize()
    if wd in (5, 6):
        return (d - timedelta(days=wd - 4)).replace(hour=16, minute=0)
    if wd in (1, 2, 3):
        return (d - timedelta(days=wd - 1)).replace(hour=12, minute=0)
    return pd.NaT


def cutoff_brt_para_utc(ts_str):
    if pd.isna(ts_str) or ts_str == "":
        return pd.NaT
    try:
        return pd.to_datetime(ts_str) + timedelta(hours=3)
    except Exception:
        return pd.NaT


def classifica(delta_dias, cutoff_utc, fd_ceiling_utc):
    if pd.notna(delta_dias) and delta_dias > 7:
        return "ADIADA"
    if pd.isna(cutoff_utc):
        return "SEM_CUTOFF"
    if pd.isna(fd_ceiling_utc):
        return "FD_INDETERMINADO"
    if fd_ceiling_utc >= cutoff_utc:
        return "RISCO_CUTOFF"
    return "COMPATIVEL_JANELA_FD"


CARTOLA_TO_CANON = {
    "FLA":"Flamengo","BOT":"Botafogo","COR":"Corinthians","BAH":"Bahia",
    "FLU":"Fluminense","VAS":"Vasco","PAL":"Palmeiras","SAO":"São Paulo",
    "SAN":"Santos","RBB":"Bragantino","CAM":"Atlético-MG","CRU":"Cruzeiro",
    "GRE":"Grêmio","INT":"Internacional","JUV":"Juventude","VIT":"Vitória",
    "CEA":"Ceará","FOR":"Fortaleza","REM":"Remo","MIR":"Mirassol",
    "SPT":"Sport Recife","CAP":"Athletico-PR","CFC":"Coritiba",
    "CHA":"Chapecoense","CRI":"Criciúma",
    "Flamengo":"Flamengo","Botafogo":"Botafogo","Corinthians":"Corinthians",
    "Bahia":"Bahia","Fluminense":"Fluminense","Vasco":"Vasco",
    "Palmeiras":"Palmeiras","São Paulo":"São Paulo","Bragantino":"Bragantino",
    "Atlético-MG":"Atlético-MG","Cruzeiro":"Cruzeiro","Grêmio":"Grêmio",
    "Internacional":"Internacional","Juventude":"Juventude","Vitória":"Vitória",
    "Criciúma":"Criciúma","Athlético-PR":"Athletico-PR","Atlético-GO":"Atlético-GO",
    "Cuiabá":"Cuiabá","Fortaleza":"Fortaleza","Ceará":"Ceará",
    "Sport Recife":"Sport Recife","Coritiba":"Coritiba",
    "Chapecoense":"Chapecoense","Remo":"Remo","Mirassol":"Mirassol",
}


def main():
    print("== Mascara temporal do Confronto v1 ==")

    # hashes ANTES
    hash_canon_antes = sha256(PARQUET_CANON)
    hash_feats_antes = sha256(FEATS_IN)
    n_linhas_canon_antes = len(pd.read_parquet(PARQUET_CANON))
    print(f"Parquet canonico SHA-256: {hash_canon_antes[:16]}...")
    print(f"Features originais SHA-256: {hash_feats_antes[:16]}...")

    # matches
    matches = pd.read_parquet(INTERIM/"matches_odds_cartola_v1.parquet")
    matches["data"] = pd.to_datetime(matches["data"], errors="coerce")

    dmin = matches.groupby(["temporada","rodada"])["data"].min().reset_index()
    dmin = dmin.rename(columns={"data":"data_min_rodada"})
    matches = matches.merge(dmin, on=["temporada","rodada"], how="left")
    matches["delta_dias"] = (matches["data"] - matches["data_min_rodada"]).dt.days

    cutoff = pd.read_csv(CSV_CUTOFF)
    cutoff["cutoff_utc"] = cutoff["cartola_cutoff"].apply(cutoff_brt_para_utc)
    matches = matches.merge(
        cutoff[["temporada","rodada","cutoff_utc"]],
        on=["temporada","rodada"], how="left")

    matches["fd_ceiling_utc"] = matches["data"].apply(fd_ceiling_utc)
    matches["classificacao"] = matches.apply(
        lambda r: classifica(r["delta_dias"], r["cutoff_utc"], r["fd_ceiling_utc"]),
        axis=1)

    # validacao de contagens
    contagem = matches["classificacao"].value_counts().to_dict()
    print()
    print("== contagens por classificacao ==")
    for k in ["COMPATIVEL_JANELA_FD","FD_INDETERMINADO","RISCO_CUTOFF","ADIADA"]:
        esperado = ESPERADO[k]
        obtido = contagem.get(k, 0)
        flag = "OK" if obtido == esperado else "DIVERGE"
        print(f"  {k}: {obtido} (esperado {esperado}) [{flag}]")

    if contagem.get("SEM_CUTOFF", 0) > 0:
        print(f"  AVISO: {contagem['SEM_CUTOFF']} partidas sem cutoff (nao esperado)")

    # match_long
    rows = []
    for _, m in matches.iterrows():
        base = {
            "temporada": int(m["temporada"]),
            "rodada": int(m["rodada"]),
            "bzzoiro_event_id": int(m["bzzoiro_event_id"]),
            "motivo_mascara": m["classificacao"] if m["classificacao"] != "COMPATIVEL_JANELA_FD" else "",
        }
        rows.append({**base, "clube_canon": m["home_team_cartola"],
                     "adversario_canon": m["away_team_cartola"]})
        rows.append({**base, "clube_canon": m["away_team_cartola"],
                     "adversario_canon": m["home_team_cartola"]})
    match_long = pd.DataFrame(rows)

    # vinculo atleta
    cart = pd.read_parquet(PARQUET_CANON)
    cart_aux = cart[["temporada","rodada","atleta_id","clube_nome"]].copy()
    cart_aux["clube_canon"] = cart_aux["clube_nome"].map(CARTOLA_TO_CANON)

    vinculo = cart_aux.merge(match_long,
                              on=["temporada","rodada","clube_canon"],
                              how="left")
    vinculo["motivo_mascara"] = vinculo["motivo_mascara"].fillna(
        "SEM_MATCH")

    # features + merge (sem duplicar colunas existentes)
    feats = pd.read_parquet(FEATS_IN)
    feats2 = feats.merge(
        vinculo[["temporada","rodada","atleta_id",
                 "motivo_mascara","adversario_canon","bzzoiro_event_id"]],
        on=["temporada","rodada","atleta_id"], how="left")

    mask = feats2["motivo_mascara"].fillna("") != ""
    n_masc = int(mask.sum())

    # check: so as 4 colunas mascaradas
    cols_mascaradas = ["p_home","p_draw","p_away","favoritismo_home_away"]
    for c in cols_mascaradas:
        feats2.loc[mask, c] = np.nan

    # validacoes
    print()
    print("== validacoes pos-mascara ==")
    print(f"linhas totais: {len(feats2)}")
    print(f"linhas mascaradas: {n_masc}")
    print(f"mando NaN: {int(feats2['mando'].isna().sum())}  "
          f"(esperado 230 = jogadores sem match em 2026/R21)")
    for c in cols_mascaradas:
        print(f"  {c} NaN: {int(feats2[c].isna().sum())}")
    print(f"motivo_mascara distintos:")
    print(feats2['motivo_mascara'].value_counts(dropna=False).to_string())

    # check: mando nao foi alterado
    soma_mando = feats2["mando"].sum()
    soma_mando_orig = feats["mando"].sum()
    print(f"\nmando sum antes/depois: {soma_mando_orig} / {soma_mando} "
          f"[{'OK' if soma_mando == soma_mando_orig else 'DIVERGE'}]")

    # salvar features masked
    saida = INTERIM/"features_confronto_v1_temporal_safe.parquet"
    feats2.to_parquet(saida, index=False)
    print(f"\nsalvo: {saida.relative_to(ROOT)}")

    # salvar auditoria
    matches_out = matches[[
        "temporada","rodada","data","home_team_cartola","away_team_cartola",
        "cutoff_utc","fd_ceiling_utc","delta_dias","classificacao"
    ]].copy()
    matches_out["margem_horas"] = (
        (matches_out["cutoff_utc"] - matches_out["fd_ceiling_utc"])
        .dt.total_seconds() / 3600)
    matches_out.to_csv(REP/"auditoria_temporal_matches.csv", index=False)

    # hash dos originais depois
    hash_canon_depois = sha256(PARQUET_CANON)
    hash_feats_depois = sha256(FEATS_IN)
    n_linhas_canon_depois = len(pd.read_parquet(PARQUET_CANON))

    canon_ok = (hash_canon_antes == hash_canon_depois
                and n_linhas_canon_antes == n_linhas_canon_depois)
    feats_ok = (hash_feats_antes == hash_feats_depois)

    print()
    print("== integridade ==")
    print(f"Parquet canonico inalterado: {canon_ok}")
    print(f"features_confronto_v1.parquet inalterado: {feats_ok}")

    # summary
    L = []
    L.append("# Confronto v1 — mascara temporal (Caminho 1)\n\n")
    L.append(f"- Gerado em: {datetime.now().isoformat(timespec='seconds')}\n\n")
    L.append("## Politica\n\n")
    L.append("COMPATIVEL_JANELA_FD mantido sob assuncao operacional.\n")
    L.append("FD_INDETERMINADO, RISCO_CUTOFF, ADIADA: mascarados.\n\n")
    L.append("## Contagens\n\n")
    for k in ["COMPATIVEL_JANELA_FD","FD_INDETERMINADO","RISCO_CUTOFF","ADIADA"]:
        L.append(f"- {k}: {contagem.get(k,0)}\n")
    L.append(f"\n- Total: {len(matches)}\n")
    L.append(f"- Cobertura mantida: {100*contagem.get('COMPATIVEL_JANELA_FD',0)/len(matches):.2f}%\n\n")
    L.append("## Impacto atleta-rodada\n\n")
    L.append(f"- Linhas totais: {len(feats2)}\n")
    L.append(f"- Linhas mascaradas: {n_masc}\n")
    L.append(f"- Linhas mantidas: {len(feats2) - n_masc}\n\n")
    L.append("## Integridade\n\n")
    L.append(f"- Parquet canonico inalterado: {canon_ok}\n")
    L.append(f"- features_confronto_v1.parquet inalterado: {feats_ok}\n")
    L.append(f"- mando preservado: {soma_mando == soma_mando_orig}\n")

    (REP/"temporal_safe_summary.md").write_text("".join(L), encoding="utf-8")
    print(f"\nsalvo: reports/temporal_safe_summary.md")
    print(f"salvo: reports/auditoria_temporal_matches.csv")


if __name__ == "__main__":
    main()
