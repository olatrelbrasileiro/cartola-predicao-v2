#!/usr/bin/env python3
"""Auditoria pre-treino do CatBoost v1.

NAO treina modelo. Apenas verifica:
- hashes de entrada
- harness (76 folds)
- temporal_safe (contagens esperadas)
- feature spec (sources e validade temporal)
- politica de elegibilidade
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml

def _detect_root():
    """Detecta raiz do projeto: tenta Termux, Codespaces, e relativo ao script."""
    candidatos = [
        Path.home()/"cartola-predicao-v2",
        Path("/workspaces/cartola-predicao-v2"),
        Path(__file__).resolve().parent.parent,
    ]
    for c in candidatos:
        if (c/"reports"/"harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz do projeto nao encontrada")

ROOT = _detect_root()
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

CANON = INTERIM/"cartola_jogador_rodada_v2.parquet"
FEATS = INTERIM/"features_confronto_v1.parquet"
TEMPORAL_SAFE = INTERIM/"features_confronto_v1_temporal_safe.parquet"
B2S_FOLDS = REP/"baseline_b2s_folds.csv"
HARNESS = REP/"harness_folds_v2.csv"
AUDIT_YAML = ROOT/"config"/"pontuacao_scouts_auditoria.yaml"
TEMPORAL_YAML = ROOT/"config"/"temporalidade_confronto_v1.yaml"
OUT_MD = REP/"auditoria_pre_catboost_v1.md"

SHAS = {
    "parquet": ("cartola_jogador_rodada_v2.parquet",
                "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"),
    "feats_orig": ("features_confronto_v1.parquet",
                   "6e4f2e6b10a060f73beb0625b5dc88e99e77aebeb4bf52e9705ae011ead60de1"),
    "temporal_safe": ("features_confronto_v1_temporal_safe.parquet",
                      "0fcc9fe7e5ec16dbdf4df2477aa5e45dc620f19708b5baf7f36b863947f5a1d2"),
    "b2s_folds": ("baseline_b2s_folds.csv",
                  "b56c6db440d7628acbbc52bb477e1b4fe81ce3fc87610e5aa0452e2110e6a142"),
    "harness": ("harness_folds_v2.csv",
                "236b7421ed83860d6921131d708b0508601a8be495ac35f175f91eaefca803a5"),
}

ESPERADO_TEMPORAL = {"COMPATIVEL_JANELA_FD": 834, "FD_INDETERMINADO": 91,
                     "RISCO_CUTOFF": 55, "ADIADA": 47}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main():
    inicio = datetime.now()
    problemas = []
    L = []
    L.append("# Auditoria pre-treino — CatBoost v1\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n\n")

    # ---- 1. hashes
    L.append("## Hashes de entrada\n\n")
    L.append("| chave | arquivo | SHA-256 (16 primeiros) | status |\n")
    L.append("|---|---|---|---|\n")
    for k, (nome, esp) in SHAS.items():
        p = REP/nome if nome.endswith(".csv") else INTERIM/nome
        if not p.exists():
            L.append(f"| {k} | {nome} | — | FALTA |\n")
            problemas.append(f"{nome} nao existe")
            continue
        h = sha256(p)
        status = "OK" if h == esp else "DIVERGE"
        if status != "OK":
            problemas.append(f"{nome}: SHA diverge (esperado {esp[:16]}, obtido {h[:16]})")
        L.append(f"| {k} | {nome} | {h[:16]}... | {status} |\n")
    L.append("\n")

    # ---- 2. harness
    L.append("## Harness\n\n")
    man = pd.read_csv(HARNESS)
    n_folds = len(man)
    por_temp = man.groupby("temporada").size().to_dict()
    L.append(f"- Folds totais: {n_folds}\n")
    for t in sorted(por_temp):
        L.append(f"- {t}: {por_temp[t]}\n")
    if n_folds != 76:
        problemas.append(f"harness tem {n_folds} folds (esperado 76)")
    if por_temp.get(2024) != 29:
        problemas.append(f"folds 2024: {por_temp.get(2024)} (esperado 29)")
    if por_temp.get(2025) != 29:
        problemas.append(f"folds 2025: {por_temp.get(2025)} (esperado 29)")
    if por_temp.get(2026) != 18:
        problemas.append(f"folds 2026: {por_temp.get(2026)} (esperado 18)")
    L.append("\n")

    # ---- 3. temporal_safe
    L.append("## Temporal-safe\n\n")
    ts = pd.read_parquet(TEMPORAL_SAFE)
    L.append(f"- Linhas: {len(ts)}\n")
    L.append(f"- Colunas: {list(ts.columns)}\n")
    L.append(f"- NaN p_home: {int(ts['p_home'].isna().sum())}\n")
    L.append(f"- NaN mando: {int(ts['mando'].isna().sum())}\n")
    L.append(f"\nContagem motivo_mascara:\n\n")
    contagem = ts["motivo_mascara"].value_counts(dropna=False).to_dict()
    for k, v in contagem.items():
        L.append(f"- {k}: {v}\n")
    L.append("\n")

    # spot: p_home/p_draw/p_away juntos
    nan_ph = ts["p_home"].isna()
    nan_pd_ = ts["p_draw"].isna()
    nan_pa = ts["p_away"].isna()
    if not (nan_ph == nan_pd_).all() or not (nan_ph == nan_pa).all():
        problemas.append("p_home/p_draw/p_away com NaN desalinhados")

    # ---- 4. elegibilidade
    L.append("## Elegibilidade\n\n")
    pq = pd.read_parquet(CANON)
    n_2025r1 = int(((pq["temporada"] == 2025) & (pq["rodada"] == 1)).sum())
    n_B = int(((pq["entrou_em_campo"] == False) & (pq["pontos_num"] != 0)).sum())
    L.append(f"- Linhas 2025/R1: {n_2025r1} (esperado 619)\n")
    L.append(f"- Linhas flag B (entrou=False & pontos!=0): {n_B} (esperado 108)\n")
    L.append("\n")

    # ---- 5. feature spec
    L.append("## Feature spec — CatBoost v1\n\n")
    L.append("| feature | grupo | fonte | fórmula | depende do target? | depende de rodada futura? |\n")
    L.append("|---|---|---|---|---|---|\n")
    spec = [
        ("hist_mean","Histórico","per-row","média(pontos_num, rodada < r)","não","não"),
        ("hist_std","Histórico","per-row","std(pontos_num, rodada < r)","não","não"),
        ("hist_n","Histórico","per-row","nº obs rodada < r","não","não"),
        ("hist_shrunk","Histórico","per-row + prior","(n*hist_mean + 7*prior_pos)/(n+7)","não","não"),
        ("rec_mean_5","Recente","per-row","média das últimas 5 obs","não","não"),
        ("rec_mean_3","Recente","per-row","média das últimas 3 obs","não","não"),
        ("rec_std_5","Recente","per-row","std das últimas 5 obs","não","não"),
        ("momentum","Recente","per-row","rec_mean_5 − hist_mean","não","não"),
        ("basic_mean","Básico","per-row","média de basic_score","não","não"),
        ("basic_rec_5","Básico","per-row","média de basic_score últimas 5","não","não"),
        ("ga_per_game","G+A","per-row","(G+A cumulativo) / jogos_num","não","não"),
        ("ga_rec_5","G+A","per-row","(G+A últimas 5) / n","não","não"),
        ("ga_shrunk","G+A","per-row + prior","(n*ga_rate + 7*prior_ga)/(n+7)","não","não"),
        ("entrou_rate","Oportunidade","per-row","média(entrou_em_campo, rodada < r)","não","não"),
        ("is_cold_start","Oportunidade","per-row","1 se hist_n == 0","não","não"),
        ("posicao_id","Posição","canonical","categórica 1–6","não","não"),
        ("p_home","Confronto","temporal_safe","probabilidade implícita","não","não"),
        ("p_draw","Confronto","temporal_safe","probabilidade implícita","não","não"),
        ("p_away","Confronto","temporal_safe","probabilidade implícita","não","não"),
        ("mando","Confronto","temporal_safe","1 casa / 0 fora","não","não"),
        ("favoritismo_home_away","Confronto","temporal_safe","log(p_home/p_away)","não","não"),
    ]
    for f_, g_, s_, form, dep_t, dep_f in spec:
        L.append(f"| {f_} | {g_} | {s_} | {form} | {dep_t} | {dep_f} |\n")
    L.append("\n")

    # ---- 6. resumo
    L.append("## Resultado\n\n")
    if problemas:
        L.append("**PROBLEMAS:**\n\n")
        for x in problemas:
            L.append(f"- {x}\n")
    else:
        L.append("**PASS** — nenhum problema detectado.\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")
    print("audit salva em:", OUT_MD.relative_to(ROOT))
    if problemas:
        print("PROBLEMAS:")
        for x in problemas:
            print(f"  - {x}")
        sys.exit(1)
    else:
        print("PASS")


if __name__ == "__main__":
    main()
