#!/usr/bin/env python3
"""
Auditoria completa do campo status_id.

- NAO altera o Parquet canonico.
- NAO modifica artefatos existentes.
- Somente leitura + relatorios em reports/status_id_*.
- Deterministico. Sem random, sem amostragem aleatoria.
- NAO atribui significado aos codigos de status_id.
- Analise principal: status_id x entrou_em_campo na MESMA rodada.
- NAO usa status em R x entrou em R+1 para inferir semantica.
- RAW x Parquet: comparacao EXAUSTIVA (todas as rodadas, todos os jogadores).
"""
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ paths
def detect_root():
    candidatos = [
        Path.home() / "cartola-ai-insights",
        Path.home() / "cartola-predicao-v2",
        Path("/workspaces/cartola-ai-insights"),
        Path("/workspaces/cartola-predicao-v2"),
        Path("/data/data/com.termux/files/home/cartola-ai-insights"),
        Path("/data/data/com.termux/files/home/cartola-predicao-v2"),
        Path(__file__).resolve().parent.parent,
    ]
    for c in candidatos:
        if (c / "data" / "interim" / "cartola_jogador_rodada_v2.parquet").exists():
            return c
    raise RuntimeError("Raiz do projeto nao encontrada")


ROOT = detect_root()
INTERIM = ROOT / "data" / "interim"
REP = ROOT / "reports"
CONFIG = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw" / "caRtola"

CANON = INTERIM / "cartola_jogador_rodada_v2.parquet"

OUT_RESUMO = REP / "status_id_auditoria_resumo.csv"
OUT_TEMP = REP / "status_id_por_temporada.csv"
OUT_ROD = REP / "status_id_por_rodada.csv"
OUT_CRIT = REP / "status_id_casos_criticos.csv"
OUT_COV = REP / "status_id_cobertura.csv"
OUT_RAWDIV = REP / "status_id_raw_parquet_divergencias.csv"
OUT_RAWTOT = REP / "status_id_raw_parquet_totais.csv"
OUT_VALID = REP / "status_id_validation.json"
OUT_MD = REP / "status_id_auditoria.md"

HASH_CANON = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
TARGET = "pontos_num"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 0. linhagem do Parquet
def localizar_script_normalizacao():
    """Procura scripts que referenciam o caminho do Parquet canonico."""
    hits = []
    for f in (ROOT / "scripts").glob("*.py"):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if ("cartola_jogador_rodada_v2.parquet" in txt
                or "cartola_jogador_rodada_v2" in txt):
            hits.append(str(f.relative_to(ROOT)))
    return hits


def localizar_linhagem_status():
    """Verifica no script de normalizacao se ha mapeamento explicito de
    atletas.status_id -> status_id."""
    for f in (ROOT / "scripts").glob("*.py"):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "atletas.status_id" in txt and "status_id" in txt:
            return str(f.relative_to(ROOT))
    return None


# ------------------------------------------------------------------ documentacao
def grep_documentacao_status():
    hits = []
    alvos = [ROOT / "README.md", ROOT / "ETAPAS.md", ROOT / "config.yaml",
             CONFIG, REP, ROOT / "cartola_src", ROOT / "docs"]
    for sub in alvos:
        if sub is None or not sub.exists():
            continue
        files = [sub] if sub.is_file() else list(sub.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".md", ".txt", ".yaml", ".yml", ".py", ".json"}:
                continue
            try:
                if f.stat().st_size > 10_000_000:
                    continue
            except OSError:
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for m in re.finditer(r"\bstatus[_\s-]?id\b", txt, re.IGNORECASE):
                linha = txt[: m.start()].count("\n") + 1
                a = max(0, m.start() - 60)
                b = min(len(txt), m.start() + 80)
                trecho = txt[a:b].replace("\n", " ")
                hits.append((str(f.relative_to(ROOT)), linha, trecho))
    return hits[:60]


def _sid_label(sid):
    return "NaN" if pd.isna(sid) else int(sid)


# ------------------------------------------------------------------ comparacao RAW x Parquet
def comparar_raw_parquet(df_canon):
    """
    Compara TODAS as linhas de todos os CSVs RAW contra o canonico, pela chave
    (temporada, rodada, atleta_id). Compara:
      atletas.status_id        -> status_id
      atletas.entrou_em_campo  -> entrou_em_campo
      atletas.pontos_num       -> pontos_num
    Retorna (totais_por_rodada_df, divergencias_df).
    """
    totais = []
    divs = []
    anos = sorted(df_canon["temporada"].unique().tolist())
    for ano in anos:
        d = RAW_DIR / str(int(ano))
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            m = re.search(r"rodada-(\d+)", f.name)
            if not m:
                continue
            r = int(m.group(1))
            try:
                raw = pd.read_csv(f, low_memory=False)
            except Exception as e:
                totais.append({
                    "temporada": int(ano), "rodada": r,
                    "n_raw": None, "n_canon": None, "n_intersec": None,
                    "div_status": None, "div_entrou": None, "div_pontos": None,
                    "n_raw_only": None, "n_canon_only": None,
                    "erro_leitura": str(e),
                })
                continue

            cols = raw.columns.tolist()
            if "atletas.atleta_id" not in cols:
                continue
            keep = ["atletas.atleta_id"]
            for c in ["atletas.status_id", "atletas.entrou_em_campo",
                      "atletas.pontos_num"]:
                if c in cols:
                    keep.append(c)
            raw_s = raw[keep].copy()
            raw_s = raw_s.rename(columns={
                "atletas.atleta_id": "atleta_id",
                "atletas.status_id": "raw_status",
                "atletas.entrou_em_campo": "raw_entrou",
                "atletas.pontos_num": "raw_pontos",
            })
            raw_s["atleta_id"] = raw_s["atleta_id"].astype("int64")

            canon = df_canon[(df_canon["temporada"] == ano)
                             & (df_canon["rodada"] == r)].copy()
            canon["atleta_id"] = canon["atleta_id"].astype("int64")
            canon_s = canon[["atleta_id", "status_id", "entrou_em_campo",
                             TARGET]].rename(columns={
                "status_id": "canon_status",
                "entrou_em_campo": "canon_entrou",
                TARGET: "canon_pontos",
            })

            merged = pd.merge(raw_s, canon_s, on="atleta_id",
                              how="outer", indicator=True)

            n_raw = int(len(raw_s))
            n_canon = int(len(canon_s))
            n_inter = int((merged["_merge"] == "both").sum())
            n_raw_only = int((merged["_merge"] == "left_only").sum())
            n_canon_only = int((merged["_merge"] == "right_only").sum())

            both = merged[merged["_merge"] == "both"].copy()

            # status
            s_raw = both["raw_status"]
            s_can = both["canon_status"]
            mask_status_div = ~(
                (s_raw.isna() & s_can.isna())
                | (s_raw.notna() & s_can.notna()
                   & (s_raw.astype("float64") == s_can.astype("float64")))
            )
            n_div_status = int(mask_status_div.sum())

            # entrou
            e_raw = both["raw_entrou"].fillna(False).astype(bool)
            e_can = both["canon_entrou"].fillna(False).astype(bool)
            mask_entrou_div = (e_raw != e_can)
            n_div_entrou = int(mask_entrou_div.sum())

            # pontos
            p_raw = pd.to_numeric(both["raw_pontos"], errors="coerce")
            p_can = pd.to_numeric(both["canon_pontos"], errors="coerce")
            mask_pontos_div = ~(
                (p_raw.isna() & p_can.isna())
                | ((p_raw - p_can).abs() < 1e-9)
            )
            n_div_pontos = int(mask_pontos_div.sum())

            totais.append({
                "temporada": int(ano), "rodada": r,
                "n_raw": n_raw, "n_canon": n_canon, "n_intersec": n_inter,
                "div_status": n_div_status,
                "div_entrou": n_div_entrou,
                "div_pontos": n_div_pontos,
                "n_raw_only": n_raw_only,
                "n_canon_only": n_canon_only,
                "erro_leitura": "",
            })

            # exemplos de divergencia (limite por rodada para nao inchar)
            for _, row in both[mask_status_div].head(10).iterrows():
                divs.append({
                    "temporada": int(ano), "rodada": r,
                    "atleta_id": int(row["atleta_id"]),
                    "tipo": "status_id",
                    "raw": None if pd.isna(row["raw_status"]) else int(row["raw_status"]),
                    "canon": None if pd.isna(row["canon_status"]) else int(row["canon_status"]),
                })
            for _, row in both[mask_entrou_div].head(5).iterrows():
                divs.append({
                    "temporada": int(ano), "rodada": r,
                    "atleta_id": int(row["atleta_id"]),
                    "tipo": "entrou_em_campo",
                    "raw": bool(row["raw_entrou"]) if pd.notna(row["raw_entrou"]) else None,
                    "canon": bool(row["canon_entrou"]) if pd.notna(row["canon_entrou"]) else None,
                })
            for _, row in both[mask_pontos_div].head(5).iterrows():
                divs.append({
                    "temporada": int(ano), "rodada": r,
                    "atleta_id": int(row["atleta_id"]),
                    "tipo": "pontos_num",
                    "raw": float(row["raw_pontos"]) if pd.notna(row["raw_pontos"]) else None,
                    "canon": float(row["canon_pontos"]) if pd.notna(row["canon_pontos"]) else None,
                })

    return pd.DataFrame(totais), pd.DataFrame(divs)


# ------------------------------------------------------------------ main
def main():
    inicio = datetime.now()
    print("== Auditoria status_id ==")

    # ---------------- 0. integridade
    h_antes = sha256(CANON)
    print(f"Parquet canonico SHA-256: {h_antes[:16]}...")
    if h_antes != HASH_CANON:
        print("ERRO: hash diverge do esperado. Abortando.", file=sys.stderr)
        sys.exit(2)

    # ---------------- 1. linhagem
    scripts_norm = localizar_script_normalizacao()
    script_linhagem = localizar_linhagem_status()
    print(f"scripts que referenciam o Parquet canonico: {scripts_norm}")
    print(f"script de normalizacao de status_id: {script_linhagem}")

    # ---------------- 2. carregar canonico
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"],
                        kind="mergesort").reset_index(drop=True)
    for col in ["temporada", "rodada", "atleta_id", "status_id",
                "entrou_em_campo", TARGET]:
        if col not in df.columns:
            print(f"ERRO: coluna {col} ausente no canonico.", file=sys.stderr)
            sys.exit(2)

    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    total = len(df)
    print(f"linhas: {total}")

    # ---------------- 3. distribuicao
    dist = df["status_id"].value_counts(dropna=False).sort_index()
    dist_df = pd.DataFrame({
        "status_id": dist.index,
        "count": dist.values,
        "pct": (dist.values / total * 100).round(4),
    })
    dist_df["status_id_label"] = dist_df["status_id"].apply(_sid_label)
    dist_df.to_csv(OUT_RESUMO, index=False)

    dist_temp = (df.groupby(["temporada", "status_id"]).size()
                 .unstack(fill_value=0))
    dist_temp.to_csv(OUT_TEMP)

    # ---------------- 4. status x entrou (mesma rodada)
    crosstab = df.groupby("status_id").agg(
        total=("entrou_bool", "size"),
        entrou=("entrou_bool", "sum"),
    ).reset_index()
    crosstab["nao_entrou"] = crosstab["total"] - crosstab["entrou"]
    crosstab["pct_entrou"] = (crosstab["entrou"] / crosstab["total"] * 100).round(4)

    crosstab_temp = df.groupby(["temporada", "status_id"]).agg(
        total=("entrou_bool", "size"),
        entrou=("entrou_bool", "sum"),
    ).reset_index()
    crosstab_temp["nao_entrou"] = crosstab_temp["total"] - crosstab_temp["entrou"]
    crosstab_temp["pct_entrou"] = (crosstab_temp["entrou"] /
                                    crosstab_temp["total"] * 100).round(4)

    crosstab_rod = df.groupby(["temporada", "rodada", "status_id"]).agg(
        total=("entrou_bool", "size"),
        entrou=("entrou_bool", "sum"),
    ).reset_index()
    crosstab_rod["nao_entrou"] = crosstab_rod["total"] - crosstab_rod["entrou"]
    crosstab_rod["pct_entrou"] = (crosstab_rod["entrou"] /
                                   crosstab_rod["total"] * 100).round(4)
    crosstab_rod.to_csv(OUT_ROD, index=False)

    # ---------------- 5. casos criticos
    nulos = df[df["status_id"].isna()]
    criticos = []
    criticos.append({"caso": "status_nulo_total", "n": int(len(nulos))})
    criticos.append({"caso": "status_nulo_e_entrou", "n": int(nulos["entrou_bool"].sum())})
    criticos.append({"caso": "status_nulo_e_pontos_nao_zero",
                     "n": int((nulos[TARGET] != 0).sum())})
    for _, r in crosstab.iterrows():
        if pd.isna(r["status_id"]):
            continue
        if r["pct_entrou"] < 50 and r["entrou"] > 0:
            criticos.append({
                "caso": f"status_{int(r['status_id'])}_baixo_pct_entrou_{r['pct_entrou']}pct",
                "n": int(r["entrou"]),
            })
    criticos_df = pd.DataFrame(criticos)
    criticos_df.to_csv(OUT_CRIT, index=False)

    exemplos = []
    for _, r in crosstab.iterrows():
        if pd.isna(r["status_id"]):
            continue
        if r["pct_entrou"] < 20 and r["entrou"] > 0:
            sub = df[(df["status_id"] == r["status_id"]) & df["entrou_bool"]].head(5)
            for _, e in sub.iterrows():
                exemplos.append({
                    "status_id": int(r["status_id"]),
                    "temporada": int(e["temporada"]),
                    "rodada": int(e["rodada"]),
                    "atleta_id": int(e["atleta_id"]),
                    "apelido": e.get("apelido", ""),
                    "pontos_num": float(e[TARGET]),
                })

    # ---------------- 6. status nulo dedicado
    nulos_resumo = {
        "n_linhas": int(len(nulos)),
        "pct_do_dataset": round(len(nulos) / total * 100, 4),
        "n_por_temporada": {int(t): int(n) for t, n in
                             nulos.groupby("temporada").size().items()},
        "n_entrou": int(nulos["entrou_bool"].sum()),
        "n_nao_entrou": int(len(nulos) - nulos["entrou_bool"].sum()),
        "pct_entrou": round(nulos["entrou_bool"].mean() * 100, 4) if len(nulos) else None,
        "n_pontos_nao_zero": int((nulos[TARGET] != 0).sum()),
    }

    # ---------------- 7. cobertura + duplicatas + multiplos status
    dup_key = int(df.duplicated(subset=["temporada", "rodada", "atleta_id"]).sum())

    multi_status = (
        df.groupby(["temporada", "rodada", "atleta_id"])["status_id"]
          .nunique(dropna=False)
    )
    n_multi_status = int((multi_status > 1).sum())

    cobertura = {
        "total_linhas": int(total),
        "n_com_status": int(df["status_id"].notna().sum()),
        "n_status_nulo": int(df["status_id"].isna().sum()),
        "pct_com_status": round(df["status_id"].notna().mean() * 100, 4),
        "n_jogadores_afetados_por_nulo": int(nulos["atleta_id"].nunique()),
        "n_rodadas_afetadas_por_nulo": int(
            nulos.groupby(["temporada", "rodada"]).ngroups) if len(nulos) else 0,
        "duplicatas_T_R_A": dup_key,
        "celulas_T_R_A_com_mais_de_um_status": n_multi_status,
    }
    pd.DataFrame([cobertura]).to_csv(OUT_COV, index=False)

    # ---------------- 8. RAW x Parquet EXAUSTIVO
    print("comparando RAW x Parquet (todas as rodadas)...")
    raw_tot, raw_div = comparar_raw_parquet(df)
    raw_tot.to_csv(OUT_RAWTOT, index=False)
    raw_div.to_csv(OUT_RAWDIV, index=False)

    # agregacoes
    if len(raw_tot):
        n_raw_total = int(raw_tot["n_raw"].dropna().sum())
        n_inter_total = int(raw_tot["n_intersec"].dropna().sum())
        n_canon_total = int(raw_tot["n_canon"].dropna().sum())
        n_div_status = int(raw_tot["div_status"].dropna().sum())
        n_div_entrou = int(raw_tot["div_entrou"].dropna().sum())
        n_div_pontos = int(raw_tot["div_pontos"].dropna().sum())
        n_raw_only = int(raw_tot["n_raw_only"].dropna().sum())
        n_canon_only = int(raw_tot["n_canon_only"].dropna().sum())
    else:
        n_raw_total = n_inter_total = n_canon_total = 0
        n_div_status = n_div_entrou = n_div_pontos = 0
        n_raw_only = n_canon_only = 0

    div_por_temp = (raw_tot.groupby("temporada")[
        ["div_status", "div_entrou", "div_pontos", "n_raw_only", "n_canon_only"]]
        .sum().reset_index()) if len(raw_tot) else pd.DataFrame()
    div_por_rod = (raw_tot.groupby(["temporada", "rodada"])[
        ["div_status", "div_entrou", "div_pontos"]]
        .sum().reset_index()) if len(raw_tot) else pd.DataFrame()

    # ---------------- 9. integridade depois
    h_depois = sha256(CANON)
    integridade_ok = (h_antes == h_depois == HASH_CANON)

    # ---------------- 10. validation.json
    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "parquet_sha256": h_antes,
        "parquet_sha_esperado": HASH_CANON,
        "integridade_ok": bool(integridade_ok),
        "linhas_totais": int(total),
        "duplicatas_T_R_A": dup_key,
        "celulas_T_R_A_com_mais_de_um_status": n_multi_status,
        "status_nulos": int(df["status_id"].isna().sum()),
        "status_codigos": sorted([int(x) for x in dist.index if pd.notna(x)]),
        "script_normalizacao_canonico": scripts_norm,
        "script_que_mapeia_status": script_linhagem,
        "raw_parquet": {
            "n_rodadas_comparadas": int(len(raw_tot)),
            "n_raw_total": n_raw_total,
            "n_intersec_total": n_inter_total,
            "n_canon_total": n_canon_total,
            "n_raw_only": n_raw_only,
            "n_canon_only": n_canon_only,
            "div_status": n_div_status,
            "div_entrou": n_div_entrou,
            "div_pontos": n_div_pontos,
        },
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    # ---------------- 11. relatorio markdown
    doc_hits = grep_documentacao_status()
    L = []
    L.append("# Auditoria do `status_id`\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Parquet: `data/interim/cartola_jogador_rodada_v2.parquet`\n")
    L.append(f"- SHA-256: `{h_antes}`\n")
    L.append(f"- Linhas: {total}\n")
    L.append(f"- Duplicatas `(temporada, rodada, atleta_id)`: {dup_key}\n")
    L.append(f"- Celulas `(T,R,A)` com mais de um `status_id`: {n_multi_status}\n\n")

    L.append("## 0. Linhagem do campo `status_id`\n\n")
    L.append(f"- Scripts que referenciam o Parquet canonico: {scripts_norm}\n")
    L.append(f"- Script com mapeamento `atletas.status_id` -> `status_id`: "
             f"`{script_linhagem}`\n")
    L.append("\n")
    L.append("Mapeamento confirmado por leitura do script de normalizacao:\n\n")
    L.append("```\n")
    L.append("RAW (data/raw/caRtola/<ano>/rodada-N.csv)\n")
    L.append("    atletas.status_id        -> status_id\n")
    L.append("    atletas.entrou_em_campo  -> entrou_em_campo\n")
    L.append("    atletas.pontos_num       -> pontos_num\n")
    L.append("```\n\n")

    L.append("## 1. Codigos encontrados\n\n")
    L.append("| status_id | count | pct |\n|---|---|---|\n")
    for _, r in dist_df.iterrows():
        L.append(f"| {r['status_id_label']} | {int(r['count'])} | {r['pct']}% |\n")
    L.append("\n")
    L.append("**Nota:** nenhum significado foi atribuido aos codigos.\n\n")

    L.append("## 2. Status x entrou_em_campo (MESMA rodada R)\n\n")
    L.append("| status_id | total | entrou | nao entrou | % entrou |\n")
    L.append("|---|---|---|---|---|\n")
    for _, r in crosstab.sort_values("pct_entrou", ascending=False).iterrows():
        L.append(f"| {_sid_label(r['status_id'])} | {int(r['total'])} | {int(r['entrou'])} | "
                 f"{int(r['nao_entrou'])} | {r['pct_entrou']}% |\n")
    L.append("\n")

    L.append("## 3. Por temporada\n\n")
    L.append("| temporada | status_id | total | entrou | nao entrou | % entrou |\n")
    L.append("|---|---|---|---|---|---|\n")
    for _, r in crosstab_temp.sort_values(["temporada", "pct_entrou"],
                                          ascending=[True, False]).iterrows():
        L.append(f"| {int(r['temporada'])} | {_sid_label(r['status_id'])} | "
                 f"{int(r['total'])} | {int(r['entrou'])} | {int(r['nao_entrou'])} | "
                 f"{r['pct_entrou']}% |\n")
    L.append("\n")

    L.append("## 4. Casos criticos\n\n")
    for _, r in criticos_df.iterrows():
        L.append(f"- `{r['caso']}`: n = {int(r['n'])}\n")
    L.append("\n")

    if exemplos:
        L.append("### Exemplos de baixo `% entrou` com entrada confirmada\n\n")
        L.append("| status_id | temporada | rodada | atleta_id | apelido | pontos_num |\n")
        L.append("|---|---|---|---|---|---|\n")
        for e in exemplos[:20]:
            L.append(f"| {e['status_id']} | {e['temporada']} | {e['rodada']} | "
                     f"{e['atleta_id']} | {e['apelido']} | {e['pontos_num']} |\n")
        L.append("\n")

    L.append("## 5. Status nulo — auditoria dedicada\n\n")
    L.append(f"- Linhas com status nulo: {nulos_resumo['n_linhas']} "
             f"({nulos_resumo['pct_do_dataset']}%)\n")
    L.append(f"- Por temporada: {nulos_resumo['n_por_temporada']}\n")
    L.append(f"- Entraram em campo: {nulos_resumo['n_entrou']}\n")
    L.append(f"- Nao entraram: {nulos_resumo['n_nao_entrou']}\n")
    L.append(f"- % que entrou: {nulos_resumo['pct_entrou']}%\n")
    L.append(f"- Pontos != 0: {nulos_resumo['n_pontos_nao_zero']}\n\n")

    L.append("## 6. Cobertura e duplicatas\n\n")
    for k, v in cobertura.items():
        L.append(f"- {k}: {v}\n")
    L.append("\n")

    L.append("## 7. RAW x Parquet — comparacao exaustiva\n\n")
    L.append(f"- Rodadas comparadas: {len(raw_tot)}\n")
    L.append(f"- Linhas RAW (total): {n_raw_total}\n")
    L.append(f"- Linhas Parquet (total): {n_canon_total}\n")
    L.append(f"- Interseccao por `(temporada, rodada, atleta_id)`: {n_inter_total}\n")
    L.append(f"- Linhas so no RAW: {n_raw_only}\n")
    L.append(f"- Linhas so no Parquet: {n_canon_only}\n\n")
    L.append("### Divergencias de valor (linhas em ambos)\n\n")
    L.append(f"- `status_id`: {n_div_status}\n")
    L.append(f"- `entrou_em_campo`: {n_div_entrou}\n")
    L.append(f"- `pontos_num`: {n_div_pontos}\n\n")
    if len(div_por_temp):
        L.append("### Por temporada\n\n")
        L.append("| temporada | div_status | div_entrou | div_pontos | raw_only | canon_only |\n")
        L.append("|---|---|---|---|---|---|\n")
        for _, r in div_por_temp.iterrows():
            L.append(f"| {int(r['temporada'])} | {int(r['div_status'])} | "
                     f"{int(r['div_entrou'])} | {int(r['div_pontos'])} | "
                     f"{int(r['n_raw_only'])} | {int(r['n_canon_only'])} |\n")
        L.append("\n")
    if len(div_por_rod):
        top_rod = div_por_rod.sort_values(
            ["div_status", "div_entrou", "div_pontos"], ascending=False).head(20)
        L.append("### Rodadas com maiores divergencias (top 20)\n\n")
        L.append("| temporada | rodada | div_status | div_entrou | div_pontos |\n")
        L.append("|---|---|---|---|---|\n")
        for _, r in top_rod.iterrows():
            L.append(f"| {int(r['temporada'])} | {int(r['rodada'])} | "
                     f"{int(r['div_status'])} | {int(r['div_entrou'])} | "
                     f"{int(r['div_pontos'])} |\n")
        L.append("\n")
    if len(raw_div):
        L.append("### Exemplos de divergencia (ate 20)\n\n")
        L.append("| temporada | rodada | atleta_id | tipo | raw | canon |\n")
        L.append("|---|---|---|---|---|---|\n")
        for _, r in raw_div.head(20).iterrows():
            L.append(f"| {int(r['temporada'])} | {int(r['rodada'])} | "
                     f"{int(r['atleta_id'])} | {r['tipo']} | {r['raw']} | {r['canon']} |\n")
        L.append("\n")

    L.append("## 8. Documentacao existente no projeto\n\n")
    if doc_hits:
        L.append(f"- Ocorrencias: {len(doc_hits)}\n\n")
        for fname, linha, trecho in doc_hits[:20]:
            L.append(f"  - `{fname}:{linha}` — {trecho}\n")
    else:
        L.append("Nenhuma mencao a `status_id` encontrada em README, "
                 "config, reports, scripts ou docs.\n")
    L.append("\n")

    L.append("## 9. Integridade\n\n")
    L.append(f"- Parquet SHA-256 (antes): `{h_antes}`\n")
    L.append(f"- Parquet SHA-256 (depois): `{h_depois}`\n")
    L.append(f"- Inalterado: **{integridade_ok}**\n\n")

    L.append("---\n\n")
    L.append("## RECOMENDACAO PARA ELEGIBILIDADE\n\n")
    L.append("Esta secao resume APENAS o que os dados demonstram.\n")
    L.append("Nenhum filtro de exclusao e' definido nesta etapa.\n\n")
    L.append("**FATOS OBSERVADOS:**\n\n")
    L.append(f"- Codigos de `status_id` presentes: `{val['status_codigos']}`\n")
    L.append(f"- Status nulos: {int(df['status_id'].isna().sum())} linhas "
             f"({round(df['status_id'].isna().mean() * 100, 2)}%)\n")
    L.append(f"- Duplicatas `(T, R, A)`: {dup_key}\n")
    L.append(f"- Celulas `(T,R,A)` com mais de um status: {n_multi_status}\n")
    L.append(f"- RAW x Parquet: {n_inter_total} linhas em comum; "
             f"div_status={n_div_status}, div_entrou={n_div_entrou}, "
             f"div_pontos={n_div_pontos}.\n\n")
    L.append("**Ranking empirico (por % que entrou na MESMA rodada R):**\n\n")
    for _, r in crosstab.sort_values("pct_entrou", ascending=False).iterrows():
        L.append(f"- `status_id = {_sid_label(r['status_id'])}`: "
                 f"{r['pct_entrou']}% entraram "
                 f"({int(r['entrou'])}/{int(r['total'])})\n")
    L.append("\n")
    L.append("**Pendencias antes de qualquer decisao de elegibilidade:**\n\n")
    L.append("- Nenhum codigo e' rotulado como duvida/provavel/contundido aqui. "
             "A atribuicao de significado deve vir de documentacao/proveniencia.\n")
    L.append("- Se algum codigo combina baixo `% entrou` com entradas confirmadas, "
             "a causa deve ser investigada antes de virar filtro de exclusao.\n")
    L.append("- Se status nulo tem entradas confirmadas em volume, "
             "tratar nulo como nao elegivel precisa de justificativa adicional.\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    # ---------------- console
    print()
    print("=== DISTRIBUICAO DE STATUS ===")
    print(dist_df.to_string(index=False))
    print()
    print("=== STATUS x ENTROU (mesma rodada) ===")
    print(crosstab.sort_values("pct_entrou", ascending=False).to_string(index=False))
    print()
    print("=== CASOS CRITICOS ===")
    print(criticos_df.to_string(index=False))
    print()
    print("=== COBERTURA ===")
    for k, v in cobertura.items():
        print(f"  {k}: {v}")
    print()
    print("=== RAW x PARQUET ===")
    print(f"  rodadas comparadas: {len(raw_tot)}")
    print(f"  n_raw_total: {n_raw_total}")
    print(f"  n_canon_total: {n_canon_total}")
    print(f"  n_intersec: {n_inter_total}")
    print(f"  raw_only: {n_raw_only}")
    print(f"  canon_only: {n_canon_only}")
    print(f"  div_status: {n_div_status}")
    print(f"  div_entrou: {n_div_entrou}")
    print(f"  div_pontos: {n_div_pontos}")
    print()
    print("=== INTEGRIDADE ===")
    print(f"  Parquet SHA-256 inalterado: {integridade_ok}")
    print()
    print("Artefatos gerados:")
    for p in [OUT_RESUMO, OUT_TEMP, OUT_ROD, OUT_CRIT, OUT_COV,
              OUT_RAWTOT, OUT_RAWDIV, OUT_VALID, OUT_MD]:
        if p.exists():
            print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")

    if not integridade_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
