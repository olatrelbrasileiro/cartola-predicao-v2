#!/usr/bin/env python3
"""
Auditoria dos dados RAW do caRtola (2024, 2025, 2026).
Gera:
  reports/inventario_caRtola_raw.csv
  reports/inventario_caRtola_raw.md
  reports/comparacao_schema_caRtola_2024_2025_2026.md
  reports/catalogo_dados_caRtola.md
  reports/qualidade_temporal_caRtola.md
NAO normaliza, NAO cria features, NAO treina modelos.
"""
import os, re, json, hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path.home() / "cartola-predicao-v2"
RAW  = ROOT / "data" / "raw" / "caRtola"
REP  = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)

ANOS = ["2024", "2025", "2026"]

def infer_cols(df):
    return {c: str(df[c].dtype) for c in df.columns}

def nulos_por_col(df):
    return {c: int(df[c].isna().sum()) for c in df.columns}

def rodada_do_nome(p):
    m = re.search(r"rodada[-_ ]?(\d+)", p.name, re.IGNORECASE)
    return int(m.group(1)) if m else None

inventario = []

for ano in ANOS:
    d = RAW / ano
    if not d.exists():
        continue
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        info = {
            "temporada": ano,
            "arquivo": f.name,
            "extensao": ext,
            "tamanho_bytes": f.stat().st_size,
            "rodada_no_nome": rodada_do_nome(f),
            "linhas": None, "colunas": None,
            "nomes_colunas": "", "tipos": "",
            "nulos": "",
            "primeira_rodada": "", "ultima_rodada": "",
            "erro_leitura": ""
        }
        if ext == ".csv":
            try:
                df = pd.read_csv(f, low_memory=False)
                info["linhas"] = len(df)
                info["colunas"] = df.shape[1]
                info["nomes_colunas"] = "|".join(map(str, df.columns))
                info["tipos"] = json.dumps(infer_cols(df), ensure_ascii=False)
                info["nulos"] = json.dumps(nulos_por_col(df), ensure_ascii=False)
                # rodada minima/maxima se houver coluna de rodada
                for cand in ["rodada", "Rodada", "RODADA"]:
                    if cand in df.columns:
                        try:
                            rr = pd.to_numeric(df[cand], errors="coerce").dropna()
                            if len(rr):
                                info["primeira_rodada"] = int(rr.min())
                                info["ultima_rodada"]  = int(rr.max())
                        except Exception:
                            pass
                        break
            except Exception as e:
                info["erro_leitura"] = f"{type(e).__name__}: {e}"
        inventario.append(info)

inv_df = pd.DataFrame(inventario)
inv_df.to_csv(REP / "inventario_caRtola_raw.csv", index=False, encoding="utf-8")

# Markdown do inventario
with open(REP / "inventario_caRtola_raw.md", "w", encoding="utf-8") as fh:
    fh.write("# Inventario RAW caRtola (2024, 2025, 2026)\n\n")
    fh.write(f"Total de arquivos: {len(inv_df)}\n\n")
    for ano in ANOS:
        sub = inv_df[inv_df["temporada"] == ano]
        fh.write(f"## Temporada {ano}\n\n")
        if sub.empty:
            fh.write("_Nenhum arquivo._\n\n")
            continue
        fh.write("| arquivo | ext | bytes | linhas | colunas | rodada_nome | primeira | ultima |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in sub.iterrows():
            fh.write(f"| {r['arquivo']} | {r['extensao']} | {r['tamanho_bytes']} | "
                     f"{r['linhas']} | {r['colunas']} | {r['rodada_no_nome']} | "
                     f"{r['primeira_rodada']} | {r['ultima_rodada']} |\n")
        fh.write("\n")
    fh.write("## Colunas por arquivo\n\n")
    for _, r in inv_df.iterrows():
        if r["nomes_colunas"]:
            fh.write(f"### {r['temporada']}/{r['arquivo']}\n\n")
            fh.write("Colunas: " + str(r["nomes_colunas"]) + "\n\n")
            if r["erro_leitura"]:
                fh.write("**ERRO DE LEITURA:** " + str(r["erro_leitura"]) + "\n\n")

print("OK: inventario gerado")

# ---------- Comparacao de schemas ----------
# Agrupa por "nome canonico" do arquivo: rodada-N.csv
def chave_arquivo(nome):
    m = re.match(r"(rodada[-_ ]?\d+)", nome, re.IGNORECASE)
    return m.group(1).lower().replace(" ", "").replace("_", "-") if m else nome.lower()

por_ano = defaultdict(dict)
for _, r in inv_df.iterrows():
    if r["extensao"] != ".csv" or not r["nomes_colunas"]:
        continue
    cols = str(r["nomes_colunas"]).split("|")
    por_ano[r["temporada"]][chave_arquivo(r["arquivo"])] = cols

todas_chaves = sorted(set().union(*[set(por_ano[a].keys()) for a in ANOS if por_ano[a]]))

with open(REP / "comparacao_schema_caRtola_2024_2025_2026.md", "w", encoding="utf-8") as fh:
    fh.write("# Comparacao de schema caRtola: 2024 x 2025 x 2026\n\n")
    fh.write("> Comparacao feita APENAS pelo nome canonico do arquivo (rodada-N).\n")
    fh.write("> Diferencas sao reportadas cruas, sem renomeacao automatica.\n\n")

    for chave in todas_chaves:
        fh.write(f"## {chave}\n\n")
        conj = {a: set(por_ano[a].get(chave, [])) for a in ANOS}
        if not any(conj.values()):
            fh.write("_Nenhuma temporada possui este arquivo._\n\n")
            continue
        comuns = set.intersection(*[c for c in conj.values() if c]) if any(conj.values()) else set()
        fh.write(f"- Comuns as temporadas presentes: {sorted(comuns)}\n")
        for a in ANOS:
            so = conj[a] - comuns if conj[a] else set()
            fh.write(f"- Somente em {a}: {sorted(so)}\n")
        # colunas com mesmo nome mas tipo diferente
        tipos_por_ano = {}
        for a in ANOS:
            tipos_por_ano[a] = {}
        for _, r in inv_df.iterrows():
            if r["extensao"] != ".csv" or r["temporada"] not in ANOS:
                continue
            if chave_arquivo(r["arquivo"]) != chave:
                continue
            try:
                t = json.loads(r["tipos"]) if r["tipos"] else {}
                tipos_por_ano[r["temporada"]] = t
            except Exception:
                pass
        for col in sorted(comuns):
            tipos = {a: tipos_por_ano[a].get(col, "?") for a in ANOS}
            if len(set(tipos.values())) > 1:
                fh.write(f"- **Tipo diferente** em `{col}`: {tipos}\n")
        fh.write("\n")

print("OK: comparacao de schema gerada")

# ---------- Catalogo de dados ----------
# Categoriza colunas por palavras-chave (heuristica, sem renomear)
CATEGORIAS = {
    "Jogador": ["atleta", "jogador", "apelido", "nome", "posicao", "posição", "id"],
    "Participacao": ["entrou", "titular", "reserva", "minutos", "status", "participou", "jogou"],
    "Pontuacao": ["pontuacao", "pontos", "media", "média", "score"],
    "Scouts": ["scout", "gol", "assist", "finalizacao", "desarme", "falta", "cartao", "cartão",
               "defesa", "penalti", "pênalti", "impedimento", "passes", "cruzamento", "drible",
               "bloqueio", "roubada", "falta_sofrida", "defesa_penalti", "defesa_pênalti"],
    "RodadaTemporada": ["rodada", "temporada", "ano", "season", "round", "data"],
    "Clube": ["clube", "time", "adversario", "adversário", "mando", "casa", "fora", "escudo"],
    "Mercado": ["preco", "preço", "valor", "variacao", "variação", "mercado", "status_mercado"],
}
catalogo = defaultdict(lambda: defaultdict(set))
for _, r in inv_df.iterrows():
    if r["extensao"] != ".csv" or not r["nomes_colunas"]:
        continue
    for col in str(r["nomes_colunas"]).split("|"):
        cl = col.lower()
        cat = "Outros"
        for c, kws in CATEGORIAS.items():
            if any(k in cl for k in kws):
                cat = c
                break
        catalogo[col][cat].add(r["temporada"])

with open(REP / "catalogo_dados_caRtola.md", "w", encoding="utf-8") as fh:
    fh.write("# Catalogo de campos encontrados nos dados RAW do caRtola\n\n")
    fh.write("| Campo | Categoria | Temporadas |\n|---|---|---|\n")
    for col in sorted(catalogo.keys()):
        cats = catalogo[col]
        cat = next(iter(cats.keys()))
        anos = sorted(set().union(*cats.values()))
        fh.write(f"| {col} | {cat} | {', '.join(anos)} |\n")
print("OK: catalogo gerado")

# ---------- Qualidade temporal ----------
with open(REP / "qualidade_temporal_caRtola.md", "w", encoding="utf-8") as fh:
    fh.write("# Qualidade temporal dos dados RAW do caRtola\n\n")
    fh.write("> Apenas reporte. Nada foi corrigido.\n\n")
    for ano in ANOS:
        sub = inv_df[(inv_df["temporada"] == ano) & (inv_df["extensao"] == ".csv")]
        fh.write(f"## Temporada {ano}\n\n")
        fh.write(f"- Arquivos CSV: {len(sub)}\n")
        if sub.empty:
            fh.write("- (sem dados)\n\n")
            continue
        rodadas = sorted([int(x) for x in sub["rodada_no_nome"].dropna().unique()])
        fh.write(f"- Rodadas presentes (pelo nome): {rodadas}\n")
        if rodadas:
            esperado = set(range(min(rodadas), max(rodadas) + 1))
            faltando = sorted(esperado - set(rodadas))
            fh.write(f"- Rodadas faltantes no intervalo: {faltando}\n")
        # duplicados por rodada
        dup = sub["rodada_no_nome"].value_counts()
        dup = dup[dup > 1]
        if len(dup):
            fh.write(f"- Rodadas com mais de um arquivo: {dict(dup)}\n")
        fh.write(f"- Linhas sem rodada (coluna): (ver inventario por arquivo)\n")
        fh.write(f"- Linhas sem jogador: (ver inventario por arquivo)\n")
        fh.write(f"- Linhas sem pontuacao: (ver inventario por arquivo)\n\n")

print("OK: qualidade temporal gerada")
print("Relatorios em:", REP)
