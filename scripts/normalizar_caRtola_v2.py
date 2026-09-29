#!/usr/bin/env python3
"""
Normalizacao dos dados RAW do caRtola para o dataset canonico
jogador x rodada x temporada.

Etapa 2.4 — SOMENTE normalizacao.
- nao cria features
- nao imputa
- nao treina modelos
- nao altera os RAWs
"""
import hashlib, sys, io
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow  # noqa: F401  (checagem de dependencia)

ROOT = Path.home() / "cartola-predicao-v2"
RAW  = ROOT / "data" / "raw" / "caRtola"
OUT  = ROOT / "data" / "interim"
REP  = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

PARQUET = OUT / "cartola_jogador_rodada_v2.parquet"
RELATORIO = REP / "normalizacao_v2.md"

ANOS = ["2024", "2025", "2026"]
ESPERADO_LINHAS = {"2024": 28499, "2025": 28589, "2026": 20165}
ESPERADO_TOTAL  = 77253
ESPERADO_RODADAS = {
    "2024": set(range(1, 39)),
    "2025": set(range(1, 39)),
    "2026": set(range(1, 28)),
}

# mapeamento de coluna original -> nome canonico
NUCLEO = {
    "atletas.atleta_id":            "atleta_id",
    "atletas.apelido":              "apelido",
    "atletas.apelido_abreviado":    "apelido_abreviado",
    "atletas.nome":                 "nome",
    "atletas.slug":                 "slug",
    "atletas.clube_id":             "clube_id",
    "atletas.clube.id.full.name":   "clube_nome",
    "atletas.posicao_id":           "posicao_id",
    "atletas.status_id":            "status_id",
    "atletas.entrou_em_campo":      "entrou_em_campo",
    "atletas.media_num":            "media_num",
    "atletas.variacao_num":         "variacao_num",
    "atletas.preco_num":            "preco_num",
    "atletas.jogos_num":            "jogos_num",
    "atletas.pontos_num":           "pontos_num",
    "atletas.rodada_id":            "rodada",
}
SCOUTS = ["A","CA","CV","DE","DS","FC","FD","FF","FS","FT","G","I",
          "PC","PS","PP","SG","V","DP","GC"]

COLS_FORA = ["Unnamed: 0","atletas.foto","atletas.minimo_para_valorizar",
             "atletas.craque","PI"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()

def coletar_raws():
    arquivos = []
    for ano in ANOS:
        for f in sorted((RAW / ano).glob("*.csv")):
            arquivos.append((ano, f))
    return arquivos

def normalizar_arquivo(ano: str, f: Path) -> pd.DataFrame:
    df = pd.read_csv(f, low_memory=False)

    # valida que rodada do nome bate com rodada da coluna (se existir)
    # e que rodada_id existe
    if "atletas.rodada_id" not in df.columns:
        raise RuntimeError(f"{f}: coluna atletas.rodada_id ausente")

    out = pd.DataFrame(index=df.index)
    out["temporada"] = ano

    # nucleo
    for orig, novo in NUCLEO.items():
        if orig in df.columns:
            out[novo] = df[orig].values
        else:
            # nao esperamos ausencia no nucleo; se ocorrer, e' erro
            raise RuntimeError(f"{f}: coluna obrigatoria {orig} ausente")

    # scouts: criar sempre, NaN quando coluna nao existe
    for s in SCOUTS:
        col = f"scout_{s}"
        if s in df.columns:
            out[col] = df[s].values
        else:
            out[col] = pd.NA

    # proveniencia
    out["arquivo_origem"] = f"{ano}/{f.name}"
    out["linha_origem"] = df.index.astype("int64")

    return out

def main():
    inicio = datetime.now()
    arquivos = coletar_raws()
    print(f"Arquivos RAW encontrados: {len(arquivos)}")

    # hash antes
    hashes_antes = {str(f): sha256(f) for _, f in arquivos}

    # processar
    partes = []
    linhas_por_ano = {}
    for ano, f in arquivos:
        dfn = normalizar_arquivo(ano, f)
        partes.append(dfn)
        linhas_por_ano.setdefault(ano, 0)
        linhas_por_ano[ano] += len(dfn)

    dados = pd.concat(partes, ignore_index=True)

    # tipos
    dados["temporada"] = dados["temporada"].astype("int64")
    dados["rodada"]    = dados["rodada"].astype("int64")
    dados["atleta_id"] = dados["atleta_id"].astype("int64")
    dados["clube_id"]  = dados["clube_id"].astype("int64")
    dados["posicao_id"] = dados["posicao_id"].astype("int64")
    dados["status_id"] = dados["status_id"].astype("int64")
    dados["jogos_num"] = dados["jogos_num"].astype("int64")
    # entrou_em_campo pode ter valores NA em algum canto? ate agora nao.
    dados["entrou_em_campo"] = dados["entrou_em_campo"].astype("boolean")
    for c in ["media_num","variacao_num","preco_num","pontos_num"]:
        dados[c] = pd.to_numeric(dados[c], errors="coerce")
    for s in SCOUTS:
        dados[f"scout_{s}"] = pd.to_numeric(dados[f"scout_{s}"], errors="coerce")
    dados["linha_origem"] = dados["linha_origem"].astype("int64")

    # ordenar por chave canonica (nao altera conteudo, so a ordem fisica)
    dados = dados.sort_values(["temporada","rodada","atleta_id"],
                              kind="mergesort").reset_index(drop=True)

    # salvar
    dados.to_parquet(PARQUET, index=False)

    # hash depois
    hashes_depois = {str(f): sha256(f) for _, f in arquivos}
    raws_inalterados = (hashes_antes == hashes_depois)

    # ---------- AUDITORIAS ----------
    total_linhas = len(dados)

    # chaves duplicadas
    dup = int(dados.duplicated(subset=["temporada","rodada","atleta_id"]).sum())

    # nulos em pontos_num
    nulos_pontos = int(dados["pontos_num"].isna().sum())

    # DP / GC estruturalmente ausentes = linhas em rodadas em que a coluna
    # original nao existia
    # reconstruimos a lista de (ano, rodada) onde faltava a coluna
    sem_dp_por_ano = {}
    sem_gc_por_ano = {}
    for ano, f in arquivos:
        cols = pd.read_csv(f, nrows=0, low_memory=False).columns.tolist()
        import re
        m = re.search(r"rodada-(\d+)", f.name)
        r = int(m.group(1)) if m else None
        if r is None:
            continue
        if "DP" not in cols:
            sem_dp_por_ano.setdefault(ano, set()).add(r)
        if "GC" not in cols:
            sem_gc_por_ano.setdefault(ano, set()).add(r)

    linhas_dp_ausente = 0
    linhas_gc_ausente = 0
    linhas_dp_nan_total = 0
    linhas_gc_nan_total = 0
    for ano, rodadas in sem_dp_por_ano.items():
        for r in rodadas:
            linhas_dp_ausente += int(
                ((dados["temporada"] == int(ano)) & (dados["rodada"] == r)).sum())
    for ano, rodadas in sem_gc_por_ano.items():
        for r in rodadas:
            linhas_gc_ausente += int(
                ((dados["temporada"] == int(ano)) & (dados["rodada"] == r)).sum())
    linhas_dp_nan_total = int(dados["scout_DP"].isna().sum())
    linhas_gc_nan_total = int(dados["scout_GC"].isna().sum())

    # distribuicao de entrou_em_campo
    dist_entrou = dados["entrou_em_campo"].value_counts(dropna=False).to_dict()

    # atletas distintos por temporada
    atletas_por_temporada = {
        str(a): int(dados[dados["temporada"] == int(a)]["atleta_id"].nunique())
        for a in ANOS
    }

    # rodadas presentes por temporada
    rodadas_por_temporada = {
        str(a): sorted(dados[dados["temporada"] == int(a)]["rodada"].unique().tolist())
        for a in ANOS
    }
    rodadas_ok = all(
        set(rodadas_por_temporada[a]) == ESPERADO_RODADAS[a] for a in ANOS
    )

    # colunas finais
    colunas_finais = list(dados.columns)

    # checagem dos NaN em DP/GC
    # linhas com DP NaN devem ser exatamente as (ano, rodada) sem coluna
    mask_dp_nan = dados["scout_DP"].isna()
    mask_dp_esperada = pd.Series(False, index=dados.index)
    for ano, rodadas in sem_dp_por_ano.items():
        for r in rodadas:
            mask_dp_esperada |= (
                (dados["temporada"] == int(ano)) & (dados["rodada"] == r))
    dp_nan_ok = bool((mask_dp_nan | mask_dp_esperada).equals(mask_dp_nan))  # NaN eh superconjunto do esperado

    mask_gc_nan = dados["scout_GC"].isna()
    mask_gc_esperada = pd.Series(False, index=dados.index)
    for ano, rodadas in sem_gc_por_ano.items():
        for r in rodadas:
            mask_gc_esperada |= (
                (dados["temporada"] == int(ano)) & (dados["rodada"] == r))
    gc_nan_ok = bool((mask_gc_nan | mask_gc_esperada).equals(mask_gc_nan))

    # reabrir parquet para validar
    reaberto = pd.read_parquet(PARQUET)
    reaberto_ok = (len(reaberto) == total_linhas)

    fim = datetime.now()

    # ---------- RELATORIO ----------
    with open(RELATORIO, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# Normalizacao v2 — caRtola\n\n")
        w(f"- **Inicio:** {inicio.isoformat(timespec='seconds')}\n")
        w(f"- **Fim:** {fim.isoformat(timespec='seconds')}\n")
        w(f"- **Duracao:** {(fim-inicio).total_seconds():.1f}s\n\n")

        w("## Arquivos processados por temporada\n\n")
        for a in ANOS:
            n = len([1 for ano, _ in arquivos if ano == a])
            w(f"- {a}: {n} CSV(s)\n")
        w("\n")

        w("## Linhas lidas por temporada\n\n")
        for a in ANOS:
            w(f"- {a}: {linhas_por_ano.get(a,0)}  "
              f"(esperado {ESPERADO_LINHAS[a]})  "
              f"{'OK' if linhas_por_ano.get(a,0)==ESPERADO_LINHAS[a] else 'DIVERGENTE'}\n")
        total_lido = sum(linhas_por_ano.values())
        w(f"- **Total lido:** {total_lido}  "
          f"(esperado {ESPERADO_TOTAL})  "
          f"{'OK' if total_lido==ESPERADO_TOTAL else 'DIVERGENTE'}\n\n")

        w("## Linhas no Parquet\n\n")
        w(f"- **Total:** {total_linhas}  "
          f"{'OK' if total_linhas==ESPERADO_TOTAL else 'DIVERGENTE'}\n\n")

        w("## DP / GC estruturalmente ausentes\n\n")
        w(f"- Linhas com `scout_DP` NaN (coluna ausente na rodada): {linhas_dp_ausente}\n")
        w(f"- Linhas com `scout_GC` NaN (coluna ausente na rodada): {linhas_gc_ausente}\n")
        w(f"- NaN em `scout_DP` (todas as causas): {linhas_dp_nan_total}\n")
        w(f"- Todas as linhas de rodadas sem coluna DP estao NaN: {dp_nan_ok}\n")
        w(f"- Distribuicao de NaN em scout_GC confere: {gc_nan_ok}\n\n")
        w("Rodadas em que a coluna **nao existia** no RAW:\n\n")
        w(f"- DP: { {a: sorted(sem_dp_por_ano.get(a, [])) for a in ANOS} }\n")
        w(f"- GC: { {a: sorted(sem_gc_por_ano.get(a, [])) for a in ANOS} }\n\n")

        w("## Distribuicao de entrou_em_campo\n\n")
        for k, v in dist_entrou.items():
            w(f"- {k}: {v}\n")
        w("\n")

        w("## Atletas distintos por temporada\n\n")
        for a in ANOS:
            w(f"- {a}: {atletas_por_temporada[a]}\n")
        w("\n")

        w("## Chaves duplicadas (temporada, rodada, atleta_id)\n\n")
        w(f"- Duplicatas: {dup}  {'OK' if dup==0 else 'PROBLEMA'}\n\n")

        w("## pontos_num nulos\n\n")
        w(f"- {nulos_pontos}  {'OK' if nulos_pontos==0 else 'PROBLEMA'}\n\n")

        w("## Rodadas presentes por temporada\n\n")
        for a in ANOS:
            rs = rodadas_por_temporada[a]
            w(f"- {a}: {len(rs)} rodadas (min={min(rs)}, max={max(rs)})  "
              f"{'OK' if set(rs)==ESPERADO_RODADAS[a] else 'DIVERGENTE'}\n")
        w(f"\n- Todas as rodadas esperadas presentes: {rodadas_ok}\n\n")

        w("## Colunas finais do canonico\n\n")
        for c in colunas_finais:
            w(f"- {c}\n")
        w("\n")

        w("## Amostra de 5 linhas\n\n")
        amostra = dados.sample(n=min(5, len(dados)), random_state=42)
        w("```\n")
        w(amostra.to_string())
        w("\n```\n\n")

        w("## Integridade dos RAWs\n\n")
        w(f"- RAWs inalterados (SHA-256 antes == depois): **{raws_inalterados}**\n")
        if not raws_inalterados:
            for k in hashes_antes:
                if hashes_antes[k] != hashes_depois[k]:
                    w(f"  - ALTERADO: {k}\n")
        w("\n")

        w("## Validacao do Parquet\n\n")
        w(f"- Existe: {PARQUET.exists()}\n")
        w(f"- Reabre com pandas/pyarrow: {reaberto_ok}\n")

    # ---------- SAIDA NO TERMINAL ----------
    print("=== RESUMO ===")
    print(f"Parquet: {PARQUET}")
    print(f"Total linhas no Parquet: {total_linhas}  (esperado {ESPERADO_TOTAL})")
    print(f"Chaves duplicadas: {dup}")
    print(f"pontos_num nulos: {nulos_pontos}")
    print(f"Linhas com scout_DP NaN (ausencia estrutural): {linhas_dp_ausente}")
    print(f"Linhas com scout_GC NaN (ausencia estrutural): {linhas_gc_ausente}")
    print(f"Distribuicao entrou_em_campo: {dist_entrou}")
    print(f"Atletas distintos por temporada: {atletas_por_temporada}")
    print(f"Rodadas OK: {rodadas_ok}")
    print(f"DP: NaN total={linhas_dp_nan_total}, esperado estrutural={linhas_dp_ausente}, contem esperado={dp_nan_ok}")
    print(f"GC: NaN total={linhas_gc_nan_total}, esperado estrutural={linhas_gc_ausente}, contem esperado={gc_nan_ok}")
    print(f"RAWs inalterados: {raws_inalterados}")
    print(f"Parquet reabre OK: {reaberto_ok}")
    print(f"Relatorio: {RELATORIO}")

    # falha dura se algo essencial quebrou
    if not raws_inalterados:
        print("ERRO: RAW alterado. Abortando.", file=sys.stderr)
        sys.exit(2)
    if total_linhas != ESPERADO_TOTAL:
        print("ERRO: total de linhas divergente.", file=sys.stderr)
        sys.exit(3)
    if dup != 0:
        print("ERRO: chave duplicada.", file=sys.stderr)
        sys.exit(4)
    if nulos_pontos != 0:
        print("ERRO: pontos_num com nulos.", file=sys.stderr)
        sys.exit(5)
    if not rodadas_ok:
        print("ERRO: rodadas divergentes.", file=sys.stderr)
        sys.exit(6)
    if not (dp_nan_ok and gc_nan_ok):
        print("ERRO: checagem de NaN em DP/GC falhou (rodadas sem coluna nao estao todas NaN).", file=sys.stderr)
        sys.exit(7)

if __name__ == "__main__":
    main()
