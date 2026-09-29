#!/usr/bin/env python3
"""
P1.3 — Auditoria numerica 2025 (dtype, precisao, representacao, estrutura).

Correcoes em relacao a versao anterior:
  - Comparacao de CHAVES RAW x canonico (por rodada)
  - Contagem RAW x canonico como criterio
  - Distincao float64 por NaN vs perda real de tipo
  - Categorizacao da representacao de cada celula bruta
  - FAIL em mismatch estrutural

Nao altera o Parquet.
"""
import hashlib, re, sys
from collections import Counter
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
RAW25 = ROOT/"data"/"raw"/"caRtola"/"2025"
OUT_MD = ROOT/"reports"/"p1_3_numeric_2025_audit.md"
OUT_KEYS = ROOT/"reports"/"p1_3_numeric_2025_keys.csv"
OUT_REPR = ROOT/"reports"/"p1_3_numeric_2025_repr.csv"
OUT_DIFFS = ROOT/"reports"/"p1_3_numeric_2025_diffs.csv"

HASH_ESPERADO = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"

COLS = [
    ("atleta_id",     "atletas.atleta_id",     "int"),
    ("rodada",        "atletas.rodada_id",     "int"),
    ("clube_id",      "atletas.clube_id",      "int"),
    ("posicao_id",    "atletas.posicao_id",    "int"),
    ("status_id",     "atletas.status_id",     "int"),
    ("jogos_num",     "atletas.jogos_num",     "int"),
    ("pontos_num",    "atletas.pontos_num",    "float"),
    ("media_num",     "atletas.media_num",     "float"),
    ("preco_num",     "atletas.preco_num",     "float"),
    ("variacao_num",  "atletas.variacao_num",  "float"),
]

RE_INT = re.compile(r"^-?\d+$")
RE_DEC = re.compile(r"^-?\d+\.\d+$")
RE_SCI = re.compile(r"^-?\d+(\.\d+)?[eE][-+]?\d+$")
RE_FLOAT = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
NULLS = {"nan","na","null","none","#n/a","#value!","#div/0!"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def rodada_from_name(name):
    m = re.search(r"rodada-(\d+)", name)
    return int(m.group(1)) if m else None


def categoriza(v):
    if v is None: return "EMPTY"
    t = str(v).strip()
    if t == "": return "EMPTY"
    if t.lower() in NULLS: return "NULL_LITERAL"
    if RE_INT.match(t): return "INTEGER"
    if RE_DEC.match(t): return "FLOAT_DECIMAL"
    if RE_SCI.match(t): return "SCIENTIFIC"
    if RE_FLOAT.match(t): return "FLOAT_OTHER"
    return "NON_NUMERIC"


def main():
    inicio = datetime.now()
    h_before = sha256(CANON)
    if h_before != HASH_ESPERADO:
        print("ERRO: hash do parquet nao bate", file=sys.stderr); sys.exit(2)

    canon = pd.read_parquet(CANON)
    canon_2025 = canon[canon["temporada"] == 2025].copy()

    files = sorted(RAW25.glob("*.csv"))
    L = []
    L.append("# P1.3 — Auditoria numerica 2025\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Arquivos RAW: {len(files)}\n\n")

    # ======================== 1. CHAVES RAW x CANON ========================
    key_rows = []
    key_issues = []
    for f in files:
        r = rodada_from_name(f.name)
        if r is None: continue
        raw = pd.read_csv(f, usecols=["atletas.atleta_id"], low_memory=False)
        raw_keys = set(int(x) for x in raw["atletas.atleta_id"].dropna().unique())
        can_keys = set(int(x) for x in
                       canon_2025[canon_2025["rodada"] == r]["atleta_id"].dropna().unique())
        only_raw = raw_keys - can_keys
        only_can = can_keys - raw_keys
        key_rows.append({
            "rodada": r,
            "n_raw_keys": len(raw_keys),
            "n_can_keys": len(can_keys),
            "only_in_raw": len(only_raw),
            "only_in_canon": len(only_can),
            "exemplos_only_raw": ",".join(map(str, sorted(only_raw)[:5])),
            "exemplos_only_canon": ",".join(map(str, sorted(only_can)[:5])),
        })
        if only_raw or only_can:
            key_issues.append({"rodada": r, "only_raw": len(only_raw),
                               "only_can": len(only_can)})

    df_keys = pd.DataFrame(key_rows)
    df_keys.to_csv(OUT_KEYS, index=False)

    L.append("## 1. Comparacao de chaves (rodada, atleta_id) RAW x canonico\n\n")
    L.append("| rodada | n_raw | n_can | only_raw | only_can |\n|---|---|---|---|---|\n")
    for _, r in df_keys.iterrows():
        flag = ""
        if r["only_in_raw"] or r["only_in_canon"]: flag = " ***"
        L.append(f"| {int(r['rodada'])} | {int(r['n_raw_keys'])} | {int(r['n_can_keys'])} | "
                 f"{int(r['only_in_raw'])} | {int(r['only_in_canon'])}{flag} |\n")
    L.append("\n")
    if key_issues:
        L.append(f"**{len(key_issues)} rodadas com divergencia de chaves.**\n\n")
    else:
        L.append("Nenhuma divergencia de chaves.\n\n")

    # ======================== 2. DTYPES ========================
    dtype_rows = []
    float_by_nan_rounds = []
    true_type_loss = []
    for f in files:
        r = rodada_from_name(f.name)
        if r is None: continue
        raw_inf = pd.read_csv(f, low_memory=False)
        raw_str = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[])
        for c_can, c_raw, tipo in COLS:
            if c_raw not in raw_inf.columns: continue
            dtype_inf = str(raw_inf[c_raw].dtype)
            has_nan = bool(raw_inf[c_raw].isna().any())
            nonempty = raw_str[c_raw][raw_str[c_raw] != ""]
            vals = nonempty.str.strip()
            is_all_int_like = bool(len(vals) and vals.str.match(RE_INT).all())
            has_non_num = bool(len(vals) and vals.str.lower().isin(NULLS).any())

            # Distinguir float64 causado por NaN
            if tipo == "int" and dtype_inf.startswith("float") and is_all_int_like:
                if has_nan:
                    nota = "float_by_nan"
                    float_by_nan_rounds.append((r, c_can))
                else:
                    nota = "int_like_como_float_SEM_NAN"
                    true_type_loss.append((r, c_can))
            elif has_non_num:
                nota = "tem_valor_nao_numerico"
            else:
                nota = ""

            dtype_rows.append({
                "rodada": r, "coluna_canon": c_can, "coluna_raw": c_raw,
                "raw_dtype_inferido": dtype_inf,
                "canon_dtype": str(canon[c_can].dtype),
                "tipo_esperado": tipo,
                "tem_nan": has_nan,
                "todos_int_like": is_all_int_like,
                "tem_nao_numerico": has_non_num,
                "nota": nota,
            })
    df_dtypes = pd.DataFrame(dtype_rows)

    L.append("## 2. Dtypes RAW (inferido) vs canonico\n\n")
    L.append(f"- colunas auditadas: {len(df_dtypes)}\n")
    L.append(f"- float_by_nan (esperado, nao e' problema): {len(float_by_nan_rounds)}\n")
    L.append(f"- perda real de tipo (float sem NaN): {len(true_type_loss)}\n\n")

    if float_by_nan_rounds:
        L.append("### float_by_nan (exemplos)\n\n")
        for r, c in float_by_nan_rounds[:10]:
            L.append(f"- rodada {r} / {c}\n")
        L.append("\n")
    if true_type_loss:
        L.append("### perda real de tipo (float sem NaN)\n\n")
        for r, c in true_type_loss[:10]:
            L.append(f"- rodada {r} / {c}\n")
        L.append("\n")

    # ======================== 3. REPRESENTACAO ========================
    repr_rows = []
    for f in files:
        r = rodada_from_name(f.name)
        if r is None: continue
        raw_str = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[])
        for c_can, c_raw, tipo in COLS:
            if c_raw not in raw_str.columns: continue
            cats = Counter(categoriza(v) for v in raw_str[c_raw])
            row = {"rodada": r, "coluna": c_can}
            for cat in ["INTEGER","FLOAT_DECIMAL","SCIENTIFIC","FLOAT_OTHER",
                        "EMPTY","NULL_LITERAL","NON_NUMERIC"]:
                row[cat] = int(cats.get(cat, 0))
            repr_rows.append(row)
    df_repr = pd.DataFrame(repr_rows)
    df_repr.to_csv(OUT_REPR, index=False)

    L.append("## 3. Categorizacao da representacao bruta (por rodada x coluna)\n\n")
    if len(df_repr):
        agg = df_repr.groupby("coluna")[
            ["INTEGER","FLOAT_DECIMAL","SCIENTIFIC","FLOAT_OTHER",
             "EMPTY","NULL_LITERAL","NON_NUMERIC"]].sum()
        L.append(agg.to_string())
        L.append("\n\n")
    else:
        L.append("Sem dados.\n\n")

    # ======================== 4. DUPLICATAS ========================
    L.append("## 4. Duplicatas (rodada, atleta_id) no RAW\n\n")
    dup_total = 0
    for f in files:
        r = rodada_from_name(f.name)
        raw = pd.read_csv(f, usecols=["atletas.atleta_id"], low_memory=False)
        d = int(raw["atletas.atleta_id"].duplicated().sum())
        if d > 0:
            dup_total += d
            L.append(f"- rodada {r}: {d}\n")
    if dup_total == 0: L.append("Nenhuma.\n")
    L.append("\n")

    # ======================== 5. COMPARACAO NUMERICA ========================
    L.append("## 5. Comparacao numerica por coluna\n\n")
    L.append("| coluna | tipo | comparadas | diffs | max_abs |\n|---|---|---|---|---|\n")

    diff_rows = []
    resumo = {}
    count_mismatch_rounds = []
    for f in files:
        r = rodada_from_name(f.name)
        if r is None: continue
        n_raw = sum(1 for _ in open(f)) - 1
        n_can = int((canon_2025["rodada"] == r).sum())
        if n_raw != n_can:
            count_mismatch_rounds.append({"rodada": r, "n_raw": n_raw, "n_can": n_can})

    for c_can, c_raw, tipo in COLS:
        if c_raw == "atletas.atleta_id":
            resumo[c_can] = {"tipo": tipo, "comparadas": 0, "diffs": 0, "max_abs": 0.0}
            L.append(f"| {c_can} | {tipo} | (index) | 0 | 0.00e+00 |\n")
            continue
        comparadas = 0; diffs = 0; max_abs = 0.0
        for f in files:
            r = rodada_from_name(f.name)
            raw = pd.read_csv(f, low_memory=False)
            if c_raw not in raw.columns: continue
            raw = raw.set_index("atletas.atleta_id")
            can_r = canon_2025[canon_2025["rodada"] == r]
            for _, row in can_r.iterrows():
                aid = int(row["atleta_id"])
                if aid not in raw.index: continue
                v_raw_s = raw.loc[aid, c_raw]
                v_raw = v_raw_s.iloc[0] if isinstance(v_raw_s, pd.Series) else v_raw_s
                v_can = row[c_can]
                if pd.isna(v_raw) and pd.isna(v_can): continue
                if pd.isna(v_raw) or pd.isna(v_can):
                    diffs += 1; continue
                comparadas += 1
                try:
                    if tipo == "int":
                        a = int(float(v_raw)); b = int(float(v_can))
                        if a != b:
                            d = abs(a - b)
                            diffs += 1
                            diff_rows.append({"rodada": r, "atleta_id": aid,
                                              "coluna": c_can, "raw": a, "canon": b,
                                              "diff_abs": d})
                            max_abs = max(max_abs, d)
                    else:
                        a = float(v_raw); b = float(v_can)
                        d = abs(a - b)
                        max_abs = max(max_abs, d)
                        if d > max(1e-9, 1e-9 * abs(a)):
                            diffs += 1
                            diff_rows.append({"rodada": r, "atleta_id": aid,
                                              "coluna": c_can, "raw": a, "canon": b,
                                              "diff_abs": d})
                except Exception:
                    diffs += 1
        resumo[c_can] = {"tipo": tipo, "comparadas": comparadas,
                         "diffs": diffs, "max_abs": max_abs}
        L.append(f"| {c_can} | {tipo} | {comparadas} | {diffs} | {max_abs:.2e} |\n")
    L.append("\n")

    L.append("## 6. Divergencias de contagem por rodada\n\n")
    if count_mismatch_rounds:
        L.append("| rodada | RAW | CANON |\n|---|---|---|\n")
        for x in count_mismatch_rounds:
            L.append(f"| {x['rodada']} | {x['n_raw']} | {x['n_can']} |\n")
    else:
        L.append("Nenhuma.\n")
    L.append("\n")

    pd.DataFrame(diff_rows).to_csv(OUT_DIFFS, index=False)

    L.append("## 7. Top deltas concretos por coluna\n\n")
    if diff_rows:
        ddf = pd.DataFrame(diff_rows)
        for c in sorted(ddf["coluna"].unique()):
            sub = ddf[ddf["coluna"] == c].nlargest(5, "diff_abs")
            L.append(f"### {c}\n\n")
            L.append(sub.to_string(index=False))
            L.append("\n\n")
    else:
        L.append("Nenhum delta acima de tolerancia.\n\n")

    # ======================== 8. CLASSIFICACAO ========================
    L.append("## 8. Classificacao\n\n")
    crit_fail = []
    crit_investigar = []
    if key_issues:
        crit_fail.append(f"divergencia de chaves em {len(key_issues)} rodadas")
    if count_mismatch_rounds:
        crit_fail.append(f"divergencia de contagem em {len(count_mismatch_rounds)} rodadas")
    if dup_total > 0:
        crit_fail.append(f"{dup_total} duplicatas no RAW")
    total_diffs = sum(v["diffs"] for v in resumo.values())
    if total_diffs > 0 and not crit_fail:
        crit_investigar.append(f"{total_diffs} diffs numericos")
    if true_type_loss:
        crit_investigar.append(f"{len(true_type_loss)} perdas reais de tipo")
    # float_by_nan nao conta como problema
    if true_type_loss or total_diffs > 0 or key_issues or count_mismatch_rounds or dup_total > 0:
        if crit_fail:
            L.append(f"**FAIL** — criterios estruturais violados:\n")
            for c in crit_fail: L.append(f"  - {c}\n")
            if crit_investigar:
                L.append(f"- e também:\n")
                for c in crit_investigar: L.append(f"  - {c}\n")
        else:
            L.append(f"**INVESTIGAR** — sem falha estrutural, mas:\n")
            for c in crit_investigar: L.append(f"  - {c}\n")
    else:
        L.append("**PASS** — sem divergencia estrutural nem numerica.\n")
        if float_by_nan_rounds:
            L.append(f"- nota: {len(float_by_nan_rounds)} casos `float_by_nan` "
                     f"foram classificados como esperados (NaN faz pandas inferir float).\n")

    h_after = sha256(CANON)
    L.append(f"\n## Integridade\n\n- Parquet inalterado: {h_before == h_after}\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    print(f"arquivos RAW: {len(files)}")
    print(f"key_issues: {len(key_issues)}   count_mismatch: {len(count_mismatch_rounds)}")
    print(f"duplicatas RAW: {dup_total}   true_type_loss: {len(true_type_loss)}   "
          f"float_by_nan: {len(float_by_nan_rounds)}")
    print("diffs por coluna:")
    for c, v in resumo.items():
        print(f"  {c:15s} tipo={v['tipo']:5s} comparadas={v['comparadas']:6d} "
              f"diffs={v['diffs']}")
    print("artefatos: reports/p1_3_numeric_2025_audit.md, _keys.csv, _repr.csv, _diffs.csv")


if __name__ == "__main__":
    main()
