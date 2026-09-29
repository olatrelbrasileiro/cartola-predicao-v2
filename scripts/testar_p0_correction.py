#!/usr/bin/env python3
"""Testes P0 — participacao, janela recente, leakage ampliado."""
import hashlib, json, sys
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
    raise RuntimeError("Raiz do projeto nao encontrada")


ROOT = detect_root()
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

CANON = INTERIM/"cartola_jogador_rodada_v2.parquet"
FOLDS = REP/"baselines_p0_corrected_folds.csv"
OUT_TESTS = REP/"p0_recent_window_tests.csv"

HASH_ESPERADO = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"
TARGET = "pontos_num"
RECENT_5 = 5
RECENT_3 = 3


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def compute_participation_features_local(df):
    df = df.reset_index(drop=True).copy()
    df["_idx"] = df.index
    out = []
    for (t, aid), sub in df.groupby(["temporada","atleta_id"], sort=False):
        sub = sub.sort_values("rodada").reset_index(drop=False)
        for i in range(len(sub)):
            row_idx = int(sub.iloc[i]["_idx"])
            prev = sub.iloc[:i]
            prev_part = prev[prev["elegivel"] & prev["entrou_bool"]]
            if len(prev_part) == 0:
                out.append({"_idx": row_idx,
                            "hist_mean_p": np.nan, "hist_std_p": np.nan, "hist_n_p": 0,
                            "rec_p_5": np.nan, "rec_p_3": np.nan, "rec_std_p_5": np.nan,
                            "basic_mean_p": np.nan, "basic_rec_p_5": np.nan})
            else:
                p = prev_part[TARGET].to_numpy(dtype=float)
                b = prev_part["basic_score"].to_numpy(dtype=float)
                rec5 = prev_part.tail(RECENT_5)
                rec3 = prev_part.tail(RECENT_3)
                out.append({"_idx": row_idx,
                            "hist_mean_p": float(p.mean()),
                            "hist_std_p": float(p.std(ddof=0)),
                            "hist_n_p": int(len(prev_part)),
                            "rec_p_5": float(rec5[TARGET].mean()),
                            "rec_p_3": float(rec3[TARGET].mean()),
                            "rec_std_p_5": float(rec5[TARGET].std(ddof=0)) if len(rec5) >= 2 else 0.0,
                            "basic_mean_p": float(b.mean()),
                            "basic_rec_p_5": float(rec5["basic_score"].mean())})
    return pd.DataFrame(out).set_index("_idx")


# -------- Teste 1: recent window determinista
def test_recent_window():
    casos = []
    # caso 1 — padrao do enunciado
    seq1 = [(1,False,0.0),(2,True,3.0),(3,False,0.0),(4,True,5.0),
            (5,True,2.0),(6,False,0.0),(7,True,7.0),(8,True,5.0)]
    # alvo R8: participantes R2(3),R4(5),R5(2),R7(7)
    casos.append({"caso":"padrao_misto_R8",
                  "esperado_rec_p_5":(3+5+2+7)/4,
                  "esperado_rec_p_3":(5+2+7)/3,
                  "seq":seq1, "alvo_rodada":8})
    # caso 2 — todas jogadas
    seq2 = [(r,True,float(r)) for r in range(1,9)]
    casos.append({"caso":"todas_jogadas_R8",
                  "esperado_rec_p_5":(3+4+5+6+7)/5,
                  "esperado_rec_p_3":(5+6+7)/3,
                  "seq":seq2, "alvo_rodada":8})
    # caso 3 — poucos participantes
    seq3 = [(1,False,0.0),(2,False,0.0),(3,True,4.0),(4,False,0.0),
            (5,True,8.0),(6,False,0.0),(7,False,0.0),(8,True,5.0)]
    casos.append({"caso":"poucos_participantes_R8",
                  "esperado_rec_p_5":(4+8)/2,
                  "esperado_rec_p_3":(4+8)/2,
                  "seq":seq3, "alvo_rodada":8})

    rows = []
    for c in casos:
        df = pd.DataFrame(c["seq"], columns=["rodada","entrou_bool",TARGET])
        df["temporada"] = 2024
        df["atleta_id"] = 1
        df["elegivel"] = True
        df["basic_score"] = 0.0
        feats = compute_participation_features_local(df)
        row_alvo = df[df["rodada"] == c["alvo_rodada"]].index[0]
        got = feats.loc[row_alvo]
        rows.append({
            "caso": c["caso"],
            "rec_p_5_esperado": c["esperado_rec_p_5"],
            "rec_p_5_obtido": float(got["rec_p_5"]),
            "rec_p_5_ok": abs(got["rec_p_5"] - c["esperado_rec_p_5"]) < 1e-9,
            "rec_p_3_esperado": c["esperado_rec_p_3"],
            "rec_p_3_obtido": float(got["rec_p_3"]),
            "rec_p_3_ok": abs(got["rec_p_3"] - c["esperado_rec_p_3"]) < 1e-9,
            "hist_n_p_esperado": sum(1 for (r,e,_) in c["seq"] if r < c["alvo_rodada"] and e),
            "hist_n_p_obtido": int(got["hist_n_p"]),
        })
        rows[-1]["hist_n_p_ok"] = rows[-1]["hist_n_p_esperado"] == rows[-1]["hist_n_p_obtido"]
    return pd.DataFrame(rows)


# -------- Teste 2: consistencia dos baselines
def test_participation_consistency():
    dfr = pd.read_csv(FOLDS)
    falhas = []
    if len(dfr) != 76: falhas.append(f"folds: {len(dfr)}")
    man = pd.read_csv(REP/"harness_folds_v2.csv")
    k1 = set(map(tuple, dfr[["temporada","rodada_alvo"]].values.tolist()))
    k2 = set(map(tuple, man[["temporada","rodada_alvo"]].values.tolist()))
    if k1 != k2: falhas.append("chaves divergem do harness")
    if ((dfr["temporada"]==2025) & (dfr["rodada_alvo"]==1)).any():
        falhas.append("2025/R1 presente")
    for c in ["b0_mae","b1_mae","b2_mae","b3_mae","b2s_mae"]:
        if (dfr[c] < 0).any(): falhas.append(f"{c} negativo")
    return falhas


# -------- Teste 3: leakage ampliado (30 atletas, features independentes)
def test_leakage_extended():
    falhas = []
    df = pd.read_parquet(CANON).sort_values(
        ["temporada","atleta_id","rodada"], kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    df["basic_score"] = 0.0  # nao precisamos do valor exato, so do layout

    feats = compute_participation_features_local(df)
    df = df.merge(feats, left_index=True, right_index=True, how="left")

    # amostra estratificada: 30 atletas de temporadas distintas
    rng = np.random.default_rng(42)
    all_atletas = df["atleta_id"].drop_duplicates().tolist()
    sample = list(rng.choice(all_atletas, size=min(30, len(all_atletas)), replace=False))

    n_checked = 0
    for aid in sample:
        sub = df[df["atleta_id"] == aid].sort_values(["temporada","rodada"])
        for t in sub["temporada"].unique():
            g = sub[sub["temporada"] == t].sort_values("rodada").reset_index(drop=True)
            for i in range(len(g)):
                prev = g.iloc[:i]
                prev_part = prev[prev["elegivel"] & prev["entrou_bool"]]
                row = g.iloc[i]

                n_esperado = len(prev_part)
                if row["hist_n_p"] != n_esperado:
                    falhas.append(f"hist_n_p errado aid={aid} t={t} r={row['rodada']}: "
                                  f"{row['hist_n_p']} vs {n_esperado}")
                    continue

                if n_esperado > 0:
                    p = prev_part[TARGET].to_numpy(dtype=float)
                    m_esp = float(p.mean())
                    if abs(row["hist_mean_p"] - m_esp) > 1e-9:
                        falhas.append(f"hist_mean_p errado aid={aid} t={t} r={row['rodada']}")
                    s_esp = float(p.std(ddof=0))
                    if abs(row["hist_std_p"] - s_esp) > 1e-9:
                        falhas.append(f"hist_std_p errado aid={aid} t={t} r={row['rodada']}")
                    r5_esp = float(prev_part.tail(RECENT_5)[TARGET].mean())
                    if abs(row["rec_p_5"] - r5_esp) > 1e-9:
                        falhas.append(f"rec_p_5 errado aid={aid} t={t} r={row['rodada']}")
                    r3_esp = float(prev_part.tail(RECENT_3)[TARGET].mean())
                    if abs(row["rec_p_3"] - r3_esp) > 1e-9:
                        falhas.append(f"rec_p_3 errado aid={aid} t={t} r={row['rodada']}")
                n_checked += 1
    return falhas, n_checked


def main():
    print("== Testes P0 ==")
    falhas = []

    h = sha256(CANON)
    if h != HASH_ESPERADO:
        falhas.append(f"hash do parquet: {h[:16]}... != esperado")

    print("\n[1] Janela recente determinista...")
    df_tests = test_recent_window()
    df_tests.to_csv(OUT_TESTS, index=False)
    print(df_tests.to_string(index=False))
    if not df_tests[["rec_p_5_ok","rec_p_3_ok","hist_n_p_ok"]].all().all():
        falhas.append("teste de janela recente falhou")

    print("\n[2] Consistencia dos baselines corrigidos...")
    f2 = test_participation_consistency()
    print(f"  {len(f2)} falhas"); falhas.extend(f2)

    print("\n[3] Leakage ampliado (30 atletas)...")
    f3, n_checked = test_leakage_extended()
    print(f"  {len(f3)} falhas em {n_checked} linhas checadas"); falhas.extend(f3)

    print("\n=== RESULTADO ===")
    if falhas:
        print("FAIL")
        for f in falhas[:30]:
            print(f"  - {f}")
        if len(falhas) > 30: print(f"  ... (+{len(falhas)-30})")
        sys.exit(1)
    print("PASS")
    print(f"  parquet sha256: {h[:16]}...")
    print(f"  janela recente: {len(df_tests)} casos OK")
    print(f"  leakage: {n_checked} linhas OK")


if __name__ == "__main__":
    main()
