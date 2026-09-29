#!/usr/bin/env python3
"""
ETAPA B — Estudo isolado do hiperparametro k do shrinkage historico.

Preditor puro:
    y_hat_k = (sum_pontos_anteriores + k * prior_pos) / (n_anteriores + k)
    fallback B0 (media dos participantes no treino) se n_anteriores == 0.

Nao usa G+A, alpha, ou outras features. Unico fator: k.
Nao altera producao, benchmark ou dataset canonico.
"""
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
    raise RuntimeError("raiz nao encontrada")


ROOT = detect_root()
INTERIM = ROOT/"data"/"interim"
REP = ROOT/"reports"

CANON      = INTERIM/"cartola_jogador_rodada_v2.parquet"
HARNESS    = REP/"harness_folds_v2.csv"
B2S_FOLDS  = REP/"baseline_b2s_folds.csv"
P0_FOLDS   = REP/"baselines_p0_corrected_folds.csv"

OUT_FOLDS    = REP/"etapa_b_k_folds.csv"
OUT_BY_SEAS  = REP/"etapa_b_k_by_season.csv"
OUT_SUMMARY  = REP/"etapa_b_k_summary.md"
OUT_SELECT   = REP/"etapa_b_k_selection.json"
OUT_VALID    = REP/"etapa_b_k_validation.json"

TARGET = "pontos_num"
K_GRADE = [0, 3, 5, 7, 10, 14, 20, 30]
K_REF = 7
TOL_MAE = 1e-4
MIN_FOLDS_MELHOR = 46
PIORA_MAX_PCT = 2.0

HASHES_ESPERADOS = {
    "canonical": "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4",
    "b2s_folds": "b56c6db440d7628acbbc52bb477e1b4fe81ce3fc87610e5aa0452e2110e6a142",
    "harness":   "236b7421ed83860d6921131d708b0508601a8be495ac35f175f91eaefca803a5",
    "p0_folds":  "8b4f61851c227d8ee8c2c76aa38f8c6389656bbb8d1109ee909ccce2d77bb689",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def mae(p, y): return float(np.mean(np.abs(p - y))) if len(y) else np.nan
def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan
def spearman(p, y):
    if len(p) < 2: return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    inicio = datetime.now()
    print("== ETAPA B — Estudo isolado de k ==")

    # ---------- integridade ANTES
    h_antes = {}
    for nome, p in [("canonical", CANON), ("b2s_folds", B2S_FOLDS),
                    ("harness", HARNESS), ("p0_folds", P0_FOLDS)]:
        h = sha256(p)
        h_antes[nome] = h
        ok = "OK" if h == HASHES_ESPERADOS[nome] else "*** DIVERGE ***"
        print(f"  {nome:12s} {h[:16]}...  {ok}")
    if any(h_antes[k] != HASHES_ESPERADOS[k] for k in HASHES_ESPERADOS):
        print("ERRO: hash divergente. Abortando.", file=sys.stderr); sys.exit(2)

    # ---------- dados
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada","atleta_id","rodada"],
                        kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)

    mask = df["elegivel"] & df["entrou_bool"]
    df["_p"] = np.where(mask, df[TARGET], 0.0)
    df["_n"] = mask.astype(int)
    g = df.groupby(["temporada","atleta_id"], sort=False)
    df["_n_prev"]   = g["_n"].cumsum() - df["_n"]
    df["_sum_prev"] = g["_p"].cumsum() - df["_p"]
    df = df.drop(columns=["_p","_n"])

    man = pd.read_csv(HARNESS)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")

    # ---------- loop
    rows = []
    for idx, (t, R) in enumerate(folds):
        mask_t = df["temporada"] == t
        train_all = df[mask_t & (df["rodada"] < R) & df["elegivel"]]
        train_part = train_all[train_all["entrou_bool"]]
        test_all = df[mask_t & (df["rodada"] == R) & df["elegivel"]]
        test = test_all[test_all["entrou_bool"]].copy()
        if len(test) == 0 or len(train_part) == 0:
            continue

        b0_const = float(train_part[TARGET].mean())
        prior_pos = train_part.groupby("posicao_id")[TARGET].mean().to_dict()

        y = test[TARGET].to_numpy(dtype=float)
        n_prev = test["_n_prev"].to_numpy(dtype=float)
        sum_prev = test["_sum_prev"].to_numpy(dtype=float)
        pos_arr = test["posicao_id"].to_numpy()
        prior_arr = np.array([prior_pos.get(int(p), b0_const) for p in pos_arr],
                             dtype=float)

        n_hist = int((n_prev > 0).sum())
        n_cold = int((n_prev == 0).sum())
        n_hist_medio = float(n_prev[n_prev > 0].mean()) if n_hist > 0 else np.nan

        for k in K_GRADE:
            pred = np.where(
                n_prev > 0,
                (sum_prev + k * prior_arr) / (n_prev + k),
                b0_const,
            )
            rows.append({
                "temporada": t, "rodada_alvo": R, "k": k,
                "n_test": int(len(test)),
                "n_hist": n_hist,
                "n_cold": n_cold,
                "n_hist_medio": n_hist_medio,
                "b0_const": b0_const,
                "mae": mae(pred, y),
                "rmse": rmse(pred, y),
                "spearman": spearman(pred, y),
            })

        if (idx + 1) % 10 == 0:
            print(f"  fold {idx+1}/{len(folds)}  {t}/R{R}")

    dfr = pd.DataFrame(rows)
    dfr.to_csv(OUT_FOLDS, index=False)

    # ---------- agregacao por temporada
    def wavg(sub, col, ncol="n_test"):
        n = sub[ncol].sum()
        return float((sub[col] * sub[ncol]).sum() / n) if n else np.nan

    linhas_seas = []
    for (k, t), sub in dfr.groupby(["k","temporada"]):
        linhas_seas.append({
            "k": k, "temporada": t,
            "n_test": int(sub["n_test"].sum()),
            "mae_pond": wavg(sub, "mae"),
            "rmse_pond": float(np.sqrt((sub["rmse"]**2 * sub["n_test"]).sum() / sub["n_test"].sum())),
            "spearman_medio": float(sub["spearman"].mean()),
        })
    df_seas = pd.DataFrame(linhas_seas).sort_values(["k","temporada"])
    df_seas.to_csv(OUT_BY_SEAS, index=False)

    # ---------- agregacao global por k
    def stats_k(sub):
        n = sub["n_test"].sum()
        return {
            "n_test": int(n),
            "mae_pond": float((sub["mae"] * sub["n_test"]).sum() / n),
            "rmse_pond": float(np.sqrt((sub["rmse"]**2 * sub["n_test"]).sum() / n)),
            "spearman_medio": float(sub["spearman"].mean()),
        }

    resumo_k = {k: stats_k(sub) for k, sub in dfr.groupby("k")}

    # por fold: delta_k = MAE_k - MAE_7
    piv = dfr.pivot_table(index=["temporada","rodada_alvo"], columns="k",
                          values="mae").reset_index()
    deltas = {}
    for k in K_GRADE:
        if k == K_REF: continue
        d = piv[k] - piv[K_REF]
        deltas[k] = d.dropna()

    # ---------- criterios
    candidatos = {}
    for k in K_GRADE:
        if k == K_REF: continue
        mae_k = resumo_k[k]["mae_pond"]
        mae_7 = resumo_k[K_REF]["mae_pond"]
        crit1 = (mae_k < mae_7 - TOL_MAE)

        d = deltas[k]
        n_melhor = int((d < -TOL_MAE).sum())
        n_empate = int((d.abs() <= TOL_MAE).sum())
        n_pior = int(len(d) - n_melhor - n_empate)
        crit2 = n_melhor >= MIN_FOLDS_MELHOR

        seas_k = df_seas[df_seas["k"] == k].set_index("temporada")
        seas_7 = df_seas[df_seas["k"] == K_REF].set_index("temporada")
        temporadas = [2024, 2025, 2026]
        crit3 = all((seas_k.loc[t, "mae_pond"] < seas_7.loc[t, "mae_pond"] - TOL_MAE)
                    for t in temporadas)
        crit4 = all((seas_k.loc[t, "mae_pond"] <= seas_7.loc[t, "mae_pond"] * (1 + PIORA_MAX_PCT/100))
                    for t in temporadas)
        mediana_delta = float(d.median())
        crit5 = mediana_delta < 0

        elegivel = all([crit1, crit2, crit3, crit4, crit5])
        candidatos[k] = {
            "mae_pond": mae_k,
            "delta_mae_vs_k7": mae_k - mae_7,
            "spearman_medio": resumo_k[k]["spearman_medio"],
            "folds_melhor": n_melhor,
            "folds_empate": n_empate,
            "folds_pior": n_pior,
            "delta_media": float(d.mean()),
            "delta_mediana": mediana_delta,
            "delta_std": float(d.std()),
            "por_temporada": {int(t): float(seas_k.loc[t, "mae_pond"]) for t in temporadas},
            "crit1_mae_global": bool(crit1),
            "crit2_folds": bool(crit2),
            "crit3_temporadas": bool(crit3),
            "crit4_piora_max": bool(crit4),
            "crit5_mediana": bool(crit5),
            "elegivel": bool(elegivel),
        }

    # ---------- decisao
    elegiveis = [k for k, c in candidatos.items() if c["elegivel"]]
    if elegiveis:
        escolhido = min(elegiveis, key=lambda k: abs(k - K_REF))
        decisao = f"RECOMMEND_K_{escolhido}"
        motivo = f"candidato(s) elegivel(eis): {elegiveis}; escolhido por menor |k-7| = {escolhido}"
    else:
        escolhido = None
        decisao = "KEEP_K7"
        motivo = "nenhum candidato cumpriu todos os 5 criterios"

    # ---------- integridade DEPOIS
    h_depois = {nome: sha256(p) for nome, p in
                [("canonical", CANON), ("b2s_folds", B2S_FOLDS),
                 ("harness", HARNESS), ("p0_folds", P0_FOLDS)]}
    integridade_ok = all(h_antes[k] == h_depois[k] for k in h_antes)

    # ---------- summary
    fim = datetime.now()
    L = []
    L.append("# Etapa B — Estudo isolado de k (shrinkage historico)\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- Folds: {len(folds)}\n")
    L.append(f"- Grade: {K_GRADE}\n")
    L.append(f"- Referencia: k={K_REF}\n")
    L.append(f"- Preditor: hist_shrunk(k) puro, fallback B0 no cold-start\n")
    L.append(f"- Teste: somente participantes\n\n")

    L.append("## Tabela final\n\n")
    L.append("| k | MAE pond | Δ MAE vs k=7 | Spearman | folds melhor | 2024 | 2025 | 2026 |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for k in K_GRADE:
        if k == K_REF:
            m = resumo_k[k]["mae_pond"]; sp = resumo_k[k]["spearman_medio"]
            L.append(f"| **{k}** | **{m:.4f}** | 0 | {sp:.4f} | — | "
                     f"{df_seas[(df_seas.k==k)&(df_seas.temporada==2024)].mae_pond.iloc[0]:.4f} | "
                     f"{df_seas[(df_seas.k==k)&(df_seas.temporada==2025)].mae_pond.iloc[0]:.4f} | "
                     f"{df_seas[(df_seas.k==k)&(df_seas.temporada==2026)].mae_pond.iloc[0]:.4f} |\n")
        else:
            c = candidatos[k]
            L.append(f"| {k} | {c['mae_pond']:.4f} | {c['delta_mae_vs_k7']:+.4f} | "
                     f"{c['spearman_medio']:.4f} | {c['folds_melhor']}/76 | "
                     f"{c['por_temporada'][2024]:.4f} | {c['por_temporada'][2025]:.4f} | "
                     f"{c['por_temporada'][2026]:.4f} |\n")
    L.append("\n")

    L.append("## Aplicacao dos 5 criterios\n\n")
    L.append("| k | crit1 MAE | crit2 folds | crit3 temporadas | crit4 piora | crit5 mediana | elegivel |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for k in K_GRADE:
        if k == K_REF:
            L.append(f"| {k} | (referencia) | — | — | — | — | — |\n"); continue
        c = candidatos[k]
        L.append(f"| {k} | {c['crit1_mae_global']} | {c['crit2_folds']} "
                 f"({c['folds_melhor']}/76) | {c['crit3_temporadas']} | "
                 f"{c['crit4_piora_max']} | {c['crit5_mediana']} "
                 f"({c['delta_mediana']:+.4f}) | {c['elegivel']} |\n")
    L.append("\n")

    L.append("## Decisao\n\n")
    L.append(f"- **{decisao}**\n")
    L.append(f"- Motivo: {motivo}\n\n")
    L.append("- Recomendacao documental. Nenhum script de producao foi alterado.\n\n")

    L.append("## Notas descritivas\n\n")
    L.append("- k=0 e' o controle sem shrinkage (`hist_shrunk(0) = hist_mean`).\n")
    L.append("- k=30 esta fora da faixa Empirical-Bayes de P1.1 (6.22–20.59) e foi "
             "incluido deliberadamente como teste de extrapolacao/saturacao.\n")
    L.append("- O efeito de k tende a ser maior para `n` pequeno e desaparecer "
             "progressivamente conforme `n` cresce. Propriedade matematica do "
             "shrinkage, nao defeito do experimento.\n")

    L.append("## Integridade\n\n")
    for k in h_antes:
        same = h_antes[k] == h_depois[k]
        L.append(f"- {k}: {'OK' if same else 'ALTERADO'}\n")

    OUT_SUMMARY.write_text("".join(L), encoding="utf-8")

    # ---------- selection.json
    sel = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "grade_avaliada": K_GRADE,
        "k_referencia": K_REF,
        "preditor": "hist_shrunk(k) puro; fallback B0 em cold-start",
        "criterios": {
            "tolerancia_empate_mae": TOL_MAE,
            "min_folds_melhores": MIN_FOLDS_MELHOR,
            "temporadas_obrigatorias": [2024, 2025, 2026],
            "piora_maxima_temporada_pct": PIORA_MAX_PCT,
            "mediana_delta_max": 0.0,
        },
        "por_candidato": {str(k): v for k, v in candidatos.items()},
        "candidatos_elegiveis": elegiveis,
        "decisao": decisao,
        "motivo": motivo,
        "detalhes": {
            "k_referencia_mae_pond": resumo_k[K_REF]["mae_pond"],
            "k_referencia_spearman": resumo_k[K_REF]["spearman_medio"],
        },
    }
    with open(OUT_SELECT, "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=2, default=str)

    # ---------- validation.json
    val = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "folds_avaliados": len(piv),
        "grade": K_GRADE,
        "hashes_antes": h_antes,
        "hashes_depois": h_depois,
        "hashes_esperados": HASHES_ESPERADOS,
        "integridade_ok": integridade_ok,
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2, default=str)

    # ---------- console
    print()
    print("=== RESULTADO ===")
    print(f"{'k':>4}  {'MAE':>10}  {'Δ vs k=7':>10}  {'Spearman':>10}  {'folds<7':>8}")
    for k in K_GRADE:
        if k == K_REF:
            print(f"{k:>4}  {resumo_k[k]['mae_pond']:>10.4f}  {'(ref)':>10}  "
                  f"{resumo_k[k]['spearman_medio']:>10.4f}  {'—':>8}")
        else:
            c = candidatos[k]
            print(f"{k:>4}  {c['mae_pond']:>10.4f}  {c['delta_mae_vs_k7']:>+10.4f}  "
                  f"{c['spearman_medio']:>10.4f}  {c['folds_melhor']:>8}")
    print()
    print(f"DECISAO: {decisao}")
    print(f"  motivo: {motivo}")
    print(f"integridade: {integridade_ok}")
    print(f"artefatos:")
    for p in [OUT_FOLDS, OUT_BY_SEAS, OUT_SUMMARY, OUT_SELECT, OUT_VALID]:
        print(f"  {p.relative_to(ROOT)}")

    if not integridade_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
