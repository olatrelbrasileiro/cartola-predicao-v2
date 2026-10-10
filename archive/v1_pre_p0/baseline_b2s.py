#!/usr/bin/env python3
"""
B2S — B2 com shrinkage walk-forward.

Spec: config/b2s_spec.yaml
"""
import hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path.home()/"cartola-predicao-v2"
PARQUET = ROOT/"data"/"interim"/"cartola_jogador_rodada_v2.parquet"
CONFIG = ROOT/"config.yaml"
SPEC = ROOT/"config"/"b2s_spec.yaml"
MANIFEST = ROOT/"reports"/"harness_folds_v2.csv"
B2_FOLDS = ROOT/"reports"/"baseline_b2_folds.csv"
REP = ROOT/"reports"

CSV_FOLDS = REP/"baseline_b2s_folds.csv"
CSV_ALPHAS = REP/"baseline_b2s_alphas.csv"
SUMMARY = REP/"baseline_b2s_summary.md"
SPEC_MD = REP/"baseline_b2s_spec.md"

TARGET = "pontos_num"
WINDOW = 5


def sha256(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def load_yaml(p):
    with open(p,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cache(df):
    """Para cada (t, r): pred_b2 (com fallback), anchor (mediana), y, n.
    Nao filtra B ainda (filtro fica no fold)."""
    cache = {}
    for t in sorted(df["temporada"].unique()):
        sub_t = df[df["temporada"] == t]
        for r in sorted(sub_t["rodada"].unique()):
            train = sub_t[(sub_t["rodada"] < r) & sub_t["elegivel"]]
            if len(train) == 0:
                continue
            recent = (train.sort_values(["atleta_id","rodada"])
                      .groupby("atleta_id").tail(WINDOW)
                      .groupby("atleta_id")[TARGET].mean())
            mean_g = float(train[TARGET].mean())
            median_g = float(train[TARGET].median())
            test = sub_t[(sub_t["rodada"] == r) & sub_t["elegivel"]].copy()
            # filtro B ambigua apenas no TESTE (metrica principal)
            test = test[~((test["entrou_em_campo"] == False) & (test[TARGET] != 0))]
            if len(test) == 0:
                continue
            pred = test["atleta_id"].map(recent).fillna(mean_g).to_numpy(dtype=float)
            cache[(int(t), int(r))] = {
                "y": test[TARGET].to_numpy(dtype=float),
                "pred_b2": pred,
                "anchor": median_g,
                "mean_g": mean_g,
                "n": int(len(test)),
            }
    return cache


def mae(pred, y):
    return float(np.mean(np.abs(pred - y)))


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def spearman(pred, y):
    if len(pred) < 2: return np.nan
    if np.all(pred == pred[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(pred).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])


def calibrar_alpha(cache, t, R, alphas, min_inner):
    inner = [r for (tt, r) in cache.keys() if tt == t and min_inner <= r <= R-1]
    if not inner:
        return 1.0, {}
    means = {}
    for a in alphas:
        ms = []
        for ri in inner:
            c = cache[(t, ri)]
            p = a * c["pred_b2"] + (1 - a) * c["anchor"]
            ms.append(mae(p, c["y"]))
        means[a] = float(np.mean(ms))
    star = min(means, key=lambda a: means[a])
    return star, means


def main():
    inicio = datetime.now()
    print("== B2S — B2 + shrinkage walk-forward ==")

    h_pq_antes = sha256(PARQUET)
    h_b2_antes = sha256(B2_FOLDS)

    cfg = load_yaml(CONFIG)
    mtr = cfg["validation"]["min_train_rounds"]
    spec = load_yaml(SPEC)
    cal = spec["calibracao_alpha"]
    min_inner = int(cal["min_inner_rounds"])
    g = cal["grid"]
    alphas = [round(a, 2) for a in np.arange(g["inicio"], g["fim"] + 1e-9, g["passo"])]
    print(f"grid: {len(alphas)} alphas, min_inner={min_inner}")

    df = pd.read_parquet(PARQUET)
    df = df.sort_values(["temporada","rodada","atleta_id"]).reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))

    print("Pre-computando cache...")
    cache = build_cache(df)

    # folds = mesmo manifesto
    man = pd.read_csv(MANIFEST)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")

    b2_of = pd.read_csv(B2_FOLDS).set_index(["temporada","rodada_alvo"])

    rows = []
    alpha_rows = []
    for (t, R) in folds:
        c = cache[(t, R)]
        a_star, means_inner = calibrar_alpha(cache, t, R, alphas, min_inner)
        p_s = a_star * c["pred_b2"] + (1 - a_star) * c["anchor"]

        # fixo 0.4 (referencia exploratoria, nao oficial)
        p_04 = 0.4 * c["pred_b2"] + (1 - 0.4) * c["anchor"]

        b2m = float(b2_of.loc[(t, R), "mae"])
        rows.append({
            "temporada": t, "rodada_alvo": R,
            "n_test": c["n"],
            "alpha_star": float(a_star),
            "anchor_median": float(c["anchor"]),
            "b2_mae": b2m,
            "b2_rmse": rmse(c["pred_b2"], c["y"]),
            "b2_spearman": spearman(c["pred_b2"], c["y"]),
            "b2s_mae": mae(p_s, c["y"]),
            "b2s_rmse": rmse(p_s, c["y"]),
            "b2s_spearman": spearman(p_s, c["y"]),
            "b2_04_mae": mae(p_04, c["y"]),
            "delta_mae_b2s_b2": mae(p_s, c["y"]) - b2m,
            "delta_mae_b2s_b2_04": mae(p_s, c["y"]) - mae(p_04, c["y"]),
            "n_inner_rounds": len([1 for (tt, r) in cache.keys()
                                   if tt == t and min_inner <= r <= R-1]),
        })
        for a in alphas:
            alpha_rows.append({
                "temporada": t, "rodada_alvo": R, "alpha": a,
                "inner_mean_mae": means_inner.get(a, np.nan),
                "selecionado": int(a == a_star),
            })

    dfr = pd.DataFrame(rows)
    dfa = pd.DataFrame(alpha_rows)
    dfr.to_csv(CSV_FOLDS, index=False)
    dfa.to_csv(CSV_ALPHAS, index=False)

    n = int(dfr["n_test"].sum())
    b2_w = float((dfr["b2_mae"]*dfr["n_test"]).sum()/n)
    b2s_w = float((dfr["b2s_mae"]*dfr["n_test"]).sum()/n)
    b204_w = float((dfr["b2_04_mae"]*dfr["n_test"]).sum()/n)
    b2_rmse_w = float(np.sqrt((dfr["b2_rmse"]**2 * dfr["n_test"]).sum()/n))
    b2s_rmse_w = float(np.sqrt((dfr["b2s_rmse"]**2 * dfr["n_test"]).sum()/n))
    b2_sp = float(dfr["b2_spearman"].mean())
    b2s_sp = float(dfr["b2s_spearman"].mean())

    n_b2s_better = int((dfr["delta_mae_b2s_b2"] < 0).sum())
    n_b2_better = int((dfr["delta_mae_b2s_b2"] > 0).sum())
    n_tie = int((dfr["delta_mae_b2s_b2"] == 0).sum())
    med_delta = float(dfr["delta_mae_b2s_b2"].median())

    alpha_counts = dfr["alpha_star"].value_counts().sort_index().to_dict()

    por_temp = dfr.groupby("temporada").apply(lambda g: pd.Series({
        "n_folds": len(g),
        "b2_mae": float((g["b2_mae"]*g["n_test"]).sum()/g["n_test"].sum()),
        "b2s_mae": float((g["b2s_mae"]*g["n_test"]).sum()/g["n_test"].sum()),
        "alpha_medio": float(g["alpha_star"].mean()),
    })).reset_index()

    h_pq_dep = sha256(PARQUET)
    h_b2_dep = sha256(B2_FOLDS)
    imut = (h_pq_antes == h_pq_dep) and (h_b2_antes == h_b2_dep)

    fim = datetime.now()
    L = []
    L.append("# B2S — B2 + shrinkage walk-forward\n\n")
    L.append(f"- Inicio: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Fim: {fim.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- Folds: {len(dfr)}  Obs: {n}\n")
    L.append(f"- Especificacao: config/b2s_spec.yaml\n\n")

    L.append("## Metrica oficial\n\n")
    L.append(f"- B2 puro:          MAE = {b2_w:.4f}  RMSE = {b2_rmse_w:.4f}  Spearman = {b2_sp:.4f}\n")
    L.append(f"- B2S walk-fwd:     MAE = {b2s_w:.4f}  RMSE = {b2s_rmse_w:.4f}  Spearman = {b2s_sp:.4f}\n")
    L.append(f"- B2S α=0.4 fixo:   MAE = {b204_w:.4f}  (referencia exploratoria, NAO oficial)\n\n")

    L.append("## Estabilidade\n\n")
    L.append(f"- α* medio: {dfr['alpha_star'].mean():.3f}\n")
    L.append(f"- α* mediana: {dfr['alpha_star'].median():.3f}\n")
    L.append(f"- Contagem: {alpha_counts}\n")
    L.append(f"- Folds B2S melhor: {n_b2s_better}/{len(dfr)}\n")
    L.append(f"- Folds B2 melhor: {n_b2_better}/{len(dfr)}\n")
    L.append(f"- Empates: {n_tie}\n")
    L.append(f"- Mediana delta (B2S - B2): {med_delta:+.4f}\n\n")

    L.append("## Por temporada\n\n")
    L.append(por_temp.to_string(index=False))
    L.append("\n\n")

    L.append("## Imutabilidade\n\n")
    L.append(f"- Parquet: {'OK' if h_pq_antes == h_pq_dep else 'MUDOU'}\n")
    L.append(f"- B2 folds: {'OK' if h_b2_antes == h_b2_dep else 'MUDOU'}\n")

    SUMMARY.write_text("".join(L), encoding="utf-8")

    spec_md = f"""# Especificacao oficial do baseline B2S

- Versao: b2s_v1
- Data: 2026-09-20
- Status: oficial

## Definicao
`pred_b2s = alpha* * pred_b2 + (1 - alpha*) * anchor`
- `pred_b2`: media das ultimas 5 observacoes do jogador (intra-temporada)
- `anchor`: mediana de pontos_num do treino do fold

## Calibracao de alpha
- Metodo: nested walk-forward
- Grid: {g['inicio']} a {g['fim']} passo {g['passo']} ({len(alphas)} valores)
- min_inner_rounds: {min_inner}
- Inner rounds: rodadas r_i em [{min_inner}, R-1], mesma temp., elegivel
- Selecao: argmin da media dos MAE internos

## Garantias
- Nenhuma rodada >= R em qualquer calculo.
- Nenhuma observacao de outra temporada.
- 2025/R1 excluida globalmente.
- alpha* calibrado apenas em rounds internos de treino.
- Fallback (train_mean_global) entra antes do shrinkage e e comprimido igualmente.

## Metricas oficiais
- MAE ponderado (principal)
- RMSE agregado (diagnostico)
- Spearman medio (referencia)
"""
    SPEC_MD.write_text(spec_md, encoding="utf-8")

    print()
    print("=== RESUMO B2S ===")
    print(f"folds: {len(dfr)}  obs: {n}")
    print(f"B2 MAE: {b2_w:.4f}  B2S MAE: {b2s_w:.4f}")
    print(f"B2 Spearman: {b2_sp:.4f}  B2S Spearman: {b2s_sp:.4f}")
    print(f"alpha* media: {dfr['alpha_star'].mean():.3f}")
    print(f"folds B2S<B2: {n_b2s_better}/{len(dfr)}")
    print(f"imutavel: {imut}")


if __name__ == "__main__":
    main()
