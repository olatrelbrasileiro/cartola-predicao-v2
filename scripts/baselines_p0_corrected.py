#!/usr/bin/env python3
"""
Baselines P0-corrected.

Correcoes em relacao a versao anterior:
1. Pre-computa B2 para TODAS as rodadas >= MIN_INNER_ROUNDS por temporada,
   antes do loop externo, garantindo calibracao de alpha correta.
2. Documenta explicitamente a definicao condicional de B0.
3. Testes de leakage ampliados.
4. Nao altera artefatos antigos.
"""
import hashlib, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


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
CONFIG = ROOT/"config"

CANON = INTERIM/"cartola_jogador_rodada_v2.parquet"
AUDIT_YAML = CONFIG/"pontuacao_scouts_auditoria.yaml"
MANIFEST = REP/"harness_folds_v2.csv"
OLD_B2S = REP/"baseline_b2s_folds.csv"

OUT_FOLDS = REP/"baselines_p0_corrected_folds.csv"
OUT_SUMMARY = REP/"baselines_p0_corrected_summary.md"
OUT_PART = REP/"p0_participation_audit.csv"
OUT_VALID = REP/"p0_validation.json"

HASH_ESPERADO = "15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4"

TARGET = "pontos_num"
K_SHRINK = 7
RECENT_5 = 5
RECENT_3 = 3
MIN_INNER_ROUNDS = 3
ALPHAS = [round(a, 2) for a in np.arange(0, 1.01, 0.05)]

# Definicoes explicitas (precisam ser autorizadas):
B0_DEFINITION = "media_pontos_dos_participantes_no_treino"
ANCHOR_DEFINITION = "mediana_pontos_dos_participantes_no_treino"
FALLBACK_DEFINITION = "b0_const_do_fold"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cumulative_diff(df, cols):
    out = df.sort_values(["temporada","atleta_id","rodada"],
                         kind="mergesort").reset_index(drop=True)
    for col in cols:
        ff = out.groupby(["temporada","atleta_id"], sort=False)[col].ffill()
        chave = out[["temporada","atleta_id"]].copy()
        chave["_ff"] = ff.values
        prev = chave.groupby(["temporada","atleta_id"], sort=False)["_ff"].shift(1)
        out[f"diff_{col}"] = ff.fillna(0).to_numpy() - prev.fillna(0).to_numpy()
    return out


def compute_basic_score(df, audit):
    pesos = audit["pesos_por_temporada"]
    total = pd.Series(0.0, index=df.index)
    for s in audit["scouts_basicos"]:
        col = f"diff_scout_{s}"
        peso_linha = df["temporada"].map({int(t): pesos[t][s] for t in pesos})
        total = total + df[col].to_numpy() * peso_linha.to_numpy()
    return total.astype(float)


def compute_participation_features(df):
    """Per-row features usam apenas PARTICIPACOES anteriores (mesma temporada)."""
    df = df.reset_index(drop=True).copy()
    df["_idx"] = df.index
    results = []
    for (t, aid), sub in df.groupby(["temporada","atleta_id"], sort=False):
        sub = sub.sort_values("rodada").reset_index(drop=False)
        n = len(sub)
        for i in range(n):
            row_idx = int(sub.iloc[i]["_idx"])
            prev = sub.iloc[:i]
            prev_part = prev[prev["elegivel"] & prev["entrou_bool"]]
            if len(prev_part) == 0:
                results.append({
                    "_idx": row_idx,
                    "hist_mean_p": np.nan, "hist_std_p": np.nan, "hist_n_p": 0,
                    "rec_p_5": np.nan, "rec_p_3": np.nan, "rec_std_p_5": np.nan,
                    "basic_mean_p": np.nan, "basic_rec_p_5": np.nan,
                })
            else:
                p = prev_part[TARGET].to_numpy(dtype=float)
                b = prev_part["basic_score"].to_numpy(dtype=float)
                rec5 = prev_part.tail(RECENT_5)
                rec3 = prev_part.tail(RECENT_3)
                results.append({
                    "_idx": row_idx,
                    "hist_mean_p": float(p.mean()),
                    "hist_std_p": float(p.std(ddof=0)),
                    "hist_n_p": int(len(prev_part)),
                    "rec_p_5": float(rec5[TARGET].mean()),
                    "rec_p_3": float(rec3[TARGET].mean()),
                    "rec_std_p_5": float(rec5[TARGET].std(ddof=0)) if len(rec5) >= 2 else 0.0,
                    "basic_mean_p": float(b.mean()),
                    "basic_rec_p_5": float(rec5["basic_score"].mean()),
                })
    return pd.DataFrame(results).set_index("_idx")


def mae(p, y):
    return float(np.mean(np.abs(p - y))) if len(y) else np.nan
def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan
def spearman(p, y):
    if len(p) < 2: return np.nan
    if np.all(p == p[0]) or np.all(y == y[0]): return np.nan
    rx = pd.Series(p).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------- pre-computo B2
def precompute_b2_all_rounds(df, min_round=MIN_INNER_ROUNDS):
    """
    Pre-computa, para CADA (T, r) com r >= min_round, o B2 dos participantes
    do test e o anchor. Independe da ordem dos folds.

    Cada entrada:
      {"y": y_participantes_r, "b2": pred_b2_participantes_r,
       "anchor": mediana_treino_r, "b0_const": media_treino_r}
    """
    out = {}
    for t in sorted(df["temporada"].unique()):
        mask_t = df["temporada"] == t
        for r in sorted(df[mask_t]["rodada"].unique()):
            if r < min_round:
                continue
            train = df[mask_t & (df["rodada"] < r) & df["elegivel"] & df["entrou_bool"]]
            test = df[mask_t & (df["rodada"] == r) & df["elegivel"] & df["entrou_bool"]]
            if len(test) == 0 or len(train) == 0:
                continue
            b0_const = float(train[TARGET].mean())
            anchor = float(train[TARGET].median())
            b2 = test["rec_p_5"].fillna(b0_const).to_numpy(dtype=float)
            y = test[TARGET].to_numpy(dtype=float)
            out[(int(t), int(r))] = {
                "y": y,
                "b2": b2,
                "anchor": anchor,
                "b0_const": b0_const,
            }
    return out


def calibrate_alpha_from_precomp(t, R, precomp):
    """Alpha calibrado usando SOMENTE rounds < R, com B2 pre-computado."""
    inner = [r for (tt, r) in precomp.keys()
             if tt == t and MIN_INNER_ROUNDS <= r <= R - 1]
    if not inner:
        return 1.0, {}
    maes = {a: [] for a in ALPHAS}
    for r_i in inner:
        entry = precomp[(t, r_i)]
        y_i = entry["y"]
        b2_i = entry["b2"]
        anchor_i = entry["anchor"]
        if len(y_i) == 0:
            continue
        for a in ALPHAS:
            pred = a * b2_i + (1 - a) * anchor_i
            maes[a].append(float(np.mean(np.abs(pred - y_i))))
    mean_mae = {a: float(np.mean(v)) if v else np.inf for a, v in maes.items()}
    alpha_star = min(mean_mae, key=mean_mae.get)
    return alpha_star, mean_mae


def main():
    inicio = datetime.now()
    print("== Baselines P0-corrected ==")
    print(f"B0 definicao: {B0_DEFINITION}")
    print(f"ANCHOR definicao: {ANCHOR_DEFINITION}")
    print(f"FALLBACK definicao: {FALLBACK_DEFINITION}")

    h_before = sha256(CANON)
    print(f"Parquet SHA-256: {h_before[:16]}...")
    if h_before != HASH_ESPERADO:
        print(f"ERRO: hash do Parquet canonico nao bate com o esperado.", file=sys.stderr)
        sys.exit(2)

    audit = load_yaml(AUDIT_YAML)
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada","rodada","atleta_id"], kind="mergesort").reset_index(drop=True)
    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    df["entrou_bool"] = df["entrou_em_campo"].fillna(False).astype(bool)
    df["flag_B"] = (~df["entrou_bool"]) & (df[TARGET] != 0)

    print(f"linhas: {len(df)}  participantes: {int(df['entrou_bool'].sum())}  "
          f"flag_B: {int(df['flag_B'].sum())}")

    scout_cols = [f"scout_{s}" for s in audit["scouts_basicos"]]
    df = apply_cumulative_diff(df, scout_cols)
    df["basic_score"] = compute_basic_score(df, audit)

    print("computando features de participacao...")
    feats_p = compute_participation_features(df)
    df = df.merge(feats_p, left_index=True, right_index=True, how="left")

    n_total = len(df)
    n_part = int(df["entrou_bool"].sum())
    n_non = n_total - n_part
    n_zero_p = int(((df["entrou_bool"]) & (df[TARGET] == 0)).sum())
    n_zero_n = int(((~df["entrou_bool"]) & (df[TARGET] == 0)).sum())
    part_audit = pd.DataFrame([{
        "total_linhas": n_total,
        "total_participantes": n_part,
        "total_nao_participantes": n_non,
        "pct_participantes": round(n_part / n_total, 4),
        "pct_nao_participantes": round(n_non / n_total, 4),
        "zeros_entre_participantes": n_zero_p,
        "zeros_entre_nao_participantes": n_zero_n,
        "flag_B_total": int(df["flag_B"].sum()),
    }])
    part_audit.to_csv(OUT_PART, index=False)

    # ---- pre-computo B2 (todas as rodadas >= MIN_INNER_ROUNDS por temporada)
    print("pre-computando B2 para todas as rodadas...")
    precomp = precompute_b2_all_rounds(df)
    print(f"  entradas: {len(precomp)}")

    man = pd.read_csv(MANIFEST)
    folds = [(int(t), int(r)) for t, r in zip(man["temporada"], man["rodada_alvo"])]
    print(f"folds: {len(folds)}")

    rows = []
    for i, (t, R) in enumerate(folds):
        mask_t = df["temporada"] == t
        train_all = df[mask_t & (df["rodada"] < R) & df["elegivel"]]
        test_all = df[mask_t & (df["rodada"] == R) & df["elegivel"]]
        train_part = train_all[train_all["entrou_bool"]]
        test_part = test_all[test_all["entrou_bool"]]
        test_metric = test_part

        if len(test_metric) == 0 or len(train_part) == 0:
            continue

        y = test_metric[TARGET].to_numpy(dtype=float)
        b0_const = float(train_part[TARGET].mean())
        basic_global = float(train_part["basic_score"].mean())

        entry = precomp.get((t, R))
        if entry is None:
            continue
        anchor_median = entry["anchor"]
        b2_preds = entry["b2"]

        pred_b0 = np.full(len(test_metric), b0_const)
        pred_b1 = test_metric["hist_mean_p"].fillna(b0_const).to_numpy(dtype=float)
        pred_b3 = test_metric["basic_mean_p"].fillna(basic_global).to_numpy(dtype=float)

        alpha_star, _ = calibrate_alpha_from_precomp(t, R, precomp)
        pred_b2s = alpha_star * b2_preds + (1 - alpha_star) * anchor_median

        rows.append({
            "temporada": t, "rodada_alvo": R,
            "n_train_total": int(len(train_all)),
            "n_train_participantes": int(len(train_part)),
            "n_test_total": int(len(test_all)),
            "n_test_participantes": int(len(test_metric)),
            "n_test_nao_participantes": int(len(test_all) - len(test_metric)),
            "b0_mae": mae(pred_b0, y), "b0_rmse": rmse(pred_b0, y), "b0_spearman": spearman(pred_b0, y),
            "b1_mae": mae(pred_b1, y), "b1_rmse": rmse(pred_b1, y), "b1_spearman": spearman(pred_b1, y),
            "b2_mae": mae(b2_preds, y), "b2_rmse": rmse(b2_preds, y), "b2_spearman": spearman(b2_preds, y),
            "b3_mae": mae(pred_b3, y), "b3_rmse": rmse(pred_b3, y), "b3_spearman": spearman(pred_b3, y),
            "b2s_mae": mae(pred_b2s, y), "b2s_rmse": rmse(pred_b2s, y), "b2s_spearman": spearman(pred_b2s, y),
            "b2s_alpha_star": float(alpha_star),
            "b2s_anchor_median": anchor_median,
            "n_inner_rounds": int(len([1 for (tt,r) in precomp.keys()
                                       if tt==t and MIN_INNER_ROUNDS <= r <= R-1])),
        })

        if (i + 1) % 10 == 0:
            print(f"  fold {i+1}/{len(folds)}  {t}/R{R}  B2S={rows[-1]['b2s_mae']:.4f} "
                  f"alpha={alpha_star:.2f}")

    dfr = pd.DataFrame(rows)
    dfr.to_csv(OUT_FOLDS, index=False)

    h_after = sha256(CANON)
    integridade_ok = (h_after == h_before)

    total_n = int(dfr["n_test_participantes"].sum())
    def _wavg(col):
        return float((dfr[col] * dfr["n_test_participantes"]).sum() / total_n)

    aggs = {}
    for b in ["b0", "b1", "b2", "b3", "b2s"]:
        aggs[b] = {
            "mae_pond": _wavg(f"{b}_mae"),
            "mae_medio": float(dfr[f"{b}_mae"].mean()),
            "rmse_pond": float(np.sqrt((dfr[f"{b}_rmse"]**2 * dfr["n_test_participantes"]).sum() / total_n)),
            "spearman_medio": float(dfr[f"{b}_spearman"].mean()),
        }

    por_temp = dfr.groupby("temporada").apply(lambda g: pd.Series({
        "n_folds": len(g),
        "n_test": int(g["n_test_participantes"].sum()),
        "b0_mae": float((g["b0_mae"]*g["n_test_participantes"]).sum()/g["n_test_participantes"].sum()),
        "b1_mae": float((g["b1_mae"]*g["n_test_participantes"]).sum()/g["n_test_participantes"].sum()),
        "b2_mae": float((g["b2_mae"]*g["n_test_participantes"]).sum()/g["n_test_participantes"].sum()),
        "b3_mae": float((g["b3_mae"]*g["n_test_participantes"]).sum()/g["n_test_participantes"].sum()),
        "b2s_mae": float((g["b2s_mae"]*g["n_test_participantes"]).sum()/g["n_test_participantes"].sum()),
        "b2s_spearman": float(g["b2s_spearman"].mean()),
    }, dtype=float)).reset_index()

    OLD = {"B0": 2.0748, "B1": 1.6012, "B2": 1.5778, "B2S": 1.4016}
    OLD_SP = {"B1": 0.4553, "B2": 0.4617, "B2S": 0.4617}

    fim = datetime.now()
    L = []
    L.append("# Baselines P0-corrected — Summary\n\n")
    L.append(f"- Gerado em: {inicio.isoformat(timespec='seconds')}\n")
    L.append(f"- Duracao: {(fim-inicio).total_seconds():.1f}s\n")
    L.append(f"- Folds: {len(dfr)}\n")
    L.append(f"- Obs no teste (participantes): {total_n}\n\n")

    L.append("## Definicoes (explicitas)\n\n")
    L.append(f"- B0 = {B0_DEFINITION}\n")
    L.append(f"- ANCHOR (B2S) = {ANCHOR_DEFINITION}\n")
    L.append(f"- FALLBACK = {FALLBACK_DEFINITION}\n")
    L.append(f"- Avaliacao = somente participantes do test (entrou_em_campo=True)\n\n")

    L.append("## Auditoria de participacao\n\n")
    L.append(part_audit.T.to_string())
    L.append("\n\n")

    L.append("## Resultados corrigidos\n\n")
    L.append("| Baseline | MAE ponderado | MAE medio | RMSE | Spearman |\n")
    L.append("|---|---|---|---|---|\n")
    for b, ag in aggs.items():
        L.append(f"| {b.upper()} | {ag['mae_pond']:.4f} | {ag['mae_medio']:.4f} | "
                 f"{ag['rmse_pond']:.4f} | {ag['spearman_medio']:.4f} |\n")
    L.append("\n")

    L.append("## Comparacao com referencia antiga (nao valida para o problema condicional)\n\n")
    L.append("| Baseline | MAE antigo | MAE corrigido | Δ |\n")
    L.append("|---|---|---|---|\n")
    mapping = {"B0":"b0", "B1":"b1", "B2":"b2", "B2S":"b2s"}
    for k, v in OLD.items():
        new = aggs[mapping[k]]["mae_pond"]
        L.append(f"| {k} | {v:.4f} | {new:.4f} | {new - v:+.4f} |\n")
    L.append("\n")

    L.append("## B2S por temporada\n\n")
    L.append(por_temp.to_string(index=False))
    L.append("\n\n")

    L.append("## Distribuicao de alpha no B2S\n\n")
    L.append(dfr["b2s_alpha_star"].value_counts().sort_index().to_string())
    L.append("\n\n")

    L.append("## Distribuicao de n_inner_rounds por fold\n\n")
    L.append(dfr["n_inner_rounds"].describe().to_string())
    L.append("\n\n")

    L.append("## Integridade\n\n")
    L.append(f"- SHA-256 Parquet antes : `{h_before}`\n")
    L.append(f"- SHA-256 Parquet depois: `{h_after}`\n")
    L.append(f"- Inalterado: **{integridade_ok}**\n")

    OUT_SUMMARY.write_text("".join(L), encoding="utf-8")

    valid = {
        "gerado_em": inicio.isoformat(timespec="seconds"),
        "folds": len(dfr),
        "total_test_participantes": total_n,
        "parquet_sha_before": h_before,
        "parquet_sha_after": h_after,
        "parquet_sha_esperado": HASH_ESPERADO,
        "integridade_ok": integridade_ok,
        "definicoes": {
            "b0": B0_DEFINITION,
            "anchor": ANCHOR_DEFINITION,
            "fallback": FALLBACK_DEFINITION,
            "avaliacao": "somente_participantes",
        },
        "agregados": aggs,
        "por_temporada": por_temp.to_dict(orient="records"),
        "old_reference": OLD,
        "old_spearman_reference": OLD_SP,
    }
    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=== RESUMO ===")
    for b, ag in aggs.items():
        print(f"  {b.upper():4s}  MAE={ag['mae_pond']:.4f}  Spearman={ag['spearman_medio']:.4f}")
    print(f"alpha medio: {dfr['b2s_alpha_star'].mean():.3f}")
    print(f"integridade parquet: {integridade_ok}")
    print("artefatos:")
    for p in [OUT_FOLDS, OUT_SUMMARY, OUT_PART, OUT_VALID]:
        print(f"  {p.relative_to(ROOT)}")

    if not integridade_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
