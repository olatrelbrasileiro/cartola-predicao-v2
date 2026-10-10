#!/usr/bin/env python3
"""
Build Scouts Granulares com Shrinkage Empírico por Scout/Posição
14 scouts × 5 parâmetros + 5 std = 75 features
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
OUT = ROOT / "data" / "interim" / "features_scouts_shrunk_v1.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

SCOUTS = ["G", "A", "FD", "FF", "FT", "FS", "DE", "SG", "DS", "FC", "DP", "GS", "CA", "CV"]
SCOUTS_STD = ["G", "A", "DE", "DP", "SG"]  # Apenas críticos
K_FLOOR, K_CEIL = 3, 50

def estimate_k_empirical(values_by_player):
    """Empirical Bayes: k = var_within / var_between (clamped)."""
    means, vars_w, ns = [], [], []
    for pid, vals in values_by_player.items():
        v = vals.dropna()
        if len(v) >= 2:
            means.append(v.mean())
            vars_w.append(v.var())
            ns.append(len(v))
    if len(means) < 5:
        return 7.0
    var_between = np.var(means)
    var_within = np.mean(vars_w) if vars_w else 0
    if var_between < 1e-9:
        return float(K_CEIL)
    k_raw = var_within / var_between
    return float(np.clip(k_raw, K_FLOOR, K_CEIL))

def main():
    print("1. Carregando dataset canônico...")
    df = pd.read_parquet(CANON)
    df = df.sort_values(["temporada", "rodada", "atleta_id"], kind="mergesort").reset_index(drop=True)
    print(f"   Shape: {df.shape}")

    # Calcular diffs para scouts que podem ser acumulados
    print("2. Calculando diffs de scouts...")
    for s in SCOUTS:
        col = f"scout_{s}"
        if col not in df.columns:
            df[col] = 0
        df[f"diff_{col}"] = df.groupby(["temporada", "atleta_id"])[col].diff().fillna(df[col])

    df["elegivel"] = ~((df["temporada"] == 2025) & (df["rodada"] == 1))
    participants = df[df["entrou_em_campo"].fillna(False) & df["elegivel"]].copy()

    print("3. Calculando k empírico por scout/posição...")
    k_map = {}
    k_report = []
    for s in SCOUTS:
        col = f"diff_scout_{s}"
        for pos in sorted(participants["posicao_id"].unique()):
            sub = participants[participants["posicao_id"] == pos]
            by_player = {pid: grp[col] for pid, grp in sub.groupby("atleta_id")}
            k = estimate_k_empirical(by_player)
            k_map[(s, pos)] = k
            k_report.append({"scout": s, "posicao_id": pos, "k_empirical": round(k, 2)})
    pd.DataFrame(k_report).to_csv(REPORTS / "scouts_k_empirical.csv", index=False)
    print(f"   k empírico salvo: {REPORTS / 'scouts_k_empirical.csv'}")

    print("4. Calculando features por fold (temporada, rodada)...")
    all_features = []
    folds = participants[["temporada", "rodada"]].drop_duplicates().sort_values(["temporada", "rodada"])

    for _, (t, R) in folds.iterrows():
        train = participants[(participants["temporada"] < t) |
                             ((participants["temporada"] == t) & (participants["rodada"] < R))]
        test = df[(df["temporada"] == t) & (df["rodada"] == R)]

        if train.empty or test.empty:
            continue

        test_feats = test[["temporada", "rodada", "atleta_id", "posicao_id"]].copy()

        for s in SCOUTS:
            col = f"diff_scout_{s}"

            # Stats por atleta no treino
            agg = train.groupby("atleta_id")[col].agg(
                mean_hist="mean", std_hist="std", n_hist="count"
            ).reset_index()

            # Últimas 3 e 5 participações
            def last_n(group, n):
                return group.tail(n).mean()
            last3 = train.groupby("atleta_id")[col].apply(lambda g: last_n(g, 3)).reset_index(name="mean_3")
            last5 = train.groupby("atleta_id")[col].apply(lambda g: last_n(g, 5)).reset_index(name="mean_5")
            std5 = train.groupby("atleta_id")[col].apply(lambda g: g.tail(5).std() if len(g) >= 2 else np.nan).reset_index(name="std_5")

            agg = agg.merge(last3, on="atleta_id", how="left")
            agg = agg.merge(last5, on="atleta_id", how="left")
            agg = agg.merge(std5, on="atleta_id", how="left")

            # Merge com test
            test_feats = test_feats.merge(agg, on="atleta_id", how="left")
            test_feats[f"scout_{s}_mean_hist"] = test_feats["mean_hist"]
            test_feats[f"scout_{s}_mean_3"] = test_feats["mean_3"]
            test_feats[f"scout_{s}_mean_5"] = test_feats["mean_5"]

            # Shrinkage com k empírico por posição
            def apply_shrink(row):
                k = k_map.get((s, row["posicao_id"]), 7.0)
                n = row["n_hist"] if pd.notna(row["n_hist"]) else 0
                m = row["mean_hist"] if pd.notna(row["mean_hist"]) else 0
                prior = train[train["posicao_id"] == row["posicao_id"]][col].mean() if n == 0 else 0
                if n == 0:
                    return prior
                return (n * m + k * prior) / (n + k)

            # Vectorized shrinkage
            prior_by_pos = train.groupby("posicao_id")[col].mean().to_dict()
            test_feats[f"_prior_{s}"] = test_feats["posicao_id"].map(prior_by_pos).fillna(0)
            test_feats[f"_k_{s}"] = test_feats["posicao_id"].map(lambda p: k_map.get((s, p), 7.0))
            n_col = test_feats["n_hist"].fillna(0)
            m_col = test_feats["mean_hist"].fillna(0)
            k_col = test_feats[f"_k_{s}"]
            pr_col = test_feats[f"_prior_{s}"]
            test_feats[f"scout_{s}_shrunk"] = np.where(
                n_col > 0,
                (n_col * m_col + k_col * pr_col) / (n_col + k_col),
                pr_col
            )

            # Momentum
            test_feats[f"scout_{s}_momentum"] = test_feats[f"scout_{s}_mean_3"] - test_feats[f"scout_{s}_mean_hist"]

            # Std_5 apenas para scouts críticos
            if s in SCOUTS_STD:
                test_feats[f"scout_{s}_std_5"] = test_feats["std_5"]

            # Limpar colunas temporárias
            for c in ["mean_hist", "std_hist", "n_hist", "mean_3", "mean_5", "std_5", f"_prior_{s}", f"_k_{s}"]:
                if c in test_feats.columns:
                    test_feats = test_feats.drop(columns=[c])

        all_features.append(test_feats)
        if len(all_features) % 10 == 0:
            print(f"   ... {len(all_features)} folds processados")

    print("5. Consolidando e salvando...")
    df_out = pd.concat(all_features, ignore_index=True)
    df_out.to_parquet(OUT, index=False)
    print(f"   ✅ Salvo: {OUT}")
    print(f"   Shape: {df_out.shape}")
    print(f"   Features geradas: {df_out.shape[1] - 4}")  # -4 para chaves

if __name__ == "__main__":
    main()
