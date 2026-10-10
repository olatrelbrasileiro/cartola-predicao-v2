#!/usr/bin/env python3
"""
Build Features de Goleiro a partir do Banco Mestre (EXP-12)
Fonte da verdade: data/interim/exp12/gol_pooling_2024_2026.parquet
Este arquivo já tem os scouts DE, GS, SG, DP limpos, por rodada (não acumulados).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Fonte da verdade do banco mestre
GOL_MASTER = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"
CANON_V2 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
OUT = ROOT / "data" / "interim" / "features_goleiro_master_v1.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

SCOUTS_GOLEIRO = ["DE", "GS", "SG", "DP"]
K_FLOOR, K_CEIL = 5, 30 # Shrinkage mais forte para goleiros (menos dados)

def estimate_k_empirical(values_by_player):
    means, vars_w = [], []
    for pid, vals in values_by_player.items():
        v = vals.dropna()
        if len(v) >= 2:
            means.append(v.mean())
            vars_w.append(v.var())
    if len(means) < 5:
        return 10.0
    var_between = np.var(means)
    var_within = np.mean(vars_w) if vars_w else 0
    if var_between < 1e-9:
        return float(K_CEIL)
    return float(np.clip(var_within / var_between, K_FLOOR, K_CEIL))

def main():
    print("1. Carregando Banco Mestre de Goleiros (EXP-12)...")
    df_gol = pd.read_parquet(GOL_MASTER)
    print(f"   Shape: {df_gol.shape}")
    
    # Filtrar apenas quem entrou em campo para calcular estatísticas reais
    df_gol = df_gol[df_gol["entrou_em_campo"] == True].copy()
    
    print("\n2. Verificando scouts de goleiro no banco mestre...")
    for s in SCOUTS_GOLEIRO:
        col = f"scout_{s}"
        if col in df_gol.columns:
            media = df_gol[col].mean()
            print(f"   ✅ {col}: Média = {media:.2f}, Max = {df_gol[col].max():.0f}")
        else:
            print(f"   ❌ {col}: NÃO ENCONTRADO")

    print("\n3. Calculando k empírico por scout (Goleiro)...")
    k_map = {}
    for s in SCOUTS_GOLEIRO:
        col = f"scout_{s}"
        if col not in df_gol.columns:
            continue
        by_player = {pid: grp[col] for pid, grp in df_gol.groupby("atleta_id")}
        k = estimate_k_empirical(by_player)
        k_map[s] = k
        print(f"   k empírico para {s}: {k:.2f}")

    print("\n4. Calculando features por fold (temporada, rodada)...")
    all_features = []
    folds = df_gol[["temporada", "rodada"]].drop_duplicates().sort_values(["temporada", "rodada"])
    
    for fold_idx, (t, R) in enumerate(folds.itertuples(index=False)):
        # Treino: temporadas < t + temporada t com rodada < R
        train = df_gol[(df_gol["temporada"] < t) | 
                       ((df_gol["temporada"] == t) & (df_gol["rodada"] < R))]
        # Teste: temporada t, rodada R
        test = df_gol[(df_gol["temporada"] == t) & (df_gol["rodada"] == R)]
        
        if train.empty or test.empty:
            continue
        
        test_feats = test[["temporada", "rodada", "atleta_id", "posicao_id"]].copy()
        
        for s in SCOUTS_GOLEIRO:
            col = f"scout_{s}"
            if col not in train.columns:
                continue
            
            # Stats por atleta no treino
            agg = train.groupby("atleta_id")[col].agg(
                mean_hist="mean", std_hist="std", n_hist="count"
            ).reset_index()
            
            # Últimas 3 e 5 participações
            last3 = train.groupby("atleta_id")[col].apply(lambda g: g.tail(3).mean()).reset_index(name="mean_3")
            last5 = train.groupby("atleta_id")[col].apply(lambda g: g.tail(5).mean()).reset_index(name="mean_5")
            std5 = train.groupby("atleta_id")[col].apply(lambda g: g.tail(5).std() if len(g) >= 2 else np.nan).reset_index(name="std_5")
            
            agg = agg.merge(last3, on="atleta_id", how="left")
            agg = agg.merge(last5, on="atleta_id", how="left")
            agg = agg.merge(std5, on="atleta_id", how="left")
            
            test_feats = test_feats.merge(agg, on="atleta_id", how="left")
            test_feats[f"scout_{s}_mean_hist"] = test_feats["mean_hist"]
            test_feats[f"scout_{s}_mean_3"] = test_feats["mean_3"]
            test_feats[f"scout_{s}_mean_5"] = test_feats["mean_5"]
            test_feats[f"scout_{s}_std_5"] = test_feats["std_5"]
            
            # Shrinkage
            prior_pos = train[col].mean()
            k = k_map.get(s, 10.0)
            n_col = test_feats["n_hist"].fillna(0)
            m_col = test_feats["mean_hist"].fillna(0)
            
            test_feats[f"scout_{s}_shrunk"] = np.where(
                n_col > 0,
                (n_col * m_col + k * prior_pos) / (n_col + k),
                prior_pos
            )
            
            # Momentum
            test_feats[f"scout_{s}_momentum"] = test_feats[f"scout_{s}_mean_3"] - test_feats[f"scout_{s}_mean_hist"]
            
            # Limpar colunas temporárias
            for c in ["mean_hist", "std_hist", "n_hist", "mean_3", "mean_5", "std_5"]:
                if c in test_feats.columns:
                    test_feats = test_feats.drop(columns=[c])
        
        all_features.append(test_feats)
        if (fold_idx + 1) % 10 == 0:
            print(f"   ... {fold_idx + 1} folds processados")
    
    print("\n5. Consolidando e salvando...")
    df_out = pd.concat(all_features, ignore_index=True)
    df_out.to_parquet(OUT, index=False)
    print(f"   ✅ Salvo: {OUT}")
    print(f"   Shape: {df_out.shape}")
    
    # Relatório rápido
    print("\n6. Validação rápida das features geradas:")
    for s in SCOUTS_GOLEIRO:
        col_shrunk = f"scout_{s}_shrunk"
        if col_shrunk in df_out.columns:
            print(f"   {col_shrunk}: Média = {df_out[col_shrunk].mean():.2f}, Max = {df_out[col_shrunk].max():.2f}")

if __name__ == "__main__":
    main()
