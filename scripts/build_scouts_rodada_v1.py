#!/usr/bin/env python3
"""
Build Scouts por Rodada (Correção Definitiva)
1. Remove duplicatas de mesma rodada.
2. Calcula o diff (valor da rodada) para scouts acumulados da API.
3. Calcula médias móveis e shrinkage empírico.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
OUT = ROOT / "data" / "interim" / "features_scouts_rodada_v1.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Todos os scouts que podem existir no dataset
ALL_SCOUTS = ["G", "A", "FD", "FF", "FT", "FS", "DE", "SG", "DS", "FC", "DP", "CA", "CV", "I", "PC", "PS", "PP"]
# Scouts que a API entrega como ACUMULADO e precisam de diff
SCOUTS_ACUMULADOS = ["G", "A", "DE", "SG", "DS", "FC", "DP", "CA", "CV", "I", "PC", "PS", "PP"]
SCOUTS_STD = ["G", "A", "DE", "DP", "SG"]
K_FLOOR, K_CEIL = 3, 50

def estimate_k_empirical(values_by_player):
    means, vars_w = [], []
    for pid, vals in values_by_player.items():
        v = vals.dropna()
        if len(v) >= 2:
            means.append(v.mean())
            vars_w.append(v.var())
    if len(means) < 5:
        return 7.0
    var_between = np.var(means)
    var_within = np.mean(vars_w) if vars_w else 0
    if var_between < 1e-9:
        return float(K_CEIL)
    return float(np.clip(var_within / var_between, K_FLOOR, K_CEIL))

def main():
    print("1. Carregando dataset canônico...")
    df = pd.read_parquet(CANON)
    print(f"   Shape original: {df.shape}")
    
    # Filtrar apenas scouts que existem
    scouts_existentes = [s for s in ALL_SCOUTS if f"scout_{s}" in df.columns]
    print(f"   Scouts encontrados: {scouts_existentes}")
    
    print("\n2. Limpando dados e resolvendo duplicatas...")
    # Ordenar para garantir consistência
    df = df.sort_values(["temporada", "atleta_id", "rodada"], kind="mergesort")
    
    # Resolver duplicatas: para mesma temporada, atleta e rodada, pegar a última linha não-nula
    # Isso corrige o problema visto no diagnóstico (ex: 3 linhas para rodada 1)
    df_clean = df.drop_duplicates(subset=["temporada", "atleta_id", "rodada"], keep="last")
    print(f"   Shape após remover duplicatas: {df_clean.shape}")
    
    print("\n3. Calculando diff para scouts acumulados (valor por rodada)...")
    for s in scouts_existentes:
        if s in SCOUTS_ACUMULADOS:
            col = f"scout_{s}"
            # Calcular diff agrupado por temporada e atleta
            df_clean[f"diff_{col}"] = df_clean.groupby(["temporada", "atleta_id"])[col].diff()
            
            # Para a primeira aparição do atleta na temporada, o diff é NaN. 
            # Assumimos que o valor acumulado inicial É o valor da primeira rodada.
            first_mask = df_clean.groupby(["temporada", "atleta_id"])["rodada"].transform("min") == df_clean["rodada"]
            df_clean.loc[first_mask, f"diff_{col}"] = df_clean.loc[first_mask, col]
            
            # Substituir NaNs restantes (ex: jogador que entrou no meio da temporada sem histórico claro) por 0
            df_clean[f"diff_{col}"] = df_clean[f"diff_{col}"].fillna(0)
            
            # Garantir que não haja valores negativos (erro de dados na API)
            df_clean[f"diff_{col}"] = df_clean[f"diff_{col}"].clip(lower=0)
            
            media_diff = df_clean[f"diff_{col}"].mean()
            print(f"   ✅ {col}: diff calculado. Média por rodada = {media_diff:.2f}")

    print("\n4. Calculando k empírico por scout/posição...")
    participants = df_clean[df_clean["entrou_em_campo"].fillna(False)].copy()
    k_map = {}
    k_report = []
    
    for s in scouts_existentes:
        col = f"diff_scout_{s}"
        if col not in participants.columns:
            continue
        for pos in sorted(participants["posicao_id"].unique()):
            sub = participants[participants["posicao_id"] == pos]
            by_player = {pid: grp[col] for pid, grp in sub.groupby("atleta_id")}
            k = estimate_k_empirical(by_player)
            k_map[(s, pos)] = k
            k_report.append({"scout": s, "posicao_id": pos, "k_empirical": round(k, 2)})
            
    pd.DataFrame(k_report).to_csv(REPORTS / "scouts_k_empirical_corrigido.csv", index=False)
    print(f"   k empírico salvo.")

    print("\n5. Calculando features por fold (temporada, rodada)...")
    all_features = []
    folds = participants[["temporada", "rodada"]].drop_duplicates().sort_values(["temporada", "rodada"])
    
    for fold_idx, (t, R) in enumerate(folds.itertuples(index=False)):
        train = participants[(participants["temporada"] < t) |
                             ((participants["temporada"] == t) & (participants["rodada"] < R))]
        test = df_clean[(df_clean["temporada"] == t) & (df_clean["rodada"] == R)]
        
        if train.empty or test.empty:
            continue
        
        test_feats = test[["temporada", "rodada", "atleta_id", "posicao_id"]].copy()
        
        for s in scouts_existentes:
            col = f"diff_scout_{s}"
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
            
            # Shrinkage
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
            
            test_feats[f"scout_{s}_momentum"] = test_feats[f"scout_{s}_mean_3"] - test_feats[f"scout_{s}_mean_hist"]
            
            if s in SCOUTS_STD:
                test_feats[f"scout_{s}_std_5"] = test_feats["std_5"]
            
            # Limpar colunas temporárias
            for c in ["mean_hist", "std_hist", "n_hist", "mean_3", "mean_5", "std_5", f"_prior_{s}", f"_k_{s}"]:
                if c in test_feats.columns:
                    test_feats = test_feats.drop(columns=[c])
        
        all_features.append(test_feats)
        if (fold_idx + 1) % 15 == 0:
            print(f"   ... {fold_idx + 1} folds processados")
    
    print("\n6. Consolidando e salvando...")
    df_out = pd.concat(all_features, ignore_index=True)
    df_out.to_parquet(OUT, index=False)
    print(f"   ✅ Salvo: {OUT}")
    print(f"   Shape: {df_out.shape}")
    
    # Validação
    print("\n7. Validação rápida (Goleiros - posicao_id == 1):")
    gk_test = df_out[df_out["posicao_id"] == 1]
    for s in ["DE", "SG", "DP"]:
        col = f"scout_{s}_shrunk"
        if col in gk_test.columns:
            print(f"   {col}: Média = {gk_test[col].mean():.2f}, Max = {gk_test[col].max():.2f}")

if __name__ == "__main__":
    main()
