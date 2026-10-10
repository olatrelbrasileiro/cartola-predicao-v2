#!/usr/bin/env python3
"""
Build Contexto Time/Adversário (16 features) + Interações (3 features)
Lê team_match_history e team_scout_history_2024_2025
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIST_26 = ROOT / "data" / "interim" / "exp12" / "team_match_history.parquet"
HIST_2425 = ROOT / "data" / "interim" / "exp12" / "team_scout_history_2024_2025.parquet"
CANON = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
OUT = ROOT / "data" / "interim" / "features_team_context_v2.parquet"

SCOUTS_CTX = ["FTT", "FA", "DS", "FC", "G", "FS", "DE"]

def main():
    print("1. Carregando históricos de time...")
    frames = []
    if HIST_26.exists():
        frames.append(pd.read_parquet(HIST_26))
        print(f"   2026: {frames[-1].shape}")
    if HIST_2425.exists():
        frames.append(pd.read_parquet(HIST_2425))
        print(f"   2024-25: {frames[-1].shape}")
    if not frames:
        raise FileNotFoundError("Nenhum histórico de time encontrado!")
    
    hist = pd.concat(frames, ignore_index=True)
    hist = hist.sort_values(["clube_id", "temporada", "rodada"], kind="mergesort").reset_index(drop=True)
    
    # Calcular clean_sheet
    if "gols_sofridos" in hist.columns:
        hist["clean_sheet"] = (hist["gols_sofridos"] == 0).astype(float)
    else:
        hist["clean_sheet"] = np.nan

    # Detectar coluna de adversário
    opp_col = None
    for c in ["adversario_id", "opp_clube_id", "opponent_id"]:
        if c in hist.columns:
            opp_col = c
            break
    
    if opp_col is None:
        raise ValueError("Coluna de adversário não encontrada no histórico!")
    
    print(f"2. Coluna de adversário detectada: {opp_col}")
    
    print("3. Calculando médias móveis por time (window=3)...")
    # Base: temporada, rodada, clube_id
    team_feats = hist[["temporada", "rodada", "clube_id", opp_col]].copy()
    team_feats = team_feats.rename(columns={opp_col: "adversario_id"})
    
    for s in SCOUTS_CTX:
        if s in hist.columns:
            col = s
        elif s == "FTT" and "FTT" not in hist.columns:
            cols_sum = [c for c in ["FD", "FF", "FT", "G", "PP"] if c in hist.columns]
            if cols_sum:
                hist["FTT"] = hist[cols_sum].sum(axis=1)
                col = "FTT"
            else:
                continue
        elif s == "FA" and "FA" not in hist.columns:
            cols_sum = [c for c in ["FD", "G"] if c in hist.columns]
            if cols_sum:
                hist["FA"] = hist[cols_sum].sum(axis=1)
                col = "FA"
            else:
                continue
        else:
            continue
        
        # Rolling mean com shift(1) para evitar leakage
        team_feats[f"team_{s}_mean_3"] = (
            hist.groupby("clube_id", sort=False)[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
    
    # Clean sheet rate
    team_feats["team_clean_sheet_rate_3"] = (
        hist.groupby("clube_id", sort=False)["clean_sheet"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    print("4. Criando features do adversário (lookup cruzado)...")
    # Renomear colunas team_* para opp_* no lookup
    opp_rename = {f"team_{s}_mean_3": f"opp_{s}_mean_3" for s in SCOUTS_CTX}
    opp_rename["team_clean_sheet_rate_3"] = "opp_clean_sheet_rate_3"
    
    # Criar lookup: para cada (temporada, rodada, clube_id), temos as features
    # Vamos renomear clube_id para _opp_clube_id para fazer o merge
    lookup = team_feats[["temporada", "rodada", "clube_id"] + list(opp_rename.keys())].copy()
    lookup = lookup.rename(columns={"clube_id": "_opp_clube_id", **opp_rename})
    
    # Agora fazemos merge: team_feats tem adversario_id, lookup tem _opp_clube_id
    team_feats = team_feats.merge(
        lookup,
        left_on=["temporada", "rodada", "adversario_id"],
        right_on=["temporada", "rodada", "_opp_clube_id"],
        how="left"
    )
    
    # Remover colunas temporárias
    team_feats = team_feats.drop(columns=["adversario_id", "_opp_clube_id"], errors="ignore")

    print("5. Calculando features de interação...")
    for s in ["FTT", "FA", "DS"]:
        t_col = f"team_{s}_mean_3"
        o_col = f"opp_{s}_mean_3"
        if t_col in team_feats.columns and o_col in team_feats.columns:
            team_feats[f"diff_{s}"] = team_feats[t_col] - team_feats[o_col]
        else:
            team_feats[f"diff_{s}"] = np.nan

    print("6. Fazendo merge com dataset canônico...")
    canon = pd.read_parquet(CANON)
    merge_cols = ["temporada", "rodada", "clube_id"]
    feat_cols = [c for c in team_feats.columns if c not in merge_cols]
    
    out = canon[["temporada", "rodada", "atleta_id", "clube_id"]].merge(
        team_feats[merge_cols + feat_cols],
        on=merge_cols,
        how="left"
    )
    
    # Preencher NaN com 0
    out[feat_cols] = out[feat_cols].fillna(0)
    
    out.to_parquet(OUT, index=False)
    print(f"   ✅ Salvo: {OUT}")
    print(f"   Shape: {out.shape}")
    print(f"   Features de contexto: {len(feat_cols)}")

if __name__ == "__main__":
    main()
