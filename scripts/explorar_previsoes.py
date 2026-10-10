#!/usr/bin/env python3
"""
Explorador Interativo de Previsões
Permite explorar as predições salvas de forma interativa.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
PRED_DB = ROOT / "data" / "predictions"

POSICOES = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}

def listar_rodadas_disponiveis():
    """Lista todas as rodadas com predições salvas"""
    files = sorted(PRED_DB.glob("predictions_*.parquet"))
    if not files:
        print("❌ Nenhuma predição encontrada em:", PRED_DB)
        return []
    
    rodadas = []
    for f in files:
        # Extrair temporada e rodada do nome do arquivo
        name = f.stem.replace("predictions_", "")
        parts = name.split("_R")
        if len(parts) == 2:
            t, R = int(parts[0]), int(parts[1])
            rodadas.append((t, R, f))
    
    return rodadas

def carregar_predicao(temporada, rodada):
    """Carrega predições de uma rodada específica"""
    file = PRED_DB / f"predictions_{temporada}_R{rodada}.parquet"
    if not file.exists():
        return None
    return pd.read_parquet(file)

def mostrar_resumo(df, temporada, rodada):
    """Mostra resumo geral da rodada"""
    print("\n" + "=" * 70)
    print(f"📊 RESUMO: Temporada {temporada} | Rodada {rodada}")
    print("=" * 70)
    
    mae = np.mean(np.abs(df["predicao"] - df["pontos_num"]))
    sp = spearmanr(df["predicao"], df["pontos_num"])[0]
    
    print(f"MAE Global: {mae:.4f}")
    print(f"Spearman Global: {sp:.4f}")
    print(f"Total de jogadores: {len(df)}")
    
    print("\n📍 Por Posição:")
    print(f"{'Posição':<12} {'N':>5} {'MAE':>8} {'Spearman':>10}")
    print("-" * 40)
    
    for pos_id, pos_nome in POSICOES.items():
        pos_df = df[df["posicao_id"] == pos_id]
        if len(pos_df) == 0:
            continue
        mae_pos = np.mean(np.abs(pos_df["predicao"] - pos_df["pontos_num"]))
        sp_pos = spearmanr(pos_df["predicao"], pos_df["pontos_num"])[0]
        print(f"{pos_nome:<12} {len(pos_df):>5} {mae_pos:>8.4f} {sp_pos:>10.4f}")

def mostrar_top_jogadores(df, posicao=None, n=10):
    """Mostra top N jogadores por previsão"""
    if posicao:
        df = df[df["posicao_id"] == posicao]
        titulo = f"TOP {n} {POSICOES.get(posicao, 'Posição')}"
    else:
        titulo = f"TOP {n} GERAL"
    
    print("\n" + "=" * 70)
    print(f"🏆 {titulo}")
    print("=" * 70)
    
    top = df.nlargest(n, "predicao")
    
    print(f"{'Rank':>4} {'Atleta':>8} {'Pos':>5} {'Prev':>7} {'Real':>7} {'Δ':>7} {'RankReal':>9} {'RankPos':>8}")
    print("-" * 70)
    
    for _, row in top.iterrows():
        delta = row["predicao"] - row["pontos_num"]
        print(f"{row['rank_geral_pred']:>4} {int(row['atleta_id']):>8} {POSICOES.get(row['posicao_id'], '?'):>5} "
              f"{row['predicao']:>7.2f} {row['pontos_num']:>7.2f} {delta:>+7.2f} "
              f"{row['rank_geral_real']:>9} {row['rank_pos_real']:>8}")

def mostrar_jogador(df, atleta_id):
    """Mostra detalhes de um jogador específico"""
    jogador = df[df["atleta_id"] == atleta_id]
    if len(jogador) == 0:
        print(f"\n❌ Jogador {atleta_id} não encontrado")
        return
    
    row = jogador.iloc[0]
    
    print("\n" + "=" * 70)
    print(f"🔍 DETALHES DO JOGADOR {atleta_id}")
    print("=" * 70)
    print(f"Posição: {POSICOES.get(row['posicao_id'], 'Desconhecida')}")
    print(f"Previsão: {row['predicao']:.2f} pontos")
    print(f"Pontuação Real: {row['pontos_num']:.2f} pontos")
    print(f"Erro: {row['predicao'] - row['pontos_num']:+.2f} pontos")
    print(f"\nRankings:")
    print(f"  Geral (previsto): #{row['rank_geral_pred']}")
    print(f"  Geral (real): #{row['rank_geral_real']}")
    print(f"  Por posição (previsto): #{row['rank_pos_pred']}")
    print(f"  Por posição (real): #{row['rank_pos_real']}")

def menu_interativo():
    """Menu principal interativo"""
    while True:
        print("\n" + "=" * 70)
        print("🔮 EXPLORADOR DE PREVISÕES")
        print("=" * 70)
        
        rodadas = listar_rodadas_disponiveis()
        if not rodadas:
            return
        
        print("\n📅 Rodadas disponíveis:")
        for i, (t, R, _) in enumerate(rodadas, 1):
            print(f"  {i}. Temporada {t} | Rodada {R}")
        
        print("\nOpções:")
        print("  1. Ver resumo de uma rodada")
        print("  2. Ver top jogadores (geral)")
        print("  3. Ver top jogadores (por posição)")
        print("  4. Ver detalhes de um jogador")
        print("  0. Sair")
        
        opcao = input("\nEscolha uma opção: ").strip()
        
        if opcao == "0":
            break
        elif opcao == "1":
            idx = int(input("Número da rodada: ")) - 1
            if 0 <= idx < len(rodadas):
                t, R, _ = rodadas[idx]
                df = carregar_predicao(t, R)
                if df is not None:
                    mostrar_resumo(df, t, R)
                else:
                    print("❌ Erro ao carregar predição")
            else:
                print("❌ Opção inválida")
        
        elif opcao == "2":
            idx = int(input("Número da rodada: ")) - 1
            n = int(input("Quantos jogadores? "))
            if 0 <= idx < len(rodadas):
                t, R, _ = rodadas[idx]
                df = carregar_predicao(t, R)
                if df is not None:
                    mostrar_top_jogadores(df, posicao=None, n=n)
                else:
                    print("❌ Erro ao carregar predição")
            else:
                print("❌ Opção inválida")
        
        elif opcao == "3":
            idx = int(input("Número da rodada: ")) - 1
            print("\nPosições:")
            for pos_id, pos_nome in POSICOES.items():
                print(f"  {pos_id}. {pos_nome}")
            pos = int(input("Posição: "))
            n = int(input("Quantos jogadores? "))
            if 0 <= idx < len(rodadas):
                t, R, _ = rodadas[idx]
                df = carregar_predicao(t, R)
                if df is not None:
                    mostrar_top_jogadores(df, posicao=pos, n=n)
                else:
                    print("❌ Erro ao carregar predição")
            else:
                print("❌ Opção inválida")
        
        elif opcao == "4":
            idx = int(input("Número da rodada: ")) - 1
            atleta = int(input("ID do atleta: "))
            if 0 <= idx < len(rodadas):
                t, R, _ = rodadas[idx]
                df = carregar_predicao(t, R)
                if df is not None:
                    mostrar_jogador(df, atleta)
                else:
                    print("❌ Erro ao carregar predição")
            else:
                print("❌ Opção inválida")
        
        else:
            print("❌ Opção inválida")
        
        input("\nPressione Enter para continuar...")

if __name__ == "__main__":
    menu_interativo()
