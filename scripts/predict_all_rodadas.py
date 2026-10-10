#!/usr/bin/env python3
"""
Predict All Rodadas (Otimizado)
Gera predições para todas as rodadas disponíveis de uma só vez.
Usa o script otimizado que salva e reutiliza modelos treinados.
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
PRED_DB = ROOT / "data" / "predictions"
CANON_V4 = INTERIM / "cartola_jogador_rodada_v4.parquet"

def main():
    print("=" * 70)
    print("🔮 PREDICT ALL RODADAS (OTIMIZADO)")
    print("=" * 70)
    
    print("\n1. Descobrindo todas as rodadas disponíveis...")
    df = pd.read_parquet(CANON_V4)
    
    rodadas = df.groupby(["temporada", "rodada"]).size().reset_index(name="count")
    rodadas = rodadas.sort_values(["temporada", "rodada"])
    rodadas_validas = rodadas[rodadas["count"] >= 50]
    
    print(f"   Total de rodadas válidas: {len(rodadas_validas)}")
    
    PRED_DB.mkdir(parents=True, exist_ok=True)
    existentes = set()
    for f in PRED_DB.glob("predictions_*.parquet"):
        name = f.stem.replace("predictions_", "")
        existentes.add(name)
    
    print(f"   Predições já existentes: {len(existentes)}")
    
    rodadas_gerar = []
    for _, row in rodadas_validas.iterrows():
        t, R = int(row["temporada"]), int(row["rodada"])
        name = f"{t}_R{R}"
        if name not in existentes:
            rodadas_gerar.append((t, R))
    
    print(f"   Rodadas a gerar: {len(rodadas_gerar)}")
    
    if len(rodadas_gerar) == 0:
        print("\n✅ Todas as rodadas já foram geradas!")
        print(f"   Use: python scripts/explorar_previsoes.py")
        return
    
    print("\n2. Gerando predições...")
    print("=" * 70)
    
    total = len(rodadas_gerar)
    sucesso = 0
    falha = 0
    
    for i, (t, R) in enumerate(rodadas_gerar, 1):
        print(f"\n[{i}/{total}] Temporada {t} | Rodada {R}")
        print("-" * 50)
        
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "predict_rodada_otimizado.py"),
            "--temporada", str(t),
            "--rodada", str(R)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Extrair MAE e Spearman do output
                for line in result.stdout.split("\n"):
                    if "MAE:" in line or "Spearman:" in line:
                        print(f"   {line.strip()}")
                sucesso += 1
            else:
                # Mostrar erro resumido
                err = result.stderr.strip().split("\n")[-1] if result.stderr else "Erro desconhecido"
                print(f"   ❌ {err[:80]}")
                falha += 1
        except subprocess.TimeoutExpired:
            print("   ❌ Timeout")
            falha += 1
        except Exception as e:
            print(f"   ❌ {e}")
            falha += 1
    
    print("\n" + "=" * 70)
    print("📊 RESUMO FINAL")
    print("=" * 70)
    print(f"Total: {total} | Sucesso: {sucesso} | Falha: {falha}")
    print(f"\n📁 Predições em: {PRED_DB}")
    print(f"📁 Modelos em: {ROOT / 'data' / 'models'}")
    print(f"\n🚀 Explore com: python scripts/explorar_previsoes.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
