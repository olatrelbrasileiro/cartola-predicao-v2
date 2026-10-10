#!/usr/bin/env python3
"""
Merge Final: Canônico v2 + Scouts Shrunk + Contexto Time/Adv → v3
"""
import pandas as pd
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON_V2 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
SCOUTS = ROOT / "data" / "interim" / "features_scouts_shrunk_v1.parquet"
CONTEXT = ROOT / "data" / "interim" / "features_team_context_v2.parquet"
OUT = ROOT / "data" / "interim" / "cartola_jogador_rodada_v3.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("1. Carregando datasets...")
    df = pd.read_parquet(CANON_V2)
    print(f"   Canônico v2: {df.shape}")
    
    df_scouts = pd.read_parquet(SCOUTS)
    print(f"   Scouts shrunk: {df_scouts.shape}")
    
    df_ctx = pd.read_parquet(CONTEXT)
    print(f"   Contexto time/adv: {df_ctx.shape}")
    
    print("2. Merge scouts...")
    scout_cols = [c for c in df_scouts.columns if c not in ["temporada", "rodada", "atleta_id", "posicao_id"]]
    df = df.merge(df_scouts[["temporada", "rodada", "atleta_id"] + scout_cols],
                  on=["temporada", "rodada", "atleta_id"], how="left")
    
    print("3. Merge contexto...")
    ctx_cols = [c for c in df_ctx.columns if c not in ["temporada", "rodada", "atleta_id", "clube_id"]]
    df = df.merge(df_ctx[["temporada", "rodada", "atleta_id"] + ctx_cols],
                  on=["temporada", "rodada", "atleta_id"], how="left")
    
    print("4. Preenchendo NaNs (0 para scouts/contexto)...")
    new_cols = scout_cols + ctx_cols
    df[new_cols] = df[new_cols].fillna(0)
    
    print("5. Validando integridade...")
    original_len = len(pd.read_parquet(CANON_V2))
    assert len(df) == original_len, f"❌ Linhas alteradas! {len(df)} vs {original_len}"
    
    print(f"   ✅ Total de linhas: {len(df)} (inalterado)")
    print(f"   ✅ Total de colunas: {df.shape[1]}")
    
    print("6. Salvando canônico v3...")
    df.to_parquet(OUT, index=False)
    
    report = REPORTS / "build_canonico_v3_report.md"
    with open(report, "w") as f:
        f.write("# Build Canônico v3 — Relatório\n\n")
        f.write(f"- Shape final: {df.shape}\n")
        f.write(f"- Features de scouts: {len(scout_cols)}\n")
        f.write(f"- Features de contexto: {len(ctx_cols)}\n")
        f.write(f"- SHA256 v2: {sha256(CANON_V2)[:16]}...\n")
        f.write(f"- SHA256 v3: {sha256(OUT)[:16]}...\n")
    
    print(f"\n✅ SUCESSO: {OUT}")
    print(f"   Relatório: {report}")

if __name__ == "__main__":
    main()
