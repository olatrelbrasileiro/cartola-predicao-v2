#!/usr/bin/env python3
"""
Atualiza o catboost_v3.py para usar o dataset v3 e todas as novas features.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SRC = SCRIPTS / "catboost_pooling.py"
DST = SCRIPTS / "catboost_v3.py"

# 1. Definir as novas features geradas pelos scripts
SCOUTS = ["G", "A", "FD", "FF", "FT", "FS", "DE", "SG", "DS", "FC", "DP", "GS", "CA", "CV"]
SCOUTS_STD = ["G", "A", "DE", "DP", "SG"]
PARAMS = ["mean_hist", "mean_3", "mean_5", "shrunk", "momentum"]

new_features = []
for s in SCOUTS:
    for p in PARAMS:
        new_features.append(f"scout_{s}_{p}")
    if s in SCOUTS_STD:
        new_features.append(f"scout_{s}_std_5")

CONTEXT_FEATURES = [
    "team_FTT_mean_3", "team_FA_mean_3", "team_DS_mean_3", "team_FC_mean_3", 
    "team_G_mean_3", "team_FS_mean_3", "team_DE_mean_3", "team_clean_sheet_rate_3",
    "opp_FTT_mean_3", "opp_FA_mean_3", "opp_DS_mean_3", "opp_FC_mean_3", 
    "opp_G_mean_3", "opp_FS_mean_3", "opp_DE_mean_3", "opp_clean_sheet_rate_3",
    "diff_FTT", "diff_FA", "diff_DS"
]

ALL_NEW_FEATURES = new_features + CONTEXT_FEATURES

def main():
    print("=" * 65)
    print("  CONFIGURANDO CATBOOST v3 OFICIAL")
    print("=" * 65)

    if not SRC.exists():
        print(f"❌ Erro: {SRC.name} não encontrado. Execute o script de promoção primeiro.")
        return

    print(f"\n1. Lendo {SRC.name}...")
    code = SRC.read_text(encoding="utf-8")

    print("2. Atualizando caminho do dataset para v3...")
    code = code.replace("cartola_jogador_rodada_v2.parquet", "cartola_jogador_rodada_v3.parquet")

    print("3. Injetando novas features em FEATURES_COM...")
    # Encontrar a lista FEATURES_COM e injetar as novas features antes do fechamento ]
    pattern = r"(FEATURES_COM\s*=\s*\[)(.*?)(\])"
    match = re.search(pattern, code, re.DOTALL)
    
    if match:
        prefix = match.group(1)
        conteudo = match.group(2)
        sufixo = match.group(3)
        
        # Filtrar features que já existem para não duplicar
        existentes = [f for f in ALL_NEW_FEATURES if f in conteudo]
        a_adicionar = [f for f in ALL_NEW_FEATURES if f not in conteudo]
        
        if a_adicionar:
            # Formatar a string de novas features
            novas_str = ",\n    ".join([f'"{f}"' for f in a_adicionar])
            conteudo_limpo = conteudo.rstrip()
            if conteudo_limpo and not conteudo_limpo.endswith(","):
                conteudo_limpo += ","
            
            novo_conteudo = (
                conteudo_limpo + 
                "\n\n    # --- NOVAS FEATURES v3: Scouts Granulares e Contexto ---\n    " + 
                novas_str + "\n"
            )
            code = code[:match.start()] + prefix + novo_conteudo + sufixo + code[match.end():]
            print(f"   ✅ {len(a_adicionar)} features injetadas com sucesso.")
            if existentes:
                print(f"   ⏭️  {len(existentes)} features já existiam (ignoradas).")
        else:
            print("   ⏭️  Todas as features já estão presentes.")
    else:
        print("   ⚠️  Não foi possível encontrar FEATURES_COM automaticamente.")

    print("4. Salvando catboost_v3.py...")
    DST.write_text(code, encoding="utf-8")
    print(f"   ✅ Salvo: {DST.name}")

    print("\n" + "=" * 65)
    print("  🎉 CONFIGURAÇÃO CONCLUÍDA!")
    print("=" * 65)
    print("\nPara treinar o modelo oficial v3 com todas as 130+ features, rode:")
    print("  python -u scripts/catboost_v3.py 2>&1 | tee reports/treino_catboost_v3.log\n")

if __name__ == "__main__":
    main()
