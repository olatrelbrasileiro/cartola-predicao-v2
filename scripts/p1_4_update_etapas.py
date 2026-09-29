#!/usr/bin/env python3
"""
P1.4 — Atualiza ETAPAS.md.

Regras:
- Backup ISO antes de alterar.
- Nao duplica P0.
- Nao sobrescreve P0 existente.
- Atualiza bloco P1 se existir; senao, adiciona.
- Nao marca [x] em P1.* automaticamente.
- So escreve o bloco P0 com valores REAIS se:
    * P0 nao existe E
    * reports/baselines_p0_corrected_folds.csv existe.
  Caso contrario, nao inventa P0.
"""
import re
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd


def detect_root():
    for c in [Path.home()/"cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c/"ETAPAS.md").exists():
            return c
    raise RuntimeError("raiz nao encontrada")


ROOT = detect_root()
ETAPAS = ROOT/"ETAPAS.md"
P0_FOLDS = ROOT/"reports"/"baselines_p0_corrected_folds.csv"


def fmt_p0_from_real(folds_csv):
    """Le os valores reais e produz bloco P0. Retorna None se nao for possivel."""
    try:
        df = pd.read_csv(folds_csv)
    except Exception:
        return None
    if len(df) == 0:
        return None
    n = df["n_test_participantes"].sum()
    def wavg(col):
        return float((df[col]*df["n_test_participantes"]).sum()/n)
    def sp(col):
        return float(df[col].mean())
    lines = []
    lines.append("\n---\n")
    lines.append("\n## ETAPA P0 — Correcao de participacao (CONCLUIDA)\n\n")
    lines.append(f"Data: {datetime.now().date().isoformat()}\n\n")
    lines.append("### Definicoes oficiais\n")
    lines.append("- Estatisticas de desempenho usam SOMENTE `entrou_em_campo=True`.\n")
    lines.append("- Recente 5/3 = ultimas 5/3 PARTICIPACOES (nao linhas).\n")
    lines.append("- Avaliacao = SOMENTE participantes no teste.\n")
    lines.append("- k=7 mantido como pre-registrado.\n")
    lines.append("- Nao participantes permanecem no dataset.\n\n")
    lines.append("### Benchmark oficial P0-corrected (valores reais)\n\n")
    lines.append("| Baseline | MAE ponderado | Spearman medio |\n|---|---|---|\n")
    for b, c_mae, c_sp in [("B0","b0_mae","b0_spearman"),
                            ("B1","b1_mae","b1_spearman"),
                            ("B2","b2_mae","b2_spearman"),
                            ("B3","b3_mae","b3_spearman"),
                            ("B2S","b2s_mae","b2s_spearman")]:
        lines.append(f"| {b} | {wavg(c_mae):.4f} | {sp(c_sp):.4f} |\n")
    lines.append(f"\nalpha medio B2S: {df['b2s_alpha_star'].mean():.3f}\n\n")
    lines.append("Arquivo: `reports/baselines_p0_corrected_folds.csv`.\n")
    lines.append("Os valores historicos (B2S MAE 1.4016) NAO sao comparaveis.\n")
    return "".join(lines)


P1_BLOCK = """

---

## ETAPA P1 — Auditoria metodologica (EM ANDAMENTO)

### Sub-etapas

- [ ] P1.1 — Justificativa quantitativa do k=7
- [ ] P1.2 — Auditoria temporal do status_id
- [ ] P1.3 — Auditoria numerica de 2025
- [ ] P1.4 — Atualizacao deste ETAPAS.md

### Regras de marcacao
- Cada sub-etapa so vira [x] apos sua auditoria rodar e passar.
- P1.2 pode ser INDETERMINADO sem falha.
- P1.4 so vira [x] apos P1.1..P1.3 terem rodado.

## Proximas etapas (nao iniciadas)

- ETAPA B — estudo de k (nao iniciada)
- Ridge v2 com base corrigida (nao iniciada)
- CatBoost v2 com base corrigida (nao iniciada)
- NDCG / pooling / oportunidade (nao iniciadas)
"""


def main():
    s = ETAPAS.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%dT%H%M")
    backup = ETAPAS.with_suffix(f".md.bak.{ts}")
    backup.write_text(s, encoding="utf-8")
    print(f"backup: {backup.name}")

    # --- P0: nao sobrescrever se existir
    if "## ETAPA P0" in s:
        print("P0 ja existe — nao alterado.")
    else:
        if P0_FOLDS.exists():
            bloco_p0 = fmt_p0_from_real(P0_FOLDS)
            if bloco_p0:
                s = s.rstrip() + bloco_p0 + "\n"
                print("P0 adicionado com valores reais.")
            else:
                print("AVISO: P0_FOLDS existe mas nao foi possivel ler. P0 nao escrito.")
        else:
            print("AVISO: P0 nao existe no ETAPAS.md e "
                  "reports/baselines_p0_corrected_folds.csv tambem nao. P0 nao escrito.")

    # --- P1: substituir se existir, senao adicionar
    if "## ETAPA P1" in s:
        # substitui do "## ETAPA P1" ate o proximo "## " ou EOF
        pattern = re.compile(r"## ETAPA P1.*?(?=\n## |\Z)", re.DOTALL)
        s = pattern.sub(P1_BLOCK.strip() + "\n", s, count=1)
        print("P1 substituido.")
    else:
        s = s.rstrip() + P1_BLOCK
        print("P1 adicionado.")

    ETAPAS.write_text(s, encoding="utf-8")
    print("ETAPAS.md atualizado.")


if __name__ == "__main__":
    main()
