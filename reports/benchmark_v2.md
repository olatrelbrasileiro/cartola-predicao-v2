# Benchmark oficial da Etapa 6+

## Referência
**B2S = B2 + shrinkage walk-forward**
- Especificação: `config/b2s_spec.yaml`
- MAE global ponderado: **1.4016**
- Spearman médio: **0.4617**
- Folds: 76 (29 + 29 + 18)

## Critério de sucesso
Um modelo é considerado progresso somente se:
1. MAE global ponderado < 1.4016.
2. Ganho consistente na maioria dos folds (≥ 39/76 melhorando).
3. Spearman não regride em relação a 0.4617.
4. Reprodutível: sem tuning contaminado, features pré-rodada, walk-forward.

## Baselines de referência
| Modelo | MAE | Spearman | Status |
|---|---|---|---|
| B0 | 2.0748 | — | histórico |
| B1 | 1.6012 | 0.4553 | histórico |
| B2 | 1.5778 | 0.4617 | histórico |
| B3 | 1.5339 | 0.3454 | diagnóstico |
| **B2S** | **1.4016** | **0.4617** | **benchmark oficial** |
