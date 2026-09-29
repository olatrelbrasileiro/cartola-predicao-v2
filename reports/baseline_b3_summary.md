# ETAPA 5 — B3 (componente basico, corrigido)

- Inicio: 2026-09-20T07:43:08
- Fim: 2026-09-20T07:43:13
- Duracao: 4.33s

## Definicao

`diff_scout_R = ffilled_cumul(R) - ffilled_cumul(R-1)`
`basic_score_R = sum(diff_scout_R * peso_scout(temporada))`
`prediction_i = mean(basic_score historico do jogador)` ou fallback B0

Scouts no canonico sao CUMULATIVOS (Etapa 3.c). Diff aplicado.
Pesos por temporada: 2024 DE=1.0, DS=1.2; 2025-2026 DE=1.3, DS=1.5.

## Folds

- Total: 76
- 2024: 29
- 2025: 29
- 2026: 18

## Metricas

- MAE medio simples: 1.5344
- **MAE global ponderado: 1.5339**
- RMSE agregado: 2.8969
- Spearman medio: 0.3454
- Folds com Spearman valido: 76
- Folds com Spearman indefinido: 0
- Observacoes avaliadas: 58182

## Coverage

- Com historico basico: 57560
- Fallback B0: 622
- Coverage: 0.9893

## Comparacao com B2

- B2 MAE: 1.5778
- B3 MAE: 1.5339
- Melhoria absoluta: +0.0439
- Melhoria percentual: +2.78%
- Folds B3 < B2: 53
- Folds B3 > B2: 23
- Empates: 0
- Mediana delta: -0.0322
- Media delta: -0.0443

## Definicao congelada dos pesos (por temporada)

| scout | peso_2024 | peso_2025 | peso_2026 | incluido_no_basico |
|---|---|---|---|---|
| DS | 1.2 | 1.5 | 1.5 | SIM |
| FT | 3.0 | 3.0 | 3.0 | SIM |
| FF | 0.8 | 0.8 | 0.8 | SIM |
| FD | 1.2 | 1.2 | 1.2 | SIM |
| DE | 1.0 | 1.3 | 1.3 | SIM |
| FC | -0.3 | -0.3 | -0.3 | SIM |
| FS | 0.5 | 0.5 | 0.5 | SIM |
| I | -0.1 | -0.1 | -0.1 | SIM |
| CV | -3.0 | -3.0 | -3.0 | SIM |
| CA | -1.0 | -1.0 | -1.0 | SIM |
| G | — | — | — | NAO |
| A | — | — | — | NAO |
| SG | — | — | — | NAO |
| PS | — | — | — | NAO |
| PP | — | — | — | NAO |
| DP | — | — | — | NAO |
| GC | — | — | — | NAO |
| GS | — | — | — | NAO |
| PC | — | — | — | NAO |
| V | — | — | — | NAO |

## Imutabilidade

- Parquet antes: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4
- Parquet depois: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4
- Imutavel: True
- B0/B1/B2 inalterados: {'b0_folds': True, 'b1_folds': True, 'b2_folds': True}
