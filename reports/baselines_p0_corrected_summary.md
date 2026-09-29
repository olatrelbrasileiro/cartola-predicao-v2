# Baselines P0-corrected — Summary

- Gerado em: 2026-09-21T12:13:04
- Duracao: 575.6s
- Folds: 76
- Obs no teste (participantes): 23691

## Definicoes (explicitas)

- B0 = media_pontos_dos_participantes_no_treino
- ANCHOR (B2S) = mediana_pontos_dos_participantes_no_treino
- FALLBACK = b0_const_do_fold
- Avaliacao = somente participantes do test (entrou_em_campo=True)

## Auditoria de participacao

                                        0
total_linhas                   77253.0000
total_participantes            31815.0000
total_nao_participantes        45438.0000
pct_participantes                  0.4118
pct_nao_participantes              0.5882
zeros_entre_participantes       2562.0000
zeros_entre_nao_participantes  45330.0000
flag_B_total                     108.0000

## Resultados corrigidos

| Baseline | MAE ponderado | MAE medio | RMSE | Spearman |
|---|---|---|---|---|
| B0 | 2.9494 | 2.9514 | 3.8277 | nan |
| B1 | 2.8587 | 2.8595 | 3.7986 | 0.2808 |
| B2 | 2.9413 | 2.9425 | 3.9378 | 0.2552 |
| B3 | 2.9287 | 2.9288 | 4.1708 | 0.1176 |
| B2S | 2.7485 | 2.7495 | 3.8057 | 0.2552 |

## Comparacao com referencia antiga (nao valida para o problema condicional)

| Baseline | MAE antigo | MAE corrigido | Δ |
|---|---|---|---|
| B0 | 2.0748 | 2.9494 | +0.8746 |
| B1 | 1.6012 | 2.8587 | +1.2575 |
| B2 | 1.5778 | 2.9413 | +1.3635 |
| B2S | 1.4016 | 2.7485 | +1.3469 |

## B2S por temporada

 temporada  n_folds  n_test   b0_mae   b1_mae   b2_mae   b3_mae  b2s_mae  b2s_spearman
      2024     29.0  9084.0 2.812543 2.724674 2.807290 2.783764 2.615037      0.249924
      2025     29.0  8881.0 3.058061 2.958945 3.037458 3.026172 2.841995      0.258150
      2026     18.0  5726.0 2.998122 2.915799 3.004955 3.007440 2.815298      0.258787

## Distribuicao de alpha no B2S

b2s_alpha_star
0.25     4
0.30    44
0.35    28

## Distribuicao de n_inner_rounds por fold

count    76.000000
mean     19.697368
std       8.132273
min       7.000000
25%      13.000000
50%      19.000000
75%      26.000000
max      35.000000

## Integridade

- SHA-256 Parquet antes : `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- SHA-256 Parquet depois: `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Inalterado: **True**
