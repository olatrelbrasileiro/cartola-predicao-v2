# H2A — alpha walk-forward vs alpha global

- Inicio: 2026-09-20T08:57:03
- Fim: 2026-09-20T08:57:07
- Duracao: 4.6s
- Folds: 76
- Obs: 58182
- Grid: 21 alphas (passo 0.05)
- min_inner_rounds: 3

## Resultado global

- B2 puro (α=1):       MAE = 1.5778
- B2 α=0.4 fixo:       MAE = 1.4002  (contaminado)
- B2 α walk-forward:   MAE = 1.4016

## Distribuicao de α*

- media: 0.401
- mediana: 0.400
- folds com α* < 1: 76/76
- folds com α* = 1: 0/76
- folds wf melhor que fixo(0.4): 7
- folds fixo(0.4) melhor que wf: 14
- empates: 55

Contagem de α* por fold:

- 0.35: 10
- 0.4: 55
- 0.45: 11

## Por temporada

 temporada  n_folds   b2_mae   wf_mae  fixo_04_mae  alpha_medio
      2024     29.0 1.498473 1.328531     1.327511     0.403448
      2025     29.0 1.619206 1.427534     1.426134     0.387931
      2026     18.0 1.638863 1.477478     1.475488     0.416667

## Imutabilidade

- Parquet SHA: OK
