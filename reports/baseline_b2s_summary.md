# B2S — B2 + shrinkage walk-forward

- Inicio: 2026-09-20T09:00:19
- Fim: 2026-09-20T09:00:24
- Duracao: 4.9s
- Folds: 76  Obs: 58182
- Especificacao: config/b2s_spec.yaml

## Metrica oficial

- B2 puro:          MAE = 1.5778  RMSE = 2.8261  Spearman = 0.4617
- B2S walk-fwd:     MAE = 1.4016  RMSE = 2.8899  Spearman = 0.4617
- B2S α=0.4 fixo:   MAE = 1.4002  (referencia exploratoria, NAO oficial)

## Estabilidade

- α* medio: 0.401
- α* mediana: 0.400
- Contagem: {0.35: 10, 0.4: 55, 0.45: 11}
- Folds B2S melhor: 74/76
- Folds B2 melhor: 2/76
- Empates: 0
- Mediana delta (B2S - B2): -0.1569

## Por temporada

 temporada  n_folds   b2_mae  b2s_mae  alpha_medio
      2024     29.0 1.498473 1.328531     0.403448
      2025     29.0 1.619206 1.427534     0.387931
      2026     18.0 1.638863 1.477478     0.416667

## Imutabilidade

- Parquet: OK
- B2 folds: OK
