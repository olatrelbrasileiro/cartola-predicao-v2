# Ridge v1 — summary

- Inicio: 2026-09-20T10:10:12
- Fim: 2026-09-20T10:25:26
- Duracao: 913.8s
- Folds: 76  Obs: 58182
- Spec: config/ridge_v1_spec.yaml

## Metricas oficiais

- Ridge  MAE = 1.6244  RMSE = 2.6912  Spearman = 0.4835
- B2S    MAE = 1.4016  Spearman = 0.4617
- Delta MAE (Ridge - B2S): +0.2228

## Comparacao com B2S

- Folds Ridge < B2S: 0/76
- Folds Ridge > B2S: 76/76
- Empates: 0
- Mediana delta MAE: +0.2174

## Alpha selecionado

- media: 33.425
- mediana: 10.000
- contagem: {0.01: 29, 10.0: 24, 100.0: 23}

## Por temporada

 temporada  n_folds  ridge_mae  b2s_mae    delta  alpha_medio
      2024     29.0   1.547445 1.328531 0.218914    25.517241
      2025     29.0   1.652451 1.427534 0.224917     0.010000
      2026     18.0   1.703170 1.477478 0.225692   100.000000

## Criterio de sucesso

- MAE < 1.4016: FAIL (1.6244)
- Folds melhor >= 39: FAIL (0/76)
- Spearman >= 0.4617: PASS (0.4835)

## Imutabilidade

- Parquet: OK
- B2S folds: OK
