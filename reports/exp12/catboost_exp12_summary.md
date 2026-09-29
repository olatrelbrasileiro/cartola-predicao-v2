# EXP-12 F3 - CatBoost + contexto F2

- base: cartola_jogador_rodada_exp12.parquet
- features: 21 FEATURES_COM + 20 F2 = 41
- CAT_FEATURES = ['posicao_id']
- protocolo: (temporada < t) OR (temporada == t AND rodada < R) & elegivel
- folds=76 fits=76 preds=23691 tempo=1095.3s

## Pooled

- MAE = 2.6323
- RMSE = 3.7510
- Spearman = 0.3415

## Cobertura F2 no dataset (pos-merge)

- 2024: 0/28499 (0.0%)
- 2025: 0/28589 (0.0%)
- 2026: 18463/20165 (91.6%)

## f2_available nas previsoes

- com F2: 5425/23691 (22.9%)
- sem F2: 18266/23691 (77.1%)

### Com F2

- n = 5425
- MAE = 2.6839
- RMSE = 3.8433
- Spearman = 0.3478

### Sem F2

- n = 18266
- MAE = 2.6170
- RMSE = 3.7231
- Spearman = 0.3389

## Por temporada

- 2024: n=9084 MAE=2.5322 RMSE=3.5769 SP=0.3202
- 2025: n=8881 MAE=2.6994 RMSE=3.8567 SP=0.3543
- 2026: n=5726 MAE=2.6871 RMSE=3.8532 SP=0.3479

## Por posicao

- posicao 1: n=1433 MAE=3.2615 RMSE=4.0582 SP=0.0877
- posicao 2: n=3607 MAE=2.9924 RMSE=3.8808 SP=0.2147
- posicao 3: n=3363 MAE=2.6516 RMSE=3.4562 SP=0.2050
- posicao 4: n=7685 MAE=2.3051 RMSE=3.4141 SP=0.3040
- posicao 5: n=6215 MAE=2.8520 RMSE=4.3620 SP=0.3353
- posicao 6: n=1388 MAE=1.8281 RMSE=2.3106 SP=0.3309

## Comparacao fold a fold

### vs catboost_v2

- folds comuns: 76
- exp12 melhor: 54
- exp12 pior: 22
- delta medio (exp12 - v2): -0.0293

### vs catboost_pooling (mesmo protocolo, sem F2)

- folds comuns: 76
- exp12 melhor: 36
- exp12 pior: 40
- delta medio (exp12 - pooling): 0.0014


## Notas

- Protocolo pooled acumulado identico a catboost_pooling.py.
- Comparacao com v2 confunde protocolo (single-season vs pooled) + F2.
- Comparacao com pooling isola o efeito das 20 F2 (se disponivel).
- Sem decisao de promocao.
