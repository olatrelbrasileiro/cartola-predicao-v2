# CatBoost v1 — Summary

- Gerado em: 2026-09-21T00:12:26
- Duracao: 0.2s
- CatBoost: 1.2.10
- Folds: 76
- Obs avaliadas: 58182
- Cache N2: 76/76 folds ja em cache antes da execucao

## Parametros (pre-registrados)

- loss_function: MAE
- iterations: 500
- depth: 6
- learning_rate: 0.05
- l2_leaf_reg: 3.0
- random_seed: 42
- thread_count: -1
- verbose: 0
- allow_writing_files: False

## Metricas oficiais

- **CatBoost MAE global ponderado: 1.3496**
- CatBoost RMSE: 2.8171
- CatBoost Spearman medio: 0.4535
- B2S MAE: 1.4016
- B2S Spearman medio: 0.4617
- Delta MAE (CB - B2S): -0.0520
- Melhoria absoluta: +0.0520
- Melhoria percentual: +3.71%

## Consistencia por fold

- Folds CB < B2S: 70/76
- Folds CB > B2S: 6/76
- Empates: 0
- Mediana delta: -0.0484
- Media delta: -0.0519

## Por temporada

 temporada  n_folds   cb_mae  cb_sem_mae  b2s_mae     delta  cb_spearman  b2s_spearman
      2024       29 1.283818    1.282428 1.328531 -0.044713     0.452091      0.462645
      2025       29 1.369995    1.374008 1.427534 -0.057539     0.446019      0.452075
      2026       18 1.422687    1.418760 1.477478 -0.054791     0.467894      0.475724

## Ablation — sem Confronto

- CatBoost com Confronto: 1.3496
- CatBoost sem Confronto: 1.3497
- Delta (sem - com): +0.0001
- Spearman com: 0.4535
- Spearman sem: 0.4510

## Feature importance (media dos 76 folds, nao e causalidade)

              feature  importance
          hist_shrunk    9.344425
           posicao_id    8.743268
           rec_mean_3    8.063730
          basic_rec_5    6.761442
           basic_mean    6.430137
          entrou_rate    6.319418
           rec_mean_5    5.777430
            hist_mean    5.260913
               p_away    5.036656
               hist_n    4.908614
               p_draw    4.842027
favoritismo_home_away    4.455483
                mando    4.376931
               p_home    4.376532
             momentum    3.747060
            ga_shrunk    3.315825
            rec_std_5    3.179060
             hist_std    3.047630
             ga_rec_5    1.183883
          ga_per_game    0.818702
        is_cold_start    0.010833

## Importancia por grupo

       grupo  importance_total                                             features
   Confronto         23.087630 p_home, p_draw, p_away, mando, favoritismo_home_away
   Histórico         22.561582             hist_mean, hist_std, hist_n, hist_shrunk
     Recente         20.767280          rec_mean_5, rec_mean_3, rec_std_5, momentum
      Básico         13.191579                              basic_mean, basic_rec_5
     Posição          8.743268                                           posicao_id
Oportunidade          6.330251                           entrou_rate, is_cold_start
         G+A          5.318410                     ga_per_game, ga_rec_5, ga_shrunk

## Integridade

- Parquet canonico inalterado: True
- Temporal-safe inalterado: True
- B2S folds inalterado: True
