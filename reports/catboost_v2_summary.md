# CatBoost v2 — Summary

- Gerado em: 2026-09-21T19:00:40
- Duracao: 439.8s
- CatBoost: 1.2.10
- Folds: 76
- Obs avaliadas (metric): 23691
- n_catboost_fits: 152

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
- cat_features: ['posicao_id']

## Metricas globais

- **CatBoost v2 MAE ponderado: 2.6612**
- CatBoost v2 RMSE ponderado: 3.7703
- CatBoost v2 Spearman medio: 0.3264
- **B2S P0-corrected MAE: 2.7485**
- Delta MAE (CB - B2S): -0.0873
- Improvement: +0.0873

## Consistencia por fold

- Folds CB < B2S: 64/76
- Folds empate: 0
- Folds CB > B2S: 12
- Mediana do delta: -0.0778

## Por temporada

 temporada  n_test      mae  mae_sem  b2s_mae     delta  spearman
      2024    9084 2.542488 2.562013 2.615037 -0.072549  0.317904
      2025    8881 2.740265 2.768264 2.841995 -0.101730  0.332769
      2026    5726 2.726851 2.745719 2.815298 -0.088447  0.329666

## Criterios de promocao

- C1 (MAE global < B2S - 1e-3): True
- C2 (>= 50/76 folds melhores): True (64/76)
- C3 (3/3 temporadas melhores): True
- C4 (nenhuma temporada > 1.5%): True
- C5 (mediana delta < -1e-3): True

**Decisao: PROMOTE_CATBOOST_V2**

Motivo: todos os 5 criterios atendidos

## Ablation — COM vs SEM confronto

- MAE com confronto: 2.6612
- MAE sem confronto: 2.6837
- Delta (sem - com): +0.0225

## Feature importance (media dos folds)

              feature  importance
             momentum   11.012537
           posicao_id    9.558600
       hist_shrunk_k7    6.623815
                mando    6.522846
          entrou_rate    5.350445
               p_draw    5.198430
           rec_mean_3    4.972139
           basic_mean    4.944128
               p_away    4.900374
             hist_std    4.692258
          basic_rec_5    4.509339
         ga_shrunk_k7    4.498584
               hist_n    4.406737
            rec_std_5    4.338153
           rec_mean_5    4.057750
            hist_mean    3.918417
favoritismo_home_away    3.604041
               p_home    3.455746
          ga_per_game    2.229653
             ga_rec_5    1.204976
        is_cold_start    0.001035

## Importancia por grupo

       grupo  importance_total                                             features
     Recente         24.380578          rec_mean_5, rec_mean_3, rec_std_5, momentum
   Confronto         23.681436 p_home, p_draw, p_away, mando, favoritismo_home_away
   Historico         19.641226          hist_mean, hist_std, hist_n, hist_shrunk_k7
     Posicao          9.558600                                           posicao_id
      Basico          9.453467                              basic_mean, basic_rec_5
          GA          7.933213                  ga_per_game, ga_rec_5, ga_shrunk_k7
Oportunidade          5.351479                           entrou_rate, is_cold_start

## Integridade

- canonical: OK
- harness: OK
- p0_folds: OK
