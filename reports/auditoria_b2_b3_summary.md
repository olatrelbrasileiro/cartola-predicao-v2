# Auditoria B2 x B3

- Inicio: 2026-09-20T08:47:25
- Fim: 2026-09-20T08:47:56
- Duracao: 31.81s
- Observacoes analisadas: 58182
- Folds: 76

## Factos observados

- B2 MAE ponderado: 1.5778
- B3 MAE ponderado: 1.5339
- Delta MAE (B3-B2): -0.0439
- B2 RMSE ponderado: 2.8261
- B3 RMSE ponderado: 2.8969
- B2 Spearman medio: 0.4617
- B3 Spearman medio: 0.3454
- Folds B3 melhor (MAE): 53/76
- Folds B2 melhor (MAE): 23/76
- Empates (MAE): 0
- Folds B3 melhor (Spearman): 0/76
- Folds B2 melhor (Spearman): 76/76

## Por temporada

 temporada  n_folds  b2_mae_ponderado  b3_mae_ponderado  delta_mae  melhoria_pct  folds_b3_melhor  folds_b2_melhor  media_delta  mediana_delta  b2_spearman_medio  b3_spearman_medio
      2024       29          1.498473          1.440580  -0.057893      3.863463               25                4    -0.058132      -0.041561           0.462648           0.346816
      2025       29          1.619206          1.585711  -0.033495      2.068606               16               13    -0.034081      -0.021687           0.452078           0.331832
      2026       18          1.638863          1.600613  -0.038250      2.333922               12                6    -0.038319      -0.019138           0.475724           0.365174

## Variancia das previsoes (media por fold)

- std B2 = 1.9512   std B3 = 0.9464
- IQR B2 = 2.3693   IQR B3 = 1.3356
- range B2 = 12.1879  range B3 = 5.3387
- n_unique B2 = 287.2  n_unique B3 = 425.7

## Por faixa do target

bucket_target     n   mae_b2    mae_b3  delta_mae_b3_b2  mean_pred_b2  mean_pred_b3  mean_y_true  folds_b3_melhor  folds_b2_melhor  empates
   pontos_0_3  9329 1.651305  0.982190        -0.669115      2.307838      1.254034     1.522814             5914             2983      432
   pontos_3_6  5113 2.297110  3.120742         0.823631      2.921206      1.338907     4.437188             1506             3418      189
  pontos_6_10  3462 4.711315  6.395743         1.684428      3.177177      1.386698     7.782441              742             2611      109
  pontos_eq_0 36408 0.737780  0.534511        -0.203269      0.721663      0.528519     0.000000            10267            12433    13708
 pontos_gt_10  1584 9.600417 11.388949         1.788531      3.535296      1.738280    13.127229              325             1225       34
  pontos_lt_0  2286 2.744269  1.962332        -0.781937      1.948676      1.168873    -0.793132             1422              737      127

## Por historico_n

bucket_historico     n   mae_b2   mae_b3  delta_mae_b3_b2  mean_pred_b2  mean_pred_b3  mean_y_true
             n_0   622 1.650934 1.650934         0.000000      1.486046      1.486046     0.513923
           n_1_2  1242 1.062786 1.033341        -0.029444      0.635298      0.261594     0.945177
           n_3_5  1916 1.092892 1.129126         0.036234      0.883303      0.380441     1.072923
          n_6_10  7868 1.419339 1.392595        -0.026744      1.356161      0.726389     1.309096
         n_11_20 23141 1.575907 1.521369        -0.054538      1.439823      0.831837     1.407880
         n_gt_20 23393 1.698202 1.650541        -0.047661      1.555989      0.900711     1.555503

## Por posicao

 posicao_id     n   mae_b2   mae_b3  delta_mae
          1  6107 1.046972 1.081252   0.034279
          2  8981 1.911405 1.759788  -0.151617
          3  9563 1.461583 1.293196  -0.168387
          4 17766 1.437821 1.363427  -0.074393
          5 14249 1.768740 1.611382  -0.157358
          6  1516 2.320505 4.808580   2.488075

## Correlações de erro

- corr(erro_b2, erro_b3) = 0.8469
- corr(abs_erro_b2, abs_erro_b3) = 0.8104

## Complementaridade

- Limiar erro alto (p75 global combinado) = 2.0133
- % obs B3 melhor que B2: 34.68%
- % obs B2 melhor que B3: 40.23%
- % empates: 25.09%
- % ambos erram alto: 16.63%

## TOP 10 folds onde B3 mais melhora (MAE)

 temporada  rodada   b2_mae   b3_mae  delta_mae
      2024      19 1.409844 1.139708  -0.270136
      2026      21 1.555929 1.300530  -0.255399
      2024      17 1.411875 1.164065  -0.247810
      2025      12 1.695158 1.470864  -0.224293
      2026      19 1.634673 1.411743  -0.222930
      2025      34 1.517487 1.305144  -0.212343
      2024      16 1.493761 1.303850  -0.189912
      2025      13 1.759334 1.584962  -0.174372
      2025      10 1.701785 1.540160  -0.161625
      2025      36 1.566969 1.408123  -0.158846

## TOP 10 folds onde B3 mais piora (MAE)

 temporada  rodada   b2_mae   b3_mae  delta_mae
      2026      18 1.740817 1.790536   0.049719
      2024      38 1.406127 1.467647   0.061519
      2025      17 1.509689 1.592206   0.082518
      2026      25 1.664786 1.751392   0.086606
      2026      24 1.687188 1.778982   0.091793
      2025      11 1.707236 1.799279   0.092043
      2024      29 1.459299 1.554730   0.095430
      2025      21 1.817915 1.946514   0.128599
      2025      33 1.439348 1.571405   0.132057
      2025      15 1.576353 1.715542   0.139189

## TOP 10 quedas de Spearman B3 vs B2

 temporada  rodada  b2_spearman  b3_spearman  delta_spearman
      2024      29     0.580669     0.387860       -0.192809
      2025      29     0.519500     0.357139       -0.162361
      2024      26     0.447178     0.286277       -0.160901
      2024      38     0.520941     0.360337       -0.160605
      2025      31     0.513268     0.355335       -0.157933
      2025      26     0.505520     0.347886       -0.157634
      2026      16     0.486063     0.329254       -0.156809
      2025      20     0.429069     0.274139       -0.154930
      2025      21     0.520613     0.369408       -0.151205
      2025      33     0.597839     0.447214       -0.150625

## TOP 10 ganhos de Spearman B3 vs B2

 temporada  rodada  b2_spearman  b3_spearman  delta_spearman
      2024      16     0.424662     0.337400       -0.087262
      2025      12     0.356303     0.269965       -0.086337
      2024      23     0.387118     0.304661       -0.082457
      2025      36     0.365355     0.283255       -0.082100
      2025      18     0.417948     0.336691       -0.081257
      2025      34     0.398045     0.320378       -0.077668
      2025      10     0.378000     0.305795       -0.072205
      2026      22     0.451079     0.384697       -0.066382
      2026      21     0.349893     0.302439       -0.047454
      2025      13     0.316302     0.272928       -0.043374

## Perfil dos 53 folds em que B3 ganha

- n_folds: 53
- mean_pct_zero: 0.6347722150596257
- mean_pct_neg: 0.03961425307036039
- mean_pct_gt6: 0.08093861840816027
- mean_pct_gt10: 0.024862899941650238
- mean_std_b2: 1.9518302810221313
- mean_std_b3: 0.9403804741702827
- mean_b2_mae: 1.569923085340115
- mean_b3_mae: 1.4838724758329922

## Perfil dos 23 folds em que B3 perde

- n_folds: 23
- mean_pct_zero: 0.6036920580858963
- mean_pct_neg: 0.03880597263909297
- mean_pct_gt6: 0.10005230957907467
- mean_pct_gt10: 0.032647252284060284
- mean_std_b2: 1.9496793252400788
- mean_std_b3: 0.960307307176999
- mean_b2_mae: 1.598684182341246
- mean_b3_mae: 1.650717348663432

## Fallback

- n_fallback B2: 622
- n_fallback B3: 622
- Mesmo conjunto: True
- MAE fallback B2: 1.6509
- MAE fallback B3: 1.6509

## Imutabilidade

- parquet: OK
- b0: OK
- b1: OK
- b2: OK
- b3: OK
- b2_sum: OK
- b2_spot: OK
- b3_sum: OK
- b3_spot: OK
