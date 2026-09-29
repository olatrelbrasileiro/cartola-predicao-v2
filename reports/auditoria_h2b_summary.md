# H2B — sinal ortogonal do B3 vs B2

- Inicio: 2026-09-20T08:57:08
- Fim: 2026-09-20T08:57:09
- Duracao: 0.3s
- Obs: 58182
- Folds: 76

## Correlacoes globais

- corr(resid_b2, innovation_b3) = -0.2717
- corr(resid_b2, prediction_b3) = 0.0926
- corr(|erro_b2|, |erro_b3|) = 0.8104
- beta (resid_b2 ~ innovation_b3) = -0.4903

## Por fold

- corr media: -0.2721
- corr mediana: -0.2725
- folds com corr > 0: 0/76
- folds com corr < 0: 76/76

## Buckets de inovacao

 q_innov     n     innov_min  innov_max  innov_mean  resid_b2_mean  resid_b2_std   y_mean  pred_b2_mean  pred_b3_mean  abs_erro_b2_mean  abs_erro_b3_mean
       0 11637 -1.235182e+01  -1.477143   -3.203331       1.317091      3.997981 3.204359      4.521450      1.318118          3.433391          3.045180
       1 11636 -1.476471e+00  -0.110000   -0.702922      -0.118296      3.202880 1.936978      1.818682      1.115760          2.211960          2.013527
       2 16958 -1.100000e-01   0.000000   -0.006909      -0.105485      1.273912 0.271937      0.166452      0.159543          0.350633          0.349794
       3  6316  9.714451e-17   0.231250    0.109236      -0.415915      2.111736 0.763062      0.347147      0.456383          0.862540          0.919133
       4 11635  2.314286e-01   5.198333    0.782347      -0.735490      2.551643 1.165286      0.429795      1.212142          1.264780          1.602451

- resid_b2_mean monotonico crescente: False
- resid_b2_mean monotonico decrescente: False
- range de resid_b2_mean entre buckets: 2.0526

## Imutabilidade

- erros CSV SHA: OK
