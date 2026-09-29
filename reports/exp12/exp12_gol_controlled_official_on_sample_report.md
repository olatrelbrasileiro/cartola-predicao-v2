# Teste Controlado 1 — CatBoost Oficial na amostra EXP-12 GOL

- Gerado em: 2026-09-26T22:35:19
- Amostra: apenas `elegivel_modelo=True` do gol_pooling (participantes)
- Features: exatamente as do catboost_v1 (FEATURES_COM)
- Hiperparâmetros: loss=MAE, depth=6, lr=0.05, iters=500, l2=3, seed=42
- Protocolo: walk-forward expansivo (mesmo da EXP-12)

## Resultado Pooled

| Métrica              | Valor    |
|----------------------|----------|
| n_folds              | 98 |
| n_pooled             | 1850 |
| MAE pooled           | 3.3177 |
| RMSE pooled          | 4.1585 |
| Spearman pooled      | 0.0594 |

## Comparação rápida com EXP-12 (braço D)

| Pipeline                        | MAE pooled | Spearman pooled |
|---------------------------------|------------|-----------------|
| EXP-12 braço D (sua última run) | 3.3378     | 0.0258          |
| **Official features nesta amostra** | **3.3177** | **0.0594** |

## Interpretação

- Se o Official nesta amostra ficar **muito melhor** que o braço D → o problema principal são as **features** da EXP-12.
- Se ficar **parecido** com o braço D → o problema principal é a **dificuldade intrínseca** de prever apenas participantes (e o MAE baixo do oficial original vinha dos zeros).
