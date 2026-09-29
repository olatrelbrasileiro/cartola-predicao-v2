# ETAPA 5 — B1 (media historica expansiva)

- Inicio: 2026-09-20T07:15:32
- Fim: 2026-09-20T07:15:34
- Duracao: 1.90s

## Definicao

`prediction_i = mean(pontos_num historico do jogador i)`
Historico: mesmo atleta_id, mesma temporada, rodada < alvo, elegivel.
Fallback: se jogador sem historico, usa `train_mean_global` do fold.
Nenhuma feature alem de (temporada, rodada, atleta_id, pontos_num).

## Folds

- Total: 76
- 2024: 29
- 2025: 29
- 2026: 18

## Metricas

- MAE medio simples: 1.6015
- **MAE global ponderado: 1.6012**  (principal oficial)
- RMSE agregado: 2.7395
- Spearman medio (folds validos): 0.4553
- Folds com Spearman valido: 76
- Folds com Spearman indefinido: 0
- Observacoes avaliadas (soma): 58182

## Coverage historico

- Soma com historico: 57560
- Soma fallback B0: 622
- Coverage global (ponderado): 0.9893
- Coverage medio por fold: 0.9892

## Comparacao com B0

- B0 MAE: 2.0748
- B1 MAE: 1.6012
- Melhoria absoluta: +0.4735
- Melhoria percentual: +22.82%
- Folds B1 < B0: 76
- Folds B1 > B0: 0
- Empates: 0
- Mediana delta MAE (B1 - B0): -0.4719

## Notas metodologicas

- Historico intra-temporada; nao ha cruzamento entre temporadas.
- Janela expansiva; nao ha rolling window.
- 2025/R1 nao entra em treino nem teste; historico de 2025 comeca em R2.
- B (108 ambiguas) permanece no TREINO; removida somente das METRICAS.
- Fallback B0 aplicado a jogadores sem historico elegivel.
- Nenhuma informacao futura utilizada.

## Imutabilidade

- Parquet SHA-256 antes: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4
- Parquet SHA-256 depois: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4
- Parquet imutavel: True
- B0 files inalterados: True
