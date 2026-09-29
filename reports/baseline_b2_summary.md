# ETAPA 5 — B2 (historico expansivo + inovacao recente)

- Inicio: 2026-09-20T07:23:19
- Fim: 2026-09-20T07:23:24
- Duracao: 5.02s

## Definicao

H = media historica expansiva do jogador (mesma temp., rodada < alvo)
R = media das ultimas 5 observacoes anteriores do jogador
innovation = R - H
prediction = H + innovation = R

Fallback: sem historico -> train_mean_global do fold.

## Folds

- Total: 76
- 2024: 29
- 2025: 29
- 2026: 18

## Metricas

- MAE medio simples: 1.5786
- **MAE global ponderado: 1.5778**  (principal)
- RMSE agregado: 2.8261
- Spearman medio: 0.4617
- Folds com Spearman valido: 76
- Folds com Spearman indefinido: 0
- Observacoes avaliadas: 58182

## Coverage

- Com historico: 57560
- Fallback B0: 622
- Coverage global: 0.9893

## Comparacao com B1

- B1 MAE: 1.6012
- B2 MAE: 1.5778
- Melhoria absoluta: +0.0234
- Melhoria percentual: +1.46%
- Folds B2 < B1: 48
- Folds B2 > B1: 28
- Empates: 0
- Mediana delta (B2 - B1): -0.0152
- Media delta (B2 - B1): -0.0228

## Notas

- Nenhuma feature externa utilizada.
- Unica mudanca em relacao ao B1: substituir H pela forma recente R.
- Janela recente fixa em 5 observacoes. Sem tuning.
- Historico intra-temporada; janela expansiva; 2025/R1 excluida.
- B (ambiguas) permanece no treino; removida so das metricas.

## Imutabilidade

- Parquet SHA-256 antes: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4
- Parquet SHA-256 depois: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4
- Parquet imutavel: True
- B0 inalterado: True
- B1 inalterado: True
