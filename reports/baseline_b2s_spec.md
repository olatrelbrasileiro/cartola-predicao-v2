# Especificacao oficial do baseline B2S

- Versao: b2s_v1
- Data: 2026-09-20
- Status: oficial

## Definicao
`pred_b2s = alpha* * pred_b2 + (1 - alpha*) * anchor`
- `pred_b2`: media das ultimas 5 observacoes do jogador (intra-temporada)
- `anchor`: mediana de pontos_num do treino do fold

## Calibracao de alpha
- Metodo: nested walk-forward
- Grid: 0.0 a 1.0 passo 0.05 (21 valores)
- min_inner_rounds: 3
- Inner rounds: rodadas r_i em [3, R-1], mesma temp., elegivel
- Selecao: argmin da media dos MAE internos

## Garantias
- Nenhuma rodada >= R em qualquer calculo.
- Nenhuma observacao de outra temporada.
- 2025/R1 excluida globalmente.
- alpha* calibrado apenas em rounds internos de treino.
- Fallback (train_mean_global) entra antes do shrinkage e e comprimido igualmente.

## Metricas oficiais
- MAE ponderado (principal)
- RMSE agregado (diagnostico)
- Spearman medio (referencia)

## Caveat

Spearman e preservado matematicamente pela transformacao monotona
`alpha*p + (1-alpha)*c`. Em float64, near-ties podem quebrar e gerar
diferencas de ordem 1e-5. Nao afeta MAE nem RMSE.
