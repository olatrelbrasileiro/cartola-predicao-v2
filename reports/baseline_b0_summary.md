# Baseline B0 — resumo

- **Inicio:** 2026-09-20T07:09:15
- **Fim:** 2026-09-20T07:09:17
- **Duracao:** 1.50s

## Definicao

`y_pred = mean(pontos_num)` do treino elegivel do fold. Aplicada como constante a todos os jogadores do teste.
Nenhuma feature usada. Nenhuma media global. Nenhum dado do teste.

## Folds

- Total: 76
- 2024: 29
- 2025: 29
- 2026: 18

## Metricas

- **MAE medio simples (por fold):** 2.0749
- **MAE global ponderado (por n):** 2.0748  ← **principal oficial**
- **RMSE agregado (por n):** 2.9881
- **Spearman:** INDEFINIDO (previsao constante dentro de cada fold; nao ha variacao entre jogadores para gerar ranking)
- Observacoes avaliadas (soma): 58182

## Estatisticas de `train_mean`

- min: 1.3311
- max: 1.7168
- mean: 1.4737
- median: 1.5113
- std: 0.0937

## Decisao B (108 linhas ambiguas)

- Permanecem no TREINO (politica atual)
- Removidas SOMENTE da metrica principal do TESTE
- Removidas no total dos testes: 84

## 2025/R1

- Excluida globalmente pelo filtro `elegivel`
- Nenhum fold usa R1 de 2025 em treino nem teste

## Imutabilidade do Parquet

- SHA-256 antes: `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- SHA-256 depois: `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Linhas antes/depois: 77253/77253
- **Imutavel: True**
