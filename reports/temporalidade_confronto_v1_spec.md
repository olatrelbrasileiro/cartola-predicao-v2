# Temporalidade do Confronto v1 — Regra congelada

## Fontes
- Odds: Football-Data `BRA.csv`, colunas `AvgCH/AvgCD/AvgCA` (closing average).
- Cutoff: pesquisa manual em GE/Cartola, registrada em `reports/auditoria_cutoff_confronto_v1.csv`.

## Regra de coleta do Football-Data (documentação oficial)
- Jogos de fim de semana (sáb/dom): coleta sexta até ~17:00 BST.
- Jogos de meio de semana (ter/qua/qui): coleta terça até ~13:00 BST.

## Conversões de fuso
- Cartola cutoff (BRT, UTC-3) → UTC somando 3h.
- FD ceiling (BST, UTC+1) → UTC subtraindo 1h.

## Regra de comparação (conservadora)
Para cada partida:
1. Determinar `fd_ceiling_utc` pelo dia da semana do jogo.
2. Comparar com `cartola_cutoff_utc`.
3. `fd_ceiling_utc < cartola_cutoff_utc` → **margem comprovada** → odds mantidas.
4. `fd_ceiling_utc >= cartola_cutoff_utc` → **margem não comprovada** → mascarar.
5. Sem cutoff pesquisado → **SEM_EVIDENCIA_TEMPORAL** → mascarar.

## Por que é conservador
A regra não afirma que a coleta ocorreu depois do cutoff. Ela afirma
que, dentro da janela documentada, **não conseguimos provar** que
ocorreu antes. Sob incerteza, mascarar é o padrão seguro.

## Diferencial em relação a opções anteriores
- **A2 (mascarar rodada inteira):** rejeitado — a unidade correta é partida.
- **A3 (assumir disponibilidade):** rejeitado — sem evidência empírica.
- **A1 refinada:** adotado — máscara por partida, com motivo auditável.

## Motivos de máscara
| motivo | significado |
|---|---|
| `""` | odds mantidas |
| `ADIADA` | partida > 7 dias após data_min da rodada |
| `RISCO_CUTOFF` | margem FD × cutoff não comprovada |
| `SEM_EVIDENCIA_TEMPORAL` | cutoff da rodada não pesquisado |

## Colunas preservadas
`mando`, `clube_canon`, `adversario_canon`, `bzzoiro_event_id`,
`tipo_match`, `motivo_mascara`.

## Status
Congelado. Aguarda preenchimento do cutoff de 2025/2026 antes de gerar
`features_confronto_v1_temporal_safe.parquet`.
