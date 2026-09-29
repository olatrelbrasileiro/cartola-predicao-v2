# Auditoria pre-treino — CatBoost v1

- Gerado em: 2026-09-20T23:28:22

## Hashes de entrada

| chave | arquivo | SHA-256 (16 primeiros) | status |
|---|---|---|---|
| parquet | cartola_jogador_rodada_v2.parquet | 15be1a317afb940c... | OK |
| feats_orig | features_confronto_v1.parquet | 6e4f2e6b10a060f7... | OK |
| temporal_safe | features_confronto_v1_temporal_safe.parquet | 0fcc9fe7e5ec16db... | OK |
| b2s_folds | baseline_b2s_folds.csv | b56c6db440d7628a... | OK |
| harness | harness_folds_v2.csv | 236b7421ed83860d... | OK |

## Harness

- Folds totais: 76
- 2024: 29
- 2025: 29
- 2026: 18

## Temporal-safe

- Linhas: 77253
- Colunas: ['temporada', 'rodada', 'atleta_id', 'clube_canon', 'p_home', 'p_draw', 'p_away', 'mando', 'favoritismo_home_away', 'tem_odds', 'motivo_mascara', 'adversario_canon', 'bzzoiro_event_id']
- NaN p_home: 14683
- NaN mando: 230

Contagem motivo_mascara:

- : 62570
- FD_INDETERMINADO: 6780
- RISCO_CUTOFF: 4162
- ADIADA: 3511
- SEM_MATCH: 230

## Elegibilidade

- Linhas 2025/R1: 619 (esperado 619)
- Linhas flag B (entrou=False & pontos!=0): 108 (esperado 108)

## Feature spec — CatBoost v1

| feature | grupo | fonte | fórmula | depende do target? | depende de rodada futura? |
|---|---|---|---|---|---|
| hist_mean | Histórico | per-row | média(pontos_num, rodada < r) | não | não |
| hist_std | Histórico | per-row | std(pontos_num, rodada < r) | não | não |
| hist_n | Histórico | per-row | nº obs rodada < r | não | não |
| hist_shrunk | Histórico | per-row + prior | (n*hist_mean + 7*prior_pos)/(n+7) | não | não |
| rec_mean_5 | Recente | per-row | média das últimas 5 obs | não | não |
| rec_mean_3 | Recente | per-row | média das últimas 3 obs | não | não |
| rec_std_5 | Recente | per-row | std das últimas 5 obs | não | não |
| momentum | Recente | per-row | rec_mean_5 − hist_mean | não | não |
| basic_mean | Básico | per-row | média de basic_score | não | não |
| basic_rec_5 | Básico | per-row | média de basic_score últimas 5 | não | não |
| ga_per_game | G+A | per-row | (G+A cumulativo) / jogos_num | não | não |
| ga_rec_5 | G+A | per-row | (G+A últimas 5) / n | não | não |
| ga_shrunk | G+A | per-row + prior | (n*ga_rate + 7*prior_ga)/(n+7) | não | não |
| entrou_rate | Oportunidade | per-row | média(entrou_em_campo, rodada < r) | não | não |
| is_cold_start | Oportunidade | per-row | 1 se hist_n == 0 | não | não |
| posicao_id | Posição | canonical | categórica 1–6 | não | não |
| p_home | Confronto | temporal_safe | probabilidade implícita | não | não |
| p_draw | Confronto | temporal_safe | probabilidade implícita | não | não |
| p_away | Confronto | temporal_safe | probabilidade implícita | não | não |
| mando | Confronto | temporal_safe | 1 casa / 0 fora | não | não |
| favoritismo_home_away | Confronto | temporal_safe | log(p_home/p_away) | não | não |

## Resultado

**PASS** — nenhum problema detectado.
