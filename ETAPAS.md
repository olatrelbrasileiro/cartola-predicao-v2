# Ordem obrigatória de implementação

Status: marcar [x] apenas quando a etapa estiver VALIDADA (com verificação
executada e resultado conferido).

- [x] ETAPA 1 — Estrutura do projeto
- [x] ETAPA 2 — Pipeline de dados jogador × rodada
- [x] ETAPA 3 — Auditoria dos dados
- [ ] ETAPA 4 — Harness walk-forward
- [ ] ETAPA 5 — Baselines B0-B5
- [ ] ETAPA 6 — Ridge (M1)
- [ ] ETAPA 7 — CatBoost (M2)
- [ ] ETAPA 8 — Experimentos adicionais (somente se justificado)

## Regras de progressão
- Não pular etapas.
- Não avançar enquanto a etapa atual não estiver validada.
- Não adicionar features fora do feature set inicial sem justificativa
  metodológica e sem passar pelo harness.
- Não substituir a metodologia oficial por outra abordagem.


## Estado da Etapa 5 (baselines)
- [x] B0  — global, MAE 2.0748
- [x] B1  — histórico expansivo, MAE 1.6012
- [x] B2  — últimas 5 observações, MAE 1.5778
- [x] B3  — basic_score, MAE 1.5339 (rebaixado a diagnóstico)
- [x] H1  — compressão identificada
- [x] H2A — α estável walk-forward
- [x] H2B — sinal ortogonal detectado, explorabilidade não demonstrada
- [x] B2S — B2 + shrinkage walk-forward, MAE 1.4016 (oficial)

## Pendências
- [ ] Explorar H2B (sinal residual do B3) — precisa de hipótese pré-registrada
- [ ] Definir próximo baseline ou componente
- [ ] Etapa 6 (Ridge) só depois de fechar a Etapa 5
---

## ETAPA P0 — Correcao de participacao (CONCLUIDA)

Data: 2026-09-21

### Definicoes oficiais
- Estatisticas de desempenho usam SOMENTE `entrou_em_campo=True`.
- Recente 5/3 = ultimas 5/3 PARTICIPACOES (nao linhas).
- Avaliacao = SOMENTE participantes no teste.
- k=7 mantido como pre-registrado.
- Nao participantes permanecem no dataset.

### Benchmark oficial P0-corrected (valores reais)

| Baseline | MAE ponderado | Spearman medio |
|---|---|---|
| B0 | 2.9494 | nan |
| B1 | 2.8587 | 0.2808 |
| B2 | 2.9413 | 0.2552 |
| B3 | 2.9287 | 0.1176 |
| B2S | 2.7485 | 0.2552 |

alpha medio B2S: 0.316

Arquivo: `reports/baselines_p0_corrected_folds.csv`.
Os valores historicos (B2S MAE 1.4016) NAO sao comparaveis.

---

## ETAPA P1 — Auditoria metodologica (EM ANDAMENTO)

### Sub-etapas

- [ ] P1.1 — Justificativa quantitativa do k=7
- [ ] P1.2 — Auditoria temporal do status_id
- [ ] P1.3 — Auditoria numerica de 2025
- [ ] P1.4 — Atualizacao deste ETAPAS.md

### Regras de marcacao
- Cada sub-etapa so vira [x] apos sua auditoria rodar e passar.
- P1.2 pode ser INDETERMINADO sem falha.
- P1.4 so vira [x] apos P1.1..P1.3 terem rodado.

## Proximas etapas (nao iniciadas)

- ETAPA B — estudo de k (nao iniciada)
- Ridge v2 com base corrigida (nao iniciada)
- CatBoost v2 com base corrigida (nao iniciada)
- NDCG / pooling / oportunidade (nao iniciadas)

---

## EXP-12 — Contexto de time e adversário

Sub-etapas:

- [x] EXP-12/F1 — Histórico de partidas por equipe, R1..R26 (CONCLUIDA)
- [x] EXP-12/F1.1 — Investigação das anomalias da auditoria (CONCLUIDA; tratamento pendente)
- [x] EXP-12/F1.2 — Auditoria temporal das partidas (CONCLUIDA; eixo temporal pendente)
- [ ] EXP-12/F2 — Features temporais de time/adversário (nao iniciada)
- [ ] EXP-12/F3 — Validação walk-forward das features novas (nao iniciada)

### F1 — Histórico de partidas por equipe (VALIDADA)

Data: 2026-09-25

- Objetivo: uma linha por equipe por partida, com scouts agregados dos
  jogadores participantes e o placar oficial da API.
- Fontes: `experiments/data/cartola_api_2026/r{1..26}.json` e
  `partidas{1..26}.json`.
- Script: `scripts/exp12_build_team_match_history.py`
  (generaliza `scripts/build_team_match_history_r1.py`, que segue como
  referência de semântica validada).
- Saídas: `data/interim/exp12/team_match_history.parquet` e
  `data/interim/exp12/team_match_history.csv`.
- Auditoria: `reports/exp12/exp12_team_match_history_audit.md` e
  `reports/exp12/exp12_team_match_history_anomalias.csv`.
- Resultado: 26 rodadas, 257 partidas, 514 linhas, 20 clubes, 27 colunas.
  A subetapa R1 do novo arquivo é idêntica, linha a linha, à saída validada
  em `team_match_history_r1.csv`.
- Semântica: `GS` é gol sofrido do goleiro e `SG` é sem gol sofrido; nenhum
  dos dois é `gols_sofridos`, que vem do placar oficial. Não existe coluna
  `clean_sheet` artificial. `FTT = FD + FF + FT + G + PP` e `FA = FD + G`.
- Nenhuma feature temporal foi criada nesta fase. O dataset canonico
  `cartola_jogador_rodada_v2.parquet` e o CatBoost v2 nao foram alterados.
- Pendencia registrada: 3 partidas da R21 sem placar oficial foram
  excluidas e 12 partidas em R2, R3, R4, R5, R19 e R21 ficaram com scouts
  zerados por payload de atletas incompleto. O tratamento dessas linhas
  precisa ser decidido antes de F2.

Regra de marcacao: F2 so comeca depois de o tratamento das linhas com scout
zerado estar decidido e documentado.

### F1.1 — Investigacao das anomalias da auditoria (CONCLUIDA, sem decisao de tratamento)

Data: 2026-09-25

- Objetivo: rastrear na fonte cada uma das 152 anomalias registradas na
  auditoria da F1 antes de criar qualquer feature temporal.
- Script: `scripts/exp12_investigar_anomalias.py`.
- Relatorio: `reports/exp12/exp12_anomalias_investigacao.md`, com evidencias
  em `exp12_inv_scouts_zerados.csv`, `exp12_inv_gols.csv`,
  `exp12_inv_goleiros.csv` e `exp12_inv_perfil_payload.csv`.
- Natureza: somente leitura. O parquet da F1 nao foi alterado
  (sha256 `ac500b47a4b9fd00feba057677357349d18f3f77a3916a5e236f6729e0c7c6e7`),
  nenhum valor foi corrigido e nenhuma feature temporal foi criada.
- Bloco A — 24 linhas / 12 partidas com scouts zerados: o `r{N}.json` traz
  apenas o tecnico (`posicao_id=6`, `entrou_em_campo=False`) para o clube
  afetado. Nao e performance zero. O mesmo defeito atinge 30 pares
  rodada-clube em 6 rodadas.
- Bloco B — 48 casos de `G` != `gols_marcados`: a identidade
  `gols_marcados = G_do_time + GC_do_adversario` reconcilia 237 de 257
  partidas. Dos 48 casos, 20 reconciliam por `GC`, 18 sao ausencia de elenco
  e 10 ficam sem explicacao aparente (diferenca -1 ou +1, payload completo,
  sem `GC`).
- Bloco C — 30 casos de `GS` != `gols_sofridos`: **todos explicados**, sem
  caso residual. 18 por elenco ausente, 6 por goleiro ausente do payload
  (clube 315, rodadas 1 a 8) e 6 por dois ou mais goleiros do mesmo clube no
  payload (clube 265, rodadas 1 a 8), com `GS` somado em ambos e, em R3 e
  R6, `GS` e `SG` atribuidos a goleiros diferentes do mesmo time na mesma
  partida. Checagem global: `GS` fecha com o placar em 470 de 470
  clube-partida com um unico goleiro em campo.
- Bloco D — transversal: 13 partidas tem `partida_data` fora da janela
  oficial da rodada em `rodadas.json` (R21 chega a reunir jogos de 29/07,
  16/09 e 03/10). A ordenacao por `rodada` nao e confiavel como eixo de tempo.
- Pendencia: o **tratamento** de cada anomalia continua nao decidido, assim
  como o eixo temporal de F2. As recomendacoes do relatorio sao propostas,
  nao aplicacoes.

Regra de marcacao: F2 nao inicia ate o auditor decidir (a) como tratar as 24
linhas sem scouts, (b) se `GS`/`SG` podem alimentar features e (c) se o eixo
de tempo e `rodada` ou `data`.

### F1.2 — Auditoria temporal das partidas (CONCLUIDA, sem escolha de eixo)

Data: 2026-09-25

- Objetivo: determinar, com evidencia, qual deve ser o eixo temporal do
  EXP-12, comparando a alternativa A (`rodada < rodada_alvo`) com a
  alternativa B (`data real < data da partida-alvo`).
- Script: `scripts/exp12_auditoria_temporal.py`.
- Relatorio: `reports/exp12/exp12_auditoria_temporal.md`, com evidencias em
  `exp12_inv_temporal_inversoes.csv` e `exp12_inv_temporal_historicos.csv`.
- Natureza: somente leitura. Nenhuma feature temporal foi criada, o parquet
  da F1 nao foi alterado (sha256
  `ac500b47a4b9fd00feba057677357349d18f3f77a3916a5e236f6729e0c7c6e7`) e
  nenhum arquivo de origem foi tocado.
- Inventario: 257 partidas, R1..R26, 20 clubes, 67 datas e 179 timestamps
  distintos. 245 partidas dentro da janela oficial da rodada e 12 fora, em 6
  rodadas (R2, R3, R4, R5, R19, R21). As 20 demais rodadas estao integralmente
  dentro da janela.
- Inversao cronologica: **6 partidas** de rodada R foram disputadas depois de
  partidas de rodadas maiores. A mais grave e a 346335 (R4, disputada em
  02/09, posicao cronologica 246 de 257), quando 25 rodadas ja tinham sido
  disputadas. A R4 tem amplitude real de 190 dias contra uma janela oficial de
  2 dias.
- Comparacao A vs B, em 514 pares (clube, partida-alvo): **126 pares** com
  historico diferente. Em **115 pares** a alternativa A inclui no historico
  partidas que foram disputadas **depois** da partida-alvo — ou seja, A vaza.
  Esses 115 pares vem de apenas 6 partidas, e sao **exatamente** as 6 partidas
  que na F1.1 tinham elenco ausente do payload. Em 12 pares A exclui passado
  real que ja tinha sido jogado.
- Mesma data: nenhum clube disputa duas partidas no mesmo dia (0 casos), entao
  o historico por clube nao tem empate. Existem 3 pares de rodadas diferentes
  disputados no mesmo dia, e 1 deles tem horario identico (R4 346337 e R19
  346538, ambos em 23/07 as 19:30, clubes diferentes).
- Granularidade da fonte: o payload de scouts e por **rodada**
  (`r{N}.json` traz os atletas de todos os clubes da rodada), nao por
  partida. A data e hora da partida sao metadados, nao dados de
  disponibilidade. A latencia de publicacao da API nao foi medida nesta
  auditoria.
- Conclusao registrada: a alternativa A tem vazamento comprovado em 115
  pares; a alternativa B nao vaza por construcao, mas sozinha nao garante
  disponibilidade, porque o limite real de informacao publicada e a rodada. As
  opcoes (A, B, B restrito pela disponibilidade da rodada, ou eixo duplo) estao
  descritas no relatorio.
- Pendencia: a escolha do eixo temporal continua **nao decidida**, assim como
  o tratamento das anomalias da F1.1. Nenhuma das opcoes foi implementada.

Regra de marcacao: F2 nao inicia ate o auditor escolher o eixo temporal e
decidir o tratamento das linhas sem scouts.

---

## EXP-12 — F2: Team Context Features — VALIDADO

**Status:** F2 validado tecnicamente.

### Implementação
- Eixo temporal oficial: `(temporada, rodada)`.
- Histórico usa apenas temporadas anteriores e rodadas anteriores da temporada corrente.
- Últimos 3/5 = últimas 3/5 partidas anteriores efetivamente disponíveis.
- Data física da partida não é utilizada como cutoff.
- Contexto do adversário calculado com o mesmo eixo temporal.
- Clean sheet definido por `gols_sofridos == 0`.
- `FTT` e `FA` ausentes por falta de roster permanecem como `NaN`.
- Partidas mascaradas por roster ausente continuam ocupando posição nas janelas last3/last5.
- Não foi introduzida variável explícita de cobertura.

### Features
Foram validadas as 20 features de contexto:

- `team_goals_scored_last3/5`
- `team_goals_conceded_last3/5`
- `team_clean_sheet_rate_last3/5`
- `team_shots_last3/5`
- `team_shots_on_target_last3/5`
- `opp_goals_scored_last3/5`
- `opp_goals_conceded_last3/5`
- `opp_clean_sheet_rate_last3/5`
- `opp_shots_last3/5`
- `opp_shots_on_target_last3/5`

### Validações
- 16 testes passaram.
- 1 teste foi skipped (`test_first`).
- 0 testes falharam.
- Ausência de duplicatas validada.
- Temporalidade validada.
- Self-inclusion/leakage validado.
- Simetria time ↔ adversário validada.
- Independência da data física validada.
- Tratamento de missing validado.
- Reconstrução manual das features validada.
- 24 linhas correspondentes aos 12 casos de roster ausente foram mascaradas corretamente em `FTT`/`FA`.

### Observação
`opp_clube_id` pode aparecer como `float` (ex.: `276.0`) devido à presença de valores ausentes. É uma questão cosmética e não altera os cálculos; correção fica como follow-up para não modificar o builder após a validação.

### Próxima etapa
Integrar as 20 features F2 ao experimento de previsão e comparar contra o CatBoost v2 oficial usando o mesmo protocolo temporal e avaliação walk-forward, sem alterar o baseline oficial.


### Em Andamento / Validação
- **EXP-11**: Validação de features de goleiro com Banco Mestre (team_scout_history) para substituir proxy de Confronto.
