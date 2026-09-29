# EXP-12 — Auditoria da construção do histórico de partidas por equipe

- Temporada: **2026**
- Rodadas: **R1..R26**
- Gerado por: `scripts/exp12_build_team_match_history.py`
- Linhas: **514** | Partidas: **257** | Clubes: **20** | Colunas: **27**

## Objetivo

Construir o histórico de partidas por equipe da temporada, uma linha por equipe por partida, agregando os scouts diretos dos jogadores que entraram em campo e o placar oficial da API. Esta etapa é **puramente descritiva**: não cria feature temporal, não usa rodada futura e não toca no dataset canônico nem no CatBoost v2.

## Fontes

- `experiments/data/cartola_api_2026/r{N}.json` — scouts dos jogadores (`atletas` vem como dicionário indexado por atleta_id)
- `experiments/data/cartola_api_2026/partidas{N}.json` — confrontos, mandante/visitante e placar oficial
- `experiments/data/cartola_api_2026/rodadas.json` — calendário de rodadas (consulta)
- `scripts/build_team_match_history_r1.py` — referência de semântica validada na R1

## Metodologia

1. Uma linha por equipe por partida: mandante `mando='C'` e visitante `mando='F'`.
2. Scouts agregados por clube a partir dos atletas com `entrou_em_campo=True` do payload da rodada. Cada clube joga uma única partida por rodada (verificado), então a agregação do clube é biunívoca com a partida.
3. Scout ausente, `{}` ou `null` = zero na categoria. Não é inventada participação.
4. `gols_marcados` e `gols_sofridos` vêm **exclusivamente** do placar oficial de `partidas{N}.json`.
5. Nenhuma linha de uma partida sem placar oficial é criada.
6. Scouts fora da lista de 14 diretos são ignorados, não renomeados.
7. Nenhuma feature temporal, nenhum uso de informação de rodada futura.

## Semântica dos campos

| campo | tipo | semantica |
|---|---|---|
| temporada | inteiro | temporada do dado, fixa em 2026 |
| rodada | inteiro | rodada da partida (campo `rodada` do arquivo, com fallback do topo do partidas{N}.json) |
| partida_id | inteiro | identificador do confronto em partidas{N}.json |
| data | texto | `partida_data` crua da API, sem normalização |
| clube_id | inteiro | equipe na linha |
| clube_nome | texto | nome do clube em `clubes` de partidas{N}.json |
| adversario_id | inteiro | equipe adversária |
| adversario_nome | texto | nome do adversário |
| mando | texto | 'C' mandante, 'F' visitante |
| gols_marcados | inteiro | placar oficial do time na partida |
| gols_sofridos | inteiro | placar oficial do adversário na partida |
| DP | inteiro | soma de defesas de pênalti dos participantes |
| GS | inteiro | soma de gol sofrido DO GOLEIRO (scout do jogador). NÃO é gols_sofridos |
| DS | inteiro | soma de desarmes |
| SG | inteiro | soma de sem gol sofrido (scout do jogador) |
| FC | inteiro | soma de faltas cometidas |
| G | inteiro | soma de gols dos participantes |
| A | inteiro | soma de assistências |
| FTT | inteiro | FD + FF + FT + G + PP |
| FA | inteiro | FD + G |
| FS | inteiro | scout FS preservado conforme API |
| I | inteiro | soma de impedimentos |
| PS | inteiro | soma de pênaltis sofridos |
| FD | inteiro | soma de finalizações defendidas |
| FF | inteiro | soma de finalizações para fora |
| FT | inteiro | soma de finalizações na trave |
| PP | inteiro | soma de pênaltis perdidos |

Pontos críticos de semântica:

- `GS` é o scout de **gol sofrido do goleiro**. Não substitui `gols_sofridos`.
- `SG` é o scout de **sem gol sofrido**. Não substitui `gols_sofridos`.
- Não existe coluna `clean_sheet` artificial: clean sheet não é derivado aqui.
- `gols_sofridos` não vem de scout nenhum.

## Validações

- [x] Exatamente 2 linhas por partida.
- [x] Sem duplicidade em (temporada, rodada, partida_id, clube_id).
- [x] Clubes/adversários inversos em id e nome.
- [x] Mando consistente: 1 linha 'C' e 1 linha 'F' por partida.
- [x] gols_marcados de A = gols_sofridos de B e vice-versa.
- [x] FTT = FD + FF + FT + G + PP.
- [x] FA = FD + G.
- [x] GS / SG / gols_sofridos separados e sem coluna clean_sheet.
- [x] Esquema com 27 colunas idêntico ao da R1 validada.
- [x] Nenhuma feature temporal criada nesta etapa.
- [x] Cobertura de todas as rodadas disponíveis na fonte.
- [x] Nenhum clube com duas partidas na mesma rodada.

## Resultado

- Rodadas construídas: **26** (R1..R26)
- Partidas lidas da fonte: **260**
- Partidas sem placar oficial (excluídas): **3**
- Partidas no dataset: **257**
- Linhas: **514**
- Clubes: **20** — [262, 263, 264, 265, 266, 267, 275, 276, 277, 280, 282, 283, 284, 285, 287, 293, 294, 315, 364, 2305]
- Colunas: **27**
- Atletas lidos: **8260**
- Participantes (`entrou_em_campo=True`): **8203**
- Não participantes ignorados: **57**
- Catálogo de clubes idêntico em todas as rodadas: **True**

### Resumo por rodada

| rodada | partidas_fonte | partidas_incluidas | partidas_sem_placar | linhas | atletas | participantes | nao_participantes | clubes_com_scouts | clubes_sem_atletas | participantes_scout_none |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 10 | 0 | 20 | 327 | 327 | 0 | 20 | 0 | 0 |
| 2 | 10 | 10 | 0 | 20 | 302 | 300 | 2 | 18 | 2 | 0 |
| 3 | 10 | 10 | 0 | 20 | 301 | 299 | 2 | 18 | 2 | 0 |
| 4 | 10 | 10 | 0 | 20 | 242 | 235 | 7 | 14 | 6 | 0 |
| 5 | 10 | 10 | 0 | 20 | 274 | 270 | 4 | 16 | 4 | 0 |
| 6 | 10 | 10 | 0 | 20 | 336 | 335 | 1 | 20 | 0 | 0 |
| 7 | 10 | 10 | 0 | 20 | 335 | 334 | 1 | 20 | 0 | 0 |
| 8 | 10 | 10 | 0 | 20 | 335 | 335 | 0 | 20 | 0 | 0 |
| 9 | 10 | 10 | 0 | 20 | 342 | 342 | 0 | 20 | 0 | 0 |
| 10 | 10 | 10 | 0 | 20 | 338 | 336 | 2 | 20 | 0 | 0 |
| 11 | 10 | 10 | 0 | 20 | 333 | 333 | 0 | 20 | 0 | 0 |
| 12 | 10 | 10 | 0 | 20 | 340 | 338 | 2 | 20 | 0 | 0 |
| 13 | 10 | 10 | 0 | 20 | 336 | 333 | 3 | 20 | 0 | 0 |
| 14 | 10 | 10 | 0 | 20 | 335 | 334 | 1 | 20 | 0 | 0 |
| 15 | 10 | 10 | 0 | 20 | 333 | 332 | 1 | 20 | 0 | 0 |
| 16 | 10 | 10 | 0 | 20 | 333 | 332 | 1 | 20 | 0 | 0 |
| 17 | 10 | 10 | 0 | 20 | 335 | 334 | 1 | 20 | 0 | 0 |
| 18 | 10 | 10 | 0 | 20 | 339 | 338 | 1 | 20 | 0 | 0 |
| 19 | 10 | 10 | 0 | 20 | 209 | 201 | 8 | 12 | 8 | 0 |
| 20 | 10 | 10 | 0 | 20 | 338 | 336 | 2 | 20 | 0 | 0 |
| 21 | 10 | 7 | 3 | 14 | 212 | 204 | 8 | 12 | 2 | 0 |
| 22 | 10 | 10 | 0 | 20 | 336 | 334 | 2 | 20 | 0 | 0 |
| 23 | 10 | 10 | 0 | 20 | 338 | 336 | 2 | 20 | 0 | 0 |
| 24 | 10 | 10 | 0 | 20 | 337 | 337 | 0 | 20 | 0 | 0 |
| 25 | 10 | 10 | 0 | 20 | 332 | 329 | 3 | 20 | 0 | 0 |
| 26 | 10 | 10 | 0 | 20 | 342 | 339 | 3 | 20 | 0 | 0 |

### Amostra

```
 temporada  rodada  partida_id                data  clube_id clube_nome  adversario_id adversario_nome mando  gols_marcados  gols_sofridos  DP  GS  DS  SG  FC  G  A  FTT  FA  FS  I  PS  FD  FF  FT  PP
      2026       1      346235 2026-01-28 19:30:00       266        FLU            284             GRE     C              2              1   0   1  12   0  13  2  1   14   6  16  1   0   4   8   0   0
      2026       1      346235 2026-01-28 19:30:00       284        GRE            266             FLU     F              1              2   0   2  17   0  12  1  1    7   3  12  2   0   2   4   0   0
      2026       1      346236 2026-01-29 21:30:00       263        BOT            283             CRU     C              4              0   0   0  14   5  10  3  3   10   4  13  0   0   1   6   0   0
      2026       1      346236 2026-01-29 21:30:00       283        CRU            263             BOT     F              0              4   0   4   6   0  12  0  0   11   4   9  5   0   4   7   0   0
      2026       1      346237 2026-01-28 21:30:00       276        SAO            262             FLA     C              2              1   0   1  28   0  14  3  1    9   5   8  1   0   2   4   0   0
      2026       1      346237 2026-01-28 21:30:00       262        FLA            276             SAO     F              1              2   0   2  13   0  11  1  1   11   5  14  0   0   4   6   0   0
      2026       1      346238 2026-01-28 20:00:00       264        COR            265             BAH     C              1              2   0   2  30   0  13  1  1   12   5  14  3   0   4   6   1   0
      2026       1      346238 2026-01-28 20:00:00       265        BAH            264             COR     F              2              1   0   3  12   0  14  2  1    7   6  15  1   1   4   1   0   0
      2026       1      346239 2026-01-29 20:00:00      2305        MIR            267             VAS     C              2              1   0   1  10   0  13  1  1   10   3   8  1   0   2   7   0   0
      2026       1      346239 2026-01-29 20:00:00       267        VAS           2305             MIR     F              1              2   0   2  12   0  10  0  1    8   4  15  2   0   4   4   0   0
```

## Anomalias encontradas

Total de registros de anomalia: **152**. Detalhe linha a linha em `reports/exp12/exp12_team_match_history_anomalias.csv`.
Nada foi corrigido silenciosamente.

```
tipo
g_scout_diferente_do_placar_oficial    48
gs_scout_diferente_de_gols_sofridos    30
clube_sem_atletas_participantes        24
linha_com_todos_os_scouts_zero         24
partida_com_flag_valida_falso          15
scout_existente_ignorado_por_escopo     6
partida_sem_placar_oficial              3
rodada_fora_de_escopo                   2
```

### 1. Partidas sem placar oficial (EXCLUÍDAS da construção)

- R21 partida 346283 SAO x 277 — placar oficial ausente (data=2026-07-29 20:30:00, valida=False); EXCLUÍDA da construção — não há resultado oficial e o safe_int da R1 transformaria isso em 0x0 fabricado
- R21 partida 346286 CAM x 280 — placar oficial ausente (data=2026-10-03 18:30:00, valida=False); EXCLUÍDA da construção — não há resultado oficial e o safe_int da R1 transformaria isso em 0x0 fabricado
- R21 partida 346290 CHA x 267 — placar oficial ausente (data=2026-07-29 20:30:00, valida=False); EXCLUÍDA da construção — não há resultado oficial e o safe_int da R1 transformaria isso em 0x0 fabricado

### 2. Clubes sem atletas participantes no payload da rodada (scouts ficam em zero)

Nestes casos a partida existe com placar oficial, mas o payload de `r{N}.json` não traz nenhum atleta com `entrou_em_campo=True` do clube — normalmente sobra apenas o técnico. As 14 colunas de scout da linha ficam zeradas por fallback, o que **não** significa performance zero.

- R2 partida 346262 COR (id 264)
- R2 partida 346262 CAP (id 293)
- R3 partida 346299 FLA (id 262)
- R3 partida 346299 VIT (id 287)
- R4 partida 346335 FLA (id 262)
- R4 partida 346335 MIR (id 2305)
- R4 partida 346337 BOT (id 263)
- R4 partida 346337 VIT (id 287)
- R4 partida 346349 BAH (id 265)
- R4 partida 346349 CHA (id 315)
- R5 partida 346405 SAN (id 277)
- R5 partida 346405 MIR (id 2305)
- R5 partida 346408 BOT (id 263)
- R5 partida 346408 CAP (id 293)
- R19 partida 346535 FLU (id 266)
- R19 partida 346535 RBB (id 280)
- R19 partida 346536 BOT (id 263)
- R19 partida 346536 SAN (id 277)
- R19 partida 346539 GRE (id 284)
- R19 partida 346539 MIR (id 2305)
- R19 partida 346543 VAS (id 267)
- R19 partida 346543 VIT (id 287)
- R21 partida 346282 BOT (id 263)
- R21 partida 346282 GRE (id 284)

### 3. Soma de `G` diferente do placar oficial

A semântica é mantida: `G` vem do scout, `gols_marcados` vem do placar oficial. A divergência é do dado de origem, não do código.

- R1 partida 346236 BOT: G=3 vs gols_marcados=4
- R1 partida 346237 SAO: G=3 vs gols_marcados=2
- R1 partida 346239 VAS: G=0 vs gols_marcados=1
- R1 partida 346239 MIR: G=1 vs gols_marcados=2
- R1 partida 346240 CAM: G=1 vs gols_marcados=2
- R2 partida 346261 BOT: G=4 vs gols_marcados=3
- R2 partida 346261 GRE: G=4 vs gols_marcados=5
- R2 partida 346262 COR: G=0 vs gols_marcados=1
- R3 partida 346291 FLU: G=2 vs gols_marcados=1
- R3 partida 346296 CAM: G=2 vs gols_marcados=3
- R3 partida 346299 FLA: G=0 vs gols_marcados=2
- R3 partida 346299 VIT: G=0 vs gols_marcados=1
- R4 partida 346335 FLA: G=0 vs gols_marcados=2
- R4 partida 346349 BAH: G=0 vs gols_marcados=2
- R5 partida 346405 SAN: G=0 vs gols_marcados=2
- R5 partida 346405 MIR: G=0 vs gols_marcados=2
- R5 partida 346408 BOT: G=0 vs gols_marcados=1
- R5 partida 346408 CAP: G=0 vs gols_marcados=4
- R7 partida 346562 INT: G=1 vs gols_marcados=2
- R7 partida 346571 GRE: G=1 vs gols_marcados=2
- R8 partida 346587 PAL: G=2 vs gols_marcados=1
- R8 partida 346589 BOT: G=1 vs gols_marcados=2
- R9 partida 346602 CFC: G=0 vs gols_marcados=1
- R10 partida 346613 PAL: G=1 vs gols_marcados=2
- R11 partida 346325 MIR: G=0 vs gols_marcados=1
- R12 partida 346355 RBB: G=2 vs gols_marcados=4
- R15 partida 346428 SAO: G=1 vs gols_marcados=2
- R15 partida 346429 CHA: G=0 vs gols_marcados=1
- R16 partida 346455 SAO: G=0 vs gols_marcados=1
- R16 partida 346464 REM: G=2 vs gols_marcados=3
- R18 partida 346513 BAH: G=1 vs gols_marcados=2
- R19 partida 346535 FLU: G=0 vs gols_marcados=1
- R19 partida 346535 RBB: G=0 vs gols_marcados=1
- R19 partida 346536 BOT: G=0 vs gols_marcados=2
- R19 partida 346536 SAN: G=0 vs gols_marcados=1
- R19 partida 346539 GRE: G=0 vs gols_marcados=1
- R19 partida 346539 MIR: G=0 vs gols_marcados=2
- R19 partida 346543 VIT: G=0 vs gols_marcados=1
- R19 partida 346544 FLA: G=3 vs gols_marcados=4
- R20 partida 346423 CHA: G=1 vs gols_marcados=2
- R21 partida 346282 BOT: G=0 vs gols_marcados=3
- R21 partida 346282 GRE: G=0 vs gols_marcados=2
- R21 partida 346285 MIR: G=1 vs gols_marcados=2
- R21 partida 346289 PAL: G=3 vs gols_marcados=4
- R22 partida 346317 SAO: G=0 vs gols_marcados=1
- R23 partida 346346 BAH: G=2 vs gols_marcados=3
- R25 partida 346454 CFC: G=2 vs gols_marcados=3
- R26 partida 346499 RBB: G=1 vs gols_marcados=2

### 4. `GS` diferente de `gols_sofridos`

Esperado por desenho (o goleiro acumula `GS`, o placar é do time), mas registrado para deixar explícito que `GS` não é `gols_sofridos`.

- R1 partida 346238 BAH: GS=3 vs gols_sofridos=1
- R1 partida 346244 CHA: GS=0 vs gols_sofridos=2
- R2 partida 346256 CHA: GS=0 vs gols_sofridos=1
- R2 partida 346262 CAP: GS=0 vs gols_sofridos=1
- R2 partida 346263 BAH: GS=2 vs gols_sofridos=1
- R3 partida 346292 BAH: GS=3 vs gols_sofridos=0
- R3 partida 346299 FLA: GS=0 vs gols_sofridos=1
- R3 partida 346299 VIT: GS=0 vs gols_sofridos=2
- R3 partida 346300 CHA: GS=0 vs gols_sofridos=3
- R4 partida 346335 MIR: GS=0 vs gols_sofridos=2
- R4 partida 346349 CHA: GS=0 vs gols_sofridos=2
- R5 partida 346403 CHA: GS=0 vs gols_sofridos=2
- R5 partida 346405 SAN: GS=0 vs gols_sofridos=2
- R5 partida 346405 MIR: GS=0 vs gols_sofridos=2
- R5 partida 346408 BOT: GS=0 vs gols_sofridos=4
- R5 partida 346408 CAP: GS=0 vs gols_sofridos=1
- R5 partida 346409 BAH: GS=3 vs gols_sofridos=1
- R6 partida 346481 BAH: GS=1 vs gols_sofridos=0
- R6 partida 346484 CHA: GS=0 vs gols_sofridos=1
- R8 partida 346591 CHA: GS=0 vs gols_sofridos=2
- R8 partida 346594 BAH: GS=6 vs gols_sofridos=4
- R19 partida 346535 FLU: GS=0 vs gols_sofridos=1
- R19 partida 346535 RBB: GS=0 vs gols_sofridos=1
- R19 partida 346536 BOT: GS=0 vs gols_sofridos=1
- R19 partida 346536 SAN: GS=0 vs gols_sofridos=2
- R19 partida 346539 GRE: GS=0 vs gols_sofridos=2
- R19 partida 346539 MIR: GS=0 vs gols_sofridos=1
- R19 partida 346543 VAS: GS=0 vs gols_sofridos=1
- R21 partida 346282 BOT: GS=0 vs gols_sofridos=2
- R21 partida 346282 GRE: GS=0 vs gols_sofridos=3

### 5. Partidas com `valida=false` na API

A construção **não** filtra por este flag (a R1 validada também não filtra). Estão listadas para auditoria.

- R2 partida 346262 CAP x 264
- R3 partida 346299 VIT x 262
- R4 partida 346335 FLA x 2305
- R4 partida 346337 BOT x 287
- R4 partida 346349 BAH x 315
- R5 partida 346405 MIR x 277
- R5 partida 346408 CAP x 263
- R19 partida 346535 FLU x 280
- R19 partida 346536 BOT x 277
- R19 partida 346539 MIR x 284
- R19 partida 346543 VIT x 267
- R21 partida 346282 BOT x 284
- R21 partida 346283 SAO x 277
- R21 partida 346286 CAM x 280
- R21 partida 346290 CHA x 267

### 6. Scouts existentes na API e fora do escopo do EXP-12

`CA, CV, DE, GC, PC, V` — mantidos fora, conforme semântica validada na R1.

### 7. Rodadas fora do escopo pedido

- R27 — arquivos de fonte existem na API mas a rodada está fora da faixa pedida (1-26)
- R28 — arquivos de fonte existem na API mas a rodada está fora da faixa pedida (1-26)

## Arquivos criados / modificados

Criados:

- `scripts/exp12_build_team_match_history.py`
- `data/interim/exp12/team_match_history.parquet`
- `data/interim/exp12/team_match_history.csv`
- `reports/exp12/exp12_team_match_history_anomalias.csv`
- `reports/exp12/exp12_team_match_history_audit.md` (este arquivo)

Não modificados (confirmado):

- `data/interim/cartola_jogador_rodada_v2.parquet`
- `scripts/catboost_v2.py`
- `scripts/build_team_match_history_r1.py`
- `data/interim/exp12/team_match_history_r1.parquet` / `.csv`

## Próximo passo

1. Decidir o tratamento das linhas com scouts zerados por payload incompleto (R2, R3, R4, R5, R19, R21) antes de usar histórico de scouts em feature.
2. Só depois disso construir as features temporais de contexto de time/adversário (janelas de 5 e 10 jogos, cobertura de janela), com corte estritamente anterior à rodada-alvo.

