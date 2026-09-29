# EXP-12 F1.2 — Auditoria temporal das partidas

- Temporada: **2026** | Rodadas: **R1..R26** | Partidas no dataset: **257**
- Gerado por: `scripts/exp12_auditoria_temporal.py`
- Natureza: **somente leitura**. Nenhuma feature temporal foi criada, nenhum dataset foi alterado e nenhuma correção foi aplicada.

## 1. Objetivo

Determinar, com evidência, qual deve ser o eixo temporal do EXP-12:

- **Alternativa A** — `rodada`: histórico = partidas com `rodada < rodada_alvo`.
- **Alternativa B** — `data` real: histórico = partidas com data real estritamente anterior à data da partida-alvo.

Este documento **não escolhe** a alternativa. Ele mostra as partidas que diferenciam A de B e o risco de leakage de cada uma.

## 2. Metodologia

1. Ler `team_match_history.parquet` e as `partidas{N}.json` / `r{N}.json` / `rodadas.json`.
2. Reconstruir uma linha por partida com `rodada`, `partida_id`, `data`, mandante, visitante e a janela oficial da rodada (`inicio`/`fim` de `rodadas.json`).
3. Classificar cada partida como dentro ou fora da janela oficial.
4. Ordenar as 257 partidas pela data real e chamar essa de **posição cronológica**, para comparar com a ordem da rodada.
5. Marcar como **inversão** toda partida de rodada R disputada depois de alguma partida de rodada maior que R.
6. Para cada par (clube, partida-alvo), montar o conjunto histórico sob A e sob B e comparar: o que A inclui e B não (`vaza_em_A`), o que B inclui e A não (`perdido_em_A`) e o que B inclui por ser da mesma rodada (`mesma_rodada_antes_em_B`).
7. Medir a granularidade real da fonte de scouts.

Nenhum valor foi corrigido. As tabelas são evidência.

## 3. Inventário das partidas

- Partidas distintas: **257**.
- Rodadas cobertas: **R1..R26**.
- Datas distintas: **67**.
- Timestamps distintos: **179**.
- Dentro da janela oficial: **245**.
- Fora da janela oficial: **12**.

### 3.1 As partidas fuera da janela oficial

| rodada | partida_id | data | mandante | visitante | janela_inicio | janela_fim | dias_da_janela | pos_cronologica | rodada_max_ja_disputada |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 346299 | 2026-02-10 21:30:00 | VIT | FLA | 2026-02-11 19:00:00 | 2026-02-12 21:30:00 | -1 | 20 | 3 |
| 2 | 346262 | 2026-02-19 19:30:00 | CAP | COR | 2026-02-04 19:00:00 | 2026-02-05 21:30:00 | 15 | 30 | 3 |
| 5 | 346405 | 2026-03-10 21:30:00 | MIR | SAN | 2026-03-11 19:00:00 | 2026-03-12 21:30:00 | -1 | 38 | 5 |
| 5 | 346408 | 2026-03-29 19:30:00 | CAP | BOT | 2026-03-11 19:00:00 | 2026-03-12 21:30:00 | 18 | 77 | 8 |
| 19 | 346536 | 2026-07-16 19:30:00 | BOT | SAN | 2026-07-21 19:30:00 | 2026-07-23 19:30:00 | -5 | 178 | 19 |
| 19 | 346543 | 2026-07-16 19:30:00 | VIT | VAS | 2026-07-21 19:30:00 | 2026-07-23 19:30:00 | -5 | 179 | 19 |
| 4 | 346349 | 2026-07-17 19:30:00 | BAH | CHA | 2026-02-25 19:00:00 | 2026-02-26 19:00:00 | 142 | 180 | 19 |
| 19 | 346535 | 2026-07-17 20:00:00 | FLU | RBB | 2026-07-21 19:30:00 | 2026-07-23 19:30:00 | -4 | 181 | 19 |
| 19 | 346539 | 2026-07-17 20:00:00 | MIR | GRE | 2026-07-21 19:30:00 | 2026-07-23 19:30:00 | -4 | 182 | 19 |
| 4 | 346337 | 2026-07-23 19:30:00 | BOT | VIT | 2026-02-25 19:00:00 | 2026-02-26 19:00:00 | 148 | 188 | 19 |
| 4 | 346335 | 2026-09-02 19:30:00 | FLA | MIR | 2026-02-25 19:00:00 | 2026-02-26 19:00:00 | 189 | 246 | 25 |
| 21 | 346282 | 2026-09-16 19:30:00 | BOT | GRE | 2026-07-29 19:30:00 | 2026-07-30 21:30:00 | 49 | 257 | 26 |

### 3.2 Todas as 257 partidas, em ordem cronológica real

`pos_cronologica` é a posição real na linha do tempo. `rodada_max_ja_disputada` é a maior rodada que já tinha jogado na data daquela partida — é o que mostra quantas rodadas o calendário oficial ja considerava vencidas.

| pos_cronologica | rodada | partida_id | data | mandante | visitante | dentro_da_janela_txt | rodada_max_ja_disputada |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 346240 | 2026-01-28 19:00:00 | CAM | PAL | sim | 1 |
| 2 | 1 | 346241 | 2026-01-28 19:00:00 | INT | CAP | sim | 1 |
| 3 | 1 | 346242 | 2026-01-28 19:00:00 | CFC | RBB | sim | 1 |
| 4 | 1 | 346243 | 2026-01-28 19:00:00 | VIT | REM | sim | 1 |
| 5 | 1 | 346235 | 2026-01-28 19:30:00 | FLU | GRE | sim | 1 |
| 6 | 1 | 346238 | 2026-01-28 20:00:00 | COR | BAH | sim | 1 |
| 7 | 1 | 346244 | 2026-01-28 20:00:00 | CHA | SAN | sim | 1 |
| 8 | 1 | 346237 | 2026-01-28 21:30:00 | SAO | FLA | sim | 1 |
| 9 | 1 | 346239 | 2026-01-29 20:00:00 | MIR | VAS | sim | 1 |
| 10 | 1 | 346236 | 2026-01-29 21:30:00 | BOT | CRU | sim | 1 |
| 11 | 2 | 346255 | 2026-02-04 19:00:00 | FLA | INT | sim | 2 |
| 12 | 2 | 346259 | 2026-02-04 19:00:00 | RBB | CAM | sim | 2 |
| 13 | 2 | 346257 | 2026-02-04 20:00:00 | SAN | SAO | sim | 2 |
| 14 | 2 | 346264 | 2026-02-04 20:00:00 | REM | MIR | sim | 2 |
| 15 | 2 | 346258 | 2026-02-04 21:30:00 | PAL | VIT | sim | 2 |
| 16 | 2 | 346261 | 2026-02-04 21:30:00 | GRE | BOT | sim | 2 |
| 17 | 2 | 346263 | 2026-02-05 19:00:00 | BAH | FLU | sim | 2 |
| 18 | 2 | 346256 | 2026-02-05 20:00:00 | VAS | CHA | sim | 2 |
| 19 | 2 | 346260 | 2026-02-05 21:30:00 | CRU | CFC | sim | 2 |
| 20 | 3 | 346299 | 2026-02-10 21:30:00 | VIT | FLA | nao | 3 |
| 21 | 3 | 346295 | 2026-02-11 19:00:00 | MIR | CRU | sim | 3 |
| 22 | 3 | 346300 | 2026-02-11 19:00:00 | CHA | CFC | sim | 3 |
| 23 | 3 | 346296 | 2026-02-11 20:00:00 | CAM | REM | sim | 3 |
| 24 | 3 | 346292 | 2026-02-11 21:30:00 | VAS | BAH | sim | 3 |
| 25 | 3 | 346293 | 2026-02-11 21:30:00 | SAO | GRE | sim | 3 |
| 26 | 3 | 346298 | 2026-02-12 19:00:00 | CAP | SAN | sim | 3 |
| 27 | 3 | 346291 | 2026-02-12 19:30:00 | FLU | BOT | sim | 3 |
| 28 | 3 | 346294 | 2026-02-12 20:00:00 | COR | RBB | sim | 3 |
| 29 | 3 | 346297 | 2026-02-12 21:30:00 | INT | PAL | sim | 3 |
| 30 | 2 | 346262 | 2026-02-19 19:30:00 | CAP | COR | nao | 3 |
| 31 | 4 | 346343 | 2026-02-25 19:00:00 | RBB | CAP | sim | 4 |
| 32 | 4 | 346350 | 2026-02-25 19:00:00 | REM | INT | sim | 4 |
| 33 | 4 | 346348 | 2026-02-25 19:30:00 | CFC | SAO | sim | 4 |
| 34 | 4 | 346345 | 2026-02-25 20:00:00 | CRU | COR | sim | 4 |
| 35 | 4 | 346341 | 2026-02-25 21:30:00 | PAL | FLU | sim | 4 |
| 36 | 4 | 346347 | 2026-02-25 21:30:00 | GRE | CAM | sim | 4 |
| 37 | 4 | 346339 | 2026-02-26 19:00:00 | SAN | VAS | sim | 4 |
| 38 | 5 | 346405 | 2026-03-10 21:30:00 | MIR | SAN | nao | 5 |
| 39 | 5 | 346406 | 2026-03-11 19:00:00 | CAM | INT | sim | 5 |
| 40 | 5 | 346409 | 2026-03-11 20:00:00 | BAH | VIT | sim | 5 |
| 41 | 5 | 346401 | 2026-03-11 21:30:00 | FLA | CRU | sim | 5 |
| 42 | 5 | 346404 | 2026-03-11 21:30:00 | COR | CFC | sim | 5 |
| 43 | 5 | 346410 | 2026-03-12 19:00:00 | REM | FLU | sim | 5 |
| 44 | 5 | 346402 | 2026-03-12 19:30:00 | VAS | PAL | sim | 5 |
| 45 | 5 | 346403 | 2026-03-12 20:00:00 | SAO | CHA | sim | 5 |
| 46 | 5 | 346407 | 2026-03-12 21:30:00 | GRE | RBB | sim | 5 |
| 47 | 6 | 346483 | 2026-03-14 18:30:00 | VIT | CAM | sim | 6 |
| 48 | 6 | 346476 | 2026-03-14 20:30:00 | BOT | FLA | sim | 6 |
| 49 | 6 | 346475 | 2026-03-15 16:00:00 | FLU | CAP | sim | 6 |
| 50 | 6 | 346477 | 2026-03-15 16:00:00 | SAN | COR | sim | 6 |
| 51 | 6 | 346481 | 2026-03-15 16:00:00 | INT | BAH | sim | 6 |
| 52 | 6 | 346478 | 2026-03-15 18:30:00 | PAL | MIR | sim | 6 |
| 53 | 6 | 346482 | 2026-03-15 18:30:00 | CFC | REM | sim | 6 |
| 54 | 6 | 346479 | 2026-03-15 20:30:00 | RBB | SAO | sim | 6 |
| 55 | 6 | 346480 | 2026-03-15 20:30:00 | CRU | VAS | sim | 6 |
| 56 | 6 | 346484 | 2026-03-16 20:00:00 | CHA | GRE | sim | 6 |
| 57 | 7 | 346564 | 2026-03-18 19:00:00 | PAL | BOT | sim | 7 |
| 58 | 7 | 346573 | 2026-03-18 19:00:00 | BAH | RBB | sim | 7 |
| 59 | 7 | 346572 | 2026-03-18 19:30:00 | CAP | CRU | sim | 7 |
| 60 | 7 | 346566 | 2026-03-18 20:00:00 | MIR | CFC | sim | 7 |
| 61 | 7 | 346570 | 2026-03-18 20:00:00 | CAM | SAO | sim | 7 |
| 62 | 7 | 346560 | 2026-03-18 21:30:00 | VAS | FLU | sim | 7 |
| 63 | 7 | 346562 | 2026-03-18 21:30:00 | SAN | INT | sim | 7 |
| 64 | 7 | 346571 | 2026-03-19 19:00:00 | GRE | VIT | sim | 7 |
| 65 | 7 | 346558 | 2026-03-19 20:00:00 | FLA | REM | sim | 7 |
| 66 | 7 | 346574 | 2026-03-19 21:30:00 | CHA | COR | sim | 7 |
| 67 | 8 | 346589 | 2026-03-21 16:00:00 | RBB | BOT | sim | 8 |
| 68 | 8 | 346585 | 2026-03-21 18:30:00 | FLU | CAM | sim | 8 |
| 69 | 8 | 346587 | 2026-03-21 21:00:00 | SAO | PAL | sim | 8 |
| 70 | 8 | 346586 | 2026-03-22 16:00:00 | VAS | GRE | sim | 8 |
| 71 | 8 | 346590 | 2026-03-22 16:00:00 | CRU | SAN | sim | 8 |
| 72 | 8 | 346592 | 2026-03-22 16:00:00 | CAP | CFC | sim | 8 |
| 73 | 8 | 346594 | 2026-03-22 16:00:00 | REM | BAH | sim | 8 |
| 74 | 8 | 346591 | 2026-03-22 18:30:00 | INT | CHA | sim | 8 |
| 75 | 8 | 346593 | 2026-03-22 18:30:00 | VIT | MIR | sim | 8 |
| 76 | 8 | 346588 | 2026-03-22 20:30:00 | COR | FLA | sim | 8 |
| 77 | 5 | 346408 | 2026-03-29 19:30:00 | CAP | BOT | nao | 8 |
| 78 | 9 | 346596 | 2026-04-01 19:30:00 | BOT | MIR | sim | 9 |
| 79 | 9 | 346601 | 2026-04-01 19:30:00 | INT | SAO | sim | 9 |
| 80 | 9 | 346600 | 2026-04-01 20:00:00 | CRU | VIT | sim | 9 |
| 81 | 9 | 346603 | 2026-04-01 20:00:00 | BAH | CAP | sim | 9 |
| 82 | 9 | 346602 | 2026-04-01 20:30:00 | CFC | VAS | sim | 9 |
| 83 | 9 | 346595 | 2026-04-01 21:30:00 | FLU | COR | sim | 9 |
| 84 | 9 | 346597 | 2026-04-02 19:00:00 | SAN | REM | sim | 9 |
| 85 | 9 | 346604 | 2026-04-02 19:00:00 | CHA | CAM | sim | 9 |
| 86 | 9 | 346598 | 2026-04-02 21:30:00 | PAL | GRE | sim | 9 |
| 87 | 9 | 346599 | 2026-04-02 21:30:00 | RBB | FLA | sim | 9 |
| 88 | 10 | 346607 | 2026-04-04 18:30:00 | SAO | CRU | sim | 10 |
| 89 | 10 | 346612 | 2026-04-04 20:30:00 | CFC | FLU | sim | 10 |
| 90 | 10 | 346606 | 2026-04-04 21:00:00 | VAS | BOT | sim | 10 |
| 91 | 10 | 346614 | 2026-04-05 16:00:00 | CHA | VIT | sim | 10 |
| 92 | 10 | 346605 | 2026-04-05 17:30:00 | FLA | SAN | sim | 10 |
| 93 | 10 | 346610 | 2026-04-05 17:30:00 | CAM | CAP | sim | 10 |
| 94 | 10 | 346608 | 2026-04-05 19:30:00 | COR | INT | sim | 10 |
| 95 | 10 | 346613 | 2026-04-05 19:30:00 | BAH | PAL | sim | 10 |
| 96 | 10 | 346609 | 2026-04-05 20:00:00 | MIR | RBB | sim | 10 |
| 97 | 10 | 346611 | 2026-04-05 20:30:00 | GRE | REM | sim | 10 |
| 98 | 11 | 346329 | 2026-04-11 16:30:00 | VIT | SAO | sim | 11 |
| 99 | 11 | 346330 | 2026-04-11 16:30:00 | REM | VAS | sim | 11 |
| 100 | 11 | 346325 | 2026-04-11 18:30:00 | MIR | BAH | sim | 11 |
| 101 | 11 | 346323 | 2026-04-11 20:00:00 | SAN | CAM | sim | 11 |
| 102 | 11 | 346327 | 2026-04-11 20:30:00 | INT | GRE | sim | 11 |
| 103 | 11 | 346328 | 2026-04-12 11:00:00 | CAP | CHA | sim | 11 |
| 104 | 11 | 346322 | 2026-04-12 16:00:00 | BOT | CFC | sim | 11 |
| 105 | 11 | 346321 | 2026-04-12 18:00:00 | FLU | FLA | sim | 11 |
| 106 | 11 | 346324 | 2026-04-12 18:30:00 | COR | PAL | sim | 11 |
| 107 | 11 | 346326 | 2026-04-12 18:30:00 | CRU | RBB | sim | 11 |
| 108 | 12 | 346352 | 2026-04-18 18:30:00 | VAS | SAO | sim | 12 |
| 109 | 12 | 346360 | 2026-04-18 18:30:00 | CHA | BOT | sim | 12 |
| 110 | 12 | 346359 | 2026-04-18 20:00:00 | VIT | COR | sim | 12 |
| 111 | 12 | 346356 | 2026-04-18 20:30:00 | CRU | GRE | sim | 12 |
| 112 | 12 | 346357 | 2026-04-19 11:00:00 | INT | MIR | sim | 12 |
| 113 | 12 | 346353 | 2026-04-19 16:00:00 | SAN | FLU | sim | 12 |
| 114 | 12 | 346358 | 2026-04-19 16:00:00 | CFC | CAM | sim | 12 |
| 115 | 12 | 346354 | 2026-04-19 18:30:00 | PAL | CAP | sim | 12 |
| 116 | 12 | 346355 | 2026-04-19 18:30:00 | RBB | REM | sim | 12 |
| 117 | 12 | 346351 | 2026-04-19 19:30:00 | FLA | BAH | sim | 12 |
| 118 | 13 | 346376 | 2026-04-25 18:30:00 | BOT | INT | sim | 13 |
| 119 | 13 | 346389 | 2026-04-25 18:30:00 | BAH | SAN | sim | 13 |
| 120 | 13 | 346390 | 2026-04-25 18:30:00 | REM | CRU | sim | 13 |
| 121 | 13 | 346378 | 2026-04-25 21:00:00 | SAO | MIR | sim | 13 |
| 122 | 13 | 346380 | 2026-04-26 16:00:00 | COR | VAS | sim | 13 |
| 123 | 13 | 346386 | 2026-04-26 16:00:00 | GRE | CFC | sim | 13 |
| 124 | 13 | 346382 | 2026-04-26 18:30:00 | RBB | PAL | sim | 13 |
| 125 | 13 | 346388 | 2026-04-26 18:30:00 | CAP | VIT | sim | 13 |
| 126 | 13 | 346374 | 2026-04-26 20:30:00 | FLU | CHA | sim | 13 |
| 127 | 13 | 346384 | 2026-04-26 20:30:00 | CAM | FLA | sim | 13 |
| 128 | 14 | 346412 | 2026-05-02 16:00:00 | BOT | REM | sim | 14 |
| 129 | 14 | 346414 | 2026-05-02 18:30:00 | PAL | SAN | sim | 14 |
| 130 | 14 | 346419 | 2026-05-02 18:30:00 | VIT | CFC | sim | 14 |
| 131 | 14 | 346418 | 2026-05-02 20:30:00 | CAP | GRE | sim | 14 |
| 132 | 14 | 346416 | 2026-05-02 21:00:00 | CRU | CAM | sim | 14 |
| 133 | 14 | 346411 | 2026-05-03 16:00:00 | FLA | VAS | sim | 14 |
| 134 | 14 | 346413 | 2026-05-03 16:00:00 | SAO | BAH | sim | 14 |
| 135 | 14 | 346417 | 2026-05-03 18:30:00 | INT | FLU | sim | 14 |
| 136 | 14 | 346420 | 2026-05-03 18:30:00 | CHA | RBB | sim | 14 |
| 137 | 14 | 346415 | 2026-05-03 20:30:00 | MIR | COR | sim | 14 |
| 138 | 15 | 346432 | 2026-05-09 16:00:00 | CFC | INT | sim | 15 |
| 139 | 15 | 346425 | 2026-05-09 18:00:00 | FLU | VIT | sim | 15 |
| 140 | 15 | 346433 | 2026-05-09 21:00:00 | BAH | CRU | sim | 15 |
| 141 | 15 | 346430 | 2026-05-10 16:00:00 | CAM | BOT | sim | 15 |
| 142 | 15 | 346434 | 2026-05-10 16:00:00 | REM | PAL | sim | 15 |
| 143 | 15 | 346427 | 2026-05-10 18:30:00 | SAN | RBB | sim | 15 |
| 144 | 15 | 346428 | 2026-05-10 18:30:00 | COR | SAO | sim | 15 |
| 145 | 15 | 346429 | 2026-05-10 18:30:00 | MIR | CHA | sim | 15 |
| 146 | 15 | 346431 | 2026-05-10 19:30:00 | GRE | FLA | sim | 15 |
| 147 | 15 | 346426 | 2026-05-10 20:30:00 | VAS | CAP | sim | 15 |
| 148 | 16 | 346460 | 2026-05-16 18:30:00 | CAM | MIR | sim | 16 |
| 149 | 16 | 346461 | 2026-05-16 18:30:00 | INT | VAS | sim | 16 |
| 150 | 16 | 346455 | 2026-05-16 19:00:00 | FLU | SAO | sim | 16 |
| 151 | 16 | 346458 | 2026-05-16 21:00:00 | PAL | CRU | sim | 16 |
| 152 | 16 | 346457 | 2026-05-17 11:00:00 | SAN | CFC | sim | 16 |
| 153 | 16 | 346456 | 2026-05-17 16:00:00 | BOT | COR | sim | 16 |
| 154 | 16 | 346463 | 2026-05-17 16:00:00 | BAH | GRE | sim | 16 |
| 155 | 16 | 346459 | 2026-05-17 18:30:00 | RBB | VIT | sim | 16 |
| 156 | 16 | 346464 | 2026-05-17 18:30:00 | CHA | REM | sim | 16 |
| 157 | 16 | 346462 | 2026-05-17 19:30:00 | CAP | FLA | sim | 16 |
| 158 | 17 | 346487 | 2026-05-23 17:00:00 | SAO | BOT | sim | 17 |
| 159 | 17 | 346493 | 2026-05-23 17:00:00 | VIT | INT | sim | 17 |
| 160 | 17 | 346489 | 2026-05-23 19:00:00 | MIR | FLU | sim | 17 |
| 161 | 17 | 346491 | 2026-05-23 19:00:00 | GRE | SAN | sim | 17 |
| 162 | 17 | 346485 | 2026-05-23 21:00:00 | FLA | PAL | sim | 17 |
| 163 | 17 | 346490 | 2026-05-24 16:00:00 | CRU | CHA | sim | 17 |
| 164 | 17 | 346494 | 2026-05-24 16:00:00 | REM | CAP | sim | 17 |
| 165 | 17 | 346488 | 2026-05-24 18:30:00 | COR | CAM | sim | 17 |
| 166 | 17 | 346486 | 2026-05-24 20:30:00 | VAS | RBB | sim | 17 |
| 167 | 17 | 346492 | 2026-05-25 20:00:00 | CFC | BAH | sim | 17 |
| 168 | 18 | 346505 | 2026-05-30 16:00:00 | FLA | CFC | sim | 18 |
| 169 | 18 | 346512 | 2026-05-30 16:00:00 | CAP | MIR | sim | 18 |
| 170 | 18 | 346511 | 2026-05-30 17:30:00 | GRE | COR | sim | 18 |
| 171 | 18 | 346513 | 2026-05-30 17:30:00 | BAH | BOT | sim | 18 |
| 172 | 18 | 346507 | 2026-05-30 20:00:00 | SAN | VIT | sim | 18 |
| 173 | 18 | 346509 | 2026-05-31 11:00:00 | RBB | INT | sim | 18 |
| 174 | 18 | 346506 | 2026-05-31 16:00:00 | VAS | CAM | sim | 18 |
| 175 | 18 | 346508 | 2026-05-31 16:00:00 | PAL | CHA | sim | 18 |
| 176 | 18 | 346510 | 2026-05-31 20:30:00 | CRU | FLU | sim | 18 |
| 177 | 18 | 346514 | 2026-05-31 20:30:00 | REM | SAO | sim | 18 |
| 178 | 19 | 346536 | 2026-07-16 19:30:00 | BOT | SAN | nao | 19 |
| 179 | 19 | 346543 | 2026-07-16 19:30:00 | VIT | VAS | nao | 19 |
| 180 | 4 | 346349 | 2026-07-17 19:30:00 | BAH | CHA | nao | 19 |
| 181 | 19 | 346535 | 2026-07-17 20:00:00 | FLU | RBB | nao | 19 |
| 182 | 19 | 346539 | 2026-07-17 20:00:00 | MIR | GRE | nao | 19 |
| 183 | 19 | 346540 | 2026-07-21 19:30:00 | CAM | BAH | sim | 19 |
| 184 | 19 | 346542 | 2026-07-22 19:30:00 | CFC | PAL | sim | 19 |
| 185 | 19 | 346537 | 2026-07-22 21:30:00 | SAO | CAP | sim | 19 |
| 186 | 19 | 346541 | 2026-07-22 21:30:00 | INT | CRU | sim | 19 |
| 187 | 19 | 346544 | 2026-07-22 21:30:00 | CHA | FLA | sim | 19 |
| 188 | 4 | 346337 | 2026-07-23 19:30:00 | BOT | VIT | nao | 19 |
| 189 | 19 | 346538 | 2026-07-23 19:30:00 | COR | REM | sim | 19 |
| 190 | 20 | 346268 | 2026-07-25 18:30:00 | CAP | INT | sim | 20 |
| 191 | 20 | 346423 | 2026-07-25 18:30:00 | SAN | CHA | sim | 20 |
| 192 | 20 | 346422 | 2026-07-25 20:30:00 | VAS | MIR | sim | 20 |
| 193 | 20 | 346266 | 2026-07-26 16:00:00 | CRU | BOT | sim | 20 |
| 194 | 20 | 346269 | 2026-07-26 16:00:00 | BAH | COR | sim | 20 |
| 195 | 20 | 346265 | 2026-07-26 18:30:00 | RBB | CFC | sim | 20 |
| 196 | 20 | 346267 | 2026-07-26 18:30:00 | GRE | FLU | sim | 20 |
| 197 | 20 | 346421 | 2026-07-26 18:30:00 | FLA | SAO | sim | 20 |
| 198 | 20 | 346270 | 2026-07-26 19:30:00 | REM | VIT | sim | 20 |
| 199 | 20 | 346424 | 2026-07-26 19:30:00 | PAL | CAM | sim | 20 |
| 200 | 21 | 346285 | 2026-07-29 19:30:00 | MIR | REM | sim | 21 |
| 201 | 21 | 346287 | 2026-07-29 19:30:00 | INT | FLA | sim | 21 |
| 202 | 21 | 346281 | 2026-07-29 21:30:00 | FLU | BAH | sim | 21 |
| 203 | 21 | 346289 | 2026-07-29 21:30:00 | VIT | PAL | sim | 21 |
| 204 | 21 | 346284 | 2026-07-30 19:30:00 | COR | CAP | sim | 21 |
| 205 | 21 | 346288 | 2026-07-30 21:30:00 | CFC | CRU | sim | 21 |
| 206 | 22 | 346317 | 2026-08-08 16:00:00 | GRE | SAO | sim | 22 |
| 207 | 22 | 346320 | 2026-08-08 18:30:00 | REM | CAM | sim | 22 |
| 208 | 22 | 346318 | 2026-08-08 20:30:00 | CFC | CHA | sim | 22 |
| 209 | 22 | 346312 | 2026-08-08 21:00:00 | BOT | FLU | sim | 22 |
| 210 | 22 | 346316 | 2026-08-09 11:00:00 | CRU | MIR | sim | 22 |
| 211 | 22 | 346314 | 2026-08-09 16:00:00 | PAL | INT | sim | 22 |
| 212 | 22 | 346319 | 2026-08-09 16:00:00 | BAH | VAS | sim | 22 |
| 213 | 22 | 346313 | 2026-08-09 18:30:00 | SAN | CAP | sim | 22 |
| 214 | 22 | 346315 | 2026-08-09 18:30:00 | RBB | COR | sim | 22 |
| 215 | 22 | 346311 | 2026-08-09 19:30:00 | FLA | VIT | sim | 22 |
| 216 | 23 | 346331 | 2026-08-15 16:30:00 | FLU | PAL | sim | 23 |
| 217 | 23 | 346342 | 2026-08-15 18:30:00 | CAP | RBB | sim | 23 |
| 218 | 23 | 346333 | 2026-08-15 21:00:00 | SAO | CFC | sim | 23 |
| 219 | 23 | 346346 | 2026-08-16 11:00:00 | CHA | BAH | sim | 23 |
| 220 | 23 | 346332 | 2026-08-16 16:00:00 | VAS | SAN | sim | 23 |
| 221 | 23 | 346338 | 2026-08-16 16:00:00 | CAM | GRE | sim | 23 |
| 222 | 23 | 346336 | 2026-08-16 18:30:00 | MIR | FLA | sim | 23 |
| 223 | 23 | 346344 | 2026-08-16 18:30:00 | VIT | BOT | sim | 23 |
| 224 | 23 | 346334 | 2026-08-16 19:30:00 | COR | CRU | sim | 23 |
| 225 | 23 | 346340 | 2026-08-17 20:00:00 | INT | REM | sim | 23 |
| 226 | 24 | 346371 | 2026-08-22 16:00:00 | FLU | REM | sim | 24 |
| 227 | 24 | 346381 | 2026-08-22 18:30:00 | INT | CAM | sim | 24 |
| 228 | 24 | 346379 | 2026-08-22 20:30:00 | CRU | FLA | sim | 24 |
| 229 | 24 | 346375 | 2026-08-23 16:00:00 | PAL | VAS | sim | 24 |
| 230 | 24 | 346377 | 2026-08-23 16:00:00 | RBB | GRE | sim | 24 |
| 231 | 24 | 346385 | 2026-08-23 16:00:00 | VIT | BAH | sim | 24 |
| 232 | 24 | 346373 | 2026-08-23 18:30:00 | SAN | MIR | sim | 24 |
| 233 | 24 | 346387 | 2026-08-23 18:30:00 | CHA | SAO | sim | 24 |
| 234 | 24 | 346383 | 2026-08-23 19:30:00 | CFC | COR | sim | 24 |
| 235 | 24 | 346372 | 2026-08-24 20:00:00 | BOT | CAP | sim | 24 |
| 236 | 25 | 346450 | 2026-08-29 18:30:00 | CAM | VIT | sim | 25 |
| 237 | 25 | 346447 | 2026-08-29 20:00:00 | SAO | RBB | sim | 25 |
| 238 | 25 | 346446 | 2026-08-29 21:20:00 | VAS | CRU | sim | 25 |
| 239 | 25 | 346452 | 2026-08-30 11:00:00 | CAP | FLU | sim | 25 |
| 240 | 25 | 346445 | 2026-08-30 16:00:00 | FLA | BOT | sim | 25 |
| 241 | 25 | 346448 | 2026-08-30 16:00:00 | COR | SAN | sim | 25 |
| 242 | 25 | 346449 | 2026-08-30 18:30:00 | MIR | PAL | sim | 25 |
| 243 | 25 | 346451 | 2026-08-30 18:30:00 | GRE | CHA | sim | 25 |
| 244 | 25 | 346453 | 2026-08-30 19:30:00 | BAH | INT | sim | 25 |
| 245 | 25 | 346454 | 2026-08-31 20:00:00 | REM | CFC | sim | 25 |
| 246 | 4 | 346335 | 2026-09-02 19:30:00 | FLA | MIR | nao | 25 |
| 247 | 26 | 346499 | 2026-09-05 16:00:00 | RBB | BAH | sim | 26 |
| 248 | 26 | 346497 | 2026-09-05 18:30:00 | SAO | CAM | sim | 26 |
| 249 | 26 | 346495 | 2026-09-05 21:00:00 | FLU | VAS | sim | 26 |
| 250 | 26 | 346502 | 2026-09-06 11:00:00 | CFC | MIR | sim | 26 |
| 251 | 26 | 346500 | 2026-09-06 16:00:00 | CRU | CAP | sim | 26 |
| 252 | 26 | 346501 | 2026-09-06 16:00:00 | INT | SAN | sim | 26 |
| 253 | 26 | 346504 | 2026-09-06 16:00:00 | REM | FLA | sim | 26 |
| 254 | 26 | 346496 | 2026-09-06 18:30:00 | BOT | PAL | sim | 26 |
| 255 | 26 | 346498 | 2026-09-06 19:30:00 | COR | CHA | sim | 26 |
| 256 | 26 | 346503 | 2026-09-07 20:00:00 | VIT | GRE | sim | 26 |
| 257 | 21 | 346282 | 2026-09-16 19:30:00 | BOT | GRE | nao | 26 |

## 4. Inversões cronológicas

Uma partida de rodada R é uma inversão quando existe uma partida de rodada maior que R que foi disputada **antes** dela.

- Partidas com inversão: **6**.
- Partidas que respeitam a ordem da rodada: **251**.

| rodada | partida_id | data | mandante | visitante | pos_cronologica | rodada_max_ja_disputada | rodadas_invertidas | n_rodadas_invertidas | dentro_da_janela |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 346262 | 2026-02-19 19:30:00 | CAP | COR | 30 | 3 | 3 | 1 | nao |
| 5 | 346408 | 2026-03-29 19:30:00 | CAP | BOT | 77 | 8 | 6,7,8 | 3 | nao |
| 4 | 346349 | 2026-07-17 19:30:00 | BAH | CHA | 180 | 19 | 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 | 15 | nao |
| 4 | 346337 | 2026-07-23 19:30:00 | BOT | VIT | 188 | 19 | 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 | 15 | nao |
| 4 | 346335 | 2026-09-02 19:30:00 | FLA | MIR | 246 | 25 | 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 | 21 | nao |
| 21 | 346282 | 2026-09-16 19:30:00 | BOT | GRE | 257 | 26 | 22,23,24,25,26 | 5 | nao |

### 4.1 Quais rodadas já tinham jogado quando cada uma dessas partidas foi disputada

- **R2 / 346262 / CAP x COR / 2026-02-19 19:30:00**: posicao cronologica 30 de 257; ja tinham sido disputadas as rodadas 1..3 (29 partidas); 1 rodada(s) maior(es) ja tinham sido disputadas antes dela: 3.
- **R5 / 346408 / CAP x BOT / 2026-03-29 19:30:00**: posicao cronologica 77 de 257; ja tinham sido disputadas as rodadas 1..8 (76 partidas); 3 rodada(s) maior(es) ja tinham sido disputadas antes dela: 6,7,8.
- **R4 / 346349 / BAH x CHA / 2026-07-17 19:30:00**: posicao cronologica 180 de 257; ja tinham sido disputadas as rodadas 1..19 (179 partidas); 15 rodada(s) maior(es) ja tinham sido disputadas antes dela: 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19.
- **R4 / 346337 / BOT x VIT / 2026-07-23 19:30:00**: posicao cronologica 188 de 257; ja tinham sido disputadas as rodadas 1..19 (187 partidas); 15 rodada(s) maior(es) ja tinham sido disputadas antes dela: 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19.
- **R4 / 346335 / FLA x MIR / 2026-09-02 19:30:00**: posicao cronologica 246 de 257; ja tinham sido disputadas as rodadas 1..25 (245 partidas); 21 rodada(s) maior(es) ja tinham sido disputadas antes dela: 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25.
- **R21 / 346282 / BOT x GRE / 2026-09-16 19:30:00**: posicao cronologica 257 de 257; ja tinham sido disputadas as rodadas 1..26 (256 partidas); 5 rodada(s) maior(es) ja tinham sido disputadas antes dela: 22,23,24,25,26.

## 5. Desvio dentro da própria rodada

`amplitude_dias` é a quantidade de dias entre a primeira e a última partida da rodada. Uma rodada com amplitude grande ou com partidas fora da janela oficial perde a propriedade de ser um bloco temporal.

| rodada | n_partidas | janela_inicio | janela_fim | data_ini | data_fim | amplitude_dias | n_fora |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 2026-01-28 | 2026-01-29 | 2026-01-28 00:00:00 | 2026-01-29 00:00:00 | 2 | 0 |
| 2 | 10 | 2026-02-04 | 2026-02-05 | 2026-02-04 00:00:00 | 2026-02-19 00:00:00 | 16 | 1 |
| 3 | 10 | 2026-02-11 | 2026-02-12 | 2026-02-10 00:00:00 | 2026-02-12 00:00:00 | 3 | 1 |
| 4 | 10 | 2026-02-25 | 2026-02-26 | 2026-02-25 00:00:00 | 2026-09-02 00:00:00 | 190 | 3 |
| 5 | 10 | 2026-03-11 | 2026-03-12 | 2026-03-10 00:00:00 | 2026-03-29 00:00:00 | 20 | 2 |
| 6 | 10 | 2026-03-14 | 2026-03-16 | 2026-03-14 00:00:00 | 2026-03-16 00:00:00 | 3 | 0 |
| 7 | 10 | 2026-03-18 | 2026-03-19 | 2026-03-18 00:00:00 | 2026-03-19 00:00:00 | 2 | 0 |
| 8 | 10 | 2026-03-21 | 2026-03-22 | 2026-03-21 00:00:00 | 2026-03-22 00:00:00 | 2 | 0 |
| 9 | 10 | 2026-04-01 | 2026-04-02 | 2026-04-01 00:00:00 | 2026-04-02 00:00:00 | 2 | 0 |
| 10 | 10 | 2026-04-04 | 2026-04-05 | 2026-04-04 00:00:00 | 2026-04-05 00:00:00 | 2 | 0 |
| 11 | 10 | 2026-04-11 | 2026-04-12 | 2026-04-11 00:00:00 | 2026-04-12 00:00:00 | 2 | 0 |
| 12 | 10 | 2026-04-18 | 2026-04-19 | 2026-04-18 00:00:00 | 2026-04-19 00:00:00 | 2 | 0 |
| 13 | 10 | 2026-04-25 | 2026-04-26 | 2026-04-25 00:00:00 | 2026-04-26 00:00:00 | 2 | 0 |
| 14 | 10 | 2026-05-02 | 2026-05-03 | 2026-05-02 00:00:00 | 2026-05-03 00:00:00 | 2 | 0 |
| 15 | 10 | 2026-05-09 | 2026-05-10 | 2026-05-09 00:00:00 | 2026-05-10 00:00:00 | 2 | 0 |
| 16 | 10 | 2026-05-16 | 2026-05-17 | 2026-05-16 00:00:00 | 2026-05-17 00:00:00 | 2 | 0 |
| 17 | 10 | 2026-05-23 | 2026-05-25 | 2026-05-23 00:00:00 | 2026-05-25 00:00:00 | 3 | 0 |
| 18 | 10 | 2026-05-30 | 2026-05-31 | 2026-05-30 00:00:00 | 2026-05-31 00:00:00 | 2 | 0 |
| 19 | 10 | 2026-07-21 | 2026-07-23 | 2026-07-16 00:00:00 | 2026-07-23 00:00:00 | 8 | 4 |
| 20 | 10 | 2026-07-25 | 2026-07-26 | 2026-07-25 00:00:00 | 2026-07-26 00:00:00 | 2 | 0 |
| 21 | 7 | 2026-07-29 | 2026-07-30 | 2026-07-29 00:00:00 | 2026-09-16 00:00:00 | 50 | 1 |
| 22 | 10 | 2026-08-08 | 2026-08-09 | 2026-08-08 00:00:00 | 2026-08-09 00:00:00 | 2 | 0 |
| 23 | 10 | 2026-08-15 | 2026-08-17 | 2026-08-15 00:00:00 | 2026-08-17 00:00:00 | 3 | 0 |
| 24 | 10 | 2026-08-22 | 2026-08-24 | 2026-08-22 00:00:00 | 2026-08-24 00:00:00 | 3 | 0 |
| 25 | 10 | 2026-08-29 | 2026-08-31 | 2026-08-29 00:00:00 | 2026-08-31 00:00:00 | 3 | 0 |
| 26 | 10 | 2026-09-05 | 2026-09-07 | 2026-09-05 00:00:00 | 2026-09-07 00:00:00 | 3 | 0 |

## 6. Comparação rodada vs data

Para cada par (clube, partida-alvo) foram montados os dois históricos. A tabela completa esta em `exp12_inv_temporal_historicos.csv`; abaixo, o agregado.

| metrica | pares |
|---|---|
| pares (clube, partida-alvo) analisados | 514 |
| historico identico em A e B | 388 |
| **historico diferente em A e B** | **126** |
| **A inclui partida disputada depois (vaza)** | **115** |
| A exclui partida ja disputada (perde passado) | 12 |
| B inclui partida da mesma rodada da alvo | 0 |
| so o dia, sem hora, mudaria o historico | 0 |

### 6.1 As partidas que A trata como passado, mas foram jogadas depois

Estas são as partidas que tornam a alternativa A um risco de leakage.

| partida_id | rodada | data | mandante | visitante | n_alvos_afetados |
|---|---|---|---|---|---|
| 346335 | 4 | 2026-09-02 19:30:00 | FLA | MIR | 42 |
| 346337 | 4 | 2026-07-23 19:30:00 | BOT | VIT | 30 |
| 346349 | 4 | 2026-07-17 19:30:00 | BAH | CHA | 28 |
| 346282 | 21 | 2026-09-16 19:30:00 | BOT | GRE | 10 |
| 346408 | 5 | 2026-03-29 19:30:00 | CAP | BOT | 6 |
| 346262 | 2 | 2026-02-19 19:30:00 | CAP | COR | 2 |

### 6.2 Casos em que A exclui passado que já tinha sido jogado

O erro oposto: a partida-alvo foi disputada **depois** de partidas de rodadas maiores, e A as exclui por serem de rodada futura. São **12 pares**, todos com a partida-alvo sendo uma das partidas adiadas.

| clube | partida_alvo | rodada_alvo | data_alvo | n_perdidas_em_A | passado_mais_antigo | passado_mais_recente |
|---|---|---|---|---|---|---|
| FLA | 346335 | 4 | 2026-09-02 19:30:00 | 21 | 346401 (R5, 2026-03-11 21:30:00) | 346445 (R25, 2026-08-30 16:00:00) |
| BOT | 346408 | 5 | 2026-03-29 19:30:00 | 3 | 346476 (R6, 2026-03-14 20:30:00) | 346589 (R8, 2026-03-21 16:00:00) |
| BOT | 346337 | 4 | 2026-07-23 19:30:00 | 15 | 346476 (R6, 2026-03-14 20:30:00) | 346536 (R19, 2026-07-16 19:30:00) |
| BOT | 346282 | 21 | 2026-09-16 19:30:00 | 5 | 346312 (R22, 2026-08-08 21:00:00) | 346496 (R26, 2026-09-06 18:30:00) |
| COR | 346262 | 2 | 2026-02-19 19:30:00 | 1 | 346294 (R3, 2026-02-12 20:00:00) | 346294 (R3, 2026-02-12 20:00:00) |
| BAH | 346349 | 4 | 2026-07-17 19:30:00 | 14 | 346409 (R5, 2026-03-11 20:00:00) | 346513 (R18, 2026-05-30 17:30:00) |
| GRE | 346282 | 21 | 2026-09-16 19:30:00 | 5 | 346317 (R22, 2026-08-08 16:00:00) | 346503 (R26, 2026-09-07 20:00:00) |
| VIT | 346337 | 4 | 2026-07-23 19:30:00 | 15 | 346409 (R5, 2026-03-11 20:00:00) | 346543 (R19, 2026-07-16 19:30:00) |
| CAP | 346262 | 2 | 2026-02-19 19:30:00 | 1 | 346298 (R3, 2026-02-12 19:00:00) | 346298 (R3, 2026-02-12 19:00:00) |
| CAP | 346408 | 5 | 2026-03-29 19:30:00 | 3 | 346475 (R6, 2026-03-15 16:00:00) | 346592 (R8, 2026-03-22 16:00:00) |
| CHA | 346349 | 4 | 2026-07-17 19:30:00 | 14 | 346403 (R5, 2026-03-12 20:00:00) | 346508 (R18, 2026-05-31 16:00:00) |
| MIR | 346335 | 4 | 2026-09-02 19:30:00 | 21 | 346405 (R5, 2026-03-10 21:30:00) | 346449 (R25, 2026-08-30 18:30:00) |

A lista completa do que B enxerga e A não, por par, está em `exp12_inv_temporal_historicos.csv`, coluna `partidas_perdidas`.

### 6.3 B nunca entra na mesma rodada da alvo

Casos em que B inclui uma partida da **mesma** rodada da alvo: **0**. Isso e estrutural, nao sorte: cada clube disputa exatamente uma partida por rodada, entao a propria rodada do alvo nunca tem outra partida do mesmo clube para entrar no historio. A diferenca entre A e B vem sempre de rodadas diferentes, nunca de jogos distintos dentro da mesma rodada.

## 7. Risco de leakage

### 7.1 Alternativa A — `rodada < rodada_alvo`

- Vazamento confirmado em **115 pares (clube, partida-alvo)**, provenientes de apenas **6 partidas** que a API devolveu com rodada antiga mas foram disputadas muito depois.
- O vazamento não é local. A partida 346335 (R4, disputada em 02/09) entra no histórico de **42 partidas-alvo** (21 do FLA e 21 do MIR), e em todas elas a data real da partida da R4 é posterior à partida-alvo.
- As 6 partidas que causam vaza em A são **exatamente** as 6 partidas que, na F1.1, tinham elenco ausente do payload. Nenhuma partida que vaza em A está fora desse conjunto, e 6 das partidas com elenco ausente (346299, 346405, 346535, 346536, 346539, 346543) **não** causam vaza em A — são as disputadas **antes** da janela da rodada, que apenas atrasam a informação em vez de antecipar o futuro. As duas anomalias do EXP-12 se concentram nas mesmas partidas adiadas.
- Além do vazamento, A **perde** informação real em 12 pares: partidas de rodadas maiores já disputadas que A exclui por serem de rodada futura.
- A é determinística e reproduz o calendário oficial, mas esse calendário não descreve a linha do tempo real dos jogos.

### 7.2 Alternativa B — `data real < data da partida-alvo`

- B não inclui nenhuma partida disputada depois da alvo: **0 vazaimentos por construção**.
- B ganha 12 pares em que a rodada manda excluir um passado que já tinha sido jogado.
- B é bem definida por clube: **nenhum clube disputa duas partidas no mesmo dia** (0 casos), então não há empate a resolver dentro do histórico de um clube. O mesmo vale ignorando a hora: usar só o dia muda o histórico em 0 pares.
- Risco próprio de B: ela assume que o dado da partida anterior estava disponivel no momento da alvo. Isso precisa ser verificado contra a granularidade da fonte (seção 8), não pode ser assumido.

### 7.3 O que as duas alternativas nunca resolvem

- **3 pares de rodadas diferentes disputados no mesmo dia.** Neles a data, sozinha, nao estabelece precedencia.
- **1 par com horario identico**: R4 346337 (BOT x VIT), R19 346538 (COR x REM). Sao clubes diferentes, entao nao afeta o historico de nenhum clube, mas prova que a fonte tem empate de timestamp.

## 8. Granularidade da fonte

O payload de scouts e por **rodada**, nao por partida: `r{N}.json` traz os atletas de todos os clubes que disputaram aquela rodada. A data e o horario da partida sao metadados, nao dados de disponibilidade.

Consequencia direta: mesmo que o eixo temporal fosse a data real com hora, a unidade de informacao realmente publicada pela API e a rodada. Uma partida disputada na terca-feira da R19 e outra na quinta da R19 compartilham o mesmo payload. Se uma feature temporal precisar de informacao que só existe depois do encerramento da rodada, a data da partida individual **nao** é o limite correto — é a data de disponibilidade do payload.

| rodada | partidas_no_arquivo | clubes_no_payload | atletas_no_payload |
|---|---|---|---|
| 1 | 10 | 20 | 327 |
| 2 | 10 | 20 | 302 |
| 3 | 10 | 20 | 301 |
| 4 | 10 | 20 | 242 |
| 5 | 10 | 20 | 274 |
| 6 | 10 | 20 | 336 |
| 7 | 10 | 20 | 335 |
| 8 | 10 | 20 | 335 |
| 9 | 10 | 20 | 342 |
| 10 | 10 | 20 | 338 |
| 11 | 10 | 20 | 333 |
| 12 | 10 | 20 | 340 |
| 13 | 10 | 20 | 336 |
| 14 | 10 | 20 | 335 |
| 15 | 10 | 20 | 333 |
| 16 | 10 | 20 | 333 |
| 17 | 10 | 20 | 335 |
| 18 | 10 | 20 | 339 |
| 19 | 10 | 20 | 209 |
| 20 | 10 | 20 | 338 |
| 21 | 10 | 20 | 212 |
| 22 | 10 | 20 | 336 |
| 23 | 10 | 20 | 338 |
| 24 | 10 | 20 | 337 |
| 25 | 10 | 20 | 332 |
| 26 | 10 | 20 | 342 |

Esta auditoria **nao** mede a latência de publicação da API. Ela aponta que a medição é necessária antes de escolher o eixo.

## 9. Recomendação técnica

O que a evidência sustenta, sem decidir o eixo:

1. **A alternativa A tem vazamento comprovado** em 115 pares, causado por 6 partidas adiadas. Não é um risco teórico: é observável nas datas.
2. **A alternativa B não vaza por construção** e não tem empate interno por clube, porque nenhum clube joga duas vezes no mesmo dia.
3. **B, sozinha, não é suficiente.** O payload é por rodada, então o limite de disponibilidade de informação é a rodada, não a partida. Usar `data` como eixo exige um segundo critério de disponibilidade.
4. **Opções a decidir pelo auditor**, com o custo de cada uma:

   - **A:** manter o eixo da rodada. Reproduz o calendário oficial, aceita os 115 vazaamentos e ignora os 12 casos de passado real perdido. Exige aceitar explicitamente o vazamento.
   - **B:** usar a data real como eixo e tratar a disponibilidade do payload como uma restrição adicional, por rodada.
   - **B restrito:** usar a data real, mas truncar o histórico no limite de disponibilidade do payload da rodada de origem, combinando os dois critérios.
   - **Eixo duplo:** manter `rodada` como eixo de janela e usar a data apenas como filtro de disponibilidade.

Nenhuma dessas foi implementada. O pipeline não foi alterado.

## 10. O que NÃO foi feito

- `data/interim/exp12/team_match_history.parquet` não foi alterado.
- Nenhum arquivo de origem (`r{N}.json`, `partidas{N}.json`, `rodadas.json`) foi alterado.
- Nenhuma feature temporal foi criada.
- Nenhuma data, rodada ou placar foi corrigido.
- A alternativa A e a alternativa B não foram implementadas nem comparadas por desempenho de modelo.
- A escolha do eixo temporal continua **pendente do auditor**.

## 11. Arquivos desta auditoria

- `scripts/exp12_auditoria_temporal.py`
- `reports/exp12/exp12_auditoria_temporal.md` (este documento)
- `reports/exp12/exp12_inv_temporal_inversoes.csv`
- `reports/exp12/exp12_inv_temporal_historicos.csv`

