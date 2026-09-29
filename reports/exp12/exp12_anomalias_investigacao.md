# EXP-12 — Investigação das anomalias do histórico de partidas

- Temporada: **2026** | Rodadas: **R1..R26**
- Gerado por: `scripts/exp12_investigar_anomalias.py`
- Natureza: **somente leitura**. Nenhum dataset foi alterado, nenhuma correção foi aplicada e nenhuma feature temporal foi criada.

## 1. Metodologia da investigação

A auditoria da Fase 1 registrou 152 anomalias. Este documento rastreia as três famílias com risco de contaminar feature:

- **A.** 24 linhas de 12 partidas com os 14 scouts diretos zerados;
- **B.** 48 linhas com soma de `G` diferente de `gols_marcados`;
- **C.** 30 linhas com `GS` diferente de `gols_sofridos`.

Procedimento, para cada caso:

1. Localizar a partida em `partidas{N}.json` e recuperar o placar oficial.
2. Abrir o payload `r{N}.json` da mesma rodada e contar, para o clube afetado: atletas no payload, atletas com `entrou_em_campo=True`, atletas com scout não vazio, goleiros (`posicao_id=1`) e quais scouts aparecem.
3. Recalcular a agregação a partir do payload bruto, sem usar o dataset, e comparar com o que está gravado.
4. Testar hipóteses explicativas: ausência de elenco, ausência de goleiro, gol contra (`GC`), troca de goleiro.
5. Classificar o caso e, quando nada explica, marcar como **sem explicação aparente**.

Nenhum valor foi corrigido. As tabelas abaixo são evidência, não correção.

## 2. Resumo dos achados

- **Bloco A:** 12 partidas / 24 linhas com scouts zerados. Causa: **elenco do clube ausente do payload** — em todos os casos o clube aparece no `partidas{N}.json`, mas o `r{N}.json` traz somente o técnico (`posicao_id=6`, `entrou_em_campo=False`). **Não é performance zero.**
- **Bloco B:** 48 casos de `G` != `gols_marcados`. A identidade `gols_marcados = G_do_time + GC_do_adversario` reconcilia 237 de 257 partidas (92.2%). Os casos remanescentes se dividem em elenco ausente e casos sem explicação aparente.
- **Bloco C:** 30 casos de `GS` != `gols_sofridos`, **todos explicados**: elenco ausente do payload (18), goleiro ausente do payload no clube 315 (6) e dois ou mais goleiros do mesmo clube no payload (6, todos do clube 265/BAH). Nenhum caso ficou sem explicação.
- **Bloco D (transversal):** 13 partidas têm data fora da janela oficial da rodada. `rodada` não é uma ordenação cronológica confiável; `data` é.

## 3. Bloco A — as 12 partidas com scouts totalmente zerados

### 3.1 Tabela por partida e clube afetado

| rodada | partida_id | mandante | visitante | placar | clube_afetado | n_atletas_no_payload | n_entrou_em_campo | n_scout_nao_vazio | existe_atleta_no_payload | scouts_no_payload | n_atletas_adversario | clubes_com_elenco_completo_na_rodada | diagnostico |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 346262 | CAP | COR | 0x1 | COR | 1 | 0 | 0 | sim | (nenhum) | 1 | 18/20 | elenco ausente do payload |
| 2 | 346262 | CAP | COR | 0x1 | CAP | 1 | 0 | 0 | sim | (nenhum) | 1 | 18/20 | elenco ausente do payload |
| 3 | 346299 | VIT | FLA | 1x2 | FLA | 1 | 0 | 0 | sim | (nenhum) | 1 | 18/20 | elenco ausente do payload |
| 3 | 346299 | VIT | FLA | 1x2 | VIT | 1 | 0 | 0 | sim | (nenhum) | 1 | 18/20 | elenco ausente do payload |
| 4 | 346335 | FLA | MIR | 2x0 | FLA | 1 | 0 | 0 | sim | (nenhum) | 1 | 14/20 | elenco ausente do payload |
| 4 | 346335 | FLA | MIR | 2x0 | MIR | 1 | 0 | 0 | sim | (nenhum) | 1 | 14/20 | elenco ausente do payload |
| 4 | 346337 | BOT | VIT | 0x0 | BOT | 1 | 0 | 0 | sim | (nenhum) | 1 | 14/20 | elenco ausente do payload |
| 4 | 346337 | BOT | VIT | 0x0 | VIT | 1 | 0 | 0 | sim | (nenhum) | 1 | 14/20 | elenco ausente do payload |
| 4 | 346349 | BAH | CHA | 2x0 | BAH | 1 | 0 | 0 | sim | (nenhum) | 1 | 14/20 | elenco ausente do payload |
| 4 | 346349 | BAH | CHA | 2x0 | CHA | 1 | 0 | 0 | sim | (nenhum) | 1 | 14/20 | elenco ausente do payload |
| 5 | 346405 | MIR | SAN | 2x2 | SAN | 1 | 0 | 0 | sim | (nenhum) | 1 | 16/20 | elenco ausente do payload |
| 5 | 346405 | MIR | SAN | 2x2 | MIR | 1 | 0 | 0 | sim | (nenhum) | 1 | 16/20 | elenco ausente do payload |
| 5 | 346408 | CAP | BOT | 4x1 | BOT | 1 | 0 | 0 | sim | (nenhum) | 1 | 16/20 | elenco ausente do payload |
| 5 | 346408 | CAP | BOT | 4x1 | CAP | 1 | 0 | 0 | sim | (nenhum) | 1 | 16/20 | elenco ausente do payload |
| 19 | 346535 | FLU | RBB | 1x1 | FLU | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 19 | 346535 | FLU | RBB | 1x1 | RBB | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 19 | 346536 | BOT | SAN | 2x1 | BOT | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 19 | 346536 | BOT | SAN | 2x1 | SAN | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 19 | 346539 | MIR | GRE | 2x1 | GRE | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 19 | 346539 | MIR | GRE | 2x1 | MIR | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 19 | 346543 | VIT | VAS | 1x0 | VAS | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 19 | 346543 | VIT | VAS | 1x0 | VIT | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 21 | 346282 | BOT | GRE | 3x2 | BOT | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |
| 21 | 346282 | BOT | GRE | 3x2 | GRE | 1 | 0 | 0 | sim | (nenhum) | 1 | 12/20 | elenco ausente do payload |

### 3.2 O que existe no payload para o clube afetado

Para cada caso, o conteúdo literal do `r{N}.json` para o clube afetado:

- **R2 / partida 346262 / COR (id 264)**: 40990/Dorival Júnior pos=6 entrou=False scout=None
- **R2 / partida 346262 / CAP (id 293)**: 92273/Odair Hellmann pos=6 entrou=False scout=None
- **R3 / partida 346299 / FLA (id 262)**: 139076/Filipe Luís pos=6 entrou=False scout=None
- **R3 / partida 346299 / VIT (id 287)**: 92180/Jair Ventura pos=6 entrou=False scout=None
- **R4 / partida 346335 / FLA (id 262)**: 88963/Leonardo Jardim pos=6 entrou=False scout=None
- **R4 / partida 346335 / MIR (id 2305)**: 104027/Rafael Guanaes pos=6 entrou=False scout=None
- **R4 / partida 346337 / BOT (id 263)**: 117965/Martín Anselmi pos=6 entrou=False scout=None
- **R4 / partida 346337 / VIT (id 287)**: 92180/Jair Ventura pos=6 entrou=False scout=None
- **R4 / partida 346349 / BAH (id 265)**: 97341/Rogério Ceni pos=6 entrou=False scout=None
- **R4 / partida 346349 / CHA (id 315)**: 73481/Gilmar Dal Pozzo pos=6 entrou=False scout=None
- **R5 / partida 346405 / SAN (id 277)**: 104754/Juan Vojvoda pos=6 entrou=False scout=None
- **R5 / partida 346405 / MIR (id 2305)**: 104027/Rafael Guanaes pos=6 entrou=False scout=None
- **R5 / partida 346408 / BOT (id 263)**: 117965/Martín Anselmi pos=6 entrou=False scout=None
- **R5 / partida 346408 / CAP (id 293)**: 92273/Odair Hellmann pos=6 entrou=False scout=None
- **R19 / partida 346535 / FLU (id 266)**: 99366/Luis Zubeldía pos=6 entrou=False scout=None
- **R19 / partida 346535 / RBB (id 280)**: 39850/Vagner Mancini pos=6 entrou=False scout=None
- **R19 / partida 346536 / BOT (id 263)**: 149990/Franclim Carvalho pos=6 entrou=False scout=None
- **R19 / partida 346536 / SAN (id 277)**: 42411/Cuca pos=6 entrou=False scout=None
- **R19 / partida 346539 / GRE (id 284)**: 88037/Luís Castro pos=6 entrou=False scout=None
- **R19 / partida 346539 / MIR (id 2305)**: 104027/Rafael Guanaes pos=6 entrou=False scout=None
- **R19 / partida 346543 / VAS (id 267)**: 128177/Pedro Emanuel pos=6 entrou=False scout=None
- **R19 / partida 346543 / VIT (id 287)**: 92180/Jair Ventura pos=6 entrou=False scout=None
- **R21 / partida 346282 / BOT (id 263)**: 149990/Franclim Carvalho pos=6 entrou=False scout=None
- **R21 / partida 346282 / GRE (id 284)**: 88037/Luís Castro pos=6 entrou=False scout=None

### 3.3 Leitura

- `n_atletas_no_payload` é **1** em todos os 24 casos, e esse único atleta é o técnico.
- `n_entrou_em_campo` é **0** e `n_scout_nao_vazio` é **0** em todos os casos.
- `scouts_no_payload` é `(nenhum)` em todos os casos: o técnico não tem scout, ou tem `{}`.
- Importante: nestas 12 partidas **os dois clubes** estão sem elenco no payload, então o adversário também aparece com `n_atletas_no_payload` igual a 1. O controle correto é a nível de rodada: a coluna `clubes_com_elenco_completo_na_rodada` mostra que, na mesma rodada, 14 a 18 dos 20 clubes têm elenco completo. O defeito é **parcial por clube**, não do arquivo inteiro.
- Nenhum caso de **performance zero** foi encontrado: as 24 linhas são 24 casos de ausência de dado e 0 casos de performance zero.
- O schema dos payloads é idêntico em R1..R26 (chaves de topo ('atletas', 'clubes', 'posicoes', 'rodada', 'total_atletas'), `atletas` como `dict`, chaves de atleta ('apelido', 'clube_id', 'entrou_em_campo', 'foto', 'pontuacao', 'posicao_id', 'scout')). **Não há variação de estrutura**: é conteúdo ausente, não formato diferente.

### 3.4 Alcance do problema além das 12 partidas

O mesmo defeito de payload aparece em 6 rodadas e atinge 30 pares (rodada, clube). As 24 linhas acima são as que entraram no dataset; as 6 restantes pertencem às 3 partidas da R21 excluídas por falta de placar oficial:

| rodada | n_atletas | elenco_ausente | n_clubes_elenco_ausente |
|---|---|---|---|
| 2 | 302 | 264,293 | 2 |
| 3 | 301 | 262,287 | 2 |
| 4 | 242 | 262,263,265,287,315,2305 | 6 |
| 5 | 274 | 263,277,293,2305 | 4 |
| 19 | 209 | 263,266,267,277,280,284,287,2305 | 8 |
| 21 | 212 | 263,267,276,277,280,282,284,315 | 8 |

## 4. Bloco B — os 48 casos de G != gols_marcados

### 4.1 Classificação

| classe | casos |
|---|---|
| B: reconcilia com GC do adversario | 20 |
| A: elenco ausente do payload | 18 |
| C: sem explicacao aparente | 10 |

Distribuição da diferença `G - gols_marcados`:

| diferenca | casos |
|---|---|
| -4 | 1 |
| -3 | 1 |
| -2 | 9 |
| -1 | 33 |
| 1 | 4 |

### 4.2 A identidade de reconciliação

Teste aplicado em todas as partidas com placar oficial:

```
gols_marcados_do_time   = G_do_time   + GC_do_adversario
gols_marcados_do_advers = G_adversario + GC_do_time
```

- Partidas testadas: **257**
- Partidas que reconciliam: **237** (92.2%)
- Partidas que não reconciliam: **20**

| rodada | partida_id | placar | predito | payload_incompleto |
|---|---|---|---|---|
| 1 | 346237 | 2x1 | 3x1 | nao |
| 1 | 346239 | 2x1 | 2x0 | nao |
| 1 | 346236 | 4x0 | 3x0 | nao |
| 2 | 346261 | 5x3 | 4x4 | nao |
| 2 | 346262 | 0x1 | 0x0 | sim |
| 3 | 346299 | 1x2 | 0x0 | sim |
| 3 | 346296 | 3x3 | 2x3 | nao |
| 3 | 346291 | 1x0 | 2x0 | nao |
| 4 | 346349 | 2x0 | 0x0 | sim |
| 4 | 346335 | 2x0 | 0x0 | sim |
| 5 | 346405 | 2x2 | 0x0 | sim |
| 5 | 346408 | 4x1 | 0x0 | sim |
| 8 | 346589 | 1x2 | 1x1 | nao |
| 8 | 346587 | 0x1 | 0x2 | nao |
| 16 | 346455 | 2x1 | 2x0 | nao |
| 19 | 346536 | 2x1 | 0x0 | sim |
| 19 | 346543 | 1x0 | 0x0 | sim |
| 19 | 346535 | 1x1 | 0x0 | sim |
| 19 | 346539 | 2x1 | 0x0 | sim |
| 21 | 346282 | 3x2 | 0x0 | sim |

Ou seja: `G` sozinho **não** é o total de gols do time. O gol contra (`GC`) é creditado ao adversário, então `G + GC_adversario` é a reconciliação correta. Isso explica a maior parte dos 48 casos sem que nenhum dado esteja errado.

### 4.3 Detalhe dos 48 casos

| rodada | partida_id | mandante | visitante | placar | clube | gols_marcados_oficial | G_agregado_time | GC_agregado_time | GC_agregado_adversario | dif_G_menos_oficial | reconcilia_com_GC_adversario | n_participantes_payload | G_jogadores | classe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 346236 | BOT (263) | CRU (283) | 4x0 | BOT | 4 | 3 | 0 | 0 | -1 | nao | 16 | 107231/Danilo Santos (G=2); 111831/Matheus Martins (G=1) | C: sem explicacao aparente |
| 1 | 346237 | SAO (276) | FLA (262) | 2x1 | SAO | 2 | 3 | 0 | 0 | 1 | nao | 17 | 80287/Luciano (G=1); 85403/Danielzinho (G=1); 95799/Artur (G=1) | C: sem explicacao aparente |
| 1 | 346239 | MIR (2305) | VAS (267) | 2x1 | VAS | 1 | 0 | 1 | 0 | -1 | nao | 14 | (ninguem com G) | C: sem explicacao aparente |
| 1 | 346239 | MIR (2305) | VAS (267) | 2x1 | MIR | 2 | 1 | 0 | 1 | -1 | sim | 15 | 68708/Eduardo (G=1) | B: reconcilia com GC do adversario |
| 1 | 346240 | CAM (282) | PAL (275) | 2x2 | CAM | 2 | 1 | 0 | 1 | -1 | sim | 15 | 116034/Victor Hugo (G=1) | B: reconcilia com GC do adversario |
| 2 | 346261 | GRE (284) | BOT (263) | 5x3 | BOT | 3 | 4 | 0 | 0 | 1 | nao | 17 | 107231/Danilo Santos (G=2); 63289/Edenílson (G=1); 92171/Arthur Cabral (G=1) | C: sem explicacao aparente |
| 2 | 346261 | GRE (284) | BOT (263) | 5x3 | GRE | 5 | 4 | 0 | 0 | -1 | nao | 16 | 105089/Tetê (G=1); 105563/Carlos Vinícius (G=3) | C: sem explicacao aparente |
| 2 | 346262 | CAP (293) | COR (264) | 0x1 | COR | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 3 | 346291 | FLU (266) | BOT (263) | 1x0 | FLU | 1 | 2 | 0 | 0 | 1 | nao | 17 | 39148/Hulk (G=1); 89837/Lucho Acosta (G=1) | C: sem explicacao aparente |
| 3 | 346296 | CAM (282) | REM (364) | 3x3 | CAM | 3 | 2 | 0 | 0 | -1 | nao | 15 | 100842/Ruan (G=1); 68920/Dudu (G=1) | C: sem explicacao aparente |
| 3 | 346299 | VIT (287) | FLA (262) | 1x2 | FLA | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 3 | 346299 | VIT (287) | FLA (262) | 1x2 | VIT | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 4 | 346335 | FLA (262) | MIR (2305) | 2x0 | FLA | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 4 | 346349 | BAH (265) | CHA (315) | 2x0 | BAH | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 5 | 346405 | MIR (2305) | SAN (277) | 2x2 | SAN | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 5 | 346405 | MIR (2305) | SAN (277) | 2x2 | MIR | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 5 | 346408 | CAP (293) | BOT (263) | 4x1 | BOT | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 5 | 346408 | CAP (293) | BOT (263) | 4x1 | CAP | 4 | 0 | 0 | 0 | -4 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 7 | 346562 | SAN (277) | INT (285) | 1x2 | INT | 2 | 1 | 0 | 1 | -1 | sim | 17 | 107989/Carbonero (G=1) | B: reconcilia com GC do adversario |
| 7 | 346571 | GRE (284) | VIT (287) | 2x0 | GRE | 2 | 1 | 0 | 1 | -1 | sim | 17 | 142004/Amuzu (G=1) | B: reconcilia com GC do adversario |
| 8 | 346587 | SAO (276) | PAL (275) | 0x1 | PAL | 1 | 2 | 0 | 0 | 1 | nao | 18 | 99274/Alexander Barboza (G=1); 99815/Jhon Arias (G=1) | C: sem explicacao aparente |
| 8 | 346589 | RBB (280) | BOT (263) | 1x2 | BOT | 2 | 1 | 0 | 0 | -1 | nao | 16 | 72493/Alex Telles (G=1) | C: sem explicacao aparente |
| 9 | 346602 | CFC (294) | VAS (267) | 1x1 | CFC | 1 | 0 | 0 | 1 | -1 | sim | 17 | (ninguem com G) | B: reconcilia com GC do adversario |
| 10 | 346613 | BAH (265) | PAL (275) | 1x2 | PAL | 2 | 1 | 0 | 1 | -1 | sim | 18 | 99815/Jhon Arias (G=1) | B: reconcilia com GC do adversario |
| 11 | 346325 | MIR (2305) | BAH (265) | 1x2 | MIR | 1 | 0 | 0 | 1 | -1 | sim | 17 | (ninguem com G) | B: reconcilia com GC do adversario |
| 12 | 346355 | RBB (280) | REM (364) | 4x2 | RBB | 4 | 2 | 0 | 2 | -2 | sim | 17 | 108133/Isidro Pitta (G=2) | B: reconcilia com GC do adversario |
| 15 | 346428 | COR (264) | SAO (276) | 3x2 | SAO | 2 | 1 | 0 | 1 | -1 | sim | 15 | 80287/Luciano (G=1) | B: reconcilia com GC do adversario |
| 15 | 346429 | MIR (2305) | CHA (315) | 1x1 | CHA | 1 | 0 | 0 | 1 | -1 | sim | 17 | (ninguem com G) | B: reconcilia com GC do adversario |
| 16 | 346455 | FLU (266) | SAO (276) | 2x1 | SAO | 1 | 0 | 0 | 0 | -1 | nao | 15 | (ninguem com G) | C: sem explicacao aparente |
| 16 | 346464 | CHA (315) | REM (364) | 2x3 | REM | 3 | 2 | 0 | 1 | -1 | sim | 17 | 105048/Jajá (G=1); 80196/Yago Pikachu (G=1) | B: reconcilia com GC do adversario |
| 18 | 346513 | BAH (265) | BOT (263) | 2x1 | BAH | 2 | 1 | 0 | 1 | -1 | sim | 17 | 91772/David Duarte (G=1) | B: reconcilia com GC do adversario |
| 19 | 346535 | FLU (266) | RBB (280) | 1x1 | FLU | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 19 | 346535 | FLU (266) | RBB (280) | 1x1 | RBB | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 19 | 346536 | BOT (263) | SAN (277) | 2x1 | BOT | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 19 | 346536 | BOT (263) | SAN (277) | 2x1 | SAN | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 19 | 346539 | MIR (2305) | GRE (284) | 2x1 | GRE | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 19 | 346539 | MIR (2305) | GRE (284) | 2x1 | MIR | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 19 | 346543 | VIT (287) | VAS (267) | 1x0 | VIT | 1 | 0 | 0 | 0 | -1 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 19 | 346544 | CHA (315) | FLA (262) | 0x4 | FLA | 4 | 3 | 0 | 1 | -1 | sim | 17 | 122321/Lorran (G=1); 94583/Pedro (G=2) | B: reconcilia com GC do adversario |
| 20 | 346423 | SAN (277) | CHA (315) | 2x2 | CHA | 2 | 1 | 0 | 1 | -1 | sim | 17 | 95332/Marcinho (G=1) | B: reconcilia com GC do adversario |
| 21 | 346282 | BOT (263) | GRE (284) | 3x2 | BOT | 3 | 0 | 0 | 0 | -3 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 21 | 346282 | BOT (263) | GRE (284) | 3x2 | GRE | 2 | 0 | 0 | 0 | -2 | nao | 0 | (ninguem com G) | A: elenco ausente do payload |
| 21 | 346285 | MIR (2305) | REM (364) | 2x1 | MIR | 2 | 1 | 0 | 1 | -1 | sim | 17 | 97460/João Victor (G=1) | B: reconcilia com GC do adversario |
| 21 | 346289 | VIT (287) | PAL (275) | 0x4 | PAL | 4 | 3 | 0 | 1 | -1 | sim | 17 | 105647/Maurício (G=1); 112863/Ramón Sosa (G=1); 99815/Arias (G=1) | B: reconcilia com GC do adversario |
| 22 | 346317 | GRE (284) | SAO (276) | 2x1 | SAO | 1 | 0 | 0 | 1 | -1 | sim | 17 | (ninguem com G) | B: reconcilia com GC do adversario |
| 23 | 346346 | CHA (315) | BAH (265) | 3x3 | BAH | 3 | 2 | 0 | 1 | -1 | sim | 17 | 104783/Acevedo (G=1); 114208/Alejo Véliz (G=1) | B: reconcilia com GC do adversario |
| 25 | 346454 | REM (364) | CFC (294) | 2x3 | CFC | 3 | 2 | 0 | 1 | -1 | sim | 17 | 101729/Fabinho (G=1); 90302/Pedro Rocha (G=1) | B: reconcilia com GC do adversario |
| 26 | 346499 | RBB (280) | BAH (265) | 2x3 | RBB | 2 | 1 | 0 | 1 | -1 | sim | 17 | 112886/Lucas Barbosa (G=1) | B: reconcilia com GC do adversario |

### 4.4 Jogadores com G e autores de GC

- R1 / 346239 / VAS: G de [(ninguem com G)], GC de [97095/Carlos Cuesta (GC=1)] (oficial 1)

### 4.5 Leitura

- Classe **A (elenco ausente)**: 18 casos — `G` zerado por falta de payload, exatamente o Bloco A.
- Classe **B (reconcilia com GC do adversário)**: 20 casos — divergência aparente, mas `G + GC_adversario` fecha com o placar.
- Classe **C (sem explicação aparente)**: 10 casos. São eles:

| rodada | partida_id | mandante | visitante | placar | clube | gols_marcados_oficial | G_agregado_time | G_agregado_adversario | dif_G_menos_oficial | G_jogadores |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 346236 | BOT (263) | CRU (283) | 4x0 | BOT | 4 | 3 | 0 | -1 | 107231/Danilo Santos (G=2); 111831/Matheus Martins (G=1) |
| 1 | 346237 | SAO (276) | FLA (262) | 2x1 | SAO | 2 | 3 | 1 | 1 | 80287/Luciano (G=1); 85403/Danielzinho (G=1); 95799/Artur (G=1) |
| 1 | 346239 | MIR (2305) | VAS (267) | 2x1 | VAS | 1 | 0 | 1 | -1 | (ninguem com G) |
| 2 | 346261 | GRE (284) | BOT (263) | 5x3 | BOT | 3 | 4 | 4 | 1 | 107231/Danilo Santos (G=2); 63289/Edenílson (G=1); 92171/Arthur Cabral (G=1) |
| 2 | 346261 | GRE (284) | BOT (263) | 5x3 | GRE | 5 | 4 | 4 | -1 | 105089/Tetê (G=1); 105563/Carlos Vinícius (G=3) |
| 3 | 346291 | FLU (266) | BOT (263) | 1x0 | FLU | 1 | 2 | 0 | 1 | 39148/Hulk (G=1); 89837/Lucho Acosta (G=1) |
| 3 | 346296 | CAM (282) | REM (364) | 3x3 | CAM | 3 | 2 | 3 | -1 | 100842/Ruan (G=1); 68920/Dudu (G=1) |
| 8 | 346587 | SAO (276) | PAL (275) | 0x1 | PAL | 1 | 2 | 0 | 1 | 99274/Alexander Barboza (G=1); 99815/Jhon Arias (G=1) |
| 8 | 346589 | RBB (280) | BOT (263) | 1x2 | BOT | 2 | 1 | 1 | -1 | 72493/Alex Telles (G=1) |
| 16 | 346455 | FLU (266) | SAO (276) | 2x1 | SAO | 1 | 0 | 2 | -1 | (ninguem com G) |

Para esses casos o payload tem elenco completo, não há `GC` e a soma de `G` continua divergindo do placar oficial. As diferenças vão de -1 a +1. Não foi encontrada explicação na fonte; ficam registrados como divergência de origem.

## 5. Bloco C — os 30 casos de GS != gols_sofridos

### 5.1 Classificação

| classe | casos |
|---|---|
| A: elenco ausente do payload | 18 |
| B: elenco presente, goleiro ausente do payload | 6 |
| C: mais de um goleiro com GS (dupla contagem) | 4 |
| D: GS e SG simultaneos no mesmo time (contradicao interna) | 2 |

### 5.2 Detalhe dos 30 casos

| rodada | partida_id | clube | adversario | placar | GS_agregado | gols_sofridos | dif | n_atletas_no_payload | n_participantes | n_goleiros | n_goleiros_com_GS | n_goleiros_com_SG | classe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 346238 | BAH | COR (264) | 1x2 | 3 | 1 | 2 | 19 | 19 | 2 | 2 | 0 | C: mais de um goleiro com GS (dupla contagem) |
| 1 | 346244 | CHA | SAN (277) | 4x2 | 0 | 2 | -2 | 16 | 16 | 0 | 0 | 0 | B: elenco presente, goleiro ausente do payload |
| 2 | 346256 | CHA | VAS (267) | 1x1 | 0 | 1 | -1 | 16 | 16 | 0 | 0 | 0 | B: elenco presente, goleiro ausente do payload |
| 2 | 346262 | CAP | COR (264) | 0x1 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 2 | 346263 | BAH | FLU (266) | 1x1 | 2 | 1 | 1 | 18 | 18 | 2 | 2 | 0 | C: mais de um goleiro com GS (dupla contagem) |
| 3 | 346292 | BAH | VAS (267) | 0x1 | 3 | 0 | 3 | 18 | 18 | 2 | 1 | 1 | D: GS e SG simultaneos no mesmo time (contradicao interna) |
| 3 | 346299 | FLA | VIT (287) | 1x2 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 3 | 346299 | VIT | FLA (262) | 1x2 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 3 | 346300 | CHA | CFC (294) | 3x3 | 0 | 3 | -3 | 16 | 16 | 0 | 0 | 0 | B: elenco presente, goleiro ausente do payload |
| 4 | 346335 | MIR | FLA (262) | 2x0 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 4 | 346349 | CHA | BAH (265) | 2x0 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 5 | 346403 | CHA | SAO (276) | 2x0 | 0 | 2 | -2 | 16 | 16 | 0 | 0 | 0 | B: elenco presente, goleiro ausente do payload |
| 5 | 346405 | SAN | MIR (2305) | 2x2 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 5 | 346405 | MIR | SAN (277) | 2x2 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 5 | 346408 | BOT | CAP (293) | 4x1 | 0 | 4 | -4 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 5 | 346408 | CAP | BOT (263) | 4x1 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 5 | 346409 | BAH | VIT (287) | 1x1 | 3 | 1 | 2 | 18 | 18 | 2 | 2 | 0 | C: mais de um goleiro com GS (dupla contagem) |
| 6 | 346481 | BAH | INT (285) | 0x1 | 1 | 0 | 1 | 18 | 18 | 2 | 1 | 1 | D: GS e SG simultaneos no mesmo time (contradicao interna) |
| 6 | 346484 | CHA | GRE (284) | 1x1 | 0 | 1 | -1 | 15 | 15 | 0 | 0 | 0 | B: elenco presente, goleiro ausente do payload |
| 8 | 346591 | CHA | INT (285) | 2x0 | 0 | 2 | -2 | 16 | 16 | 0 | 0 | 0 | B: elenco presente, goleiro ausente do payload |
| 8 | 346594 | BAH | REM (364) | 4x1 | 6 | 4 | 2 | 18 | 18 | 3 | 2 | 0 | C: mais de um goleiro com GS (dupla contagem) |
| 19 | 346535 | FLU | RBB (280) | 1x1 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 19 | 346535 | RBB | FLU (266) | 1x1 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 19 | 346536 | BOT | SAN (277) | 2x1 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 19 | 346536 | SAN | BOT (263) | 2x1 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 19 | 346539 | GRE | MIR (2305) | 2x1 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 19 | 346539 | MIR | GRE (284) | 2x1 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 19 | 346543 | VAS | VIT (287) | 1x0 | 0 | 1 | -1 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 21 | 346282 | BOT | GRE (284) | 3x2 | 0 | 2 | -2 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |
| 21 | 346282 | GRE | BOT (263) | 3x2 | 0 | 3 | -3 | 1 | 0 | 0 | 0 | 0 | A: elenco ausente do payload |

### 5.3 Goleiros presentes no payload

- R1 / 346238 / BAH (GS=3, sofridos=1): 71043/Léo Vieira entrou=True GS=2 SG=0 DE=4 DP=0 scout={'CA': 1, 'DE': 4, 'GS': 2}; 91101/Ronaldo entrou=True GS=1 SG=0 DE=4 DP=0 scout={'DE': 4, 'GS': 1}
- R1 / 346244 / CHA (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R2 / 346256 / CHA (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R2 / 346262 / CAP (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R2 / 346263 / BAH (GS=2, sofridos=1): 71043/Léo Vieira entrou=True GS=1 SG=0 DE=10 DP=0 scout={'DE': 10, 'FS': 2, 'GS': 1}; 91101/Ronaldo entrou=True GS=1 SG=0 DE=3 DP=0 scout={'DE': 3, 'GS': 1}
- R3 / 346292 / BAH (GS=3, sofridos=0): 71043/Léo Vieira entrou=True GS=3 SG=0 DE=2 DP=0 scout={'DE': 2, 'GS': 3}; 91101/Ronaldo entrou=True GS=0 SG=1 DE=4 DP=0 scout={'CA': 1, 'DE': 4, 'SG': 1}
- R3 / 346299 / FLA (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R3 / 346299 / VIT (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R3 / 346300 / CHA (GS=0, sofridos=3): (nenhum goleiro posicao_id=1 no payload)
- R4 / 346335 / MIR (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R4 / 346349 / CHA (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R5 / 346403 / CHA (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R5 / 346405 / SAN (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R5 / 346405 / MIR (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R5 / 346408 / BOT (GS=0, sofridos=4): (nenhum goleiro posicao_id=1 no payload)
- R5 / 346408 / CAP (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R5 / 346409 / BAH (GS=3, sofridos=1): 71043/Léo Vieira entrou=True GS=2 SG=0 DE=5 DP=0 scout={'DE': 5, 'GS': 2}; 91101/Ronaldo entrou=True GS=1 SG=0 DE=1 DP=0 scout={'DE': 1, 'GS': 1}
- R6 / 346481 / BAH (GS=1, sofridos=0): 71043/Léo Vieira entrou=True GS=1 SG=0 DE=3 DP=0 scout={'DE': 3, 'FS': 1, 'GS': 1}; 91101/Ronaldo entrou=True GS=0 SG=1 DE=2 DP=0 scout={'CA': 1, 'DE': 2, 'FS': 1, 'SG': 1}
- R6 / 346484 / CHA (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R8 / 346591 / CHA (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R8 / 346594 / BAH (GS=6, sofridos=4): 71043/Léo Vieira entrou=True GS=2 SG=0 DE=5 DP=0 scout={'DE': 5, 'GS': 2}; 85425/João Paulo entrou=True GS=4 SG=0 DE=4 DP=0 scout={'DE': 4, 'FS': 1, 'GS': 4}; 91101/Ronaldo entrou=True GS=0 SG=0 DE=2 DP=0 scout={'DE': 2}
- R19 / 346535 / FLU (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R19 / 346535 / RBB (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R19 / 346536 / BOT (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R19 / 346536 / SAN (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R19 / 346539 / GRE (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R19 / 346539 / MIR (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R19 / 346543 / VAS (GS=0, sofridos=1): (nenhum goleiro posicao_id=1 no payload)
- R21 / 346282 / BOT (GS=0, sofridos=2): (nenhum goleiro posicao_id=1 no payload)
- R21 / 346282 / GRE (GS=0, sofridos=3): (nenhum goleiro posicao_id=1 no payload)

### 5.4 Leitura

- Classe **A (elenco ausente)**: 18 casos — `GS` zerado por falta de payload.
- Classe **B (elenco presente, goleiro ausente)**: 6 casos, todos do **clube 315 (CHA/Fortaleza)**, que tem 15 a 16 atletas no payload mas **nenhum com `posicao_id=1`** nas rodadas 1 a 8. O elenco de linha está lá; o goleiro não está. Isso zera estruturalmente `GS`, `SG` e `DP` para esse clube nessas rodadas — inclusive na R1 já validada.
- Classe **C (dupla contagem na troca de goleiro)**: 4 casos, todos do **clube 265 (BAH/Bahia)**, e em todos há **dois goleiros com `GS` na mesma partida**. O `GS` é somado para os dois e estoura o total de gols sofridos do time:

- R1 / 346238: GS agregado 3 contra 1 gols sofridos (2 goleiros com GS > 0)
- R2 / 346263: GS agregado 2 contra 1 gols sofridos (2 goleiros com GS > 0)
- R5 / 346409: GS agregado 3 contra 1 gols sofridos (2 goleiros com GS > 0)
- R8 / 346594: GS agregado 6 contra 4 gols sofridos (2 goleiros com GS > 0)

- Classe **D (contradição interna do payload)**: 2 casos, também do **clube 265 (BAH)**. São partidas em que a API devolve **dois goleiros da mesma equipe, ambos com `entrou_em_campo=True`**, e ainda atribui `GS` a um e `SG` ao outro **na mesma partida** — estados mutuamente exclusivos:

- R3 / 346292 / BAH: placar 0x1 => 0 gols sofridos, mas GS agregado 3 (1 goleiro com GS) **e** 1 goleiro com SG. 71043/Léo Vieira entrou=True GS=3 SG=0 DE=2 DP=0 scout={'DE': 2, 'GS': 3}; 91101/Ronaldo entrou=True GS=0 SG=1 DE=4 DP=0 scout={'CA': 1, 'DE': 4, 'SG': 1}
- R6 / 346481 / BAH: placar 0x1 => 0 gols sofridos, mas GS agregado 1 (1 goleiro com GS) **e** 1 goleiro com SG. 71043/Léo Vieira entrou=True GS=1 SG=0 DE=3 DP=0 scout={'DE': 3, 'FS': 1, 'GS': 1}; 91101/Ronaldo entrou=True GS=0 SG=1 DE=2 DP=0 scout={'CA': 1, 'DE': 2, 'FS': 1, 'SG': 1}

Esse é o único mecanismo que explica as 30 anomalias de `GS`, e ele é estrutural. A checagem global de `GS` contra `gols_sofridos` em todos os clube-partida do payload dá:

- **1 goleiro com `entrou_em_campo=True`**: 470 de 470 fecham exatamente com o placar oficial (100%).
- **2 ou mais goleiros com `entrou_em_campo=True`**: 7 de 13 fecham. Os 6 que **não** fecham são:

| rodada | partida_id | clube | gols_sofridos | GS_agregado | goleiros |
|---|---|---|---|---|---|
| 1 | 346238 | BAH | 1 | 3 | Léo Vieira entrou=True GS=2; Ronaldo entrou=True GS=1 |
| 2 | 346263 | BAH | 1 | 2 | Léo Vieira entrou=True GS=1; Ronaldo entrou=True GS=1 |
| 3 | 346292 | BAH | 0 | 3 | Léo Vieira entrou=True GS=3; Ronaldo entrou=True GS=None |
| 5 | 346409 | BAH | 1 | 3 | Léo Vieira entrou=True GS=2; Ronaldo entrou=True GS=1 |
| 6 | 346481 | BAH | 0 | 1 | Léo Vieira entrou=True GS=1; Ronaldo entrou=True GS=None |
| 8 | 346594 | BAH | 4 | 6 | Léo Vieira entrou=True GS=2; João Paulo entrou=True GS=4; Ronaldo entrou=True GS=None |

As 6 divergências são **todas do clube 265 (BAH)**, e apenas nas rodadas 1 a 8, que é exatamente a janela em que a API devolve 2 a 3 goleiros do Bahia no payload (Léo Vieira 71043, Ronaldo 91101 e, na R8, João Paulo 85425). A partir da R9 o Bahia passa a ter **um único goleiro por partida** e o `GS` fecha em todas as rodadas restantes. Os outros clubes com 2+ goleiros no payload (294 na R4, 283 na R5, 2305 na R12, 263 e 284 na R18, 285 na R24, 265 na R15 e R17) **não** geram contradição: o `GS` é creditado só a um deles.

Portanto `GS` **não é** uma medida de gols sofridos do time. É o contador por goleiro e ele: (a) desaparece quando o goleiro não está no payload; (b) é somado em todos os goleiros marcados como `entrou_em_campo=True` na partida; (c) pode coexistir com `SG` do colega na mesma partida. É exatamente por isso que a semântica validada proíbe usar `GS` como `gols_sofridos`.

- Classe **E (sem explicação aparente)**: 0 casos.

## 6. Bloco D — rodada não é ordem cronológica

13 partidas têm `partida_data` fora da janela oficial de rodada definida em `rodadas.json`:

| rodada | janela_oficial | data_min | data_max | n_fora_da_janela | datas_fora |
|---|---|---|---|---|---|
| 2 | 2026-02-04..2026-02-05 | 2026-02-04 | 2026-02-19 | 1 | 2026-02-19 |
| 3 | 2026-02-11..2026-02-12 | 2026-02-10 | 2026-02-12 | 1 | 2026-02-10 |
| 4 | 2026-02-25..2026-02-26 | 2026-02-25 | 2026-09-02 | 3 | 2026-07-17,2026-07-23,2026-09-02 |
| 5 | 2026-03-11..2026-03-12 | 2026-03-10 | 2026-03-29 | 2 | 2026-03-10,2026-03-29 |
| 19 | 2026-07-21..2026-07-23 | 2026-07-16 | 2026-07-23 | 4 | 2026-07-16,2026-07-17 |
| 21 | 2026-07-29..2026-07-30 | 2026-07-29 | 2026-10-03 | 2 | 2026-09-16,2026-10-03 |

São partidas adiadas que a API devolve dentro do arquivo de outra rodada. A R21 é o caso extremo: reúne jogos de 29/07, 16/09 e 03/10/2026. Para qualquer feature temporal, a ordenação por `rodada` não é segura; a ordenação por `data` também precisa tratar partidas que ocorram depois da rodada que as indexou.

## 7. Padrões encontrados

1. **Ausência de elenco no payload** (6 rodadas, 30 pares rodada-clube, 24 linhas no dataset). O único atleta presente é o técnico.
2. **Ausência de goleiro no payload** (clube 315 nas rodadas 1 a 8, com elenco de linha completo). Zera `GS`, `SG` e `DP` estruturalmente.
3. **Reconciliação de gols depende de `GC`.** `G` isolado não fecha com o placar em 7.8% das partidas, mas `G + GC_adversario` fecha em 92.2%.
4. **`GS` é contador por goleiro**, não por time: fecha em 100% dos clube-partida com um único goleiro em campo e quebra justamente nos casos de 2+ goleiros do mesmo time.
5. **Schema uniforme**: o problema é sempre ausência de conteúdo, nunca mudança de formato.
6. **`rodada` não ordena o tempo**: 13 partidas estão fora da janela da própria rodada.

## 8. Casos sem explicação aparente

- Bloco B, classe C: **10** casos de `G` != `gols_marcados` com payload completo e sem `GC` no jogo. Diferenças de -1 e +1.
- Bloco C, classe E: 0 casos. As 2 divergências que pareciam sem causa no primeiro corte eram o clube 265 com `GS` e `SG` simultâneos, e passaram a ter causa identificada (classe D).

Nenhum desses foi corrigido. Ficam registrados para decisão do auditor.

## 9. Recomendação técnica por tipo de anomalia

As recomendações abaixo **não foram aplicadas**. Cada uma é uma decisão metodológica que precisa ser aprovada antes de virar código.

### 9.1 Scouts zerados por elenco ausente (Bloco A)

- **Não** tratar como zero de performance. O valor zero seria uma afirmação falsa sobre o time.
- **Recomendação:** marcar essas linhas com um indicador de ausência de dado (`scouts_disponiveis = 0`) e, nas features temporais, excluir a partida da janela de scouts do clube em vez de somar zero. A cobertura da janela passa a ser explícita, o que já é exigido pelo EXP-12.
- O placar oficial dessas linhas continua válido e pode ser usado como está.
- Nenhuma mudança no artefato `team_match_history.parquet` nesta fase: o indicador é derivado na etapa de features, com o payload como fonte.

### 9.2 Goleiro ausente do payload (achado do Bloco C, classe B)

- `GS`, `SG` e `DP` desses clubes são **estruturalmente zero**, não zero de performance.
- **Recomendação:** nunca derivar clean sheet, goleiro ou defesa a partir de `GS`/`SG` sem antes checar a presença de goleiro no payload da rodada. Para contexto de adversário, `gols_sofridos` do placar oficial é o campo confiável.
- Se o EXP-12 precisar de features de goleiro, o payload de `posicao_id=1` tem de ser uma pré-condição verificável, e as rodadas 1 a 8 do clube 315 seriam lacunas conhecidas.

### 9.3 G != gols_marcados (Bloco B)

- **Não** alterar `G` nem `gols_marcados`.
- **Recomendação:** manter os dois campos com suas fontes distintas e documentar a identidade `gols_marcados = G + GC_adversario` como a regra de reconciliação. Se o EXP-12 quiser um total de gols por time a partir de scouts, ele precisa somar `GC` do adversário — mas isso é decisão de feature, não correção desta base.
- As 10 sem explicação aparente devem ser tratadas como ruído de fonte e nunca como sinal.

### 9.4 GS != gols_sofridos (Bloco C)

- **Não** usar `GS` como gols sofridos e **não** criar coluna `clean_sheet`.
- O ponto a favoring dessa decisão ficou forte: `GS` fecha com o placar em 470/470 clube-partida com um único goleiro em campo, e quebra em 6 dos 13 casos com 2+ goleiros. A série é confiável só sob a condição de goleiro único.
- **Recomendação:** manter `GS` e `SG` como scouts brutos com a semântica atual e registrar a divergência. Se alguma feature de contexto de adversário precisar de clean sheet, ela deve vir de `gols_sofridos == 0` do placar oficial, com a restrição de que a informação de goleiro está ausente em rodadas específicas.

### 9.5 Rodada não cronológica (Bloco D)

- **Recomendação:** antes de qualquer janela temporal, decidir explicitamente se a ordenação é por `rodada` (ordem oficial da competição) ou por `data` (ordem real dos jogos). As 13 partidas fora de janela mudam o resultado de qualquer janela que use rodada como eixo de tempo.
- Este ponto é **bloqueante** para a próxima fase e precisa ser resolvido antes de construir features.

## 10. O que NÃO foi feito

- `data/interim/exp12/team_match_history.parquet` não foi alterado.
- Nenhuma feature temporal foi criada.
- Nenhum valor de scout, gol ou placar foi corrigido.
- `cartola_jogador_rodada_v2.parquet` e `catboost_v2.py` não foram tocados.
- A decisão de tratamento continua **pendente do auditor**.

## 11. Arquivos desta investigação

- `scripts/exp12_investigar_anomalias.py`
- `reports/exp12/exp12_anomalias_investigacao.md` (este documento)
- `reports/exp12/exp12_inv_scouts_zerados.csv`
- `reports/exp12/exp12_inv_gols.csv`
- `reports/exp12/exp12_inv_goleiros.csv`
- `reports/exp12/exp12_inv_perfil_payload.csv`

