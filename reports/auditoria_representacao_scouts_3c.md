# Auditoria da representacao temporal dos scouts — Etapa 3.c

- Data: 2026-09-20T07:40:27
- Dataset: data/interim/cartola_jogador_rodada_v2.parquet
- Linhas: 77253
- Metodo: para cada (temporada, atleta_id), ordenar por rodada;
  comparar valores de scout em rodadas consecutivas;
  medir frequencia de nao-decrescimento.

## Tabela de monotonicidade

| scout | pares_validos | nao_decrescente | proporcao | delta<0 | semantica |
|---|---|---|---|---|---|
| A | 15174 | 15174 | 1.0000 | 0 | CUMULATIVO |
| CA | 32741 | 32741 | 1.0000 | 0 | CUMULATIVO |
| CV | 4242 | 4242 | 1.0000 | 0 | CUMULATIVO |
| DE | 3246 | 3246 | 1.0000 | 0 | CUMULATIVO |
| DS | 44256 | 44256 | 1.0000 | 0 | CUMULATIVO |
| FC | 44780 | 44780 | 1.0000 | 0 | CUMULATIVO |
| FD | 29565 | 29565 | 1.0000 | 0 | CUMULATIVO |
| FF | 36213 | 36213 | 1.0000 | 0 | CUMULATIVO |
| FS | 44881 | 44881 | 1.0000 | 0 | CUMULATIVO |
| FT | 7841 | 7841 | 1.0000 | 0 | CUMULATIVO |
| G | 18059 | 18059 | 1.0000 | 0 | CUMULATIVO |
| I | 18591 | 18591 | 1.0000 | 0 | CUMULATIVO |
| PC | 3633 | 3633 | 1.0000 | 0 | CUMULATIVO |
| PS | 2552 | 2552 | 1.0000 | 0 | CUMULATIVO |
| PP | 838 | 838 | 1.0000 | 0 | CUMULATIVO |
| SG | 15708 | 15708 | 1.0000 | 0 | CUMULATIVO |
| V | 1703 | 1703 | 1.0000 | 0 | CUMULATIVO |
| DP | 493 | 493 | 1.0000 | 0 | CUMULATIVO |
| GC | 798 | 798 | 1.0000 | 0 | CUMULATIVO |

## Conclusao

Todos os 19 scouts apresentam 100% de nao-decrescimento em rodadas
consecutivas. Isso e evidencia definitiva de que os scouts no
dataset canonico sao CUMULATIVOS (total acumulado desde o inicio
da temporada ate a rodada da linha).

## Consequencia

Qualquer uso dos scouts como feature ou componente de pontuacao
precisa primeiro converter o valor cumulativo em contribuicao
por rodada, via diferenca entre observacoes consecutivas do
mesmo jogador na mesma temporada.

## Achado secundario

O scout `GS` (Gol Sofrido) existe no RAW 2024-2026 mas esta
ausente do dataset canonico. Isso e uma lacuna da Etapa 2.4
a ser registrada. Nao afeta o B3 (GS e excluido por definicao).
