# Elegibilidade pre-rodada via `status_id` — v1

- Gerado em: 2026-09-21T21:13:58
- Parquet: `data/interim/cartola_jogador_rodada_v2.parquet`
- SHA-256: `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Linhas: 77253

## 0. Regra congelada

| status_id | nome | elegivel |
|---|---|---|
| 7 | Provavel | True |
| 2 | Duvida | True |
| 5 | Contundido | False |
| 3 | Suspenso | False |
| 6 | Nulo | False |

O filtro aplica-se SOMENTE ao universo da rodada-alvo. `status_id` nao e' feature de regressao. Nao se infere semantica a partir de resultados historicos.

## A. Cobertura por (temporada, rodada)

- Rodadas cobertas: 103
- Cobertura elegivel media: 31.06%
- Rodadas com elegivel vazio: 0

Arquivo: `reports/status_elegibilidade_v1_cobertura.csv`

## B. Distribuicao por temporada

 temporada  status_2  status_3  status_5  status_6  status_7  total
      2024       730       511      1631     17598      8029  28499
      2025       614       497      2057     17250      8171  28589
      2026       620       320      1558     11902      5765  20165

Arquivo: `reports/status_elegibilidade_v1_distribuicao.csv`

## C. Integridade

- Linhas totais: 77253
- Nulos reais em `status_id`: 0
- Codigos presentes: [2, 3, 5, 6, 7]
- Codigos desconhecidos: []
- Duplicatas (T,R,A): 0
- Celulas (T,R,A) com mais de um status: 0

## D. Teste de invariancia do filtro as variaveis de resultado

Para cada um dos 76 folds, o conjunto elegivel foi recalculado em uma copia da rodada-alvo com `pontos_num`, `entrou_em_campo`, `scout_*`, `media_num`, `variacao_num` e `preco_num` perturbados. Isso prova que o filtro depende exclusivamente de `status_id`. NAO substitui a auditoria de proveniencia temporal do snapshot, que foi realizada separadamente (`reports/status_id_auditoria.md`).

- **PASS**: conjunto elegivel invariante em 76 folds.

## E. Integracao com o harness

- Folds processados: 76
- Folds por temporada (observado): {2024: 29, 2025: 29, 2026: 18}
- Folds por temporada (esperado):  {2024: 29, 2025: 29, 2026: 18}
- Cobertura elegivel media nos folds: 30.22%

Arquivo por fold: `reports/status_elegibilidade_v1_folds.csv`

## F. Criterios de PASS

- parquet_sha_ok: OK
- sem_nulos_reais: OK
- sem_desconhecidos: OK
- sem_duplicatas: OK
- sem_multi_status: OK
- folds_76: OK
- nenhum_fold_sem_elegiveis: OK
- todos_folds_invariantes: OK
- elegiveis_apenas_7_2: OK
- nao_elegiveis_3_5_6: OK

**Resultado: PASS**

## G. Integridade do Parquet canonico

- SHA-256 (antes): `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- SHA-256 (depois): `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Inalterado: **True**

---

## PENDENCIAS / RISCOS

- **Semantica dos codigos**: a atribuicao `7=Provavel`, `2=Duvida`,   `5=Contundido`, `3=Suspenso`, `6=Nulo` deve ser documentada com a   fonte (documentacao oficial ou decisao humana rastreavel). Este   relatorio apenas aplica a regra, nao a valida contra a fonte.
- **Consequencia do filtro no MAE**: NAO medida nesta etapa por   decisao metodologica. O efeito do filtro na performance do modelo   sera objeto de etapa posterior.
- **Universo elegivel e resultado historico**: o filtro pode excluir   jogadores que efetivamente entraram em campo. Isso e' esperado e   nao indica erro nos dados; reflete o snapshot de mercado.
