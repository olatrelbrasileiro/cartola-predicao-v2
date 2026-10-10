# Cartola FC — Previsão de Pontuação (v2)

## Objetivo
Prever a pontuação de jogadores do Cartola FC e, posteriormente, usar essas
previsões para auxiliar na escalação.

Esta é uma reescrita do zero. A tentativa anterior misturava muitas features e
modelos sem prova de ganho fora da amostra. A v2 é metodologicamente rigorosa
e incremental.

## Princípio central
Primeiro construir corretamente -> depois medir -> depois provar -> só então
adicionar complexidade.

Cada etapa precisa provar que adicionou informação **fora da amostra**.

---

## Feature set inicial (somente estes 5 grupos)

1. CONFRONTO
   - Odds da Bzorro
   - Mando de campo

2. HISTÓRICO
   - Pontuações de todas as participações
   - Scouts de todas as participações

3. RECENTE
   - Pontuações das últimas 5 participações/rodadas válidas
   - Scouts das últimas 5 participações/rodadas válidas

4. BÁSICO
   - Média/componente de pontos sem G/A/SG

5. PARTICIPAÇÃO EM GOLS
   - G+A por jogo

### Proibido adicionar inicialmente
xG, xA, FotMob, força detalhada do time, adversário detalhado, dezenas de
features derivadas, features externas adicionais, redes neurais, transformers,
RNN/LSTM, Kalman filter, modelos hierárquicos complexos, learning-to-rank.

---

## Regras de dados (obrigatórias)

Distinguir sempre:
- (A) jogador NÃO participou
- (B) jogador entrou em campo e fez 0 pontos
- (C) jogador entrou em campo e pontuou

"Não participou" NÃO é pontuação 0.
Não participante não pode ser transformado artificialmente em observação zero
para calcular Histórico ou Recente.
Se entrou e fez 0, é observação real e deve entrar.

As features usadas para prever a rodada R usam SOMENTE informações
disponíveis antes de R.
Nunca calcular média da temporada inteira e depois fazer split temporal.
As odds precisam respeitar um timestamp operacional definido antes da previsão.

---

## Validação oficial

Walk-forward com janela expansiva:

    treinar R1..R9   -> prever R10
    treinar R1..R10  -> prever R11
    treinar R1..R11  -> prever R12
    ...

- Todas as transformações de features respeitam o corte temporal.
- Proibido split aleatório.
- Proibido k-fold aleatório.
- Nenhuma informação da rodada futura pode entrar nas features.
- O harness de validação é construído e testado ANTES de qualquer modelo.

---

## Métricas

- Principal: MAE
- Secundária: Spearman por rodada (ranking)
- Diagnóstico: RMSE

Sempre reportar:
- resultado agregado
- resultado por rodada
- MAE por rodada
- Spearman por rodada

Quando possível, separar desempenho em jogos com evento de gol/assistência e
sem esses eventos.

MAPE NÃO é métrica principal.

---

## Baselines obrigatórios (antes de ML complexo)

- B0: Histórico
- B1: Recente
- B2: Histórico + Recente
- B3: Básico + Participação em Gols
- B4: B3 + Confronto
- B5: Conjunto completo por posição vs modelo único

Todos usam exatamente o mesmo harness walk-forward.

---

## Modelos (somente após baselines)

- M1: Ridge / regressão linear regularizada
- M2: CatBoost

Abordagens mais sofisticadas só se houver evidência consistente de ganho fora
da amostra.

---

## Critério de sucesso

Uma abordagem só é progresso se houver ganho consistente fora da amostra.
Observar:
- maioria dos folds/rodadas
- MAE
- Spearman
- estabilidade
- ausência de leakage
- reprodutibilidade

Se uma feature não acrescenta informação de forma consistente, deve ser
removida.
Se um modelo complexo não supera os baselines, deve ser descartado.

---

## Estrutura

    cartola-predicao-v2/
    ├── data/raw/         dados brutos (nunca editados à mão)
    ├── data/interim/     dados parcialmente tratados
    ├── data/processed/   dataset jogador × rodada final
    ├── experiments/      saídas por experimento
    ├── models/           modelos treinados serializados
    ├── reports/          métricas, tabelas, gráficos, auditorias
    ├── scripts/          código executável
    ├── notebooks/        inspeção (opcional)
    ├── config.yaml       configuração mínima
    ├── ETAPAS.md         checklist das etapas
    └── README.md         este arquivo

## 🏆 Modelo Oficial de Produção

- **Versão**: CatBoost v3 (com Pooling entre Temporadas).
- **Baseline**: B2S (corrigido para P0 - avaliação apenas em participantes).
- **Destaque Goleiros**: O pipeline agora incorpora **features de contexto de time** (via Banco Mestre `team_scout_history`) para a posição de Goleiro, substituindo a dependência excessiva do proxy "Confronto" e melhorando o sinal preditivo (MAE e Spearman).

## 🧹 Estado do Repositório
- Arquivos legados (v1, baselines pré-P0) foram movidos para `archive/v1_pre_p0/`.
- O pipeline oficial reside em `scripts/` com sufixo `_v2` ou `_v3`.
