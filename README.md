# Previsão de churn em SaaS financeiro

Pipeline completo de ciência de dados, da geração e limpeza dos dados à modelagem e a um painel executivo em Streamlit, para responder a duas perguntas de negócio:

1. Quais clientes têm maior probabilidade de cancelar nos próximos 30 dias?
2. O que mais explica esse cancelamento?

> **Aviso importante:** a base é **sintética**, criada para fins de portfólio. O efeito dos erros de API sobre o cancelamento foi definido por mim na geração dos dados. O que o projeto demonstra é o **método** (desenho sem vazamento, validação, interpretação e comunicação), não uma descoberta sobre clientes reais.

## Resultados em resumo

| Pergunta | Resposta nesta base |
|---|---|
| Qual o principal fator de churn? | Erros de API nos últimos 30 dias |
| Quanto esse grupo pesa? | 11,2% dos clientes (3 ou mais erros) concentram 37,1% dos cancelamentos |
| Qual o risco desse grupo? | Churn de 27,8%, contra 5,9% nos demais (4,7 vezes maior) |
| Qual o impacto? | Cerca de 2.444 cancelamentos (29,2% do total) em excesso, ~R$ 1,15 mi de receita mensal |
| E se os erros fossem contidos? | Simulação do modelo: 23,1% menos cancelamentos (8.379 para 6.447) |
| Qual a qualidade do modelo? | AUC de 0,75; nos 10% de maior risco, captura 37% dos cancelamentos (lift de 3,7x) |

Os erros de API são **uma** das causas principais, não a única. A seção "Limitações e próximos passos" lista outras frentes a explorar.

## Painel

![Visão executiva](docs/imagens/visao_executiva.png)
![Fatores de churn](docs/imagens/fatores_de_churn.png)

O painel tem cinco abas: Visão executiva (conclusões, importância das variáveis, churn por faixa de erros, previsto contra observado, receita em risco por plano), Fatores de churn, Carteira em risco (lista exportável), Simulador de retenção e Metodologia.

## Estrutura do repositório

```
churn-prediction-saas/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                  # data de corte, janelas, semente e caminhos
│   ├── 01_gera_dados_churn.py     # gera os dados brutos (sintéticos, com falhas propositais)
│   ├── 02_limpeza_dados.py        # limpeza e validação
│   ├── 03_feature_engineering.py  # variáveis de comportamento
│   └── 04_modelagem_churn.py      # modelos, métricas, importância e pontuação da carteira
├── app/
│   └── app_dashboard.py           # painel em Streamlit
├── docs/imagens/                  # prints do painel
├── data/                          # gerado localmente (não versionado)
└── outputs/                       # gerado localmente (não versionado)
```

## Etapas do pipeline

**1. Geração dos dados (`01`).** 100 mil clientes e 3,5 milhões de eventos de uso (login, consulta ao painel, criação de transação, exportação de relatório, erro de API, abertura de chamado). O comportamento é gerado primeiro e o cancelamento é sorteado depois, a partir de uma função de risco que combina erros de API, uso, chamados, plano, tempo de casa e canal, mais um termo de ruído que representa motivos não observados. Idade e renda não influenciam o cancelamento. Os arquivos saem com falhas propositais: nulos, duplicatas, idades impossíveis, datas em dois formatos, registros órfãos, caixa e espaços inconsistentes e fusos horários misturados.

**2. Limpeza (`02`).** Remove duplicatas (500 usuários e 14.321 eventos), trata idades inválidas (153) e nulas (4.081) com a mediana, preenche renda nula (6.142) com a mediana do plano, padroniza as ações, remove eventos de usuários inexistentes (28.415) e converte todos os horários para UTC, respeitando o fuso de cada registro.

**3. Variáveis (`03`).** Calculadas somente com os 90 dias anteriores à data de corte (29/09/2025).

| Variável | Construção |
|---|---|
| `api_errors_30d` | Contagem de eventos `error_api_500` nos 30 dias antes do corte |
| `api_errors_prev_60d` | Mesma contagem nos 60 dias anteriores a esses |
| `events_30d`, `events_prev_60d` | Eventos de uso (sem erros e chamados) em cada período |
| `activity_trend` | Ritmo diário recente dividido pelo ritmo dos 60 dias anteriores, com suavização (+1) e teto de 5. Vale 1 se o uso se manteve e menos de 0,5 em queda forte |
| `days_since_last_activity` | Dias entre o último evento de uso e a data de corte |
| `support_calls_90d` | Chamados abertos nos 90 dias |
| `tenure_days`, `plan`, `acquisition_channel`, `age`, `income` | Perfil do cliente |

O alvo (`churn`) indica se o cliente cancelou nos 30 dias **seguintes** à data de corte. Nenhuma informação posterior ao corte entra nas variáveis.

**4. Modelagem (`04`).** Compara uma regra simples (3 ou mais erros de API), uma regressão logística (baseline) e um gradient boosting (`HistGradientBoostingClassifier`, profundidade 3, taxa de aprendizado 0,05, até 400 iterações com parada antecipada e regularização L2). Separação dos dados: 80% treino (80 mil clientes) e 20% teste (20 mil), estratificados pelo alvo, com semente 42. Para pontuar toda a carteira, usa validação cruzada de 5 partes: cada cliente é pontuado por um modelo que não o viu no treino.

**5. Painel (`app`).** Lê as pontuações e as métricas geradas e apresenta as conclusões.

## Ferramentas

Python, pandas e NumPy (dados e variáveis), scikit-learn (modelos, validação e importância por permutação), SciPy (calibração da taxa de churn na geração), Plotly (gráficos) e Streamlit (painel).

## Decisões de projeto

| Decisão | Motivo |
|---|---|
| Data de corte com janelas de 90 dias antes e 30 depois | Evita vazamento de dados: o modelo só vê o que existia no dia da previsão |
| Cancelamento gerado a partir do comportamento, com ruído | Evita o acerto artificial de bases em que o alvo é derivado de um "perfil" |
| Métricas em dados não vistos (teste de 20% e validação cruzada) | Estimar o desempenho real, não o de memorização |
| Probabilidades da carteira em validação cruzada | Nenhum cliente é pontuado por um modelo treinado com ele |
| Importância por permutação no teste | Mede o impacto real na capacidade de prever, não apenas o uso da variável pela árvore |
| Comparação com regra simples e regressão logística | Mostrar quanto o modelo complexo realmente acrescenta |
| Simulador de retenção com parâmetros editáveis | A taxa de sucesso de uma ação só pode ser medida em teste A/B real, então é uma hipótese explícita |

## Resultados detalhados

### Desempenho dos modelos (conjunto de teste)

| Modelo | ROC-AUC | AUC-PR |
|---|---|---|
| Regra simples (3+ erros de API) | 0,644 | 0,159 |
| Regressão logística | 0,743 | 0,299 |
| Gradient boosting | 0,748 | 0,312 |
| Gradient boosting, validação cruzada (5 partes) | 0,749 | n/d |

A taxa de churn da base é de 8,4%, que é o AUC-PR de um modelo aleatório. O AUC é a probabilidade de o modelo atribuir risco maior a um cliente que cancelou do que a um que ficou, tomando um par ao acaso.

### Precisão, recall e F1 por limiar de decisão

| Limiar de probabilidade | Clientes sinalizados | Precisão | Recall | F1 |
|---|---|---|---|---|
| 15% | 11,0% | 29,6% | 38,9% | 0,34 |
| 30% (risco alto) | 3,8% | 50,5% | 22,7% | 0,31 |
| 50% | 1,7% | 63,5% | 12,9% | 0,21 |

Nos 10% de clientes de maior risco, a precisão é de 31,0% e o recall é de 36,9% (lift de 3,7x). O modelo é uma ferramenta de **priorização**, não de acerto individual: o churn tem ruído e motivos não observados, então a escolha do limiar equilibra quantos clientes abordar e quantos cancelamentos capturar.

### Fatores de churn

| Erros de API (30 dias) | Churn |
|---|---|
| 0 | 4,8% |
| 1 a 2 | 8,0% |
| 3 a 4 | 17,7% |
| 5 ou mais | 44,3% |

| Variável | Queda de AUC ao embaralhar |
|---|---|
| Erros de API (30 dias) | 0,191 |
| Dias sem uso | 0,024 |
| Plano | 0,019 |
| Tendência de uso | 0,014 |
| Demais (canal, tempo de casa, chamados) | até 0,004 |
| Idade e renda | próximo de 0 |

Os chamados ao suporte têm AUC isolado de 0,65, mas perdem peso no modelo conjunto porque acompanham os erros de API.

### Faixas de risco

| Faixa | Clientes | Churn observado | Probabilidade média |
|---|---|---|---|
| Baixo (< 12%) | 84.558 | 5,2% | 5,4% |
| Médio (12% a 30%) | 11.766 | 18,0% | 17,6% |
| Alto (30% ou mais) | 3.676 | 49,5% | 48,1% |

A proximidade entre probabilidade prevista e churn observado indica boa calibração.

## Associação não é causalidade

O projeto mede **associação estatística**: clientes com mais erros de API cancelam mais. Isso não prova que o erro causa o cancelamento, pois outros fatores podem influenciar as duas coisas. Nesta base a relação causal foi definida por construção, mas em dados reais seria preciso análise causal ou um experimento (por exemplo, estabilizar a API para um grupo e comparar com outro).

A redução de **23,1%** nos cancelamentos é uma **simulação do modelo**: ele foi reaplicado aos mesmos clientes com os erros limitados a 2 em 30 dias, e a diferença nas probabilidades foi somada. Não é o resultado de uma intervenção real.

## Limitações e próximos passos

- Os erros de API são uma das causas principais. Outras frentes a explorar com dados reais: preço e mudanças de plano, tempo de resposta e resolução do suporte, satisfação (NPS), onboarding e primeiros dias de uso, uso de funcionalidades específicas, sazonalidade, tipo de contrato e ofertas de concorrentes.
- Contar só erros 500 simplifica o problema. Em dados reais, entrariam erros 5xx, timeouts e falhas de integração, possivelmente ponderados pela gravidade.
- Validar a hipótese causal e a taxa de sucesso das ações com um teste A/B.
- Evoluir a modelagem: análise de sobrevivência, explicabilidade individual (SHAP), calibração formal e monitoramento do modelo ao longo do tempo.

## Como executar

```bash
git clone <url-do-repositorio>
cd churn-prediction-saas
python3 -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt

python src/01_gera_dados_churn.py
python src/02_limpeza_dados.py
python src/03_feature_engineering.py
python src/04_modelagem_churn.py

streamlit run app/app_dashboard.py
```

O pipeline leva cerca de 2 minutos e gera aproximadamente 220 MB em `data/`. Os resultados são reproduzíveis (semente 42).
