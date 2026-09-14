# MNIST Preditor

Pipeline de machine learning para classificação de dígitos manuscritos (MNIST), com benchmark comparativo de três modelos e testes de robustez em dados fora da distribuição de treino.

> **Mini Projeto do Módulo 2 do curso de Desenvolvimento de IA para Análise Preditiva do programa SCTEC**.
> Desenvolvido para consolidar o ciclo completo de um projeto de classificação supervisionada:
> análise exploratória, pré-processamento, treinamento e ajuste de múltiplos modelos,
> avaliação comparativa por métricas de classificação, e análise crítica do comportamento
> do classificador diante de dados que fogem da distribuição de treinamento.

## 1. Problema

Reconhecer automaticamente dígitos de 0 a 9 escritos à mão, a partir de imagens 28 x 28 em escala de cinza. Cada imagem pertence a exatamente uma das dez classes, configurando um problema de **classificação multiclasse**.

Além do benchmark, o projeto investiga duas questões de robustez: o que o classificador faz ao receber uma classe ausente do treinamento, e se o pipeline se sustenta em imagens fotografadas fora do protocolo controlado da base.

- **Dataset:** MNIST (70.000 imagens), obtido via `fetch_openml`
- **Variável-alvo:** `label` (categórica, 10 classes)
- **Por que importa:** o reconhecimento de caracteres manuscritos está na base de sistemas de triagem postal, digitalização de formulários e leitura de cheques. Mais relevante que a acurácia, porém, é saber **quando o modelo não sabe** — um classificador que erra com alta confiança é mais perigoso que um que erra hesitando.

## 2. Como executar

```bash
git clone https://github.com/jhonatanalves/miniprojeto2_sctec.git
cd miniprojeto2_sctec
pip install -r requirements.txt
jupyter notebook data/notebooks/mnist_pipeline.ipynb
```

No Jupyter, execute todas as células em ordem (Kernel → Restart & Run All). O MNIST é baixado automaticamente na primeira execução e fica em cache local. As imagens manuscritas do Desafio C já acompanham o repositório. A execução completa leva cerca de 20 minutos, a maior parte na busca de hiperparâmetros da rede neural.

## 3. Estrutura do repositório

```
.
├── data/
│   ├── manuscritos/          # dígitos escritos à mão (Desafio C)
│   └── notebooks/
│       └── mnist_pipeline.ipynb
├── requirements.txt
├── .gitignore
├── PLANO.md                  # organização do trabalho e fluxo de branches
└── README.md
```

O desenvolvimento seguiu fluxo com branch `develop` e uma feature branch por etapa, integradas por Pull Request e preservadas no grafo do repositório.

## 4. Pipeline e decisões técnicas

O desenvolvimento segue as fases exigidas no enunciado. Cada decisão está justificada em detalhe nas células de texto do notebook; o resumo abaixo cobre o essencial.

### Fase 1 — Análise Exploratória
Dimensionalidade, faixa de intensidade dos pixels, distribuição das classes e grade de amostras. **Achados principais:** a base é aproximadamente balanceada (razão de 1,25:1 entre a classe mais e a menos frequente, desvio-padrão de 0,57 p.p.), o que torna a acurácia global uma métrica confiável; o achatamento em 784 features descarta a vizinhança espacial dos pixels, limitação que se manifesta nos erros da Fase 4.

### Fase 2 — Pré-processamento
Divisão estratificada em 70/10/20 (treino, validação e teste) e normalização dos pixels para [0, 1] por divisão constante. **Optou-se por não usar `StandardScaler`**: as 784 features já compartilham unidade e faixa, e padronizar por pixel distorceria as bordas de variância quase nula, amplificando ruído sem ganho.

### Fase 3 — Treinamento dos modelos
KNN, Random Forest e MLP (`MLPClassifier`), cada um com dois hiperparâmetros ajustados por validação. A busca do KNN usou subamostra de 10 mil exemplos, já que seu custo está na predição. **Resultado contraintuitivo:** na Random Forest, o melhor desempenho veio *sem* limite de profundidade: restringir a 10 níveis causou subajuste, pois dez decisões binárias não bastam para isolar regiões relevantes em 784 features.

### Fase 4 — Avaliação Comparativa
Matrizes de confusão, relatório por classe e tabela comparativa sobre o conjunto de teste, tocado uma única vez. **O par 4 → 9 foi o erro mais frequente nos três modelos**, sem exceção: formas que compartilham laço superior e haste descendente, distinguíveis apenas por uma região pequena que se dilui no vetor achatado.

### Fase 5 — Desafios A e B: generalização extrema
Os dígitos 4 e 7 foram removidos do treino e o modelo restrito foi avaliado exclusivamente sobre eles. Análise da distribuição de rótulos atribuídos e da calibração da confiança (ver Seção 7).

### Fase 6 — Desafio C: imagens próprias
Dez dígitos escritos à mão e fotografados. O pipeline reproduz a especificação do MNIST: binarização por Otsu, inversão, recorte pela bounding box, redimensionamento para 20 px com proporção preservada e centralização pelo **centro de massa**, etapa crítica, já que os modelos não têm invariância a translação.

## 5. Tecnologias

Python, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, OpenCV, SciPy.

## 6. Resultados

Modelo final: **MLP (512→256, lr=0,001)**, escolhido por vencer o KNN e a Random Forest em todas as métricas.

| Modelo | Acurácia | Precisão | Revocação | F1-Score |
|---|---|---|---|---|
| **MLP** | **0,9797** | **0,9798** | **0,9797** | **0,9797** |
| KNN (k=3, distance) | 0,9721 | 0,9724 | 0,9721 | 0,9721 |
| Random Forest (300 árvores) | 0,9664 | 0,9663 | 0,9664 | 0,9663 |

Métricas ponderadas sobre 14.000 imagens de teste.

**Sobre o custo computacional:** a diferença de acurácia entre os três é menor que 1,5 ponto percentual, mas o perfil de custo é radicalmente distinto. O KNN não treina (apenas armazena a base) e gasta segundos por predição; a MLP leva minutos treinando e prediz em milissegundos. Para um sistema que treina uma vez e prediz continuamente, a MLP domina duplamente — melhor acurácia e inferência mais barata.

**Sobre a escolha da arquitetura:** a rede 512→256 superou a 256→128 em apenas 0,10 p.p., consumindo cerca de sete vezes o tempo de treino. O critério adotado foi a acurácia de validação; em produção, a escolha defensável seria a arquitetura menor.

## 7. Análise de robustez

Esta seção concentra a contribuição analítica do projeto.

**Desafio A/B — classes nunca vistas.** Removidos 4 e 7 do treino, o modelo restrito manteve 97,22% de acurácia nas oito classes restantes, mas não tem como acertar as ocultas: seu vocabulário não as contém e as probabilidades somam 1 por construção. As substituições não foram aleatórias, **92,4% dos quatros viraram nove**, contra os ~12,5% esperados de uma atribuição ao acaso. O modelo aprendeu features genuínas de forma e migra para o vizinho morfológico mais próximo, sem jamais sinalizar novidade.

A calibração, porém, surpreendeu. A confiança média nas amostras fora da distribuição foi de **0,560**, com apenas 3,0% acima de 0,90, contra média de **0,838** e 52,4% acima de 0,90 nas classes conhecidas. A votação entre 300 árvores gera discordância mensurável diante do desconhecido, o que dá à Random Forest uma estimativa rudimentar de incerteza. Um limiar de rejeição em torno de 0,70 capturaria a maior parte dos casos.

**Desafio C — imagens próprias.** A MLP acertou **8 de 10**, contra 97,97% no teste do MNIST. A queda é o *domain shift* esperado: iluminação, espessura de traço e ruído de sensor que a base nunca apresentou. Com dez amostras, o valor está no diagnóstico dos erros, não na métrica.

Os dois erros foram de naturezas opostas. O 3 virou 9 com 69,5% de confiança (hesitação apropriada, dada a curva superior fechada do traço). Já o **0 virou 9 com 99,9% de confiança**, sem qualquer ambiguidade morfológica plausível, apontando falha de pré-processamento e não do classificador.

**A conclusão que atravessa os dois desafios:** a Random Forest atribuiu confiança de 0,560 a dígitos que nunca vira; a MLP errou com 99,9% de certeza. A softmax aplica exponenciais sobre as ativações finais, amplificando diferenças pequenas em probabilidades próximas de 1, e não há ensemble para discordar consigo mesmo. Um limiar de rejeição calibrado sobre a floresta capturaria boa parte dos casos problemáticos; o mesmo limiar na MLP deixaria passar o erro mais grave. **A escolha do modelo não afeta só a acurácia, afeta a confiabilidade do sinal de incerteza que ele emite.** Um sistema que precise saber quando não sabe pode preferir o modelo menos preciso.

## 8. Melhorias para versões futuras

- **v2:** rede convolucional (CNN), preservando a estrutura espacial das imagens. É a limitação mais evidente do projeto — a confusão 4/9 e a dependência da centralização manual decorrem do achatamento em 784 features.
- **v2:** *data augmentation* com rotações, deslocamentos e variações de espessura, aproximando o treino da variabilidade da escrita real.
- **v3:** limiar de rejeição e calibração de confiança (*temperature scaling*, estimativa por ensemble). O projeto diagnosticou a falsa certeza, mas não a mitigou.
- **Validação:** ampliar a amostra de imagens próprias para dezenas de exemplares por classe, de autores diferentes, permitindo medir o *domain shift* em vez de apenas observá-lo.

## 9. Autor

Jhonatan Alves