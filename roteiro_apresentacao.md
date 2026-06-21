# Roteiro de Apresentação — Trabalho Prático 2 de IA

**Aluno:** João Victor Borges Carvalho
**Tempo estimado de apresentação:** 5-7 minutos
**Dataset:** Breast Cancer Wisconsin (Original) — 699 amostras, 9 features, 1 target

---

## ESTRUTURA DA APRESENTAÇÃO

```
Slide 1 (30s)  → O que é, qual dataset, restrições
Slide 2 (2min) → Pipeline de pré-processamento + algoritmos
Slide 3 (2min) → Resultados e comparação
Perguntas      → O que responder se perguntarem
```

---

## SLIDE 1 — TÍTULO E CONTEXTO (30 segundos)

### O que falar:

"Meu trabalho é sobre pré-processamento e análise preditiva aplicados ao dataset
Breast Cancer Wisconsin. São 699 amostras de tecido mamário, cada uma com 9
características celulares avaliadas por patologistas numa escala de 1 a 10, e o
objetivo é classificar se o tumor é benigno ou maligno."

"A restrição principal imposta pelo professor é que **nada de scikit-learn,
pandas ou numpy** — tudo foi implementado do zero com Python puro + módulo math.
A única exceção é o matplotlib pra plotagem."

### Conceitos que você precisa dominar:

| Termo | Significado |
|-------|-------------|
| **Amostra** | Uma linha do dataset = um paciente |
| **Feature / Atributo** | Uma coluna = uma característica medida (ex: Uniformity of cell size) |
| **Target / Classe** | O que queremos prever: benigno (0) ou maligno (1) |
| **Pré-processamento** | Preparar os dados antes de aplicar algoritmos |
| **Classificação binária** | Problema com apenas 2 classes possíveis |

### As 9 features (decore os nomes e o que significam):

| Feature | O que significa (em português simples) |
|---------|---------------------------------------|
| Clump_thickness | Espessura dos aglomerados de células |
| Uniformity_of_cell_size | Uniformidade do tamanho das células |
| Uniformity_of_cell_shape | Uniformidade do formato das células |
| Marginal_adhesion | Aderência nas margens do tecido |
| Single_epithelial_cell_size | Tamanho das células epiteliais isoladas |
| Bare_nuclei | Núcleos nus (sem citoplasma) |
| Bland_chromatin | Textura da cromatina |
| Normal_nucleoli | Tamanho dos nucléolos |
| Mitoses | Taxa de divisão celular (mitose) |

**Dica:** Pense assim → quanto MAIOR a nota (1-10), PIOR é a característica.
Uma célula cancerosa tende a ter notas altas em várias dessas features.

---

## SLIDE 2 — METODOLOGIA (2 minutos)

### PARTE A: Pipeline de Pré-processamento

Explique cada etapa em uma frase. O slide já tem o resumo visual, você complementa:

**1. Identificação do target (Class → 0/1)**
"A coluna `Class` originalmente tem valores 2 (benigno) e 4 (maligno). Converti
pra 0 e 1 porque é o padrão em classificação binária — facilita distância
euclidiana no K-NN e função sigmoide na rede neural."

**2. Análise de tipos, escala e estatísticas**
"Classifiquei todas as features como quantitativas discretas em escala racional.
Calculei média, mediana, moda, variância, desvio padrão e IQR pra cada uma. O
que chamou atenção: a moda da maioria é 1 — ou seja, a maior parte das amostras
tem baixa severidade. E o Bare_nuclei é o atributo mais disperso."

**3. Split estratificado 80/20 (seed=42)**
"Separei 80% pra treino e 20% pra teste, mantendo a proporção de malignos
(~35%) igual nos dois. Isso se chama amostragem estratificada. Seed fixa
garante que toda vez que rodar, a divisão é a mesma."

**4. Limpeza: 184 duplicatas removidas**
"Removi exemplos duplicados porque eles inflam artificialmente a importância de
certos padrões. Se o mesmo paciente aparece 3 vezes, o algoritmo dá 3x mais peso
pra ele. Nenhuma coluna foi removida."

**5. Outliers mantidos (contexto médico)**
"Detectei 68 outliers pelo método IQR, mas decidi MANTER todos. Uma nota 10 não
é erro de medição — é uma avaliação clínica válida de alta severidade. No
contexto médico, outliers são SINAIS, não ruído."

**6. Normalização Min-Max [0, 1]**
"Coloquei todos os valores na escala de 0 a 1 usando Min-Max. Isso é essencial
pro K-NN (senão features com magnitude maior dominam a distância) e pra MLP
(senão a sigmoide satura). Os parâmetros foram calculados SÓ no treino e depois
aplicados no teste — sem data leakage."

**7. Correlação de Pearson — 9/9 features mantidas**
"Calculei a correlação de cada feature com o target. Todas tiveram |r| ≥ 0.36.
Bare_nuclei foi a mais correlacionada (0.77) e Mitoses a menos (0.36). Como
nenhuma ficou abaixo do limiar de 0.10, mantive todas as 9."

### Como interpretar a correlação de Pearson:

| Valor de r | Interpretação |
|------------|---------------|
| 0.00 a 0.10 | Muito fraca — candidata a remoção |
| 0.10 a 0.30 | Fraca |
| 0.30 a 0.50 | Moderada |
| 0.50 a 0.70 | Forte |
| 0.70 a 1.00 | Muito forte |

**r positivo** = quando a feature aumenta, a chance de ser maligno também aumenta.
**r negativo** = quando a feature aumenta, a chance de ser maligno diminui.

No nosso caso, TODAS as correlações são positivas — faz sentido: quanto maior a
nota em qualquer feature, maior a chance de malignidade.

---

### PARTE B: Algoritmos Implementados

**K-NN (K-Nearest Neighbors):**
"Algoritmo baseado em similaridade. Pra classificar um novo paciente, ele calcula
a distância euclidiana até todos os 362 exemplos de treino, pega os K vizinhos
mais próximos, e faz uma votação: a classe que aparecer mais vezes ganha. Testei
K de 1 a 15. K=3 foi o melhor."

**Árvore C4.5:**
"Constrói uma árvore de decisão. Em cada nó, escolhe a feature e o ponto de
corte que melhor separam benigno de maligno, usando Gain Ratio. A árvore é
podada por profundidade pra evitar overfitting. Profundidade 3 foi a ideal:
apenas 15 nós."

**MLP (Rede Neural):**
"Rede com 9 entradas, 8 neurônios ocultos com função sigmoide, e 1 saída.
Treinei por 500 épocas com SGD e backpropagation manual. A rede aprendeu —
loss caiu de ~0.10 pra ~0.03."

---

## OS GRÁFICOS — O QUE ELES MOSTRAM

### 1. `distribuicao_classes.png` — Gráfico de barras
**O que é:** Duas barras mostrando quantos casos benignos (azul) e malignos
(vermelho) existem no dataset.

**Como ler:** A barra azul é maior (458 benignos, 65.5%), a vermelha menor
(241 malignos, 34.5%). A razão é ~1.9:1.

**O que falar:** "O dataset é relativamente balanceado. A razão está abaixo de
2:1, então não precisei aplicar técnicas de balanceamento artificial como SMOTE."

### 2. `histogramas_por_classe.png` — Grid 3×3 de histogramas
**O que é:** 9 gráficos, um pra cada feature. Cada gráfico tem barras azuis
(benignos) e vermelhas (malignos) sobrepostas.

**Como ler:** Eixo X = valor da feature (1 a 10). Eixo Y = frequência (quantos
pacientes têm aquele valor). Se as barras azuis estão concentradas nos valores
baixos e as vermelhas nos valores altos, a feature DISCRIMINA bem as classes.

**O que falar:** "Dá pra ver claramente que os casos benignos (azul) tendem a
ter notas baixas (1-3), enquanto os malignos (vermelho) se espalham mais pros
valores altos (6-10). Features como Bare_nuclei e Uniformity of cell shape têm
uma separação bem nítida entre as duas distribuições."

**Exemplo concreto:** No histograma do Bare_nuclei, os benignos estão quase
todos em 1, enquanto os malignos se distribuem de 1 a 10. Isso indica que
Bare_nuclei é um excelente discriminador — e a correlação de 0.77 confirma.

### 3. `boxplots_por_classe.png` — Grid 3×3 de boxplots
**O que é:** 9 boxplots, um pra cada feature. Cada boxplot tem duas caixas:
azul (benigno) e vermelha (maligno).

**Como ler um boxplot:**
- A **caixa** vai do Q1 (25º percentil) ao Q3 (75º percentil)
- A **linha no meio** da caixa é a mediana
- Os **bigodes** vão até o valor mais extremo não-outlier
- **Pontos isolados** além dos bigodes são outliers

**O que falar:** "Os boxplots confirmam o que os histogramas sugerem: pra
features muito discriminativas como Bare_nuclei, a caixa azul fica bem embaixo
(mediana=1) e a vermelha bem mais alta (mediana ~6-7). Já pra Mitoses, as
caixas são quase idênticas — o que faz sentido, é a feature menos correlacionada."

**Detalhe importante:** Nos boxplots, os benignos quase não têm outliers. Já os
malignos têm vários pontos acima do bigode superior. Esses são exatamente os
outliers que eu decidi MANTER — pacientes com notas extremas (10) em várias
features, que são justamente os casos mais graves.

### 4. `mlp_convergencia.png` — Curva de convergência da rede neural
**O que é:** Um gráfico de linha mostrando o loss (erro) da MLP ao longo das
500 épocas de treinamento.

**Como ler:** Eixo X = número da época (0 a 500). Eixo Y = loss MSE (Mean
Squared Error). A curva deve cair e estabilizar.

**O que falar:** "A curva mostra que a rede convergiu bem. O loss caiu
rapidamente nas primeiras 50 épocas e depois foi diminuindo mais devagar até
estabilizar por volta de 0.03. Se a curva estivesse subindo ou oscilando
muito, seria sinal de que a taxa de aprendizado está alta demais."

---

## SLIDE 3 — RESULTADOS (2 minutos)

### A tabela principal:

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Baseline | 0.3504 | 0.3504 | 1.0000 | 0.5189 |
| **K-NN (K=3)** 🏆 | **0.9781** | **1.0000** | **0.9375** | **0.9677** |
| C4.5 (d=3) | 0.9708 | 0.9783 | 0.9375 | 0.9574 |
| MLP (8 ocultos) | 0.9635 | 0.9778 | 0.9167 | 0.9462 |

### O que cada métrica significa (decore isso):

| Métrica | Fórmula | Significado |
|---------|---------|-------------|
| **Acurácia** | (TP+TN)/Total | % de acertos geral |
| **Precisão** | TP/(TP+FP) | Dos que eu disse "maligno", quantos realmente são? |
| **Recall** | TP/(TP+FN) | Dos malignos reais, quantos eu encontrei? |
| **F1-Score** | 2×P×R/(P+R) | Média harmônica — equilíbrio entre precisão e recall |

**TP = True Positive:** paciente é maligno e eu acertei (disse maligno)
**TN = True Negative:** paciente é benigno e eu acertei (disse benigno)
**FP = False Positive:** paciente é benigno mas eu disse maligno (alarme falso)
**FN = False Negative:** paciente é maligno mas eu disse benigno (PERIGOSO!)

### O que falar sobre cada modelo:

**Baseline:**
"O baseline simplesmente chuta a classe majoritária pra todo mundo. Serve como
piso — qualquer modelo decente precisa superar F1 = 0.52. Ele tem recall 100%
porque chuta 'maligno' sempre, mas precisão péssima."

**K-NN:**
"Foi o melhor modelo. Precisão perfeita — zero falsos positivos. Isso significa
que ele não assustou nenhuma paciente saudável com diagnóstico errado. Recall
de 93.75%: dos 48 casos malignos, acertou 45. Só 3 escaparam."

**C4.5:**
"Desempenho quase idêntico ao K-NN, mas com uma vantagem crucial: é totalmente
interpretável. A árvore tem só 15 nós. A raiz é 'Uniformity of cell size ≤
0.2778'. Dá pra explicar pro médico exatamente por que cada decisão foi tomada."

**MLP:**
"Ficou um pouco atrás em recall (91.67%). A rede convergiu bem, mas é uma
caixa preta — você não consegue explicar facilmente por que ela classificou
alguém como maligno."

### Os 3 cards de insight (slide):

**"Precisão perfeita"** → "O K-NN não gerou nenhum falso positivo. No contexto
médico, isso significa que nenhuma paciente saudável recebeu diagnóstico
equivocado de câncer. Falso positivo gera estresse, biópsias desnecessárias,
custo emocional e financeiro."

**"93.75% de recall"** → "Apenas 3 falsos negativos em 48 casos malignos. O
recall alto é a métrica mais crítica aqui: um falso negativo significa deixar
um câncer passar despercebido, e as consequências podem ser fatais."

**"C4.5 interpretável"** → "15 nós, raiz clara. Em aplicação médica real, a
interpretabilidade muitas vezes é mais importante que 1% a mais de acurácia.
Você precisa conseguir justificar cada diagnóstico."

---

## DECISÕES QUE TOMEI (possíveis perguntas do professor)

### 1. "Por que moda e não média pra imputação?"
"Porque os dados são discretos — notas inteiras de 1 a 10. A média geraria
valores como 3.7, que não existem na prática clínica. Um patologista nunca
dá nota 3.7."

### 2. "Por que manteve os outliers?"
"No contexto médico, uma nota 10 não é erro — é sinal de severidade. Se eu
removesse os outliers, estaria jogando fora justamente os casos mais
informativos pra detecção de câncer."

### 3. "Por que Min-Max e não Z-score?"
"A escala original já é limitada (1-10), então o Min-Max preserva a estrutura
da distribuição. Além disso, é essencial pro K-NN (features com maior magnitude
dominariam a distância) e pra MLP (valores grandes saturam a sigmoide)."

### 4. "Por que Pearson e não PCA?"
"Pearson dá pra implementar do zero com Python puro. PCA exigiria decomposição
em autovalores/autovetores — bem mais complexo. E a interpretação do Pearson
é direta: cada coeficiente te diz exatamente quanto aquela feature se relaciona
com o target."

### 5. "Por que hold-out e não cross-validation?"
"Como implementei tudo na mão, 5-fold CV com 3 algoritmos seria bem mais
custoso. O ganho marginal não justificaria. Pra 362 exemplos de treino, o
hold-out é suficiente."

### 6. "Por que K=3 e não K=5 ou K=7?"
"K=1 sofre overfitting (recall cai pra 85%). K≥5 começa a suavizar demais a
fronteira de decisão. K=3 é o ponto ideal. E K ímpar evita empate na votação
binária."

### 7. "Como funciona o Gain Ratio da C4.5?"
"O Information Gain puro tem um viés: favorece features com muitos valores
distintos. O Gain Ratio corrige isso dividindo o IG pelo Split Info — uma
medida de quantos valores distintos a feature tem. É como 'penalizar' features
que naturalmente teriam IG alto só por terem muitos valores."

### 8. "Como implementou o backpropagation?"
"Regra da cadeia do cálculo, aplicada camada por camada. A derivada do erro
em relação a cada peso é calculada propagando o gradiente da saída até a
entrada. Usei SGD: atualizo os pesos a cada exemplo, não por batch."

---

## RESUMO FINAL — O QUE FALAR NA CONCLUSÃO

"O K-NN com K=3 foi o melhor modelo no geral, mas a C4.5 entrega resultado
quase idêntico com a vantagem de ser interpretável. Se eu tivesse que escolher
um modelo pra colocar em produção num contexto médico real, iria de C4.5 pela
transparência — você consegue justificar cada classificação olhando a árvore.

Implementar tudo na mão deu trabalho, especialmente o backpropagation da MLP,
mas valeu a pena. Entender exatamente o que cada algoritmo está fazendo, sem
caixa preta, muda completamente a relação com os resultados."

---

## DICAS PRÁTICAS PRA APRESENTAR

1. **Treine em voz alta** pelo menos 2 vezes antes. Cronometre.
2. **Não leia os slides** — eles são apoio visual. Você explica com suas palavras.
3. **Aponte pros gráficos** quando falar deles. "Aqui no boxplot do Bare_nuclei
   dá pra ver claramente que..."
4. **Fale da restrição com orgulho**, não como desculpa. "Tudo do zero" é
   diferencial, não limitação.
5. **Se perguntarem algo que você não sabe**, responda: "Boa pergunta, eu não
   testei esse cenário específico, mas a abordagem que eu usaria seria..."
6. **Leve o notebook aberto** no Jupyter — se perguntarem pra ver o código, você
   mostra na hora.
7. **Os slides têm 3 minutos de conteúdo.** Se o professor deixar mais tempo,
   use pra detalhar os gráficos e as decisões.
