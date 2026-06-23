# Trabalho Pratico 2 -- Pre-processamento e Analise Preditiva

**Disciplina:** Inteligencia Artificial
**Aluno:** Joao Victor Borges Carvalho
**Dataset:** Breast Cancer Wisconsin (Original) -- UCI Machine Learning Repository (id=15)

---

## Sobre o projeto

Este trabalho implementa um pipeline completo de pre-processamento e analise preditiva sobre a base Breast Cancer Wisconsin, com 699 amostras de tecido mamario classificadas como benignas ou malignas a partir de 9 caracteristicas celulares avaliadas por patologistas.

Toda a parte matematica e todos os algoritmos de aprendizado de maquina foram implementados do zero, usando apenas Python puro e o modulo `math`. Nenhuma biblioteca com formulas prontas foi utilizada (scikit-learn, pandas para calculos, numpy, scipy). A unica excecao e o `matplotlib`, usado exclusivamente para plotagem dos graficos.

---

## Estrutura do projeto

```
trabalho2IA/
├── data/
│   └── breast_cancer_wisconsin.csv    # Dataset original (699 linhas x 10 colunas)
├── plots/
│   ├── boxplots_por_classe.png         # Boxplots comparativos benigno vs maligno
│   ├── distribuicao_classes.png        # Distribuicao das classes (barras)
│   ├── histogramas_por_classe.png      # Histogramas sobrepostos por classe
│   └── mlp_convergencia.png            # Curva de convergencia da MLP
├── utils.py                            # Biblioteca de funcoes matematicas e algoritmos (~1.160 linhas)
├── main.py                             # Pipeline completo (script unico, ~1.200 linhas)
├── 01_analise_exploratoria.ipynb       # Notebook 1: Etapas 1 a 5 (exploracao)
├── 02_preprocessamento.ipynb           # Notebook 2: Etapas 6 a 10 (limpeza e normalizacao)
├── 03_modelagem_preditiva.ipynb        # Notebook 3: Etapas 2.1 a 2.7 (algoritmos)
├── documentacao_final.md               # Documentacao detalhada com justificativas
├── apresentacao.pptx                   # Slides de 3 minutos para apresentacao
├── trabalho.md                         # Enunciado original do professor
└── README.md                           # Este arquivo
```

---

## Dataset

O **Breast Cancer Wisconsin (Original)** possui 699 amostras de tecido mamario. Cada amostra tem 9 features, todas notas de 1 a 10 atribuidas por patologistas:

| Feature | Descricao |
|---------|-----------|
| Clump_thickness | Espessura do aglomerado |
| Uniformity_of_cell_size | Uniformidade do tamanho celular |
| Uniformity_of_cell_shape | Uniformidade do formato celular |
| Marginal_adhesion | Adesao marginal |
| Single_epithelial_cell_size | Tamanho de celula epitelial unica |
| Bare_nuclei | Nucleos nus (16 valores ausentes) |
| Bland_chromatin | Cromatina suave |
| Normal_nucleoli | Nucleolos normais |
| Mitoses | Taxa de divisao celular |

**Target:** `Class` -- 2 (benigno) ou 4 (maligno), convertido para 0/1.

---

## Pipeline -- Parte 1: Pre-processamento

| Etapa | Resultado |
|-------|-----------|
| 1. Atributo alvo | `Class` convertido: 2 -> 0 (benigno), 4 -> 1 (maligno) |
| 2. Tipos de dados | Todos Quantitativos Discretos |
| 3. Escala | Intervalar Discreta (intervalos aproximadamente constantes, sem zero absoluto verdadeiro) |
| 4. Estatisticas | Media, mediana, moda, variancia, IQR + histogramas e boxplots |
| 5. Split treino/teste | Hold-out 80/20 estratificado (seed=42) |
| 6. Eliminacao | 0 colunas removidas, 184 duplicatas removidas |
| 7. Desbalanceamento | 1.09:1 -- dataset balanceado, sem necessidade de SMOTE |
| 8. Limpeza | Outliers mantidos (contexto medico); 0 inconsistentes; 0 ausentes no treino |
| 9. Normalizacao | Min-Max [0, 1] (parametros do treino aplicados ao teste) |
| 10. Dimensionalidade | Correlacao de Pearson -- 9/9 features mantidas (|r| >= 0.36) |

Resultado final: 362 exemplos de treino x 9 features normalizadas, 137 exemplos de teste.

---

## Pipeline -- Parte 2: Analise Preditiva

### Algoritmos implementados do zero

- **Baseline:** Classe majoritaria (sempre prediz a classe mais frequente)
- **K-NN:** Distancia euclidiana + votacao majoritaria (K=3, impar para evitar empate)
- **Arvore C4.5:** Gain Ratio (Information Gain / Split Info) + poda por profundidade
- **MLP:** Rede neural 9->8->1 com sigmoide, backpropagation manual (regra da cadeia), SGD, 500 epocas

### Metricas implementadas do zero

Matriz de confusao (TP, TN, FP, FN), acuracia, precisao, recall (sensibilidade), especificidade, F1-score.

Classe positiva = 1 (Maligno). Recall e a metrica mais critica no contexto medico (falso negativo = cancer nao detectado).

### Resultados no conjunto de teste (137 exemplos)

| Modelo | Acuracia | Precisao | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Baseline (classe majoritaria) | 0.3504 | 0.3504 | 1.0000 | 0.5189 |
| **K-NN (K=3)** | **0.9781** | **1.0000** | **0.9375** | **0.9677** |
| C4.5 (d=3) | 0.9708 | 0.9783 | 0.9375 | 0.9574 |
| MLP (8 ocultos) | 0.9635 | 0.9778 | 0.9167 | 0.9462 |

**K-NN com K=3 foi o melhor modelo** (F1=0.9677), com precisao perfeita -- zero falsos positivos.

A C4.5 com profundidade 3 entrega resultado quase identico (F1=0.9574) com apenas 15 nos, sendo totalmente interpretavel. A raiz da arvore e `Uniformity_of_cell_size <= 0.2778`.

A MLP teve uma unica configuracao de hiperparametros testada (limitacao de tempo), entao o F1 de 0.9462 e um limite inferior do que a rede pode alcancar com tuning adequado.

---

## Decisoes e justificativas

1. **Escala intervalar, nao racional:** Nao ha zero absoluto verdadeiro -- nota 1 e o piso da escala, nao ausencia total. Mas a literatura medica trata como intervalar para permitir estatistica parametrica.

2. **Moda para imputacao, nao media:** Dados sao discretos (1-10). A media geraria valores decimais inexistentes na pratica clinica (ex: 3.7).

3. **Outliers mantidos:** No contexto medico, nota 10 nao e erro de medicao -- e sinal de alta severidade. Remove-los eliminaria os casos mais informativos.

4. **Min-Max [0,1], nao Z-score:** Escala original ja e limitada (1-10), o Min-Max preserva a estrutura. Essencial para K-NN (features com maior magnitude dominariam a distancia) e MLP (evita saturacao da sigmoide).

5. **Correlacao de Pearson, nao PCA:** Implementavel do zero com Python puro. PCA exigiria decomposicao em autovalores -- mais complexa e com perda de interpretabilidade.

6. **Hold-out 80/20 estratificado, nao cross-validation:** Com 3 algoritmos implementados manualmente, 5-fold CV seria muito custoso. O ganho marginal nao justifica para 362 exemplos.

7. **K=3 (impar):** Evita empate na votacao binaria. K=1 sofre overfitting; K>=5 suaviza demais a fronteira de decisao.

8. **Gain Ratio, nao so Information Gain:** O IG puro favorece features com muitos valores distintos. O Gain Ratio corrige esse vies dividindo pelo Split Info.

---

## Como executar

### Requisitos

- Python 3.10+
- matplotlib
- jupyter (opcional, para os notebooks)

```bash
pip install matplotlib jupyter
```

### Opcao 1: Jupyter Notebooks (recomendado)

```bash
jupyter notebook
```

Execute na ordem: `01_analise_exploratoria.ipynb` -> `02_preprocessamento.ipynb` -> `03_modelagem_preditiva.ipynb`

Cada notebook e autocontido -- se quiser pular direto para o 3, ele recarrega os dados do zero e reproduz todo o pipeline.

### Opcao 2: Script unico

```bash
python main.py
```

Executa o pipeline completo e imprime todos os resultados no terminal. Leva cerca de 30-60 segundos (a MLP e a parte mais lenta).

---

## Implementacao from scratch

Todas as funcoes abaixo foram implementadas manualmente em `utils.py` (Python puro + `math`):

**Estatistica:** media, mediana, moda, variancia (amostral), desvio padrao, minimo, maximo, amplitude, percentil com interpolacao linear, quartis, IQR, correlacao de Pearson.

**Pre-processamento:** contador de frequencia manual, stratified split com LCG proprio (Fisher-Yates determinista), deteccao de outliers (IQR), imputacao pela moda, remocao de colunas constantes, remocao de duplicatas, normalizacao Min-Max.

**Metricas:** matriz de confusao (TP/TN/FP/FN), acuracia, precisao, recall, especificidade, F1-score.

**Algoritmos:** baseline (classe majoritaria), K-NN (distancia euclidiana, votacao), C4.5 (entropia, Information Gain, Split Info, Gain Ratio, arvore recursiva com poda), MLP (forward pass, backpropagation com regra da cadeia, SGD, inicializacao de pesos).

---

## Licenca

Projeto academico -- uso livre para estudo e referencia.

Dataset: Breast Cancer Wisconsin (Original) -- UCI Machine Learning Repository.
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
