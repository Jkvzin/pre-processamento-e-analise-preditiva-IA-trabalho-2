# Contexto do Projeto

## Trabalho Prático 2 — Pré-processamento e Análise Preditiva

**Disciplina:** Inteligência Artificial  
**Aluno:** João Victor Borges Carvalho  
**Dataset:** Breast Cancer Wisconsin (Original) — UCI ML Repository (id=15)  

**Restrição principal:** Nada de scikit-learn, pandas (pras contas) ou qualquer lib com fórmulas prontas. Tudo implementado do zero — só Python puro + módulo `math`. A única exceção é o `matplotlib` pra plotagem.

---

## O que esse projeto entrega

O trabalho tem duas partes grandes:

1. **Parte 1 — Pré-processamento (10 etapas):** exploração, limpeza, normalização e redução de dimensionalidade
2. **Parte 2 — Análise Preditiva (7 etapas):** implementação manual de baseline + 3 algoritmos (K-NN, C4.5, MLP), avaliação e comparação

---

## Estrutura

```
trabalho2IA/
├── data/breast_cancer_wisconsin.csv    ← Dataset (699 linhas × 10 colunas)
├── plots/                              ← 4 gráficos gerados
├── utils.py                            ← 40+ funções "from scratch" (~1.170 linhas)
├── main.py                             ← Pipeline completo (~1.200 linhas)
├── documentacao_final.md               ← Documentação detalhada com justificativas
├── README.md                           ← Visão geral do projeto
├── contexto.md                         ← Este arquivo (resumo executivo)
├── trabalho.md                         ← Enunciado original do professor
├── roteiro_apresentacao.md             ← Roteiro completo pra apresentar
├── explicacao_graficos.md              ← Guia visual explicando cada gráfico
├── apresentacao.pptx                   ← Slides de 3 minutos (3 slides)
├── 01_analise_exploratoria.ipynb       ← Notebook 1: Etapas 1 a 5
├── 02_preprocessamento.ipynb           ← Notebook 2: Etapas 6 a 10
└── 03_modelagem_preditiva.ipynb        ← Notebook 3: Etapas 2.1 a 2.7
```

**Nota:** Arquivos auxiliares (`import_data.py`, `notebook.ipynb` antigo, `gen_slides.js`, `node_modules/`) foram removidos na limpeza final — o projeto contém apenas o necessário pra entrega.

---

## Dataset

O **Breast Cancer Wisconsin (Original)** tem 699 amostras de tecido mamário com 9 features (notas de 1 a 10 dadas por patologistas) e 1 target (`Class`).

Features: Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Single_epithelial_cell_size, Bare_nuclei (16 valores ausentes), Bland_chromatin, Normal_nucleoli, Mitoses.

Target: `Class` — 2 (benigno) ou 4 (maligno), convertido pra 0/1.

---

## Pipeline — Parte 1

| Etapa | O que fiz | Resultado |
|-------|-----------|-----------|
| 1 | Atributo alvo | `Class` → 2→0, 4→1 |
| 2 | Tipos de dados | Todos Quantitativos Discretos |
| 3 | Escala | Racional Discreta |
| 4 | Estatísticas + gráficos | Média, mediana, variância, IQR, histogramas, boxplots |
| 5 | Split treino/teste | Hold-out 80/20 estratificado |
| 6 | Eliminação | 0 colunas removidas, 184 duplicatas removidas |
| 7 | Desbalanceamento | 1.09:1 — balanceado |
| 8 | Limpeza | Outliers mantidos, 0 inconsistentes, 0 ausentes no treino |
| 9 | Normalização | Min-Max [0, 1] |
| 10 | Dimensionalidade | Pearson — 9/9 features mantidas |

Resultado: 362 exemplos de treino × 9 features, 137 de teste.

---

## Pipeline — Parte 2

### Algoritmos (todos do zero)

- **Baseline:** classe majoritária (sempre prediz Maligno)
- **K-NN:** distância euclidiana + votação majoritária
- **C4.5:** entropia, Information Gain, Split Info, Gain Ratio, poda por profundidade
- **MLP:** forward pass, backpropagation (regra da cadeia), SGD

### Resultados

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Baseline | 0.3504 | 0.3504 | 1.0000 | 0.5189 |
| **K-NN (K=3)** 🏆 | **0.9781** | **1.0000** | **0.9375** | **0.9677** |
| C4.5 (d=3) | 0.9708 | 0.9783 | 0.9375 | 0.9574 |
| MLP (8 ocultos) | 0.9635 | 0.9778 | 0.9167 | 0.9462 |

---

## Decisões que tomei

1. **Moda pra imputação** (não média): dados são discretos (1-10), média geraria valores clinicamente inexistentes
2. **Outliers mantidos:** nota 10 não é erro — é sinal de severidade no contexto médico
3. **Min-Max [0,1]** (não Z-score): escala já limitada, preserva estrutura, essencial pra K-NN e MLP
4. **Pearson** (não PCA): implementável do zero, interpretação direta
5. **Estratificação no split:** mantém proporção de malignos (~35%) nos dois conjuntos
6. **K ímpar no K-NN:** evita empate na votação binária
7. **Gain Ratio** (não só Information Gain): evita viés por features com muitos valores distintos

---

## Como rodar

```bash
# Notebooks (recomendado)
jupyter notebook
# Execute 01 → 02 → 03

# Ou script único
python main.py
```

Dependências: só `matplotlib` e `jupyter`.
