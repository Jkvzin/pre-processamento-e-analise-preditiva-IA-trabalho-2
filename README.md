# Pré-processamento e Análise Preditiva — Trabalho Prático 2

**Disciplina:** Inteligência Artificial  
**Aluno:** João Victor Borges Carvalho  
**Dataset:** Breast Cancer Wisconsin (Original) — UCI Machine Learning Repository  

---

## Do que se trata

Esse é o segundo trabalho prático da disciplina de IA. A proposta é pegar uma base de dados real, fazer todo o pré-processamento na mão e depois implementar três algoritmos de classificação do zero — sem usar scikit-learn, pandas (pras contas) ou qualquer biblioteca que já entregue as fórmulas prontas.

O dataset que peguei foi o **Breast Cancer Wisconsin (Original)**, com 699 amostras de tecido mamário classificadas como benignas ou malignas a partir de 9 características celulares avaliadas por patologistas.

---

## A restrição mais importante

O professor foi bem claro: **nada de usar bibliotecas com fórmulas prontas**. Isso significa que:

- Média, mediana, variância, desvio padrão, correlação de Pearson → tudo calculado com Python puro + `math`
- K-NN, Árvore C4.5, Rede Neural MLP → implementados do zero, incluindo backpropagation manual
- Métricas (acurácia, precisão, recall, F1, matriz de confusão) → também na mão

A única exceção é o **matplotlib**, que só uso pra plotar os gráficos — ele não faz nenhum cálculo estatístico por baixo dos panos.

---

## Estrutura do projeto

```
├── data/
│   └── breast_cancer_wisconsin.csv   ← Dataset original
├── plots/
│   ├── histogramas_por_classe.png
│   ├── boxplots_por_classe.png
│   ├── distribuicao_classes.png
│   └── mlp_convergencia.png
├── utils.py                          ← Biblioteca com 40+ funções "from scratch"
├── main.py                           ← Pipeline completo (roda tudo de uma vez)
├── documentacao_final.md             ← Documentação detalhada com justificativas
├── contexto.md                       ← Resumo executivo
├── trabalho.md                       ← Enunciado original do trabalho
├── 01_analise_exploratoria.ipynb     ← Notebook 1: Etapas 1 a 5
├── 02_preprocessamento.ipynb         ← Notebook 2: Etapas 6 a 10
├── 03_modelagem_preditiva.ipynb      ← Notebook 3: Etapas 2.1 a 2.7
└── apresentacao.pptx                 ← Slides de 3 minutos pra apresentar
```

---

## Pipeline resumido

### Parte 1 — Pré-processamento

1. **Atributo alvo:** `Class` → converti 2 (benigno) pra 0 e 4 (maligno) pra 1
2. **Tipos de dados:** todos quantitativos discretos (notas de 1 a 10)
3. **Escala:** racional discreta (zero absoluto, intervalos constantes)
4. **Estatísticas:** média, mediana, moda, variância, IQR + histogramas e boxplots
5. **Split:** hold-out 80/20 estratificado (seed=42)
6. **Limpeza:** 184 duplicatas removidas, nenhuma coluna eliminada
7. **Balanceamento:** razão 1.09:1 — dataset já balanceado, não precisei de SMOTE
8. **Outliers:** detectei 68, mas mantive todos (no contexto médico, nota 10 não é erro — é sinal de severidade)
9. **Normalização:** Min-Max [0, 1], parâmetros do treino aplicados no teste
10. **Dimensionalidade:** correlação de Pearson — todas as 9 features mantidas (|r| ≥ 0.36)

### Parte 2 — Modelagem

Implementei 4 modelos do zero:

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Baseline (classe majoritária) | 0.3504 | 0.3504 | 1.0000 | 0.5189 |
| **K-NN (K=3)** 🏆 | **0.9781** | **1.0000** | **0.9375** | **0.9677** |
| C4.5 (d=3) | 0.9708 | 0.9783 | 0.9375 | 0.9574 |
| MLP (8 ocultos) | 0.9635 | 0.9778 | 0.9167 | 0.9462 |

---

## O que eu aprendi com esse trabalho

O K-NN com K=3 foi o melhor modelo (F1=0.9677), o que me surpreendeu um pouco — eu esperava que a rede neural fosse levar. Mas faz sentido: o dataset é pequeno (362 exemplos de treino) e o K-NN, sendo um algoritmo baseado em similaridade, funciona muito bem nesse cenário. A precisão perfeita (zero falsos positivos) é particularmente interessante no contexto médico — você não quer assustar uma paciente saudável com um diagnóstico errado.

A árvore C4.5 com profundidade 3 também me impressionou: só 15 nós e desempenho quase idêntico ao K-NN, com a vantagem de ser totalmente interpretável. Dá pra inspecionar cada regra de decisão e entender exatamente por que o modelo classificou cada caso.

Implementar backpropagation na mão foi a parte mais desafiadora. Depurar derivadas parciais com print statement não é exatamente divertido, mas depois que funciona dá uma satisfação grande.

---

## Como rodar

### Requisitos

- Python 3.10+
- matplotlib
- jupyter (opcional, pros notebooks)

Instalação rápida:

```bash
pip install matplotlib jupyter
```

### Opção 1: Jupyter Notebooks (recomendado pra entrega)

```bash
jupyter notebook
```

Execute na ordem: `01_analise_exploratoria.ipynb` → `02_preprocessamento.ipynb` → `03_modelagem_preditiva.ipynb`

Cada notebook é autocontido — se quiser pular direto pro 3, ele recarrega os dados do zero.

### Opção 2: Script Python

```bash
python main.py
```

Roda o pipeline completo de uma vez e imprime todos os resultados no terminal. Leva cerca de 30-60 segundos (a MLP é a parte mais lenta).

---

## Licença

Projeto acadêmico — uso livre pra estudo e referência.

Dataset: Breast Cancer Wisconsin (Original) — UCI Machine Learning Repository.  
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
