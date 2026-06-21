# Explicação dos Gráficos — Trabalho Prático 2 de IA

**Guia pra entender cada gráfico gerado e saber explicar na apresentação.**

---

## GRÁFICO 1: `distribuicao_classes.png` — O MAIS SIMPLES

### O que é
Duas barras lado a lado mostrando quantos pacientes têm tumor benigno e quantos têm maligno.

### O que você vê
```
BARRA AZUL (esquerda)     ████████████████████████  458 casos  (65.5%)
BARRA VERMELHA (direita)  ████████████              241 casos  (34.5%)
```

### O que significa
Tem quase o DOBRO de casos benignos (azul) do que malignos (vermelho). Mas a razão é 1.9:1 — abaixo de 2:1, então o dataset é considerado balanceado.

### Por que isso importa
Se a diferença fosse muito grande (tipo 9:1), eu precisaria aplicar técnicas de balanceamento (criar dados falsos ou descartar exemplos). Como tá perto de 1:1, posso seguir sem artifícios.

### O que falar na apresentação
"Aqui dá pra ver que o dataset tem mais benignos que malignos, mas a diferença não é tão grande a ponto de precisar de técnica de balanceamento artificial. A razão é 1.9:1."

---

## GRÁFICO 2: `histogramas_por_classe.png` — O MAIS IMPORTANTE

### O que é
Uma grade 3×3 com 9 gráficos de barras, um pra cada característica do tumor.

### Como ler cada mini-gráfico
- **Eixo X (horizontal):** valor da característica, de 1 a 10
- **Eixo Y (vertical):** quantos pacientes têm aquele valor
- **Barras AZUIS:** tumores BENIGNOS
- **Barras VERMELHAS:** tumores MALIGNOS

### O truque pra interpretar
Compare ONDE as barras azuis e vermelhas estão concentradas:

```
Feature BOA (ex: Bare_nuclei):         Feature RUIM (ex: Mitoses):

  AZUL ████                              AZUL ████████
       ████                                   ██
       ██           VERMELHA  ██              ██   VERMELHA  ██
       █                     ██               █              ██
       1  2  3  4  5  6  7  8  9  10          1  2  3  4  5  6  7  8  9  10

  AZUIS nos valores baixos (1-3)           AZUIS e VERMELHOS no mesmo lugar
  VERMELHOS nos valores altos (6-10)       Não dá pra separar!
  → Essa feature DISCRIMINA bem           → Essa feature NÃO ajuda muito
```

### O que significa cada uma das 9 features (da mais útil pra menos)

| # | Feature | Tradução | O que você vê | Qualidade |
|---|---------|----------|---------------|-----------|
| 1 | Bare_nuclei | Núcleos nus | Azul todo no 1, vermelho espalhado até 10 | ⭐⭐⭐⭐⭐ |
| 2 | Uniformity of cell shape | Uniformidade do formato | Azul concentrado 1-3, vermelho espalhado | ⭐⭐⭐⭐ |
| 3 | Uniformity of cell size | Uniformidade do tamanho | Parecido com o de cima, boa separação | ⭐⭐⭐⭐ |
| 4 | Bland_chromatin | Textura da cromatina | Azul mais baixo, vermelho mais distribuído | ⭐⭐⭐⭐ |
| 5 | Clump_thickness | Espessura dos aglomerados | Diferença visível, mas não gritante | ⭐⭐⭐ |
| 6 | Marginal_adhesion | Aderência nas margens | Azul maioria 1, vermelho um pouco mais espalhado | ⭐⭐⭐ |
| 7 | Normal_nucleoli | Tamanho dos nucléolos | Padrão parecido com os anteriores | ⭐⭐⭐ |
| 8 | Single_epithelial_cell_size | Tamanho célula epitelial | Menos diferença entre azul e vermelho | ⭐⭐ |
| 9 | Mitoses | Taxa de divisão celular | Azul e vermelho quase iguais, ambos no 1-2 | ⭐ |

### O que falar na apresentação
"Olha que interessante: no Bare_nuclei, os benignos estão quase todos no valor 1, enquanto os malignos se distribuem de 1 a 10. As duas cores quase não se sobrepõem — isso indica que Bare_nuclei é um excelente discriminador. Já no Mitoses, as duas barras são praticamente iguais: todo mundo tem valor 1 ou 2, independente da classe. Isso sugere que Mitoses é a feature menos útil pra classificação."

### Detalhe extra: correlação confirma
O Bare_nuclei tem r = 0.77 (muito forte), Mitoses tem r = 0.36 (moderada). A correlação de Pearson confirma exatamente o que os histogramas mostram visualmente.

---

## GRÁFICO 3: `boxplots_por_classe.png` — O MAIS TÉCNICO

### O que é
Mesma grade 3×3, mas agora com boxplots em vez de barras. Cada mini-gráfico tem DUAS caixas coloridas.

### Como ler um boxplot (guia visual)

```
    Valor 10 ─     ·  ← outlier (ponto isolado acima da caixa)
               │
    Valor 8  ─ ┌───┐  ← bigode superior (valor máximo "normal")
               │   │
    Valor 6  ─ │ ▄ │  ← Q3 (75% dos dados estão ABAIXO dessa linha)
               │ █ │      A caixa vai de Q1 até Q3
    Valor 4  ─ │─█─│  ← MEDIANA (linha grossa no meio)
               │ █ │      É o valor do "meio" — metade acima, metade abaixo
    Valor 2  ─ │ ▀ │  ← Q1 (25% dos dados estão ABAIXO dessa linha)
               │   │
               └───┘  ← bigode inferior (valor mínimo "normal")
    Valor 1  ─
```

**Traduzindo:**
- A **caixa** mostra onde está a MAIORIA dos pacientes (os 50% centrais)
- A **linha grossa** é a mediana (o valor "típico" daquele grupo)
- Os **bigodes** vão até os valores extremos "normais"
- Os **pontinhos** são outliers — valores muito diferentes do resto

### Cada mini-gráfico tem DUAS caixas
- **Caixa AZUL** (esquerda) = tumores BENIGNOS
- **Caixa VERMELHA** (direita) = tumores MALIGNOS

### Como interpretar (exemplo com Bare_nuclei)

```
Benignos (azul):           Malignos (vermelho):
   10 ─                       10 ─    ·
     │                         │    · ·  ← vários outliers!
    8 ─                        8 ─ ┌───┐
     │                         │   │   │
    6 ─                        6 ─ │ █ │  ← caixa lá em cima
     │                         │   │ █ │
    4 ─                        4 ─ │─█─│  ← mediana ~6
     │                         │   │ █ │
    2 ─ ┌───┐                  2 ─ │ ▀ │
     │ │   │                   │   │   │
    1 ─ │─█─│ ← mediana = 1    1 ─ └───┘
     │ │ ▀ │
       └───┘
       
    Caixa grudada embaixo      Caixa bem mais alta
    Quase todo mundo nota 1    Maioria entre 4-8
```

### O que você fala na apresentação
"Olha o boxplot do Bare_nuclei: a caixa azul tá lá embaixo, com mediana 1 — quase todos os benignos têm nota 1 nessa característica. A caixa vermelha tá bem mais alta, mediana perto de 6. As caixas mal se sobrepõem. Isso mostra que Bare_nuclei separa MUITO bem as duas classes."

"Agora olha o Mitoses: as duas caixas estão no mesmo lugar, mesma altura, mesma mediana. Essa feature é praticamente inútil pra diferenciar benigno de maligno."

### Detalhe importante: os pontinhos (outliers)
Repare nos pontinhos acima dos bigodes vermelhos — principalmente em Single_epithelial_cell_size (23 outliers) e Mitoses (45 outliers). Esses são pacientes com notas extremas (10) em várias características.

**Eu decidi MANTER esses outliers.** Uma nota 10 não é erro de medição — é uma avaliação clínica indicando altíssima severidade. Se eu removesse, estaria jogando fora justamente os casos mais graves de câncer.

---

## GRÁFICO 4: `mlp_convergencia.png` — CURVA DE APRENDIZADO

### O que é
Uma única linha verde descendo da esquerda pra direita, mostrando como o erro da rede neural diminuiu ao longo do treinamento.

### O desenho do que você vê
```
Loss (erro)
        ↑
  0.12 ─╲
        ╲
  0.08 ─ ╲___
           ╲  ```````````````````````
  0.04 ─    ╲__________________________
             
  0.00 ─|────|────|────|────|────|────|→
         0   100   200   300   400   500   Épocas
```

### O que cada parte significa
- **Eixo X (horizontal):** número da época — cada "rodada" completa de treinamento. A rede passou 500 vezes por todos os dados de treino.
- **Eixo Y (vertical):** loss (erro) — medido como MSE (Mean Squared Error). Quanto MAIS BAIXO, MELHOR. Significa que a rede está errando menos.
- **Linha verde:** o erro diminuindo a cada época.

### Como interpretar a curva em 3 fases

**Fase 1 — Queda rápida (épocas 0 a 50):**
O erro caiu de ~0.10 pra ~0.03. A rede aprendeu os padrões MAIS ÓBVIOS logo de cara.

**Fase 2 — Ajuste fino (épocas 50 a 300):**
O erro continua caindo, mas mais devagar. A rede está refinando os detalhes.

**Fase 3 — Estabilização (épocas 300 a 500):**
O erro quase não muda mais. A rede convergiu — treinar além disso não traria melhora. Chegou no limite do que essa arquitetura consegue aprender.

### O que um formato de curva diz sobre o treinamento

| Formato da curva | Significado | O que fazer |
|-----------------|-------------|-------------|
| Caindo e estabilizando ✅ | Rede convergiu, tudo certo | Nada — é o ideal |
| Caindo muito devagar | Taxa de aprendizado baixa | Aumentar learning_rate |
| Subindo e descendo (zigue-zague) | Taxa de aprendizado alta | Diminuir learning_rate |
| Reta horizontal | Rede não está aprendendo | Verificar arquitetura/dados |
| Caindo no treino mas erro alto no teste | Overfitting | Simplificar a rede ou usar regularização |

Nossa curva é do primeiro tipo ✅ — caindo suave e estabilizando.

### O que falar na apresentação
"A curva mostra que a rede convergiu direitinho. O erro caiu de ~0.10 pra ~0.03 ao longo de 500 épocas — uma redução de 70%. A queda foi suave, sem oscilações bruscas, e estabilizou no final. Isso indica que a taxa de aprendizado (0.1) tava adequada e a arquitetura (9→8→1) foi suficiente pro problema."

"Se a curva estivesse subindo ou oscilando muito, seria sinal de overfitting ou learning_rate mal calibrado. Mas ela caiu limpa e estabilizou — exatamente o que a gente quer ver."

---

## RESUMO FINAL — COMO EXPLICAR OS 4 GRÁFICOS EM 1 MINUTO

1. **Distribuição de classes:** "Dataset com 65% benigno e 35% maligno. Razão 1.9:1 — balanceado, sem necessidade de SMOTE."

2. **Histogramas:** "Vejam como Bare_nuclei separa bem as classes: azul concentrado no 1, vermelho espalhado até 10. Já Mitoses não separa nada — as duas cores estão sobrepostas nos valores 1 e 2."

3. **Boxplots:** "Confirmam a mesma análise. Destaque pros pontinhos (outliers) que eu mantive — são os casos mais severos de câncer, e fazem sentido clinicamente."

4. **Convergência da MLP:** "Rede neural aprendeu direitinho. Erro caiu 70% ao longo de 500 épocas, estabilizando em ~0.03. Sem overfitting, sem oscilação."
