Roteiro do Trabalho Prático 2 - Pré-processamento e Análise Preditiva

Neste trabalho, você deverá realizar o pré-processamento e análise preditiva sobre a base de dados que foi atribuída para você. Cada aluno é responsável por uma base de dados, cujo link para acesso está disponível na página da disciplina. Durante a etapa de pré-processamento e análise de dados, você deverá seguir o seguinte roteiro como guia para suas atividades. Mesmo que a base de dados não apresente as características analisadas, você deverá realizar a análise e mostrar que elas não se aplicam:

Parte 1: Pré-processamento e Análise de Dados

Identificação do atributo alvo (saída);

Identificação dos tipos de dados dos atributos de entrada (quantitativo, qualitativo);

Identificação da escala de dados dos atributos de entrada (nominal, ordinal, intervalar, racional);

Exploração dos dados através de medidas de localidade;

Exploração dos dados através de medidas de espalhamento;

Exploração dos dados através de medidas de distribuição;

Identificação e separação do conjunto de teste, que será utilizado para testar o desempenho dos modelos – o conjunto de testes deve ser representativo e ter as características da população completa. Caso sua base de dados já tenha o conjunto de teste definido, analisar se este segue as características do conjunto de treinamento;

Identificação e eliminação de atributos não necessários;

Identificação e eliminação de exemplos não necessários;

Análise e aplicação de técnicas de amostragem de dados (caso não seja necessário, analisar o porquê);

Identificação e aplicação de técnicas para minimizar problemas de desbalanceamento (caso não seja necessário, analisar o porquê);

Limpeza de dados:

a. Identificação e eliminação de ruídos ou outliers;

b. Identificação e eliminação de dados inconsistentes;

c. Identificação e eliminação de dados redundantes;

d. Identificação e resolução de dados incompletos (ausentes) – utilização de alguma técnica de preenchimento e justificar;

Identificação e conversão dos tipos de dados (caso não seja necessário, analisar o porquê). Os tipos de conversão que podem ser utilizados são:

a. Conversão de tipos (simbólico para numérico, ordinal para numérico, nominal para binário, numérico para ordinal);

b. Normalização dos dados (re-escala ou padronização);

Análise e aplicação de alguma técnica para redução de dimensionalidade – pesquisar alguma técnica utilizada na literatura e aplicar.

Parte 2: Análise Preditiva

Durante essa etapa de análise preditiva, você deverá seguir o seguinte roteiro como guia para suas atividades:

Definição da técnica de validação a ser utilizada (cross-validation, hold-out, leave-one-out, etc);

Definição das métricas a serem utilizadas para avaliar os resultados preditivos dos modelos (acurácia, precisão, recall, matriz de confusão, etc);

Definição de um algoritmo base (baseline), que será utilizado como base para análise dos resultados – algoritmo classe majoritária;

Criação de modelo preditivo utilizando algoritmo de indução baseado nos vizinhos mais próximos e similaridade de dados (K-NN);

Criação de modelo preditivo utilizando algoritmo de indução baseado em árvores de decisão (decisiontree ou árvore C4.5);

Criação de modelo preditivo utilizando algoritmo de indução redes neurais artificiais (MLP);

Análise dos resultados do algoritmo baseline;

Análise dos resultados dos três algoritmos de aprendizado de máquina supracitados.

Parte 3: Itens Necessários para Entrega e Apresentação

A seguir, são listados os itens necessários para entrega e apresentação do Trabalho Prático 2:

Código fonte em linguagem Python, R ou Java;

Documentação detalhada seguindo os itens expostos no roteiro, na ordem em que são apresentados, contendo a justificativa para todas as definições e decisões acerca do projeto;

Apresentação contendo 3 slides explicando a definição do problema, a metodologia de resolução e análise dos resultados. Estes slides deverão ser utilizados para apresentação do trabalho para a turma, cujo tempo máximo será limitado em 3 minutos.