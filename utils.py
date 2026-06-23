"""
utils.py — Funções matemáticas e estatísticas implementadas "na mão"
para o Trabalho Prático 2 — Pré-processamento e Análise Preditiva.

Todas as funções usam apenas Python puro (listas, loops, dicts) + módulo math.
NENHUMA biblioteca externa (pandas, numpy, scipy) é usada para cálculos.
"""

import csv
import math


# ---------------------------------------------------------------------------
# 0. CONTADOR DE FREQUÊNCIA MANUAL (substitui collections.Counter)
# ---------------------------------------------------------------------------

def _count_frequencies(items):
    """
    Conta a frequência de cada item em uma lista.
    Substitui collections.Counter com implementação manual (dict).
    """
    freq = {}
    for item in items:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    return freq


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO DE DADOS
# ---------------------------------------------------------------------------

def load_csv(filepath):
    """
    Lê um arquivo CSV e retorna:
      - header: lista com nomes das colunas
      - rows:   lista de listas, cada sublista é uma linha (valores como string)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]  # ignora linhas vazias
    return header, rows


def parse_dataset(filepath):
    """
    Lê o CSV do Breast Cancer Wisconsin, converte Class (2→0, 4→1) e
    trata valores ausentes representados como string vazia ('' → None).

    Retorna:
      - header:   nomes das colunas
      - X:        lista de listas com features (float ou None)
      - y:        lista com target (0 ou 1)
      - mask:     lista de booleanos indicando linhas válidas (sem None)
    """
    header, rows = load_csv(filepath)

    X = []
    y = []
    mask = []

    for row in rows:
        features = []
        valid = True
        for i, val in enumerate(row[:-1]):       # todas menos a última (Class)
            if val.strip() == '':
                features.append(None)
                valid = False
            else:
                features.append(float(val))
        X.append(features)

        target_val = row[-1].strip()
        if target_val == '':
            y.append(None)
            valid = False
        else:
            class_val = int(float(target_val))
            y.append(0 if class_val == 2 else 1)   # 2→0 (benigno), 4→1 (maligno)

        mask.append(valid)

    return header, X, y, mask


# ---------------------------------------------------------------------------
# 2. MEDIDAS DE LOCALIDADE (Tendência Central)
# ---------------------------------------------------------------------------

def calc_mean(data):
    """Média aritmética: sum(x) / n"""
    n = len(data)
    if n == 0:
        return None
    return sum(data) / n


def calc_median(data):
    """Mediana: valor central após ordenação"""
    n = len(data)
    if n == 0:
        return None
    sorted_data = sorted(data)
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    else:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2


def calc_mode(data):
    """Moda: valor(es) mais frequente(s)"""
    if len(data) == 0:
        return []
    freq = _count_frequencies(data)
    max_count = max(freq.values())
    modes = [k for k, v in freq.items() if v == max_count]
    return modes


# ---------------------------------------------------------------------------
# 3. MEDIDAS DE ESPALHAMENTO (Dispersão)
# ---------------------------------------------------------------------------

def calc_variance(data, sample=True):
    """
    Variância.
      - sample=True  → variância amostral (divide por n-1)
      - sample=False → variância populacional (divide por n)
    """
    n = len(data)
    if n < 2:
        return None
    mean = calc_mean(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    divisor = n - 1 if sample else n
    return sum(squared_diffs) / divisor


def calc_std(data, sample=True):
    """Desvio padrão: raiz quadrada da variância"""
    var = calc_variance(data, sample)
    if var is None:
        return None
    return math.sqrt(var)


def calc_min(data):
    """Valor mínimo"""
    if len(data) == 0:
        return None
    return min(data)


def calc_max(data):
    """Valor máximo"""
    if len(data) == 0:
        return None
    return max(data)


def calc_range(data):
    """Amplitude: max - min"""
    if len(data) == 0:
        return None
    return calc_max(data) - calc_min(data)


def calc_percentile(data, p):
    """
    Calcula o p-ésimo percentil (0 <= p <= 100) usando interpolação linear.
    Método equivalente ao `numpy.percentile` com interpolation='linear'.
    """
    n = len(data)
    if n == 0:
        return None
    sorted_data = sorted(data)
    rank = (p / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_data[lower]
    frac = rank - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def calc_quartiles(data):
    """Retorna (Q1, Q2/mediana, Q3)"""
    return (
        calc_percentile(data, 25),
        calc_percentile(data, 50),
        calc_percentile(data, 75),
    )


def calc_iqr(data):
    """Intervalo interquartil: Q3 - Q1"""
    q1, _, q3 = calc_quartiles(data)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


# ---------------------------------------------------------------------------
# 4. RESUMO ESTATÍSTICO COMPLETO
# ---------------------------------------------------------------------------

def describe(column_name, data):
    """
    Retorna um dicionário com todas as medidas descritivas para uma coluna.
    """
    return {
        'atributo': column_name,
        'n': len(data),
        'media': calc_mean(data),
        'mediana': calc_median(data),
        'moda': calc_mode(data),
        'variancia': calc_variance(data),
        'desvio_padrao': calc_std(data),
        'minimo': calc_min(data),
        'maximo': calc_max(data),
        'amplitude': calc_range(data),
        'q1': calc_percentile(data, 25),
        'q2': calc_percentile(data, 50),
        'q3': calc_percentile(data, 75),
        'iqr': calc_iqr(data),
    }


# ---------------------------------------------------------------------------
# 5. UTILITÁRIOS
# ---------------------------------------------------------------------------

def extract_column(X, col_idx, mask=None):
    """
    Extrai uma coluna numérica de X, filtrando valores None.
    Se mask for fornecida, usa apenas linhas onde mask[i] == True.
    """
    values = []
    for i, row in enumerate(X):
        if mask is not None and not mask[i]:
            continue
        val = row[col_idx]
        if val is not None:
            values.append(val)
    return values


def count_target(y):
    """Conta ocorrências de cada classe em y."""
    counts = _count_frequencies(y)
    return dict(counts)


def format_percent(value, total):
    """Formata valor/total como percentual."""
    if total == 0:
        return "0.00%"
    return f"{(value / total) * 100:.2f}%"


# ---------------------------------------------------------------------------
# 6. AMOSTRAGEM E DIVISÃO DE DADOS
# ---------------------------------------------------------------------------

def stratified_split(X, y, test_ratio=0.2, seed=None):
    """
    Divide os dados em treino e teste de forma estratificada (mantém a
    proporção das classes). Implementação pura, sem bibliotecas externas.

    Parâmetros:
      - X: lista de listas com features
      - y: lista com targets (0 ou 1)
      - test_ratio: fração destinada ao teste (default 0.2 = 20%)
      - seed: semente para reprodutibilidade

    Retorna:
      - X_train, X_test, y_train, y_test
    """
    # Separa índices por classe
    idx_class0 = [i for i, val in enumerate(y) if val == 0]
    idx_class1 = [i for i, val in enumerate(y) if val == 1]

    # Embaralhamento determinístico (Fisher-Yates com seed)
    if seed is not None:
        rng = _create_rng(seed)

        def shuffle(lst):
            for i in range(len(lst) - 1, 0, -1):
                j = rng() % (i + 1)
                lst[i], lst[j] = lst[j], lst[i]
    else:
        import random
        def shuffle(lst):
            random.shuffle(lst)

    shuffle(idx_class0)
    shuffle(idx_class1)

    # Calcula quantos exemplos de cada classe vão para teste
    n_test0 = max(1, round(len(idx_class0) * test_ratio))
    n_test1 = max(1, round(len(idx_class1) * test_ratio))

    # Índices de teste
    test_idx = set(idx_class0[:n_test0] + idx_class1[:n_test1])
    train_idx = [i for i in range(len(y)) if i not in test_idx]
    test_idx = sorted(test_idx)

    # Monta os conjuntos
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


def _create_rng(seed):
    """
    Gerador de números pseudoaleatórios (Linear Congruential Generator)
    implementado do zero para garantir reprodutibilidade entre execuções.

    Retorna uma função rand() que gera um inteiro de 31 bits.
    """
    state = seed

    def rand():
        nonlocal state
        # Parâmetros do LCG (glibc/ANSI C)
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state

    return rand


# ---------------------------------------------------------------------------
# 7. CORRELAÇÃO DE PEARSON (para redução de dimensionalidade)
# ---------------------------------------------------------------------------

def calc_correlation(x, y):
    """
    Calcula o coeficiente de correlação de Pearson entre duas listas x e y.

    Fórmula:
      r = Σ((xi - x̄)(yi - ȳ)) / √(Σ(xi - x̄)² * Σ(yi - ȳ)²)

    Retorna valor entre -1 e 1.
    """
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = calc_mean(x)
    mean_y = calc_mean(y)

    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0

    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        sum_xy += dx * dy
        sum_x2 += dx * dx
        sum_y2 += dy * dy

    denominator = math.sqrt(sum_x2 * sum_y2)
    if denominator == 0:
        return 0.0

    return sum_xy / denominator


def pearson_correlation_matrix(X, header):
    """
    Calcula a correlação de Pearson de cada feature com o target.

    Parâmetros:
      - X: lista de listas (apenas features)
      - header: nomes das colunas (features)

    Retorna: lista de tuplas (nome, correlação, valor_absoluto) ordenada por |r| decrescente
    """
    n_features = len(header)
    n = len(X)

    correlations = []
    for col_idx in range(n_features):
        x_vals = []
        for row in X:
            if row[col_idx] is not None:
                x_vals.append(row[col_idx])

        # Target é a última coluna de X se X incluir target? Não.
        # Precisamos que o target venha separado.
        # Esta função espera X só com features + y separado.
        # Vamos criar uma versão mais flexível abaixo.

    return correlations


def feature_target_correlations(X, y, feature_names):
    """
    Calcula a correlação de Pearson entre cada feature e o target.

    Retorna: lista de (nome, r, |r|) ordenada por |r| decrescente.
    """
    n_features = len(X[0])
    results = []

    for col_idx in range(n_features):
        xs = []
        ys = []
        for row, target in zip(X, y):
            val = row[col_idx]
            if val is not None and target is not None:
                xs.append(val)
                ys.append(target)

        r = calc_correlation(xs, ys)
        results.append((feature_names[col_idx], r, abs(r)))

    results.sort(key=lambda tup: tup[2], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 8. LIMPEZA DE DADOS
# ---------------------------------------------------------------------------

def impute_mode(X, col_idx):
    """
    Preenche valores None na coluna col_idx com a moda (valor mais frequente)
    dos valores válidos. Retorna uma NOVA matriz X (não modifica a original).

    Justificativa para o Breast Cancer Wisconsin:
      Todos os atributos são pontuações discretas (1-10). A moda garante que
      o valor imputado seja um inteiro clinicamente válido, ao contrário da
      média que geraria valores decimais inexistentes no domínio do problema.
    """
    # Extrai valores válidos para calcular a moda
    valid_vals = []
    for row in X:
        val = row[col_idx]
        if val is not None:
            valid_vals.append(val)

    if not valid_vals:
        return X  # sem valores válidos, retorna original

    modes = calc_mode(valid_vals)
    fill_value = modes[0]  # usa o primeiro em caso de empate

    # Cria nova matriz com valores preenchidos
    X_new = []
    for row in X:
        new_row = list(row)
        if new_row[col_idx] is None:
            new_row[col_idx] = fill_value
        X_new.append(new_row)

    return X_new


def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detecta outliers usando o método IQR (Intervalo Interquartil).

    Um valor é considerado outlier se:
      valor < Q1 - (multiplier * IQR)  ou  valor > Q3 + (multiplier * IQR)

    Retorna: (lower_bound, upper_bound, outlier_indices, outlier_values)
    """
    q1, q2, q3 = calc_quartiles(data)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    outliers_idx = []
    outliers_val = []
    for i, val in enumerate(data):
        if val < lower or val > upper:
            outliers_idx.append(i)
            outliers_val.append(val)

    return lower, upper, outliers_idx, outliers_val


def remove_constant_columns(X, feature_names):
    """
    Remove colunas com variância zero (valor constante), pois não contribuem
    para a predição.

    Retorna: (X_new, feature_names_new, removed_names)
    """
    n_features = len(X[0])
    to_remove = []

    for col_idx in range(n_features):
        values = set()
        for row in X:
            if row[col_idx] is not None:
                values.add(row[col_idx])
        if len(values) <= 1:
            to_remove.append(col_idx)

    if not to_remove:
        return X, feature_names, []

    # Remove colunas (percorrendo em ordem reversa para não quebrar índices)
    X_new = [list(row) for row in X]
    for col_idx in reversed(to_remove):
        for row in X_new:
            row.pop(col_idx)

    removed_names = [feature_names[i] for i in to_remove]
    feature_names_new = [name for i, name in enumerate(feature_names) if i not in to_remove]

    return X_new, feature_names_new, removed_names


def remove_duplicate_rows(X, y):
    """
    Remove exemplos duplicados (mesmas features e mesmo target).
    Mantém apenas a primeira ocorrência.

    Retorna: (X_new, y_new, n_removed)
    """
    seen = set()
    X_new = []
    y_new = []
    n_removed = 0

    for row, target in zip(X, y):
        key = tuple(row) + (target,)
        if key not in seen:
            seen.add(key)
            X_new.append(list(row))
            y_new.append(target)
        else:
            n_removed += 1

    return X_new, y_new, n_removed


# ---------------------------------------------------------------------------
# 9. NORMALIZAÇÃO
# ---------------------------------------------------------------------------

def normalize_minmax(X):
    """
    Normaliza as features para o intervalo [0, 1] usando a fórmula:
      X_norm = (X - X_min) / (X_max - X_min)

    Calcula min/max por coluna e aplica a transformação.

    Justificativa: Min-Max preserva a estrutura da distribuição original e
    é adequado para dados com escala limitada conhecida (1-10). Essencial
    para evitar que features com maior magnitude dominem o K-NN e para
    evitar saturação na MLP.

    Retorna: (X_normalized, mins, maxs)
    """
    n_features = len(X[0])
    n = len(X)

    # Calcula min e max de cada coluna
    mins = []
    maxs = []
    for col_idx in range(n_features):
        col_vals = [row[col_idx] for row in X if row[col_idx] is not None]
        mins.append(min(col_vals))
        maxs.append(max(col_vals))

    # Aplica normalização
    X_norm = []
    for row in X:
        new_row = []
        for col_idx, val in enumerate(row):
            denom = maxs[col_idx] - mins[col_idx]
            if denom == 0:
                new_row.append(0.0)
            else:
                new_row.append((val - mins[col_idx]) / denom)
        X_norm.append(new_row)

    return X_norm, mins, maxs


# ---------------------------------------------------------------------------
# 10. CONVERSÃO DE DADOS (features para matriz numérica pura)
# ---------------------------------------------------------------------------

def to_matrix(X, y):
    """
    Converte as estruturas para listas puras (garante que não há None).
    Útil como passo final de pré-processamento antes da modelagem.
    """
    X_clean = []
    for row in X:
        X_clean.append([float(v) if v is not None else 0.0 for v in row])
    y_clean = [int(v) for v in y]
    return X_clean, y_clean


# ═══════════════════════════════════════════════════════════════════════════
# PARTE 2: ANÁLISE PREDITIVA — ALGORITMOS "FROM SCRATCH"
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# 11. MÉTRICAS DE AVALIAÇÃO
# ---------------------------------------------------------------------------

def confusion_matrix_values(y_true, y_pred):
    """
    Calcula Verdadeiro Positivo (TP), Verdadeiro Negativo (TN),
    Falso Positivo (FP) e Falso Negativo (FN) a partir de duas listas.

    Convenção: classe positiva = 1 (maligno), classe negativa = 0 (benigno).

    Retorna: (TP, TN, FP, FN)
    """
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    return tp, tn, fp, fn


def calc_accuracy(tp, tn, fp, fn):
    """Acurácia: (TP + TN) / Total"""
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


def calc_precision(tp, fp):
    """Precisão: TP / (TP + FP) — quantos positivos preditos são realmente positivos."""
    denom = tp + fp
    if denom == 0:
        return 0.0
    return tp / denom


def calc_recall(tp, fn):
    """Recall (Sensibilidade): TP / (TP + FN) — quantos positivos reais foram detectados."""
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def calc_specificity(tn, fp):
    """Especificidade: TN / (TN + FP) — quantos negativos reais foram detectados."""
    denom = tn + fp
    if denom == 0:
        return 0.0
    return tn / denom


def calc_f1_score(precision, recall):
    """F1-Score: média harmônica entre precisão e recall."""
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * (precision * recall) / denom


def evaluate_model(y_true, y_pred):
    """
    Calcula todas as métricas de uma vez.

    Retorna um dicionário com:
      TP, TN, FP, FN, accuracy, precision, recall, specificity, f1_score
    """
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy':    calc_accuracy(tp, tn, fp, fn),
        'precision':   calc_precision(tp, fp),
        'recall':      calc_recall(tp, fn),
        'specificity': calc_specificity(tn, fp),
        'f1_score':    calc_f1_score(calc_precision(tp, fp), calc_recall(tp, fn)),
    }


# ---------------------------------------------------------------------------
# 12. BASELINE — CLASSE MAJORITÁRIA
# ---------------------------------------------------------------------------

def majority_class_baseline(y_train):
    """
    Identifica a classe majoritária no treino e retorna uma função
    que sempre prediz essa classe para qualquer entrada.

    Retorna: (predict_fn, majority_class, majority_ratio)
    """
    counts = _count_frequencies(y_train)
    majority = max(counts, key=counts.get)
    ratio = counts[majority] / len(y_train)
    return (lambda x: majority), majority, ratio


# ---------------------------------------------------------------------------
# 13. DISTÂNCIA EUCLIDIANA (para K-NN)
# ---------------------------------------------------------------------------

def euclidean_distance(a, b):
    """
    Calcula a distância euclidiana entre dois vetores a e b.

    Fórmula: d(a,b) = √(Σ(ai - bi)²)
    """
    if len(a) != len(b):
        raise ValueError(f"Vetores de tamanhos diferentes: {len(a)} vs {len(b)}")
    sum_sq = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sum_sq += diff * diff
    return math.sqrt(sum_sq)


# ---------------------------------------------------------------------------
# 14. K-NN (K-NEAREST NEIGHBORS)
# ---------------------------------------------------------------------------

def knn_predict(X_train, y_train, x_test, k=5):
    """
    Prediz a classe de um único exemplo x_test usando K-NN.

    Algoritmo:
      1. Calcula distância euclidiana de x_test para todos os pontos de treino
      2. Seleciona os K vizinhos mais próximos
      3. Votação majoritária entre os K vizinhos

    Retorna: 0 (benigno) ou 1 (maligno)
    """
    n_train = len(X_train)
    # Calcula todas as distâncias
    distances = []
    for i in range(n_train):
        dist = euclidean_distance(x_test, X_train[i])
        distances.append((dist, y_train[i]))

    # Ordena por distância crescente
    distances.sort(key=lambda pair: pair[0])

    # Votação entre os K vizinhos mais próximos
    k = min(k, len(distances))
    votes = [distances[i][1] for i in range(k)]

    # Soma dos votos: 0 = benigno, 1 = maligno
    # Se empate, vence a classe 0 (benigno — decisão conservadora)
    sum_votes = sum(votes)
    if sum_votes > k / 2:
        return 1
    else:
        return 0


def knn_predict_batch(X_train, y_train, X_test, k=5):
    """
    Prediz classes para múltiplos exemplos de teste.

    Retorna: lista de predições (0 ou 1)
    """
    return [knn_predict(X_train, y_train, x, k) for x in X_test]


# ---------------------------------------------------------------------------
# 15. ÁRVORE DE DECISÃO C4.5
# ---------------------------------------------------------------------------

def calc_entropy(labels):
    """
    Calcula a entropia de um conjunto de rótulos.

    Fórmula: H(S) = -Σ p(c) * log2(p(c))

    Retorna: valor em bits (0 = conjunto puro, 1 = máximo para binário)
    """
    n = len(labels)
    if n == 0:
        return 0.0

    counts = _count_frequencies(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calc_split_info(values):
    """
    Calcula a informação de split (SplitInfo) para a razão de ganho C4.5.

    SplitInfo = -Σ (|Sv|/|S|) * log2(|Sv|/|S|)
    """
    n = len(values)
    if n == 0:
        return 0.0
    counts = _count_frequencies(values)
    split_info = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            split_info -= p * math.log2(p)
    return split_info


def best_split_c45(X, y, feature_indices=None):
    """
    Encontra o melhor ponto de corte (feature + threshold) usando a Razão
    de Ganho (Gain Ratio) do algoritmo C4.5.

    GainRatio = InformationGain / SplitInfo

    Para cada feature, testa divisões binárias em cada valor único.

    Retorna: (best_feature_idx, best_threshold, best_gain_ratio)
             ou (None, None, -1) se nenhum split melhorar a entropia.
    """
    n_samples = len(y)
    if n_samples <= 1:
        return None, None, -1.0

    current_entropy = calc_entropy(y)

    if feature_indices is None:
        feature_indices = list(range(len(X[0])))

    best_gain_ratio = -1.0
    best_feature = None
    best_threshold = None

    for feat_idx in feature_indices:
        # Coleta valores únicos ordenados da feature
        col_vals = [row[feat_idx] for row in X]
        unique_vals = sorted(set(col_vals))

        # Testa cada threshold (média entre valores consecutivos)
        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0

            # Divide os dados
            left_y = []
            right_y = []
            left_vals = []
            right_vals = []
            for j, val in enumerate(col_vals):
                if val <= threshold:
                    left_y.append(y[j])
                    left_vals.append(val)
                else:
                    right_y.append(y[j])
                    right_vals.append(val)

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # Ganho de Informação
            left_entropy = calc_entropy(left_y)
            right_entropy = calc_entropy(right_y)
            p_left = len(left_y) / n_samples
            p_right = len(right_y) / n_samples
            info_gain = current_entropy - (p_left * left_entropy + p_right * right_entropy)

            # SplitInfo (combina os valores da feature no split)
            split_vals = ['L'] * len(left_y) + ['R'] * len(right_y)
            split_info = calc_split_info(split_vals)

            if split_info == 0:
                continue

            gain_ratio = info_gain / split_info

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = feat_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain_ratio


def build_tree_c45(X, y, feature_names, max_depth=10, min_samples=2, depth=0):
    """
    Constrói recursivamente uma árvore de decisão C4.5.

    Estrutura do nó:
      {
        'type': 'leaf' | 'node',
        'prediction': int (classe majoritária, para leaf),
        'feature_idx': int (índice da feature de split, para node),
        'feature_name': str,
        'threshold': float,
        'left': dict (subárvore esquerda: valor <= threshold),
        'right': dict (subárvore direita: valor > threshold),
      }

    Critérios de parada:
      - max_depth atingida
      - Nó puro (todos mesma classe)
      - Menos de min_samples exemplos
      - Nenhum split melhora o ganho
    """
    n_samples = len(y)

    # Critérios de parada: nó folha
    unique_classes = set(y)
    if len(unique_classes) == 1:
        majority = max(_count_frequencies(y).items(), key=lambda kv: kv[1])[0]
        return {'type': 'leaf', 'prediction': majority}
    if n_samples < min_samples or depth >= max_depth:
        majority = max(_count_frequencies(y).items(), key=lambda kv: kv[1])[0]
        return {'type': 'leaf', 'prediction': majority}

    # Encontra o melhor split
    feat_idx, threshold, gain_ratio = best_split_c45(X, y)

    # Se nenhum split melhora, vira folha
    if feat_idx is None or gain_ratio <= 0:
        majority = max(_count_frequencies(y).items(), key=lambda kv: kv[1])[0]
        return {'type': 'leaf', 'prediction': majority}

    # Divide os dados
    X_left, y_left, X_right, y_right = [], [], [], []
    for i in range(n_samples):
        if X[i][feat_idx] <= threshold:
            X_left.append(X[i])
            y_left.append(y[i])
        else:
            X_right.append(X[i])
            y_right.append(y[i])

    # Constrói subárvores recursivamente
    return {
        'type': 'node',
        'feature_idx': feat_idx,
        'feature_name': feature_names[feat_idx] if feature_names else f'f{feat_idx}',
        'threshold': threshold,
        'left': build_tree_c45(X_left, y_left, feature_names, max_depth, min_samples, depth + 1),
        'right': build_tree_c45(X_right, y_right, feature_names, max_depth, min_samples, depth + 1),
    }


def predict_tree_c45(tree, x):
    """
    Prediz a classe de um exemplo percorrendo a árvore C4.5.
    """
    node = tree
    while node['type'] == 'node':
        if x[node['feature_idx']] <= node['threshold']:
            node = node['left']
        else:
            node = node['right']
    return node['prediction']


def predict_tree_batch(tree, X):
    """
    Prediz classes para múltiplos exemplos.
    """
    return [predict_tree_c45(tree, x) for x in X]


def count_tree_nodes(tree):
    """Conta o número total de nós na árvore (para análise)."""
    if tree['type'] == 'leaf':
        return 1
    return 1 + count_tree_nodes(tree['left']) + count_tree_nodes(tree['right'])


# ---------------------------------------------------------------------------
# 16. REDE NEURAL ARTIFICIAL — MLP (Multilayer Perceptron)
# ---------------------------------------------------------------------------

def sigmoid(x):
    """
    Função de ativação sigmoide: σ(x) = 1 / (1 + e^(-x))
    Limita overflow: para x < -100, retorna ~0; para x > 100, retorna ~1.
    """
    if x < -100:
        return 0.0
    if x > 100:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    """
    Derivada da sigmoide dado o valor já computado σ(x):
    σ'(x) = σ(x) * (1 - σ(x))
    O parâmetro 'x' aqui é σ(x), não x original.
    """
    return x * (1.0 - x)


def _init_weights(n_input, n_hidden, n_output, seed=None):
    """
    Inicializa pesos com pequenos valores aleatórios usando Xavier/Glorot-like.
    """
    if seed is not None:
        rng = _create_rng(seed)
    else:
        import random
        rng = lambda: random.randint(0, 0x7FFFFFFF)

    def rand_weight():
        """Pequeno valor aleatório entre -0.5 e 0.5 escalado por fan-in."""
        return ((rng() % 10000) / 10000.0 - 0.5) / math.sqrt(n_input)

    # Pesos: input → hidden
    w1 = []
    for _ in range(n_input):
        row = []
        for _ in range(n_hidden):
            row.append(rand_weight())
        w1.append(row)

    # Bias: hidden
    b1 = [0.0] * n_hidden

    # Pesos: hidden → output
    w2 = []
    for _ in range(n_hidden):
        row = []
        for _ in range(n_output):
            row.append(rand_weight())
        w2.append(row)

    # Bias: output
    b2 = [0.0] * n_output

    return w1, b1, w2, b2


def mlp_train(X, y, hidden_neurons=8, learning_rate=0.1, epochs=200, seed=None, verbose=False):
    """
    Treina uma MLP (Multilayer Perceptron) com uma camada oculta usando
    backpropagation e gradiente descendente estocástico (SGD).

    Arquitetura:
      - Entrada: n_features
      - Oculta: hidden_neurons (com ativação sigmoide)
      - Saída: 1 neurônio (sigmoid → probabilidade da classe 1)

    Retorna: (w1, b1, w2, b2, history)
      - history: lista de (epoch, loss) para gráfico de convergência
    """
    n_samples = len(X)
    n_features = len(X[0])
    n_output = 1
    n_hidden = hidden_neurons

    w1, b1, w2, b2 = _init_weights(n_features, n_hidden, n_output, seed)
    history = []

    for epoch in range(epochs):
        total_loss = 0.0

        # SGD: percorre cada exemplo
        for i in range(n_samples):
            # ── FORWARD PASS ──
            # Camada oculta
            hidden_out = [0.0] * n_hidden
            for h in range(n_hidden):
                z = b1[h]
                for f in range(n_features):
                    z += X[i][f] * w1[f][h]
                hidden_out[h] = sigmoid(z)

            # Camada de saída
            output_in = b2[0]
            for h in range(n_hidden):
                output_in += hidden_out[h] * w2[h][0]
            output_out = sigmoid(output_in)

            # Erro
            target = y[i]
            error = target - output_out
            total_loss += error * error  # MSE acumulado

            # ── BACKPROPAGATION ──
            # Gradiente da camada de saída
            delta_out = error * sigmoid_derivative(output_out)

            # Gradiente da camada oculta
            delta_hidden = [0.0] * n_hidden
            for h in range(n_hidden):
                delta_hidden[h] = delta_out * w2[h][0] * sigmoid_derivative(hidden_out[h])

            # Atualiza pesos hidden → output
            for h in range(n_hidden):
                w2[h][0] += learning_rate * delta_out * hidden_out[h]
            b2[0] += learning_rate * delta_out

            # Atualiza pesos input → hidden
            for f in range(n_features):
                for h in range(n_hidden):
                    w1[f][h] += learning_rate * delta_hidden[h] * X[i][f]
            for h in range(n_hidden):
                b1[h] += learning_rate * delta_hidden[h]

        avg_loss = total_loss / n_samples
        history.append((epoch, avg_loss))

        if verbose and (epoch + 1) % 50 == 0:
            # Calcula acurácia no treino para feedback
            preds = mlp_predict(X, w1, b1, w2, b2)
            correct = sum(1 for p, t in zip(preds, y) if p == t)
            acc = correct / n_samples
            print(f'    Época {epoch+1:>4d}/{epochs} | Loss: {avg_loss:.6f} | Acurácia treino: {acc:.4f}')

    return w1, b1, w2, b2, history


def mlp_predict(X, w1, b1, w2, b2, threshold=0.5):
    """
    Prediz classes para múltiplos exemplos usando a MLP treinada.

    Retorna: lista de 0 ou 1
    """
    n_hidden = len(b1)
    predictions = []

    for x in X:
        # Forward pass
        hidden_out = []
        for h in range(n_hidden):
            z = b1[h]
            for f in range(len(x)):
                z += x[f] * w1[f][h]
            hidden_out.append(sigmoid(z))

        output_in = b2[0]
        for h in range(n_hidden):
            output_in += hidden_out[h] * w2[h][0]
        prob = sigmoid(output_in)

        predictions.append(1 if prob >= threshold else 0)

    return predictions


def mlp_predict_proba(X, w1, b1, w2, b2):
    """
    Retorna probabilidades (valores entre 0 e 1) em vez de classes.
    Útil para análise de thresholds.
    """
    n_hidden = len(b1)
    probs = []

    for x in X:
        hidden_out = []
        for h in range(n_hidden):
            z = b1[h]
            for f in range(len(x)):
                z += x[f] * w1[f][h]
            hidden_out.append(sigmoid(z))

        output_in = b2[0]
        for h in range(n_hidden):
            output_in += hidden_out[h] * w2[h][0]
        probs.append(sigmoid(output_in))

    return probs
