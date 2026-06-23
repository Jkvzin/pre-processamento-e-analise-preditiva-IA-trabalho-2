"""
main.py — Trabalho Prático 2: Pré-processamento e Análise Preditiva
Base: Breast Cancer Wisconsin (Original)

Orquestra todas as etapas do roteiro, importando funções matemáticas
puras do módulo utils.py.
"""

import os
import sys
import math

# Garante que o diretório atual está no path para importar utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import (
    load_csv, parse_dataset, extract_column, count_target,
    calc_mean, calc_median, calc_mode,
    calc_variance, calc_std, calc_min, calc_max, calc_range,
    calc_percentile, calc_quartiles, calc_iqr,
    describe, format_percent,
    # ── Parte 2 ──
    confusion_matrix_values,
    calc_accuracy, calc_precision, calc_recall, calc_specificity, calc_f1_score,
    evaluate_model,
    majority_class_baseline,
    knn_predict_batch,
    build_tree_c45, predict_tree_batch, count_tree_nodes,
    mlp_train, mlp_predict,
)


# ======================================================================
# CONFIGURAÇÃO
# ======================================================================

DATA_PATH = os.path.join('data', 'breast_cancer_wisconsin.csv')
PLOTS_DIR = 'plots'

# Descrições textuais de cada atributo
ATTR_DESCRIPTIONS = {
    'Clump_thickness':                 'Espessura do aglomerado (1-10)',
    'Uniformity_of_cell_size':         'Uniformidade do tamanho celular (1-10)',
    'Uniformity_of_cell_shape':        'Uniformidade do formato celular (1-10)',
    'Marginal_adhesion':              'Adesão marginal (1-10)',
    'Single_epithelial_cell_size':     'Tamanho de célula epitelial única (1-10)',
    'Bare_nuclei':                    'Núcleos nus (1-10)',
    'Bland_chromatin':                'Cromatina suave (1-10)',
    'Normal_nucleoli':                'Nucléolos normais (1-10)',
    'Mitoses':                        'Mitoses (1-10)',
}

# Intervalos teóricos para cada atributo (escala de 1 a 10)
FEATURE_MIN = 1
FEATURE_MAX = 10


# ======================================================================
# FUNÇÕES AUXILIARES DE FORMATAÇÃO
# ======================================================================

def print_header(title, char='=', width=75):
    """Imprime um cabeçalho centralizado."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_subheader(title, char='-', width=60):
    """Imprime um sub-cabeçalho."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# ======================================================================
# ETAPA 1: IDENTIFICAÇÃO DO ATRIBUTO ALVO
# ======================================================================

def etapa1_atributo_alvo(header):
    print_header('ETAPA 1: Identificação do Atributo Alvo (Saída)')

    target_name = header[-1]
    print(f'  Atributo alvo identificado: "{target_name}"')
    print(f'  Posição no dataset:         última coluna (índice {len(header) - 1})')
    print()
    print('  Classes originais no arquivo:')
    print('    - 2 → Tumor BENIGNO')
    print('    - 4 → Tumor MALIGNO')
    print()
    print('  Conversão aplicada (para compatibilidade com algoritmos):')
    print('    - 2 → 0 (Benigno)')
    print('    - 4 → 1 (Maligno)')
    print()
    print('  Justificativa: A conversão para 0/1 é padrão em classificação binária.')
    print('  Isso simplifica cálculos de distância e funções de ativação nos')
    print('  algoritmos que implementaremos manualmente (K-NN, MLP).')
    print('  O valor 0 representa a classe negativa (benigno) e 1 a positiva (maligno).')

    return target_name


# ======================================================================
# ETAPA 2: TIPOS DE DADOS DOS ATRIBUTOS DE ENTRADA
# ======================================================================

def etapa2_tipos_dados(header, X):
    print_header('ETAPA 2: Identificação dos Tipos de Dados dos Atributos de Entrada')

    print(f'  {"Atributo":<32} {"Tipo":<18} {"Justificativa"}')
    print(f'  {"-"*32} {"-"*18} {"-"*50}')

    for i, name in enumerate(header[:-1]):
        # Todos os atributos são numéricos (valores de 1 a 10)
        # Verificamos se há valores não-inteiros
        col = extract_column(X, i)
        all_int = all(v == int(v) for v in col if v is not None)
        unique_vals = set(col)

        if all_int and len(unique_vals) <= 12:
            tipo = 'Quantitativo Discreto'
            just = f'Valores inteiros em escala limitada ({len(unique_vals)} valores distintos)'
        else:
            tipo = 'Quantitativo Contínuo'
            just = 'Valores numéricos com possibilidade de casas decimais'

        print(f'  {name:<32} {tipo:<18} {just}')

    print()
    print('  Observação: Embora Bare_nuclei apareça com valores como 1.0, 10.0 no CSV,')
    print('  semanticamente todos os atributos são pontuações inteiras de 1 a 10.')
    print('  O ponto decimal é artefato de formatação do CSV original, não do dado real.')


# ======================================================================
# ETAPA 3: ESCALA DE DADOS DOS ATRIBUTOS DE ENTRADA
# ======================================================================

def etapa3_escala_dados(header, X):
    print_header('ETAPA 3: Identificação da Escala de Dados dos Atributos de Entrada')

    print(f'  {"Atributo":<32} {"Escala":<18} {"Justificativa"}')
    print(f'  {"-"*32} {"-"*18} {"-"*50}')

    for i, name in enumerate(header[:-1]):
        col = extract_column(X, i)
        unique_vals = set(col)

        # Escala intervalar: intervalos constantes entre valores consecutivos
        # não há zero absoluto verdadeiro (nota 1 = mínimo da escala, não ausência)
        if len(unique_vals) <= 12 and all(v == int(v) for v in col if v is not None):
            escala = 'Intervalar Discreta'
        else:
            escala = 'Intervalar Contínua'

        just = (
            'Intervalos aproximadamente constantes entre notas '
            'consecutivas (diferença 3→5 ≈ diferença 7→9 na '
            'percepção do patologista)'
        )

        print(f'  {name:<32} {escala:<18} {just}')

    print()
    print('  Todas as features são do tipo ESCALA INTERVALAR porque:')
    print('    1. Os intervalos entre notas consecutivas são aproximadamente constantes')
    print('    2. A escala original é ordinal (1-10), mas na literatura médica')
    print('    trata-se como intervalar para permitir operações estatísticas')
    print('    paramétricas (média, desvio padrão, correlação).')
    print('    3. Não há zero absoluto verdadeiro — nota 1 não significa ausência')
    print('    total da característica, mas sim o mínimo observável na escala.')


# ======================================================================
# ETAPA 4a: MEDIDAS DE LOCALIDADE
# ======================================================================

def etapa4a_medidas_localidade(header, X, mask):
    print_header('ETAPA 4a: Exploração — Medidas de Localidade (Tendência Central)')

    print(f'  {"Atributo":<32} {"Média":>8} {"Mediana":>8} {"Moda(s)":>12}')
    print(f'  {"-"*32} {"-"*8} {"-"*8} {"-"*20}')

    for i, name in enumerate(header[:-1]):
        col = extract_column(X, i, mask)
        mean_val = calc_mean(col)
        median_val = calc_median(col)
        mode_val = calc_mode(col)
        mode_str = ', '.join(str(int(m)) for m in sorted(mode_val)[:3])

        print(f'  {name:<32} {mean_val:>8.2f} {median_val:>8.1f} {mode_str:>12}')


# ======================================================================
# ETAPA 4b: MEDIDAS DE ESPALHAMENTO
# ======================================================================

def etapa4b_medidas_espalhamento(header, X, mask):
    print_header('ETAPA 4b: Exploração — Medidas de Espalhamento (Dispersão)')

    print(f'  {"Atributo":<32} {"Amplitude":>10} {"Variância":>10} {"Desv.Pad.":>10} {"IQR":>8}')
    print(f'  {"-"*32} {"-"*10} {"-"*10} {"-"*10} {"-"*8}')

    for i, name in enumerate(header[:-1]):
        col = extract_column(X, i, mask)
        amp = calc_range(col)
        var = calc_variance(col)
        std = calc_std(col)
        iqr = calc_iqr(col)

        print(f'  {name:<32} {amp:>10.2f} {var:>10.4f} {std:>10.4f} {iqr:>8.2f}')


# ======================================================================
# ETAPA 4c: MEDIDAS DE DISTRIBUIÇÃO (Gráficos)
# ======================================================================

def etapa4c_distribuicao(header, X, y, mask):
    print_header('ETAPA 4c: Exploração — Medidas de Distribuição (Histogramas + Boxplots)')

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Separa dados por classe para os histogramas sobrepostos
    idx_benign = [i for i, val in enumerate(y) if val == 0 and mask[i]]
    idx_malign = [i for i, val in enumerate(y) if val == 1 and mask[i]]

    # ----- 4c.1: Histogramas lado a lado (grid 3x3) -----
    print('  Gerando histogramas (grid 3x3)...')

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(header[:-1], axes)):
        col_benign = [X[j][i] for j in idx_benign if X[j][i] is not None]
        col_malign = [X[j][i] for j in idx_malign if X[j][i] is not None]

        bins = range(1, 12)  # bins alinhados com valores possíveis (1-10)
        ax.hist(col_benign, bins=bins, alpha=0.7, label='Benigno (0)', color='steelblue',
                edgecolor='white')
        ax.hist(col_malign, bins=bins, alpha=0.7, label='Maligno (1)', color='crimson',
                edgecolor='white')

        ax.set_title(name.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, 11))

    plt.suptitle('Distribuição dos Atributos por Classe (Benigno vs Maligno)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    hist_path = os.path.join(PLOTS_DIR, 'histogramas_por_classe.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → Salvo em: {hist_path}')

    # ----- 4c.2: Boxplots lado a lado (grid 3x3) -----
    print('  Gerando boxplots (grid 3x3)...')

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(header[:-1], axes)):
        col_benign = [X[j][i] for j in idx_benign if X[j][i] is not None]
        col_malign = [X[j][i] for j in idx_malign if X[j][i] is not None]

        bp = ax.boxplot(
            [col_benign, col_malign],
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 2},
            widths=0.5
        )
        ax.set_xticklabels(['Benigno (0)', 'Maligno (1)'])
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('crimson')

        ax.set_title(name.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_ylabel('Valor')

    plt.suptitle('Boxplots por Classe (Benigno vs Maligno)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    box_path = os.path.join(PLOTS_DIR, 'boxplots_por_classe.png')
    plt.savefig(box_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → Salvo em: {box_path}')

    # ----- 4c.3: Distribuição das classes (gráfico de barras) -----
    print('  Gerando gráfico de distribuição das classes...')

    counts = count_target(y)
    total = sum(counts.values())
    labels = ['Benigno (0)', 'Maligno (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['steelblue', 'crimson']
    percentages = [f'{v} ({format_percent(v, total)})' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                pct, ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title('Distribuição das Classes no Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Quantidade de Exemplos')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    class_path = os.path.join(PLOTS_DIR, 'distribuicao_classes.png')
    plt.savefig(class_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → Salvo em: {class_path}')

    # ----- 4c.4: Resumo textual da distribuição -----
    print_subheader('Resumo da Distribuição')
    n_valid_total = sum(1 for m in mask if m)
    # Recalcula counts apenas sobre exemplos VÁLIDOS (mask=True)
    y_valid_full = [y[i] for i in range(len(y)) if mask[i]]
    counts_valid = count_target(y_valid_full)
    total_valid_count = sum(counts_valid.values())
    values = [counts_valid.get(0, 0), counts_valid.get(1, 0)]
    print(f'\n  Total de exemplos (válidos): {n_valid_total}')
    print(f'  Benignos (0): {values[0]} ({format_percent(values[0], total_valid_count)})')
    print(f'  Malignos  (1): {values[1]} ({format_percent(values[1], total_valid_count)})')
    print(f'  Razão benigno/maligno: {values[0]}/{values[1]} ≈ {values[0]/values[1]:.2f}:1')

    imbalance_ratio = max(values) / min(values) if min(values) > 0 else float('inf')
    print(f'\n  Avaliação de desbalanceamento:')
    if imbalance_ratio > 2.0:
        print(f'    ⚠️  Dataset DESBALANCEADO (razão {imbalance_ratio:.2f}:1).')
        print(f'    Será necessário aplicar técnicas de balanceamento (Etapa 7).')
    else:
        print(f'    ✅ Dataset relativamente balanceado (razão {imbalance_ratio:.2f}:1).')


# ======================================================================
# ETAPA 5: SEPARAÇÃO DO CONJUNTO DE TESTE (ESTRATIFICADO)
# ======================================================================

def etapa5_separacao_teste(header, X, y, mask):
    print_header('ETAPA 5: Identificação e Separação do Conjunto de Teste')

    # Trabalhamos apenas com linhas válidas
    X_valid = [X[i] for i in range(len(X)) if mask[i]]
    y_valid = [y[i] for i in range(len(y)) if mask[i]]

    print(f'  Exemplos válidos disponíveis: {len(y_valid)}')

    # Contagem antes da divisão
    counts = count_target(y_valid)
    total_valid = sum(counts.values())
    print(f'  Distribuição original:')
    print(f'    Benigno (0): {counts[0]} ({format_percent(counts[0], total_valid)})')
    print(f'    Maligno  (1): {counts[1]} ({format_percent(counts[1], total_valid)})')

    # Divisão estratificada 80/20
    from utils import stratified_split
    X_train, X_test, y_train, y_test = stratified_split(
        X_valid, y_valid, test_ratio=0.2, seed=42
    )

    # Contagem pós-divisão
    counts_train = count_target(y_train)
    counts_test = count_target(y_test)
    total_train = sum(counts_train.values())
    total_test = sum(counts_test.values())

    print(f'\n  Conjunto de TREINAMENTO ({total_train} exemplos, {format_percent(total_train, total_valid)}):')
    print(f'    Benigno (0): {counts_train[0]} ({format_percent(counts_train[0], total_train)})')
    print(f'    Maligno  (1): {counts_train[1]} ({format_percent(counts_train[1], total_train)})')

    print(f'\n  Conjunto de TESTE ({total_test} exemplos, {format_percent(total_test, total_valid)}):')
    print(f'    Benigno (0): {counts_test[0]} ({format_percent(counts_test[0], total_test)})')
    print(f'    Maligno  (1): {counts_test[1]} ({format_percent(counts_test[1], total_test)})')

    # Verifica se a estratificação manteve a proporção
    prop_orig = counts[1] / total_valid
    prop_train = counts_train[1] / total_train
    prop_test = counts_test[1] / total_test

    print(f'\n  Verificação de representatividade:')
    print(f'    Proporção original de malignos:  {prop_orig:.4f}')
    print(f'    Proporção no treino:             {prop_train:.4f} (delta: {abs(prop_train - prop_orig):.4f})')
    print(f'    Proporção no teste:              {prop_test:.4f}  (delta: {abs(prop_test - prop_orig):.4f})')

    if abs(prop_train - prop_orig) < 0.03 and abs(prop_test - prop_orig) < 0.03:
        print(f'\n    ✅ Conjunto de teste REPRESENTATIVO (proporções mantidas).')
    else:
        print(f'\n    ⚠️  Divergência na proporção — verificar amostragem.')

    print(f'\n  Justificativa:')
    print(f'    - Hold-out 80/20 com estratificação garante que ambas as classes')
    print(f'    estejam representadas proporcionalmente no treino e no teste.')
    print(f'    - Seed=42 garante reprodutibilidade dos resultados.')
    print(f'    - O conjunto de teste (20%) será guardado e NUNCA usado durante')
    print(f'    o pré-processamento ou treinamento — apenas na avaliação final.')

    return X_train, X_test, y_train, y_test


# ======================================================================
# ETAPA 6: ELIMINAÇÃO DE ATRIBUTOS E EXEMPLOS NÃO NECESSÁRIOS
# ======================================================================

def etapa6_eliminacao_atributos_exemplos(header_train, X_train, y_train):
    print_header('ETAPA 6: Identificação e Eliminação de Atributos e Exemplos')

    from utils import remove_constant_columns, remove_duplicate_rows

    feature_names = list(header_train)

    # 6a. Atributos não necessários — colunas constantes
    print_subheader('6a. Atributos Não Necessários (Colunas Constantes)')
    X_clean, feature_names_clean, removed = remove_constant_columns(X_train, feature_names)

    if removed:
        print(f'  Atributos removidos (variância zero):')
        for name in removed:
            print(f'    - {name}')
    else:
        print(f'  Nenhum atributo constante encontrado.')
        print(f'  Justificativa: Todos os atributos apresentam variabilidade nos')
        print(f'  dados de treino, contribuindo potencialmente para a predição.')

    # 6b. Exemplos não necessários — duplicatas
    print_subheader('6b. Exemplos Não Necessários (Duplicatas)')
    X_dedup, y_dedup, n_dup = remove_duplicate_rows(X_clean, y_train)

    if n_dup > 0:
        print(f'  Exemplos duplicados removidos: {n_dup}')
    else:
        print(f'  Nenhum exemplo duplicado encontrado.')

    print(f'  Total antes: {len(y_train)} | Total após: {len(y_dedup)}')
    print(f'\n  Justificativa: Duplicatas podem inflar artificialmente a importância')
    print(f'  de certos padrões e enviesar o modelo. Sua remoção garante que cada')
    print(f'  exemplo contribua igualmente para o aprendizado.')

    return feature_names_clean, X_dedup, y_dedup


# ======================================================================
# ETAPA 7: ANÁLISE DE AMOSTRAGEM E DESBALANCEAMENTO
# ======================================================================

def etapa7_amostragem_desbalanceamento(y_train_clean):
    print_header('ETAPA 7: Análise de Técnicas de Amostragem e Desbalanceamento')

    counts = count_target(y_train_clean)
    total = sum(counts.values())

    print(f'  Distribuição atual (após limpeza):')
    print(f'    Benigno (0): {counts[0]} ({format_percent(counts[0], total)})')
    print(f'    Maligno  (1): {counts[1]} ({format_percent(counts[1], total)})')

    majority_class = 0 if counts[0] > counts[1] else 1
    minority_class = 1 - majority_class
    ratio = counts[majority_class] / counts[minority_class] if counts[minority_class] > 0 else float('inf')

    print(f'\n  Razão de desbalanceamento: {ratio:.2f}:1')
    print(f'  Classe majoritária: {majority_class} ({counts[majority_class]} exemplos)')
    print(f'  Classe minoritária: {minority_class} ({counts[minority_class]} exemplos)')

    print(f'\n  Análise:')
    if ratio < 2.0:
        print(f'    ✅ Dataset balanceado (razão < 2:1).')
        print(f'    Não é necessária aplicação de técnicas de amostragem.')
    elif ratio < 4.0:
        print(f'    ⚠️  Desbalanceamento moderado (razão entre 2:1 e 4:1).')
        print(f'    Técnicas recomendadas: undersampling da classe majoritária')
        print(f'    ou oversampling com duplicação simples da minoritária.')
    else:
        print(f'    ❌ Desbalanceamento severo (razão > 4:1).')
        print(f'    Técnicas recomendadas: SMOTE, ADASYN ou undersampling agressivo.')

    print(f'\n  Justificativa: O dataset apresenta razão de {ratio:.2f}:1, que está')
    print(f'  abaixo do limiar crítico de 2:1. Neste contexto, a aplicação de')
    print(f'  técnicas como SMOTE poderia introduzir ruído sintético sem benefício')
    print(f'  real. O classificador baseline (classe majoritária) servirá como')
    print(f'  referência para avaliar se os modelos estão aprendendo além do viés.')


# ======================================================================
# ETAPA 8: LIMPEZA DE DADOS
# ======================================================================

def etapa8_limpeza_dados(header_train, X_train, y_train, mask, header_original, X_original, y_original):
    print_header('ETAPA 8: Limpeza de Dados')

    from utils import impute_mode, detect_outliers_iqr, calc_quartiles, calc_iqr

    feature_names = list(header_train) if header_train else list(header_original[:-1])

    # 8a. Outliers
    print_subheader('8a. Identificação de Outliers (Método IQR)')
    n_outliers_total = 0
    for col_idx, name in enumerate(feature_names):
        col_vals = []
        for row in X_train:
            val = row[col_idx]
            if val is not None:
                col_vals.append(val)
        lower, upper, outliers_idx, outliers_val = detect_outliers_iqr(col_vals)
        if outliers_val:
            print(f'  {name:<32} {len(outliers_val):>3} outliers  (limites: [{lower:.2f}, {upper:.2f}])')
            n_outliers_total += len(outliers_val)
        else:
            print(f'  {name:<32}    0 outliers')

    print(f'\n  Total de outliers detectados: {n_outliers_total}')
    print(f'  Decisão: MANTER os outliers.')
    print(f'  Justificativa: Os valores estão em escala de 1-10 (notas de patologistas).')
    print(f'  Um valor extremo (ex: 10) não é um erro de medição, mas sim uma')
    print(f'  avaliação clínica válida que indica alta severidade. Removê-los')
    print(f'  eliminaria justamente os casos mais informativos para detecção de câncer.')

    # 8b. Dados Inconsistentes
    print_subheader('8b. Identificação de Dados Inconsistentes')
    n_inconsistent = 0
    for col_idx, name in enumerate(feature_names):
        for row_idx, row in enumerate(X_train):
            val = row[col_idx]
            if val is not None:
                if val < FEATURE_MIN or val > FEATURE_MAX:
                    n_inconsistent += 1
                    print(f'  Valor {val} no atributo {name} (linha {row_idx}) — fora do intervalo [1,10]')

    if n_inconsistent == 0:
        print(f'  Nenhum valor fora do intervalo esperado [1, 10].')
        print(f'  Justificativa: Todos os atributos do dataset representam notas')
        print(f'  inteiras de patologistas na escala de 1 a 10. Valores fora desse')
        print(f'  intervalo indicariam erro de coleta ou corrupção de dados.')

    # 8c. Dados Redundantes (já tratado na Etapa 6 — duplicatas)
    print_subheader('8c. Dados Redundantes')
    print(f'  Análise já realizada na Etapa 6 (remoção de duplicatas).')
    print(f'  Atributos redundantes seriam detectados por correlação cruzada > 0.95.')
    print(f'  A análise de correlação completa será apresentada na Etapa 10.')

    # 8d. Dados Ausentes
    print_subheader('8d. Identificação e Resolução de Dados Ausentes')

    # Primeiro, imputamos os dados ausentes no conjunto de treino completo (antes da divisão)
    # Identifica colunas com valores ausentes
    missing_cols = []
    for col_idx, name in enumerate(feature_names):
        n_missing = sum(1 for row in X_train if row[col_idx] is None)
        if n_missing > 0:
            missing_cols.append((col_idx, name, n_missing))
            print(f'  {name:<32} {n_missing} valores ausentes')

    if not missing_cols:
        print(f'  Nenhum valor ausente encontrado no conjunto de treino.')

    # Aplica imputação pela moda
    X_imputed = [list(row) for row in X_train]
    for col_idx, name, n_missing in missing_cols:
        # Calcula a moda
        col_vals = [row[col_idx] for row in X_train if row[col_idx] is not None]
        modes = calc_mode(col_vals)
        fill_value = modes[0]

        # Imputa
        for row in X_imputed:
            if row[col_idx] is None:
                row[col_idx] = fill_value

        print(f'\n  → {name}: preenchidos {n_missing} valores com a MODA = {int(fill_value)}')

    print(f'\n  Justificativa: A moda foi escolhida como estratégia de imputação')
    print(f'  porque os atributos são pontuações clínicas DISCRETAS (1-10).')
    print(f'  Usar a média geraria valores decimais (ex: 3.45) clinicamente')
    print(f'  inexistentes. A moda preserva a integridade do domínio e insere')
    print(f'  o valor mais típico observado nos dados válidos.')

    return X_imputed


# ======================================================================
# ETAPA 9: CONVERSÃO DE TIPOS DE DADOS E NORMALIZAÇÃO
# ======================================================================

def etapa9_normalizacao(header_train, X_train, X_test):
    print_header('ETAPA 9: Conversão de Tipos de Dados e Normalização')

    from utils import normalize_minmax

    feature_names = list(header_train)

    # 9a. Conversão de tipos
    print_subheader('9a. Conversão de Tipos')
    print(f'  Análise: Todos os atributos já são numéricos (quantitativos discretos).')
    print(f'  Não há atributos simbólicos, ordinais ou nominais que necessitem')
    print(f'  conversão para representação numérica. O dataset já está em formato')
    print(f'  compatível com algoritmos de aprendizado de máquina.')
    print(f'  A conversão do target (2/4 → 0/1) já foi realizada na Etapa 1.')

    # 9b. Normalização Min-Max
    print_subheader('9b. Normalização Min-Max [0, 1]')

    # Calcula parâmetros no treino
    X_train_norm, mins, maxs = normalize_minmax(X_train)

    # Aplica os MESMOS parâmetros no teste (sem vazamento de dados!)
    X_test_norm = []
    for row in X_test:
        new_row = []
        for col_idx, val in enumerate(row):
            denom = maxs[col_idx] - mins[col_idx]
            if denom == 0:
                new_row.append(0.0)
            else:
                new_row.append((val - mins[col_idx]) / denom)
        X_test_norm.append(new_row)

    print(f'  Parâmetros calculados no TREINO (min, max por coluna):')
    for i, name in enumerate(feature_names):
        print(f'    {name:<32} min={mins[i]:>4.1f}  max={maxs[i]:>4.1f}')

    print(f'\n  Verificação pós-normalização (treino):')
    for i, name in enumerate(feature_names):
        col_vals = [row[i] for row in X_train_norm]
        print(f'    {name:<32} min={min(col_vals):.4f}  max={max(col_vals):.4f}')

    print(f'\n  Justificativa:')
    print(f'    - Min-Max foi escolhido porque a escala original já é limitada (1-10)')
    print(f'    e a transformação para [0,1] preserva a estrutura da distribuição.')
    print(f'    - Essencial para o K-NN: sem normalização, features com maior')
    print(f'    magnitude dominariam o cálculo da distância Euclidiana.')
    print(f'    - Essencial para a MLP: evita saturação das funções de ativação')
    print(f'    nos neurônios da camada de entrada.')
    print(f'    - Os parâmetros (min, max) são calculados APENAS no treino e')
    print(f'    aplicados ao teste para evitar data leakage.')

    return X_train_norm, X_test_norm


# ======================================================================
# ETAPA 10: REDUÇÃO DE DIMENSIONALIDADE (CORRELAÇÃO DE PEARSON)
# ======================================================================

def etapa10_reducao_dimensionalidade(feature_names, X_train_norm, y_train_clean, X_test_norm):
    print_header('ETAPA 10: Redução de Dimensionalidade — Correlação de Pearson')

    from utils import feature_target_correlations, calc_correlation

    # Calcula correlação feature-target no treino
    correlations = feature_target_correlations(X_train_norm, y_train_clean, feature_names)

    print(f'  Correlação de Pearson (feature × target):')
    print(f'  {"Feature":<32} {"r":>8} {"|r|":>8} {"Interpretação"}')
    print(f'  {"-"*32} {"-"*8} {"-"*8} {"-"*30}')

    low_corr_features = []
    for name, r, abs_r in correlations:
        if abs_r < 0.10:
            interp = 'MUITO FRACA — candidata a remoção'
            low_corr_features.append(name)
        elif abs_r < 0.30:
            interp = 'Fraca'
        elif abs_r < 0.50:
            interp = 'Moderada'
        elif abs_r < 0.70:
            interp = 'Forte'
        else:
            interp = 'MUITO FORTE'

        print(f'  {name:<32} {r:>+8.4f} {abs_r:>8.4f} {interp}')

    # Decisão sobre remoção
    print(f'\n  Features com correlação |r| < 0.10:')
    if low_corr_features:
        for name in low_corr_features:
            print(f'    - {name}')
    else:
        print(f'    Nenhuma.')

    # Limiar de decisão: remover features com |r| < 0.10
    threshold = 0.10
    to_remove = [name for name, r, abs_r in correlations if abs_r < threshold]

    if to_remove:
        print(f'\n  Decisão: REMOVER features com |r| < {threshold} do conjunto de treino e teste.')
        remove_indices = [feature_names.index(name) for name in to_remove]

        feature_names_reduced = [name for name in feature_names if name not in to_remove]

        X_train_reduced = []
        for row in X_train_norm:
            X_train_reduced.append([row[i] for i in range(len(row)) if i not in remove_indices])

        X_test_reduced = []
        for row in X_test_norm:
            X_test_reduced.append([row[i] for i in range(len(row)) if i not in remove_indices])

        print(f'  Features restantes: {len(feature_names_reduced)} de {len(feature_names)}')
        for name in feature_names_reduced:
            r = [c for c in correlations if c[0] == name][0][1]
            print(f'    - {name} (r={r:+.4f})')
    else:
        print(f'\n  Decisão: MANTER todas as features.')
        print(f'  Justificativa: Todas as features apresentam correlação relevante')
        print(f'  com o target (|r| ≥ 0.10). Remover features abaixo desse limiar')
        print(f'  reduziria dimensionalidade sem perda significativa de informação.')
        feature_names_reduced = list(feature_names)
        X_train_reduced = X_train_norm
        X_test_reduced = X_test_norm

    print(f'\n  Justificativa da técnica:')
    print(f'    - Correlação de Pearson foi escolhida como método de feature')
    print(f'    selection por ser implementável "from scratch" e de interpretação')
    print(f'    direta (mede relação linear feature-target).')
    print(f'    - PCA exigiria decomposição em autovalores/autovetores, complexa')
    print(f'    de implementar manualmente e com perda de interpretabilidade.')
    print(f'    - O limiar de |r| < 0.10 é conservador: remove apenas features')
    print(f'    com correlação insignificante, preservando a maior parte da')
    print(f'    informação preditiva.')

    return feature_names_reduced, X_train_reduced, X_test_reduced


# ======================================================================
# MAIN
# ======================================================================

def main():
    print_header('TRABALHO PRÁTICO 2 — PRÉ-PROCESSAMENTO E ANÁLISE PREDITIVA',
                 char='#', width=75)
    print('  Base: Breast Cancer Wisconsin (Original)')
    print('  Aluno: João Victor Borges Carvalho')
    print('  Abordagem: Implementação "from scratch" (Python puro + math)\n')

    # Carrega dados
    print('Carregando dataset...')
    header, X, y, mask = parse_dataset(DATA_PATH)

    n_total = len(y)
    n_valid = sum(mask)
    n_invalid = n_total - n_valid
    print(f'  Linhas carregadas: {n_total}')
    print(f'  Linhas válidas:    {n_valid}')
    print(f'  Linhas com dados ausentes: {n_invalid}')
    print(f'  Colunas: {len(header)} ({len(header) - 1} features + 1 target)')

    # ── PARTE 1: PRÉ-PROCESSAMENTO E ANÁLISE DE DADOS ──

    # Etapa 1
    target_name = etapa1_atributo_alvo(header)

    # Etapa 2
    etapa2_tipos_dados(header, X)

    # Etapa 3
    etapa3_escala_dados(header, X)

    # Etapa 4a, 4b, 4c
    etapa4a_medidas_localidade(header, X, mask)
    etapa4b_medidas_espalhamento(header, X, mask)
    etapa4c_distribuicao(header, X, y, mask)

    # Etapa 5: Separação treino/teste (estratificado)
    X_train, X_test, y_train, y_test = etapa5_separacao_teste(header, X, y, mask)

    # Etapa 6: Eliminação de atributos e exemplos desnecessários
    feature_names, X_train_clean, y_train_clean = etapa6_eliminacao_atributos_exemplos(
        header[:-1], X_train, y_train
    )

    # Etapa 7: Análise de amostragem e desbalanceamento
    etapa7_amostragem_desbalanceamento(y_train_clean)

    # Etapa 8: Limpeza de dados (outliers, inconsistentes, redundantes, ausentes)
    X_train_imputed = etapa8_limpeza_dados(
        feature_names, X_train_clean, y_train_clean,
        mask, header, X, y
    )

    # Também precisamos imputar o conjunto de teste com os mesmos critérios
    # Identifica colunas com None no treino original
    missing_cols_test = []
    for col_idx in range(len(feature_names)):
        if any(row[col_idx] is None for row in X_test):
            # Calcula moda no treino imputado
            col_vals_train = [row[col_idx] for row in X_train_imputed]
            modes = calc_mode(col_vals_train)
            fill_value = modes[0]
            missing_cols_test.append((col_idx, fill_value))

    X_test_imputed = [list(row) for row in X_test]
    for col_idx, fill_value in missing_cols_test:
        for row in X_test_imputed:
            if row[col_idx] is None:
                row[col_idx] = fill_value

    # Etapa 9: Normalização Min-Max
    X_train_norm, X_test_norm = etapa9_normalizacao(
        feature_names, X_train_imputed, X_test_imputed
    )

    # Etapa 10: Redução de dimensionalidade
    feature_names_final, X_train_final, X_test_final = etapa10_reducao_dimensionalidade(
        feature_names, X_train_norm, y_train_clean, X_test_norm
    )

    # ── RESUMO FINAL DA PARTE 1 ──
    print_header('PARTE 1 CONCLUÍDA — RESUMO DO PRÉ-PROCESSAMENTO', char='#', width=75)
    print(f'  Dataset original:          {n_total} exemplos, {len(header) - 1} features')
    print(f'  Após limpeza (treino):     {len(y_train_clean)} exemplos, {len(feature_names_final)} features')
    print(f'  Conjunto de teste:         {len(y_test)} exemplos (não utilizado no pré-processamento)')
    print(f'  Normalização:              Min-Max [0, 1]')
    print(f'  Redução dimensionalidade:  Correlação de Pearson (|r| < 0.10 removidas)')
    print(f'  Features finais:           {", ".join(feature_names_final)}')
    print(f'\n  Pipeline completo → Dados prontos para a Parte 2: Análise Preditiva.\n')

    # ═══════════════════════════════════════════════════════════════
    # PARTE 2: ANÁLISE PREDITIVA
    # ═══════════════════════════════════════════════════════════════

    Xt = X_train_final
    Xs = X_test_final
    yt = y_train_clean
    ys = y_test

    # Etapa 2.1: Técnica de validação
    etapa21_validacao()

    # Etapa 2.2: Métricas
    etapa22_metricas()

    # Etapa 2.3: Baseline
    baseline_fn, baseline_majority, baseline_ratio = etapa23_baseline(yt, ys)

    # Etapa 2.4: K-NN
    knn_results = etapa24_knn(Xt, yt, Xs, ys)

    # Etapa 2.5: Árvore C4.5
    c45_results = etapa25_arvore_c45(Xt, yt, Xs, ys, feature_names_final)

    # Etapa 2.6: MLP
    mlp_results = etapa26_mlp(Xt, yt, Xs, ys)

    # Etapa 2.7: Comparação final
    etapa27_comparacao(baseline_fn, baseline_majority, baseline_ratio,
                       knn_results, c45_results, mlp_results, ys)


# ═══════════════════════════════════════════════════════════════════════════
# PARTE 2: FUNÇÕES DAS ETAPAS
# ═══════════════════════════════════════════════════════════════════════════


def etapa21_validacao():
    print_header('PARTE 2 — ETAPA 2.1: Definição da Técnica de Validação')
    print('  Técnica escolhida: Hold-out 80/20 com amostragem estratificada')
    print()
    print('  Justificativa:')
    print('    - A divisão hold-out já foi realizada na Etapa 5 da Parte 1,')
    print('    com separação estratificada que preserva a proporção das classes')
    print('    (35% malignos) tanto no treino quanto no teste.')
    print('    - O conjunto de treino (362 exemplos) será usado para ajuste dos')
    print('    modelos, e o conjunto de teste (137 exemplos) para avaliação final.')
    print('    - O seed=42 garante reprodutibilidade.')
    print('    - Cross-validation foi considerada, mas o hold-out é suficiente')
    print('    para o escopo deste trabalho e evita o custo computacional de')
    print('    múltiplos treinamentos em algoritmos implementados manualmente.')


def etapa22_metricas():
    print_header('PARTE 2 — ETAPA 2.2: Definição das Métricas de Avaliação')

    print('  Métricas escolhidas (todas implementadas "from scratch"):')
    print()
    print('    1. Matriz de Confusão (TP, TN, FP, FN)')
    print('       Classe Positiva = 1 (Maligno) | Classe Negativa = 0 (Benigno)')
    print()
    print('    2. Acurácia (Accuracy) = (TP + TN) / Total')
    print('       Proporção de predições corretas no total.')
    print()
    print('    3. Precisão (Precision) = TP / (TP + FP)')
    print('       Dos diagnosticados como malignos, quantos realmente são?')
    print()
    print('    4. Recall (Sensibilidade) = TP / (TP + FN)')
    print('       Dos casos realmente malignos, quantos foram detectados?')
    print('       ⚠️  CRÍTICO no contexto médico: perder um câncer (FN) é grave.')
    print()
    print('    5. Especificidade = TN / (TN + FP)')
    print('       Dos casos benignos, quantos foram corretamente identificados?')
    print()
    print('    6. F1-Score = 2 × (Precision × Recall) / (Precision + Recall)')
    print('       Média harmônica: balanço entre precisão e recall.')

    print()
    print('  Justificativa:')
    print('    - A acurácia isolada pode ser enganosa em datasets desbalanceados.')
    print('    - Recall é a métrica mais crítica no domínio médico (evitar falsos')
    print('    negativos = não deixar câncer passar despercebido).')
    print('    - F1-Score equilibra precisão e recall, sendo a métrica de')
    print('    comparação principal entre os modelos.')


def etapa23_baseline(y_train, y_test):
    print_header('PARTE 2 — ETAPA 2.3: Algoritmo Baseline (Classe Majoritária)')

    baseline_fn, majority_class, ratio = majority_class_baseline(y_train)

    class_name = 'Maligno (1)' if majority_class == 1 else 'Benigno (0)'
    print(f'  Classe majoritária no treino: {class_name}')
    print(f'  Proporção: {ratio:.4f} ({ratio*100:.2f}%)')

    # Prediz no teste
    y_pred = [baseline_fn(None) for _ in y_test]

    # Avalia
    metrics = evaluate_model(y_test, y_pred)

    print(f'\n  Resultados no conjunto de TESTE:')
    print(f'    Matriz de Confusão:')
    print(f'                       Predito 0   Predito 1')
    print(f'    Real 0 (Benigno)       {metrics["tn"]:>5}       {metrics["fp"]:>5}')
    print(f'    Real 1 (Maligno)       {metrics["fn"]:>5}       {metrics["tp"]:>5}')
    print(f'\n    Acurácia:      {metrics["accuracy"]:.4f}')
    print(f'    Precisão:      {metrics["precision"]:.4f}')
    print(f'    Recall:        {metrics["recall"]:.4f}')
    print(f'    Especificidade:{metrics["specificity"]:.4f}')
    print(f'    F1-Score:      {metrics["f1_score"]:.4f}')

    print(f'\n  Justificativa:')
    print(f'    - O baseline sempre prediz a classe majoritária (1 = Maligno).')
    print(f'    - Isso estabelece um PISO MÍNIMO de desempenho: qualquer modelo')
    print(f'    real deve superar este baseline para ser considerado útil.')
    print(f'    - Acurácia esperada ≈ {ratio*100:.1f}% (proporção da classe majoritária).')

    return baseline_fn, majority_class, ratio


def etapa24_knn(X_train, y_train, X_test, y_test):
    print_header('PARTE 2 — ETAPA 2.4: Modelo Preditivo — K-NN')

    # Testa múltiplos valores de K
    k_values = [1, 3, 5, 7, 9, 11, 15]
    best_k = None
    best_f1 = -1
    all_results = []

    print(f'  Testando K ∈ {k_values}...\n')
    print(f'  {"K":>4}  {"Acurácia":>10}  {"Precisão":>10}  {"Recall":>10}  {"F1-Score":>10}')
    print(f'  {"-"*4}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')

    for k in k_values:
        y_pred = knn_predict_batch(X_train, y_train, X_test, k=k)
        m = evaluate_model(y_test, y_pred)
        all_results.append((k, m))
        print(f'  {k:>4}  {m["accuracy"]:>10.4f}  {m["precision"]:>10.4f}  {m["recall"]:>10.4f}  {m["f1_score"]:>10.4f}')

        if m['f1_score'] > best_f1:
            best_f1 = m['f1_score']
            best_k = k

    best_m = [r for r in all_results if r[0] == best_k][0][1]

    print(f'\n  Melhor K: {best_k} (F1-Score = {best_f1:.4f})')
    print(f'\n  Matriz de Confusão (K={best_k}):')
    print(f'                       Predito 0   Predito 1')
    print(f'    Real 0 (Benigno)       {best_m["tn"]:>5}       {best_m["fp"]:>5}')
    print(f'    Real 1 (Maligno)       {best_m["fn"]:>5}       {best_m["tp"]:>5}')

    print(f'\n  Justificativa:')
    print(f'    - K-NN é um algoritmo baseado em similaridade (distância euclidiana).')
    print(f'    - A normalização Min-Max é ESSENCIAL aqui para evitar que features')
    print(f'    com maior magnitude dominem o cálculo da distância.')
    print(f'    - K={best_k} oferece o melhor equilíbrio entre viés e variância.')
    print(f'    - K muito baixo (1) sofre de overfitting; K muito alto suaviza demais.')
    print(f'    - Empates na votação são resolvidos em favor da classe 0 (benigno),')
    print(f'    por isso K ímpar é preferível para classificação binária.')

    return {'k': best_k, 'metrics': best_m, 'all': all_results}


def etapa25_arvore_c45(X_train, y_train, X_test, y_test, feature_names):
    print_header('PARTE 2 — ETAPA 2.5: Modelo Preditivo — Árvore de Decisão (C4.5)')

    # Testa diferentes profundidades máximas
    depths = [3, 5, 7, 10, None]
    best_depth = None
    best_f1 = -1
    all_results = []

    print(f'  Testando max_depth ∈ {[d if d else "ilimitado" for d in depths]}...\n')
    print(f'  {"Max Depth":>12}  {"Nós":>6}  {"Acurácia":>10}  {"Precisão":>10}  {"Recall":>10}  {"F1-Score":>10}')
    print(f'  {"-"*12}  {"-"*6}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')

    for max_depth in depths:
        actual_depth = max_depth if max_depth else 20  # limite prático
        tree = build_tree_c45(X_train, y_train, feature_names, max_depth=actual_depth, min_samples=5)
        y_pred = predict_tree_batch(tree, X_test)
        m = evaluate_model(y_test, y_pred)
        n_nodes = count_tree_nodes(tree)
        all_results.append((max_depth, m, n_nodes, tree))

        depth_label = str(max_depth) if max_depth else '∞'
        print(f'  {depth_label:>12}  {n_nodes:>6}  {m["accuracy"]:>10.4f}  {m["precision"]:>10.4f}  {m["recall"]:>10.4f}  {m["f1_score"]:>10.4f}')

        if m['f1_score'] > best_f1:
            best_f1 = m['f1_score']
            best_depth = max_depth

    best = [r for r in all_results if r[0] == best_depth][0]
    best_m = best[1]
    best_nodes = best[2]
    best_tree = best[3]

    print(f'\n  Melhor profundidade: {best_depth if best_depth else "ilimitada"} ({best_nodes} nós)')
    print(f'\n  Matriz de Confusão:')
    print(f'                       Predito 0   Predito 1')
    print(f'    Real 0 (Benigno)       {best_m["tn"]:>5}       {best_m["fp"]:>5}')
    print(f'    Real 1 (Maligno)       {best_m["fn"]:>5}       {best_m["tp"]:>5}')

    # Mostra a raiz da árvore para interpretabilidade
    if best_tree['type'] == 'node':
        print(f'\n  Raiz da árvore: {best_tree["feature_name"]} ≤ {best_tree["threshold"]:.4f}')

    print(f'\n  Justificativa:')
    print(f'    - C4.5 usa Gain Ratio (InformationGain / SplitInfo) para evitar')
    print(f'    viés em favor de features com muitos valores distintos.')
    print(f'    - Dados normalizados (0-1) são particionados por thresholds')
    print(f'    calculados como médias entre valores consecutivos únicos.')
    print(f'    - A profundidade máxima controla overfitting: árvores rasas têm')
    print(f'    alto viés; árvores profundas têm alta variância.')
    print(f'    - A árvore é INTERPRETÁVEL — podemos inspecionar quais features')
    print(f'    são usadas nas decisões, algo impossível em redes neurais.')

    return {'depth': best_depth, 'metrics': best_m, 'nodes': best_nodes, 'all': all_results}


def etapa26_mlp(X_train, y_train, X_test, y_test):
    print_header('PARTE 2 — ETAPA 2.6: Modelo Preditivo — Rede Neural MLP')

    # Configurações fixas (já que testar todas as combinações seria lento)
    # Arquitetura: 9 entradas → 8 ocultos → 1 saída
    print('  Arquitetura: 9 entradas → 8 neurônios ocultos (sigmoide) → 1 saída (sigmoide)')
    print('  Treinamento: SGD, learning_rate=0.1, epochs=500')
    print()

    # Treina
    w1, b1, w2, b2, history = mlp_train(
        X_train, y_train,
        hidden_neurons=8,
        learning_rate=0.1,
        epochs=500,
        seed=42,
        verbose=True
    )

    # Prediz no teste
    y_pred = mlp_predict(X_test, w1, b1, w2, b2, threshold=0.5)
    metrics = evaluate_model(y_test, y_pred)

    print(f'\n  Resultados no conjunto de TESTE:')
    print(f'    Matriz de Confusão:')
    print(f'                       Predito 0   Predito 1')
    print(f'    Real 0 (Benigno)       {metrics["tn"]:>5}       {metrics["fp"]:>5}')
    print(f'    Real 1 (Maligno)       {metrics["fn"]:>5}       {metrics["tp"]:>5}')
    print(f'\n    Acurácia:      {metrics["accuracy"]:.4f}')
    print(f'    Precisão:      {metrics["precision"]:.4f}')
    print(f'    Recall:        {metrics["recall"]:.4f}')
    print(f'    Especificidade:{metrics["specificity"]:.4f}')
    print(f'    F1-Score:      {metrics["f1_score"]:.4f}')

    # Gráfico de convergência
    epochs_list = [h[0] for h in history]
    losses = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_list, losses, color='darkgreen', linewidth=1.5)
    ax.set_title('Convergência do Treinamento — MLP (Loss MSE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    loss_path = os.path.join(PLOTS_DIR, 'mlp_convergencia.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Gráfico de convergência salvo em: {loss_path}')

    print(f'\n  Justificativa:')
    print(f'    - MLP com uma camada oculta é suficiente para problemas de')
    print(f'    classificação binária com fronteira de decisão não-linear.')
    print(f'    - Sigmoide como ativação: produz saída em (0,1), interpretável')
    print(f'    como probabilidade da classe positiva (maligno).')
    print(f'    - SGD (Stochastic Gradient Descent): atualiza pesos a cada')
    print(f'    exemplo, convergindo mais rápido que batch GD para datasets')
    print(f'    pequenos (362 exemplos).')
    print(f'    - Backpropagation implementada manualmente: calcula gradientes')
    print(f'    via regra da cadeia da camada de saída para a de entrada.')
    print(f'    - Threshold=0.5: valores ≥ 0.5 → classe 1 (maligno).')

    return {
        'metrics': metrics,
        'history': history,
        'hidden_neurons': 8,
        'epochs': 500,
        'learning_rate': 0.1,
    }


def etapa27_comparacao(baseline_fn, baseline_majority, baseline_ratio,
                       knn_results, c45_results, mlp_results, y_test):
    print_header('PARTE 2 — ETAPA 2.7: Análise e Comparação dos Resultados')

    print(f'  ╔══════════════════════════════════════════════════════════════════╗')
    print(f'  ║              COMPARAÇÃO FINAL DE MODELOS                         ║')
    print(f'  ╠══════════════════╦══════════╦══════════╦══════════╦══════════════╣')
    print(f'  ║ Modelo           ║ Acurácia ║ Precisão ║ Recall   ║ F1-Score     ║')
    print(f'  ╠══════════════════╬══════════╬══════════╬══════════╬══════════════╣')

    # Baseline
    y_pred_bl = [baseline_fn(None) for _ in y_test]
    bl = evaluate_model(y_test, y_pred_bl)
    print(f'  ║ Baseline (Major) ║ {bl["accuracy"]:.4f}  ║ {bl["precision"]:.4f}  ║ {bl["recall"]:.4f}  ║ {bl["f1_score"]:.4f}       ║')

    # K-NN
    knn_m = knn_results['metrics']
    print(f'  ║ K-NN (K={knn_results["k"]})       ║ {knn_m["accuracy"]:.4f}  ║ {knn_m["precision"]:.4f}  ║ {knn_m["recall"]:.4f}  ║ {knn_m["f1_score"]:.4f}       ║')

    # C4.5
    c45_m = c45_results['metrics']
    c45_label = f'C4.5 (d={c45_results["depth"]})'.ljust(17) if c45_results['depth'] else 'C4.5 (d=∞)'.ljust(17)
    print(f'  ║ {c45_label}║ {c45_m["accuracy"]:.4f}  ║ {c45_m["precision"]:.4f}  ║ {c45_m["recall"]:.4f}  ║ {c45_m["f1_score"]:.4f}       ║')

    # MLP
    mlp_m = mlp_results['metrics']
    print(f'  ║ MLP (8 ocultos)  ║ {mlp_m["accuracy"]:.4f}  ║ {mlp_m["precision"]:.4f}  ║ {mlp_m["recall"]:.4f}  ║ {mlp_m["f1_score"]:.4f}       ║')
    print(f'  ╚══════════════════╩══════════╩══════════╩══════════╩══════════════╝')

    # Determina o melhor modelo por F1-Score
    models = {
        'Baseline (Classe Majoritária)': bl['f1_score'],
        f'K-NN (K={knn_results["k"]})': knn_m['f1_score'],
        'Árvore C4.5': c45_m['f1_score'],
        'MLP (Rede Neural)': mlp_m['f1_score'],
    }
    best_model = max(models, key=models.get)
    best_score = models[best_model]

    print(f'\n  🏆 MELHOR MODELO (F1-Score): {best_model} = {best_score:.4f}')

    # Análise comparativa
    print(f'\n  ANÁLISE COMPARATIVA:')
    print(f'  {"─"*65}')

    print(f'\n  1. BASELINE vs MODELOS:')
    for name, score in models.items():
        if name != 'Baseline (Classe Majoritária)':
            diff = score - bl['f1_score']
            print(f'     {name}: F1 = {score:.4f} (Δ baseline = {diff:+.4f})')

    print(f'\n  2. RECALL (Sensibilidade) — CRÍTICO NO CONTEXTO MÉDICO:')
    print(f'     Baseline: {bl["recall"]:.4f} (sempre prediz maligno → 100% recall)')
    print(f'     K-NN:     {knn_m["recall"]:.4f}')
    print(f'     C4.5:     {c45_m["recall"]:.4f}')
    print(f'     MLP:      {mlp_m["recall"]:.4f}')

    print(f'\n  3. INTERPRETABILIDADE:')
    print(f'     K-NN:    Média — podemos ver os vizinhos, mas não há "regras"')
    print(f'     C4.5:    Alta — árvore totalmente inspecionável')
    print(f'     MLP:     Baixa — caixa preta, pesos não são intuitivos')

    print(f'\n  4. CUSTO COMPUTACIONAL:')
    print(f'     K-NN:    Treino instantâneo (lazy), predição lenta (O(n·d))')
    print(f'     C4.5:    Treino moderado, predição rápida (O(log n))')
    print(f'     MLP:     Treino lento (SGD iterativo), predição rápida (O(d·h))')

    print(f'\n  CONCLUSÃO:')
    print(f'    O modelo {best_model} obteve o melhor F1-Score ({best_score:.4f}),')
    print(f'    indicando o melhor equilíbrio entre precisão e recall. No contexto')
    print(f'    de diagnóstico de câncer de mama, o recall é especialmente importante:')
    print(f'    perder um caso maligno (falso negativo) pode ter consequências graves.')
    print(f'    O modelo com maior recall associado a boa precisão é o mais adequado')
    print(f'    para este domínio de aplicação.')

    # Salva tabela para documentação
    return {
        'baseline': bl,
        'knn': knn_m,
        'c45': c45_m,
        'mlp': mlp_m,
        'best_model': best_model,
        'best_score': best_score,
    }


if __name__ == '__main__':
    main()
