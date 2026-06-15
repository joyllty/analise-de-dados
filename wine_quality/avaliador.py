import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def avaliacao_cross_validation(modelo, atributos, classes, nome_modelo, cv):
    '''
    avalia o modelo com cross validation
    '''

    scoring = ['accuracy',
            'precision_weighted',
            'recall_weighted',
            'f1_weighted',
            'f1_macro']

    score_cross = cross_validate(
        modelo,
        atributos,
        classes,
        scoring=scoring,
        cv=cv,
        verbose=0,
        n_jobs=1
    )

    matriz_confusao, metricas_classes = gerar_matriz_confusao(
    modelo,
    atributos,
    classes,
    cv
    )

    resultados = {
        'modelo': nome_modelo,
        'accuracy': score_cross['test_accuracy'].mean(),
        'precision_weighted': score_cross['test_precision_weighted'].mean(),
        'recall_weighted': score_cross['test_recall_weighted'].mean(),
        'f1_weighted': score_cross['test_f1_weighted'].mean(),
        'f1_macro': score_cross['test_f1_macro'].mean(),
        'matriz_confusao': matriz_confusao,
        'metricas_classes': metricas_classes
    }

    print('\n================================')
    print(f'Cross Validation - {nome_modelo}')
    print('================================')
    print(f">> Accuracy:           {resultados['accuracy']:.4f}")
    print(f">> Precision weighted: {resultados['precision_weighted']:.4f}")
    print(f">> Recall weighted:    {resultados['recall_weighted']:.4f}")
    print(f">> F1 weighted:        {resultados['f1_weighted']:.4f}")
    print(f">> F1 macro:           {resultados['f1_macro']:.4f}")

    print('\n====== MATRIZ DE CONFUSÃO ======')
    print(matriz_confusao)

    print('\n====== MÉTRICAS POR CLASSE ======')

    for _, linha in metricas_classes.iterrows():
        print(
            f"classe quality {int(linha['quality'])} -> "
            f"suporte: {int(linha['suporte'])}, "
            f"acertos: {int(linha['acertos'])}, "
            f"acurácia/recall: {linha['acuracia_classe']:.4f}, "
            f"precision: {linha['precision']:.4f}, "
            f"f1: {linha['f1_score']:.4f}"
        )
    return resultados


def gerar_matriz_confusao(modelo, atributos, classes, cv):
    '''
    gera a matriz de confusão usando predições geradas por cross validation

    também calcula as métricas por classe:

    a acurácia por classe calculada pela matriz de confusão é equivalente ao recall da classe.
    '''

    predicoes = cross_val_predict(
        modelo,
        atributos,
        classes,
        cv=cv,
        n_jobs=1
    )

    rotulos = sorted(classes.unique())

    matriz = confusion_matrix(
        classes,
        predicoes,
        labels=rotulos
    )

    df_matriz = pd.DataFrame(
        matriz,
        index=[f'Real {classe}' for classe in rotulos],
        columns=[f'Previsto {classe}' for classe in rotulos]
    )

    precision, recall, f1, suporte = precision_recall_fscore_support(
        classes,
        predicoes,
        labels=rotulos,
        zero_division=0
    )

    metricas_classes = []

    for i, classe in enumerate(rotulos):
        total_classe = matriz[i].sum()
        acertos_classe = matriz[i][i]

        if total_classe == 0:
            acuracia = 0
        else:
            acuracia = acertos_classe / total_classe

        metricas_classes.append({
            'quality': classe,
            'suporte': total_classe,
            'acertos': acertos_classe,
            'acuracia_classe': acuracia,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i]
        })

    df_metricas_classes = pd.DataFrame(metricas_classes)

    return df_matriz, df_metricas_classes

def comparar_modelos(resultados_list):
    '''
    compara os modelos e seleciona o melhor pela accuracy
    '''

    df_resultados = pd.DataFrame(resultados_list)
    df_resultados = df_resultados.set_index('modelo')

    metricas = [
        'accuracy',
        'precision_weighted',
        'recall_weighted',
        'f1_weighted',
        'f1_macro'
    ]

    print('\n======================================')
    print('>> COMPARAÇÃO DOS MODELOS')
    print(df_resultados[metricas].round(4))

    melhor_modelo = df_resultados['accuracy'].idxmax()

    print('\n======================================')
    print('RECOMENDAÇÃO')

    print(f'>> O modelo mais adequado considerando a maior acurácia é: {melhor_modelo}')

    return df_resultados, melhor_modelo
