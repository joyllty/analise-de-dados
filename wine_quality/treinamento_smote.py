import json
import os
import pickle as pkl
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from avaliador import avaliacao_cross_validation, comparar_modelos


# ======= paths =======
PATH_TINTO = './dados/winequality-red.csv'
PATH_BRANCO = './dados/winequality-white.csv'
PATH_NOMES = './dados/winequality.names'
PATH_DADOS_UNIFICADOS = './dados/winequality_red_white_unificado.csv'

PATH_MODELO = './modelos_salvos/modelo_wine_quality.pkl'
PATH_COLUNAS = './modelos_salvos/colunas_modelo.json'

RANDOM_STATE = 42
N_ITER_RANDOM_SEARCH = 20
CV_FOLDS = 5


# ======= carregamento e pré-processamento =======
def carregar_dados(path_tinto, path_branco):
    '''
    Lê os dois arquivos CSV, adiciona a coluna tipo_vinho e junta tudo em um único dataframe.

    tipo_vinho:
    0 = vinho tinto
    1 = vinho branco
    '''

    dados_tinto = pd.read_csv(path_tinto, sep=';')
    dados_branco = pd.read_csv(path_branco, sep=';')

    dados_tinto['tipo_vinho'] = 0
    dados_branco['tipo_vinho'] = 1

    dados = pd.concat([dados_tinto, dados_branco], ignore_index=True)

    return dados


def salvar_dados_unificados(dados, caminho):
    '''
    salva o CSV final contendo os dados dos vinhos tintos e brancos juntos
    '''

    dados.to_csv(caminho, index=False)


def separar_atributos_classe(dados):
    '''
    separa os atributos da coluna alvos
    '''

    classes = dados['quality']
    atributos = dados.drop(columns=['quality'])

    return atributos, classes


def salvar_modelo(modelo, caminho):
    '''
    salva o melhor modelo treinado em .pkl
    '''

    with open(caminho, 'wb') as arquivo:
        pkl.dump(modelo, arquivo)


def salvar_colunas(colunas, caminho):
    '''
    salva os nomes das colunas usadas no treinamento
    '''

    with open(caminho, 'w', encoding='utf-8') as arquivo:
        json.dump(list(colunas), arquivo, indent=4, ensure_ascii=False)


# ======= metaestimadores e grids =======
def criar_metaestimadores():
    '''
    cria os três metaestimadores usados na atividade

    os três modelos escolhidos são baseados em árvores de decisão, porque a base Wine Quality
    é composta principalmente por atributos numéricos físico-químicos dos vinhos e modelos baseados 
    em árvores são ótimos para esse tipo de dado porque conseguem capturar relações não lineares entre os atributos
    
    Random Forest:
    foi priorizado porque é um ensemble robusto. combina várias árvores de decisão treinadas com amostras
    diferentes dos dados, reduzindo overfitting e melhorando a capacidade de generalização

    Extra Trees:
    também é um ensemble de árvores, parecido com a Random Forest mas adiciona mais aleatoriedade na criação 
    das árvores. serviu como comparação direta com a Random Forest

    HistGradientBoosting:
    modelo de boosting do próprio scikit-learn. diferente da Random Forest e Extra Trees, que treinam várias árvores
    de forma mais independente, o boosting constrói árvores sequencialmente, tentando corrigir os erros 
    cometidos pelas árvores anteriores

    '''

    rf_grid = {
        'smote__k_neighbors': [2, 3],
        'classificador__n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'classificador__criterion': ['gini', 'entropy'],
        'classificador__min_samples_split':[int(x) for x in np.linspace(start=2, stop=10, num=2)],
        'classificador__max_depth':  [int(x) for x in np.linspace(start=5, stop=50, num=10)] + [None],
        'classificador__max_features':  ['sqrt', 'log2']
    }

    extra_trees_grid = {
        'smote__k_neighbors': [2, 3],
        'classificador__n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'classificador__criterion': ['gini', 'entropy'],
        'classificador__min_samples_split':[2, 5, 10],
        'classificador__min_samples_leaf': [1, 2, 4],
        'classificador__max_depth': [int(x) for x in np.linspace(start=5, stop=50, num=10)] + [None],
        'classificador__max_features': ['sqrt', 'log2']
    }

    hgb_grid = {
        'smote__k_neighbors': [2, 3],
        'classificador__max_iter': [50, 100, 150, 200],
        'classificador__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classificador__max_leaf_nodes': [15, 31, 63],
        'classificador__max_depth': [None, 5, 10, 20],
        'classificador__min_samples_leaf': [10, 20, 30],
        'classificador__l2_regularization': [0.0, 0.01, 0.1]
    }

    modelos = {
        'Random Forest': {
            'modelo': Pipeline([
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classificador', RandomForestClassifier(random_state=RANDOM_STATE))
            ]),
            'parametros': rf_grid
        },

        'Extra Trees': {
            'modelo': Pipeline([
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classificador', ExtraTreesClassifier(random_state=RANDOM_STATE))
            ]),
            'parametros': extra_trees_grid
        },

        'HistGradientBoosting': {
            'modelo': Pipeline([
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classificador', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
            ]),
            'parametros': hgb_grid
        }
    }

    return modelos


# ======= hiperparametrização =======
def hiperparametrizar_modelo(modelo, parametros, atributos, classes, nome_modelo):
    '''
    busca os melhores hiperparâmetros usando RandomizedSearchCV
    '''

    print('\n================================')
    print(f'>> Hiperparametrização - {nome_modelo}')
    print('================================')

    search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=parametros,
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring='f1_weighted',
        cv=CV_FOLDS,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(atributos, classes)

    print('\n>> Melhores parâmetros:')
    pprint(search.best_params_)

    return search.best_estimator_, search.best_params_


# ======= treinamento =======
def treinar_modelos(atributos, classes):
    '''
    hiperparametriza os três modelos com RandomizedSearchCV e depois avalia cada um com 
    cross_validate no módulo avaliador
    '''

    modelos = criar_metaestimadores()
    resultados = []
    modelos_treinados = {}

    for nome_modelo, config in modelos.items():
        modelo, parametros = hiperparametrizar_modelo(
            config['modelo'],
            config['parametros'],
            atributos,
            classes,
            nome_modelo
        )

        resultado = avaliacao_cross_validation(
            modelo,
            atributos,
            classes,
            nome_modelo,
            CV_FOLDS
        )

        resultados.append(resultado)
        modelos_treinados[nome_modelo] = modelo

    return resultados, modelos_treinados


# ======= main =======
def main():
    '''
    foi escolhida a abordagem sem holdout e com cross validation
    '''

    os.makedirs('./modelos_salvos', exist_ok=True)
    os.makedirs('./resultados', exist_ok=True)

    dados = carregar_dados(PATH_TINTO, PATH_BRANCO)

    # os dados são embaralhados antes da cross validate para evitar que os folds ficassem enviesados pela ordem dos arquivos    
    dados = dados.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    salvar_dados_unificados(dados, PATH_DADOS_UNIFICADOS)

    atributos, classes = separar_atributos_classe(dados)

    print('\n====== COLUNAS DO DATASET ======')
    print(dados.columns)

    print('\n====== FREQUÊNCIA DAS CLASSES ======')
    print(Counter(classes))

    # todos os modelos escolhidos são baseados em árvores por isso os dados não foram normalizados
    resultados, modelos_treinados, melhores_parametros = treinar_modelos(atributos, classes)

    melhor_modelo_nome = comparar_modelos(resultados)

    melhor_modelo = modelos_treinados[melhor_modelo_nome]

    salvar_modelo(melhor_modelo, PATH_MODELO)
    salvar_colunas(atributos.columns, PATH_COLUNAS)


if __name__ == '__main__':
    main()