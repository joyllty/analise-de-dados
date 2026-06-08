import pandas as pd
import numpy as np
import pickle as pkl
from pprint import pprint
from collections import Counter

from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate

# ======= carregamento e balanceamento da base =======

def carregar_dados():
    '''
    carrega a base Bank Marketing do repositório UCI

    a base possui dados de campanhas de marketing feitas por telefone, o objetivo é prever se o cliente 
    vai assinar ou não um depósito a prazo

    '''
    bank_marketing = fetch_ucirepo(id=222)

    dados_atributos = bank_marketing.data.features

    dados_classe = bank_marketing.data.targets

    dados_classe = dados_classe['y']

    # junta atributos e classe apenas para visualização
    dados = pd.concat([dados_atributos, dados_classe], axis=1)

    return dados, dados_atributos, dados_classe


def balancear_dados(dados_atributos, dados_classe):
    '''
    balanceia os dados com SMOTE
    '''

    resampler = SMOTE(random_state=42)

    atributos_b, classes_b = resampler.fit_resample(dados_atributos, dados_classe)

    print(Counter(classes_b))

    return atributos_b, classes_b


# ======= tratamento dos dados =======

def tratar_dados(dados_atributos, dados_classe):
    # converte colunas categóricas em colunas numéricas
    atributos_tratados = pd.get_dummies(dados_atributos, dtype=int)

    # converte a classe de texto para número
    classes_tratadas = dados_classe.map({
        'no': 0,
        'yes': 1
    })

    return atributos_tratados, classes_tratadas


# ======= hiperparametrização =======

def hiperparametrizar_random_forest(atributos, classes):
    '''
    busca os melhores hiperparâmetros para a Random Forest

    '''

    # domínio de valores que serão testados
    rf_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=200, num=4)],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'max_depth': [10, 20, 30, None],
        'max_features': ['sqrt', 'log2']
    }

    # modelo base
    rf = RandomForestClassifier(random_state=42)

    # busca aleatória pelos melhores hiperparâmetros
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_grid,
        n_iter=10,
        cv=5,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    search.fit(atributos, classes)

    print("\n>> Melhores parâmetros:")
    pprint(search.best_params_)

    return search.best_params_


# ======= treinamento =======

def treinar_random_forest(atributos, classes, parametros):
    '''
    treina o modelo Random Forest usando os melhores parâmetros
    '''

    modelo_rf = RandomForestClassifier(
        **parametros,
        random_state=42
    )

    modelo_rf.fit(atributos, classes)

    return modelo_rf    

def avaliar_modelo(modelo, atributos, classes):
    '''
    avalia o modelo usando Cross Validation
    '''

    scoring = [
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'accuracy'
    ]

    score_cross = cross_validate(
        modelo,
        atributos,
        classes,
        scoring=scoring,
        cv=10,
        verbose=0,
        n_jobs=-1
    )

    resultados = {
        'precision': score_cross['test_precision_macro'].mean(),
        'recall': score_cross['test_recall_macro'].mean(),
        'f1_score': score_cross['test_f1_macro'].mean(),
        'accuracy': score_cross['test_accuracy'].mean()
    }

    print('\n====== RESULTADOS COM CROSS VALIDATION ======')
    print(f'Precision: {resultados["precision"]:.4f}')
    print(f'Recall:    {resultados["recall"]:.4f}')
    print(f'F1-score:  {resultados["f1_score"]:.4f}')
    print(f'Accuracy:  {resultados["accuracy"]:.4f}')

    return resultados


def salvar_modelo(modelo, caminho):
    pkl.dump(modelo, open(caminho, 'wb'))


def main():
    dados, dados_atributos, dados_classe = carregar_dados()
    #print(dados)
    #print(dados_classe)

    #print(Counter(dados_classe))  # 0: 39922 / 1: 5289
    #print(Counter(dados_atributos))

    atributos_tratados, classes_tratadas = tratar_dados(dados_atributos, dados_classe)
    #print(atributos_tratados)

    atributos_b, classes_b = balancear_dados(atributos_tratados, classes_tratadas)
    #print(atributos_b)

    # ======== encontrar os melhores parametros pra rf ========
    parametros_rf = hiperparametrizar_random_forest(atributos_b, classes_b)

    # treinamento usando os dados balanceados
    modelo_rf = treinar_random_forest(atributos_b, classes_b, parametros_rf)

    # avaliação do modelo com cross validation
    avaliar_modelo(modelo_rf, atributos_b, classes_b)

    salvar_modelo(modelo_rf, 'modelo_bank.pkl')

if __name__ == '__main__':
    main()