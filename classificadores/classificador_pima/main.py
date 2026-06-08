import pandas as pd
import numpy as np
import pickle as pkl
from collections import Counter
from pprint import pprint

# from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 


# ======= carregamento e pré-processamento =======
def carregar_dados(path):
    '''
    lê o csv e separa os atributos da coluna de classe
    '''
    dados = pd.read_csv(path, sep = ',')
    dados_classe = dados['Outcome']
    dados_atributos = dados.drop(columns=['Outcome'])

    return dados, dados_classe, dados_atributos


def balancear_dados(dados_atributos, dados_classe):
    '''
    balancear os dados com SMOTE.
    cria novos exemplos sintéticos da classe com menos dados
    '''
    # objeto SMOTE - balanceador
    resampler = SMOTE(random_state=42) 

    # executa o balanceamento
    atributos_b, classes_b = resampler.fit_resample(dados_atributos, dados_classe)

    print(atributos_b)
    print(classes_b)

    print('\n====== FREQUENCIA DAS CLASSES APÓS BALANCEAMENTO ======')

    # conta quantos exemplos existem de cada classe depois do smote
    class_count = Counter(classes_b)
    print(class_count) # 0: 500 / 1: 500

    return atributos_b, classes_b, 

'''  
def segmentar_dados(atributos_b, classes_b, test_size=0.3):
    
    divide dados em treino e teste (hold-out)
    
    return train_test_split(atributos_b, classes_b, test_size=test_size, random_state=42)
'''
 
def salvar_modelo(modelo, caminho: str):
    pkl.dump(modelo, open(caminho, 'wb'))
 
def normalizar_dados(atributos_b):
    '''
    normaliza os dados com StandardScaler
    '''
    scaler = StandardScaler()
    # ===== hold out nao utilizado =====
    #atributos_train_norm = scaler.fit_transform(atributos_train)

    # transforma com o fit treinado dos dados de treino, pra nao perder a escala
    #atributos_test_norm = scaler.transform(atributos_test) 
    
    atributos_norm = scaler.fit_transform(atributos_b)
    return atributos_norm, scaler


# ======= hiperparametrização =======

def hiperparametrizar_random_forest(atributos_b, classe_b):
    '''
    busca os melhores hiperparâmetros para a Random Forest com RandomizedSearchCV
    random forest - multiplas arvores de decisao

    - n_estimators: quantidade de árvores na floresta, mais árvores = mais estável, mas mais lento para treinar
    - criterion: critério de divisão dos nós. gini mede impureza, entropy ganho de informação
    - min_samples_split: mínimo de exemplos para dividir um nó
    - max_depth: profundidade máxima de cada árvore. limitar evita overfitting
    - max_features: quantos atributos considerar em cada divisão. 'sqrt' e 'log2' introduzem aleatoriedade, aumentando a diversidade entre as árvores
 
    cv=5: cada combinação é testada com 5-fold cross validation internamente
    n_iter=20: testa 20 combinações 
    '''

    rf_grid = {
        'n_estimators':     [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'criterion':        ['gini', 'entropy'],
        'min_samples_split':[int(x) for x in np.linspace(start=2, stop=10, num=2)],
        'max_depth':        [int(x) for x in np.linspace(start=5, stop=50, num=10)] + [None],
        'max_features':     ['sqrt', 'log2'],
    }
    rf = RandomForestClassifier(random_state=42)

    search = RandomizedSearchCV(rf, rf_grid, n_iter=20, cv=5, verbose=1, n_jobs=-1, random_state=42)
    search.fit(atributos_b, classe_b)

    print("\n>> Melhores parâmetros:")
    pprint(search.best_params_)

    return search.best_params_
 

def hiperparametrizar_svm(atributos_norm, classes_b):
    '''
    busca os melhores hiperparâmetros para o SVM com RandomizedSearchCV 
    '''

    svm_grid = {
        'C':      [0.01, 0.1, 1, 10, 100], # regularização do modelo, evita overfitting
        'kernel': ['rbf', 'linear', 'poly'], # projeta os dados em espaço de maior dimensão pra encontrar uma separação linear
        'gamma':  ['scale', 'auto', 0.001, 0.01, 0.1], # raio de influência de cada exemplo, valores altos -> overfitting
    }
    svm = SVC(random_state=42)
    search = RandomizedSearchCV(svm, svm_grid, n_iter=20, cv=5,
                                verbose=1, n_jobs=-1, random_state=42)
    search.fit(atributos_norm, classes_b)
    print("\n[SVM] Melhores parâmetros:")
    pprint(search.best_params_)

    return search.best_params_
 


def hiperparametrizar_logistic_regression(atributos_norm, classes_b):
    '''
    busca os melhores hiperparâmetros para Logistic Regression

    a Logistic Regression é adequada para classificação binária, ele estima a probabilidade
    de uma amostra pertencer a uma classe e, a partir dessa probabilidade, decide se o resultado
    será 0 ou 1

    é um modelo simples, rápido, interpretável e serve como uma boa comparação com modelos mais
    complexos como RF e SVM

    recebe atributos normalizados porque o modelo usa regularização, e a escala dos atributos pode influenciar o resultado
    '''

    lr_grid = {
        'C': [0.01, 0.1, 1, 10, 100], # regularização do modelo, ajuda a evitar overfitting
        'solver': ['lbfgs', 'liblinear'], # encontrar os melhores coeficientes da LR
        'max_iter': [500, 1000] # define o número máximo de iterações para o modelo convergir 
    }

    lr = LogisticRegression(random_state=42)

    search = RandomizedSearchCV(
        lr,
        lr_grid,
        n_iter=10,
        cv=5,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    search.fit(atributos_norm, classes_b)

    print("\n[Regressão Logística] Melhores parâmetros:")
    pprint(search.best_params_)

    return search.best_params_


# ======== treinamento ========

def treinamento_rf(atributos_b, classe_b, best_params):
    '''
    treina a Random Forest com os melhores hiperparâmetros
    '''
    modelo_rf = RandomForestClassifier(**best_params, random_state=42)

    modelo_rf.fit(atributos_b, classe_b)

    return modelo_rf
 
 
def treinar_svm(atributos_norm, classes_b, best_params):
    '''
    treina SVM com os melhores hiperparâmetros, recebe dados normalizados
    '''
    modelo = SVC(**best_params, random_state=42, probability=True)
    modelo.fit(atributos_norm, classes_b)

    return modelo
 
 
def treinar_logistic_regression(atributos_norm, classes_b, best_params):
    '''
    treina a Regressão Logística com os melhores hiperparâmetros, recebe dados normalizados
    '''
    modelo = LogisticRegression(**best_params, random_state=42)
    modelo.fit(atributos_norm, classes_b)

    return modelo
 

# ======== avaliando ========

def avaliacao_cross_validation(modelo, atributos, classes, nome_modelo):
    '''
    avalia o modelo com Cross Validation de 10 folds
    '''

    scoring = ['precision_macro', # mede quantas das predições positivas realmente são positivas
               'recall_macro',  # mede quantos exemplos positivos reais foram detectados, um falso negativo nessa base é crítico
               'f1_macro',  # média harmonica entre precision e recall.
               'accuracy',  # proporção de classificados corretamente
               'roc_auc']   # avalia a qualidade de separação entre as classes, independente do threshold escolhido

    score_cross = cross_validate(modelo, 
                                atributos, 
                                classes,
                                scoring=scoring, # métricas
                                cv=10,  # estratégia de divisão, numero de folds
                                verbose=0, # nivel de verbosidade
                                n_jobs=-1) # usar todos os processadores
 
    resultados = {
        'modelo': nome_modelo,
        'matriz scoring': score_cross,
        'precision': score_cross['test_precision_macro'].mean(),
        'recall':    score_cross['test_recall_macro'].mean(),
        'f1_score':  score_cross['test_f1_macro'].mean(),
        'accuracy':  score_cross['test_accuracy'].mean(),
        'roc_auc':   score_cross['test_roc_auc'].mean(),
    }

    print('\n================================')
    print(f'Cross Validation - {nome_modelo}')
    print('================================')
    print(f'>> Precision: {resultados['precision']:.4f}')
    print(f'>> Recall: {resultados['recall']:.4f}')
    print(f'>> F1-Score: {resultados['f1_score']:.4f}')
    print(f'>> Accuracy: {resultados['accuracy']:.4f}')
    print(f'>> ROC AUC: {resultados['roc_auc']:.4f}')

    return resultados


# ======== comparando os modelos ========

def comparar_modelos(resultados_list):
    '''
    compara os resultados dos modelos avaliados em relação a accuracy
    '''
    # df com os resultados dos modelos
    df_res = pd.DataFrame(resultados_list)

    # define o nome do modelo como índice da tabela
    df_res = df_res.set_index('modelo')

    metricas = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    print('\n======================================')
    print('       COMPARAÇÃO DOS MODELOS')
    print('======================================')
    print(df_res[metricas].round(4))

    # escolhendo o melhor modelo com base na accuracy
    melhor_modelo = df_res['accuracy'].idxmax()
    
    print('\n======================================')
    print('RECOMENDAÇÃO PARA PRODUÇÃO')

    print(f'>> O modelo mais adequado para produção considerando a maior acurácia é: {melhor_modelo}')

    return df_res


def main():
    '''
    ======== decisão na abordagem de avaliação ========

    utilizei cross validation sem hold out!!! a base Pima tem 768 exemplos, é relativamente pequena
    e utilizar o hold out e reservar 30% pra teste reduziria o treino pra mais ou menos 530 dados, fazendo com que a 
    performance possa se tornar meio instável e depender bastante da sorte no split dos dados.

    com o cross validation (10 folds) cada dado participa do teste exatamente 1 vez e do treino 9 vezes, 
    a base é divida em 10 rodadas, e em cada rodada os dados são divididos em 9 partes pra treino e 1 parte pra teste,
    sendo que essa parte de teste nunca participou do treino daquela rodada. no final, tem-se uma média de todas 
    as 10 estimativas dessas rodadas, e acredito eu, que acaba produzindo uma performance muito mais estável.
    '''

    dados, dados_classe, dados_atributos = carregar_dados('diabetes.csv')
    # print(dados)
    # print(dados_classe)

    # print(Counter(dados_classe))  # 0: 500 / 1: 268
    # print(Counter(dados_atributos))
    
    # ======== balancear os dados ========
    atributos_b, classes_b = balancear_dados(dados_atributos, dados_classe)
    print(dados.columns) # rotulos das colunas

    # caso fosse utilizar o hold out
    # atributos_train, atributos_test, classes_train, classes_test = segmentar_dados(atributos_b, classes_b)

    # ======== normalização ========
    atributos_norm, scaler = normalizar_dados(atributos_b)
    salvar_modelo(scaler, 'scaler_diabetes.pkl')

    print(atributos_norm)

    # ====== Random Forest ======
    print('\n================================')
    print('>> Modelo 1: Random Forest\n')

    # ======== encontrar os melhores parametros pra rf ========
    # passando dados nao normalizados porque nao faz diferença, a RF vai analisar se o valor é maior ou menor que X, então 
    # estando normalizado ou nao, a escala não importa
    rf_parametros = hiperparametrizar_random_forest(atributos_norm, classes_b)
    modelo_rf  = treinamento_rf(atributos_b, classes_b, rf_parametros)
    resultado_rf = avaliacao_cross_validation(modelo_rf, atributos_b, classes_b, 'Random Forest')


    # ======== SVM ========
    print('\n================================')
    print('>> Modelo 2: SVM\n')

    # passandoa atributos normalizados 
    svm_parametros = hiperparametrizar_svm(atributos_norm, classes_b)
    modelo_svm = treinar_svm(atributos_norm, classes_b, svm_parametros)
    resultado_svm = avaliacao_cross_validation(modelo_svm, atributos_norm, classes_b, 'SVM')

    # ======== Logistic Regression ========
    print('\n================================')
    print('>> Modelo 3: Logistic Regression')

    # passando atributos normalizados 
    lr_parametros = hiperparametrizar_logistic_regression(atributos_norm, classes_b)
    modelo_lr = treinar_logistic_regression(atributos_norm, classes_b, lr_parametros)
    resultado_lr = avaliacao_cross_validation(modelo_lr, atributos_norm, classes_b, 'Logistic Regression')

    # comparando os modelos e decidindo qual é o melhor para produção
    comparar_modelos([resultado_rf, resultado_svm, resultado_lr])

    # salvando todos os modelos
    salvar_modelo(modelo_rf, 'modelo_rf_diabetes.pkl')
    salvar_modelo(modelo_svm, 'modelo_svm_diabetes.pkl')
    salvar_modelo(modelo_lr, 'modelo_lr_diabetes.pkl')


if __name__ == '__main__':
    main()