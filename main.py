import pandas as pd
import numpy as np
import pickle as pkl
from collections import Counter
from pprint import pprint

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



# ======= carregamento e pré-processamento =======
def carregar_dados(path):
    '''
    lê o csv e separa os atributos da coluna de classe
    '''
    dados = pd.read_csv(path, sep = ';')

    # substitui nulos pela mediana de cada coluna numérica
    dados = dados.fillna(dados.median(numeric_only=True))

    dados_classe = dados['default payment next month']
    dados_atributos = dados.drop(columns=['default payment next month', 'ID'])

    return dados, dados_classe, dados_atributos


def tratar_categoricas(dados_atributos):
    '''
    Converte as colunas categóricas em códigos numéricos
    as colunas SEX, EDUCATION e MARRIAGE são categóricas. como utilizarei o SMOTENC, ele precisa que 
    essas colunas estejam codificadas em números, mas ainda marcadas como categóricas

    o encoder é ajustado apenas nos dados de treino e depois aplicado nos dados de teste para evitar data leakage
    '''

    colunas_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']

    dados_atributos = dados_atributos.copy()

    dados_atributos = pd.get_dummies(dados_atributos, columns=colunas_categoricas, dtype=int)

    print(f"\n====== Colunas após get_dummies ======")
    print(dados_atributos.columns.tolist())
 
 
    return dados_atributos


def normalizar_dados_numericos(atributos_train, atributos_teste):
    '''
    normaliza apenas as colunas numéricas

    o scaler é ajustado apenas nos dados de treino com fit_transform e depois o mesmo scaler é aplicado nos 
    dados de teste com transform, evitando data leakage
    '''
  
    scaler = StandardScaler()

    atributos_train_norm = scaler.fit_transform(atributos_train)  # aprende + transforma
    atributos_teste_norm = scaler.transform(atributos_teste)  # só transforma

    return atributos_train_norm, atributos_teste_norm, scaler


def balancear_dados(atributos_train, classes_train):
    '''
    balancear os dados com SMOTE
    cria novos exemplos sintéticos da classe com menos dados
    '''

    # objeto SMOTE - balanceador
    resampler = SMOTE(random_state=42) 

    # executa o balanceamento
    atributos_b, classes_b = resampler.fit_resample(atributos_train, classes_train)

    print(atributos_b)
    print(classes_b)

    print('\n====== FREQUENCIA DAS CLASSES APÓS BALANCEAMENTO ======')

    # conta quantos exemplos existem de cada classe depois do smote
    class_count = Counter(classes_b)
    print(class_count)

    return atributos_b, classes_b, 


def segmentar_dados(atributos_b, classes_b, test_size=0.3):
    '''
    divide dados em treino e teste (hold-out)
    '''
    return train_test_split(atributos_b, classes_b, test_size=test_size, random_state=42)
 

def salvar_modelo(modelo, caminho: str):
    pkl.dump(modelo, open(caminho, 'wb'))


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
 

def hiperparametrizar_xgboost(atributos_train, classe_train):
    '''
    busca os melhores hiperparâmetros para o XGBoost com RandomizedSearchCV
 
    parâmetros buscados:
    - n_estimators: número de árvores (rodadas de boosting)
    - max_depth: profundidade máxima de cada árvore
    - learning_rate: tamanho do passo de cada árvore
    - subsample: fração dos exemplos usada em cada árvore. introduz aleatoriedadee reduz overfitting.
    - colsample_bytree: fração dos atributos considerada em cada árvore
      Também introduz aleatoriedade, semelhante ao max_features da RF
    - scale_pos_weight: peso da classe positiva (inadimplente). alternativa ao SMOTE para lidar com desbalanceamento
    '''
    xgb_grid = {
        'n_estimators':     [int(x) for x in np.linspace(50, 300, 6)],
        'max_depth':        [3, 4, 5, 6, 7, 8],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'scale_pos_weight': [1],
    }
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')

    search = RandomizedSearchCV(xgb, xgb_grid, n_iter=20, cv=5, verbose=0, n_jobs=-1, random_state=42)
    search.fit(atributos_train, classe_train)

    print("\n[XGB] Melhores parâmetros:")
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
 
 
def treinar_xgboost(atributos_train, classe_train, best_params):
    '''
    treina XGBoost com os melhores hiperparâmetros, recebe dados sem normalização
    '''
    modelo = XGBClassifier(**best_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    modelo.fit(atributos_train, classe_train)

    return modelo
 

def avaliacao_modelo(modelo, atributos_teste, classe_test, nome_modelo):
    '''
    métricas avaliadas:
    - Accuracy: proporção de classificações corretas
    - Precision (macro): qualidade das predições de inadimplência
    - Recall (macro): proporção de inadimplentes reais detectados , CRÍTICO para o banco
    - F1-Score (macro): equilíbrio entre precision e recall
    '''
    preds = modelo.predict(atributos_teste)
    probs = modelo.predict_proba(atributos_teste)[:, 1]
 
    resultados = {
        'modelo':    nome_modelo,
        'precision': precision_score(classe_test, preds, average='macro'),
        'recall':    recall_score(classe_test, preds, average='macro'),
        'f1_score':  f1_score(classe_test, preds, average='macro'),
        'accuracy':  accuracy_score(classe_test, preds),
        'roc_auc':   roc_auc_score(classe_test, probs),
    }
 
    print(f'\n===================================')
    print(f'Resultados Hold-out - {nome_modelo}')
    print(f'Precision: {resultados['precision']:.4f}')
    print(f'Recall: {resultados['recall']:.4f}')
    print(f'F1-Score: {resultados['f1_score']:.4f}')
    print(f'Accuracy: {resultados['accuracy']:.4f}')
 
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

    metricas = ['accuracy', 'precision', 'recall', 'f1_score']

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
    dados, dados_classe, dados_atributos = carregar_dados('default_of_credit_card_clients.csv')
    #print(dados_atributos.columns)
    #print(dados_classe)

    dados_atributos = tratar_categoricas(dados_atributos)

    atributos_train, atributos_teste, classes_train, classes_teste = segmentar_dados(dados_atributos, dados_classe)

    print(atributos_train)

    #print(Counter(dados_classe)) / 0: 23364, 1:6636
    atributos_train_b, classes_train_b = balancear_dados(atributos_train, classes_train)

    # ====== Random Forest ======
    print('\n================================')
    print('>> Modelo 1: Random Forest\n')

    # ======== RANDOM FOREST ========
    # passando dados nao normalizados porque nao faz diferença, a RF vai analisar se o valor é maior ou menor que X, então 
    # estando normalizado ou nao, a escala não importa
    rf_parametros = hiperparametrizar_random_forest(atributos_train_b, classes_train_b)
    modelo_rf  = treinamento_rf(atributos_train_b, classes_train_b, rf_parametros)
    resultado_rf = avaliacao_modelo(modelo_rf, atributos_teste, classes_teste, 'Random Forest')

    # ======== XGBOOST ========
    print('\n================================')
    print('\n>> Modelo 2: XGBOOST')
    xgb_params = hiperparametrizar_xgboost(atributos_train_b, classes_train_b)
    modelo_xgb = treinar_xgboost(atributos_train_b, classes_train_b, xgb_params)
    resultado_xgb= avaliacao_modelo(modelo_xgb, atributos_teste, classes_teste, 'XGBoost')
 
    comparar_modelos([resultado_rf, resultado_xgb])

    # salvando todos os modelos
    salvar_modelo(modelo_rf, 'modelo_rf_bank.pkl')
    salvar_modelo(modelo_xgb, 'modelo_xgboost_bank.pkl')


if __name__ == '__main__':
    main()