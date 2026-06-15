import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import matplotlib.pyplot as plt
import math

'''
cada linha representa um paciente com informações clínicas

Colunas da base:
age - idade do paciente
anaemia - variável binária indicando anemia
        1 -> possui anemia
        0 -> não possui anemia
creatinine_phosphokinase - nível da enzima CPK no sangue
diabetes - variável binária indicando diabetes
        1 -> possui diabetes
        0 -> não possui diabetes
ejection_fraction - porcentagem de sangue que sai do coração a cada contração
high_blood_pressure - variável binária indicando hipertensão
        1 -> possui pressão alta
        0 -> não possui pressão alta
platelets - quantidade de plaquetas no sangue
serum_creatinine - nível de creatinina sérica
serum_sodium - nível de sódio sérico
sex - variável binária referente ao sexo biológico registrado na base
        1 -> masculino
        0 -> feminino
smoking - variável binária indicando tabagismo
        1 -> fuma
        0 -> não fuma
time - período de acompanhamento do paciente
DEATH_EVENT - desfecho do acompanhamento
        1 -> ocorreu morte durante o período observado
        0 -> não ocorreu morte durante o período observado

ATENÇÃO:
DEATH_EVENT não é usado como atributo de entrada no treinamento, porque o objetivo é receber os dados de um paciente desconhecido.
Para um paciente novo, normalmente não faria sentido informar se ele morreu ou não durante o acompanhamento.
Essa coluna fica separada apenas para ajudar na interpretação dos clusters depois do treinamento.

As colunas binárias são mantidas como 0 e 1.
Como o normalizador utilizado é MinMaxScaler, variáveis que já estão em 0 e 1 continuam em 0 e 1.
'''


def carregar_dados(path):
    '''
    carrega o csv, separa a coluna DEATH_EVENT e devolve os dados que serão usados no agrupamento
    '''
    dados = pd.read_csv(path, sep=',')

    # DEATH_EVENT é removido do treinamento porque é um desfecho, não uma característica de entrada
    # time é removido porque não representa um dado clínico inicial, da pra ser usado pra interpretar cluster depois
    colunas_removidas = ['DEATH_EVENT', 'time']

    dados_atributos = dados.drop(columns=colunas_removidas)

    return dados, dados_atributos


def normalizar_dados(dados_atributos):
    '''
    treina o normalizador, salva o normalizador e aplica a normalização nos atributos do paciente
    retorna um array

    as variáveis binárias são mantidas no dataset porque fazem parte do perfil do paciente
    como elas já estão em 0 e 1 o MinMaxScaler não altera sua escala
    '''
    scaler = MinMaxScaler()
    normalizador = scaler.fit(dados_atributos)

    # salvar o normalizador para usar depois na inferência de novos pacientes
    pickle.dump(normalizador, open('scaler_heart_failure.pkl', 'wb'))

    dados_norm = normalizador.fit_transform(dados_atributos)

    return dados_norm


def calcular_distorcoes(dados_norm):
    '''
    calcula a distorção para diferentes quantidades de clusters
    '''
    distorcoes = []

    # intervalo de pontos da reta da função distorcoes | numero de clusters
    K = range(1, 19)

    for i in K:
        # treinando interativamente e aumentando o numero de clusters
        # testando quantidades de centroides para descobrir a ideal
        modelo = KMeans(n_clusters=i, random_state=42).fit(dados_norm)

        # média das distâncias de cada ponto até o centroide mais próximo
        distorcoes.append(
            sum(
                np.min(cdist(dados_norm, modelo.cluster_centers_, 'euclidean'), axis=1) / dados_norm.shape[0]
            )
        )

    fig, ax = plt.subplots()
    ax.plot(K, distorcoes)
    ax.set_xlabel('Número de Clusters')
    ax.set_ylabel('Distorção')
    ax.set_title('Método do Cotovelo - Heart Failure')
    ax.grid()
    plt.show()

    return distorcoes, K


def calcular_numero_clusters(distorcoes, K):
    '''
    calcular o numero ideal de clusters, utilizando o metodo do cotovelo -> encontrar o ponto mais distante
    em linha reta da reta de clusters, isto é, o ponto onde adicionar mais um centroide reduzirá minimamente
    as distorções -> clusters bem definidos
    '''

    # definindo os pontos das retas
    x0 = K[0]
    y0 = distorcoes[0]
    xn = K[-1]
    yn = distorcoes[-1]

    distancias = []

    for i in range(len(distorcoes)):
        x= K[i]
        y= distorcoes[i]
        # FORMULA DISTANCIA PONTO-RETA (reta de dois pontos)
        numerador = abs(      # abs -> positivo independente do resultado
            (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
        )
        denominador = math.sqrt(
            (yn-y0)**2 + (xn-x0)**2
        )
        distancias.append(numerador/denominador)

    numero_clusters_otimo = K[distancias.index(np.max(distancias))]
    print('\n>> Número ótimo de clusters = ', numero_clusters_otimo)

    return numero_clusters_otimo


def treinar_kmeans(numero_clusters_otimo, dados_norm):
    '''
    treina o modelo de clusters usando KMeans
    '''
    cluster_heart_failure = KMeans(n_clusters=numero_clusters_otimo,random_state=42).fit(dados_norm)

    return cluster_heart_failure


def salvar_modelo(cluster_heart_failure):
    '''
    salva o modelo de clusters treinado
    '''
    pickle.dump(cluster_heart_failure, open('cluster_heart_failure.pkl', 'wb'))
