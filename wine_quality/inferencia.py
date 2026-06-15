import json
import pickle as pkl

import pandas as pd


PATH_MODELO = './modelos_salvos/modelo_wine_quality.pkl'
PATH_COLUNAS = './modelos_salvos/colunas_modelo.json'


def carregar_modelo(path_modelo):
    '''
    carrega o modelo treinado 
    '''

    with open(path_modelo, 'rb') as arquivo:
        modelo = pkl.load(arquivo)

    return modelo


def carregar_colunas(path_colunas):
    '''
    carrega a ordem das colunas usadas do treinamento
    '''

    with open(path_colunas, 'r', encoding='utf-8') as arquivo:
        colunas = json.load(arquivo)

    return colunas


def montar_amostra():
    '''
    monta uma amostra de vinho para inferência

    tipo_vinho:
    0 = vinho tinto
    1 = vinho branco
    '''

    amostra = {
        'fixed acidity': 7.0,
        'volatile acidity': 0.27,
        'citric acid': 0.36,
        'residual sugar': 20.7,
        'chlorides': 0.045,
        'free sulfur dioxide': 45.0,
        'total sulfur dioxide': 170.0,
        'density': 1.001,
        'pH': 3.0,
        'sulphates': 0.45,
        'alcohol': 8.8,
        'tipo_vinho': 1
    }

    return amostra


def prever_qualidade(modelo, colunas, amostra):
    '''
    recebe uma amostra de vinho e retorna a classe quality prevista
    '''

    dados_amostra = pd.DataFrame([amostra])
    dados_amostra = dados_amostra[colunas]

    predicao = modelo.predict(dados_amostra)[0]

    return predicao


def main():
    modelo = carregar_modelo(PATH_MODELO)
    colunas = carregar_colunas(PATH_COLUNAS)
    amostra = montar_amostra()

    predicao = prever_qualidade(modelo, colunas, amostra)


    print(pd.DataFrame([amostra]))

    print('\n====== RESULTADO DA INFERÊNCIA ======')
    print(f'>> Qualidade prevista do vinho: {predicao}')


if __name__ == '__main__':
    main()
