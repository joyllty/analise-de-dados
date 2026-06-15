import pandas as pd


def normalizar_novo_dado(novo_dado, normalizador):
    '''
    aplica o normalizador de dados ao novo dado
    '''
    novo_dado_norm = normalizador.transform(novo_dado)

    novo_dado_norm = pd.DataFrame(novo_dado_norm, columns=novo_dado.columns)

    return novo_dado_norm


def prever_cluster(modelo, novo_dado_norm):
    '''
    retorna o cluster previsto do novo dado
    '''
    cluster = modelo.predict(novo_dado_norm)

    # uma linha tem apenas uma previsão
    return cluster[0]
