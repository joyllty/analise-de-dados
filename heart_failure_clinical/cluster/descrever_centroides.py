import pickle
import pandas as pd


def carregar_modelo(nome_modelo):
    '''
    carregar o modelo de clusters
    '''
    modelo = pickle.load(open(nome_modelo, 'rb'))

    return modelo


def carregar_normalizador(nome_normalizador):
    '''
    carregar o normalizador dos dados
    '''
    normalizador = pickle.load(open(nome_normalizador, 'rb'))

    return normalizador


def desnormalizar_centroides(centroides, normalizador):
    '''
    desnormaliza o dataframe dos centroides para melhor analise em escala real dos dados
    '''
    centroides_desnorm = normalizador.inverse_transform(centroides)

    return centroides_desnorm


def comparar_valores(valor_cluster, media, margem=0.10):
    '''
    compara cada valor de uma linha de cluster com a media geral de cada coluna dos dados originais,
    respeitando uma margem de 10% para evitar classificar pequenas variações como altas ou baixas
    '''
    if valor_cluster > media * (1 + margem):
        return 'alto'

    elif valor_cluster < media * (1 - margem):
        return 'baixo'

    else:
        return 'medio'


def interpretar_binario(valor_cluster):
    '''
    interpreta colunas binárias dentro dos centroides.
    como o centroide representa a média do grupo, valores próximos de 1 indicam maior presença da característica.
    '''
    if valor_cluster >= 0.60:
        return 'alto'

    elif valor_cluster <= 0.40:
        return 'baixo'

    else:
        return 'medio'


def interpretar_cluster(linha, media_geral, indice):
    '''
    interpreta cada linha dos clusters, de acordo com a comparação entre cada coluna do dataset e com a média geral
    '''
    descricoes = mensagens_colunas()
    colunas_binarias = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

    print(f'\n>> Cluster: {indice}')

    for coluna in linha.index:

        # ignora colunas sem descrição
        if coluna not in descricoes:
            continue

        valor_cluster = linha[coluna]
        media_coluna = media_geral[coluna]

        if coluna in colunas_binarias:
            resultado = interpretar_binario(valor_cluster)
        else:
            resultado = comparar_valores(valor_cluster, media_coluna)

        frase = descricoes[coluna][resultado]

        print(f'- {coluna}: {valor_cluster:.2f} -> {frase}')


def mensagens_colunas():
    '''
    descrições usadas para interpretar o perfil de cada cluster
    '''
    descricoes_cluster = {
        'age': {
            'alto': 'pacientes com idade mais alta',
            'baixo': 'pacientes mais jovens',
            'medio': 'pacientes com idade próxima da média geral'
        },

        'anaemia': {
            'alto': 'maior presença de anemia no grupo',
            'baixo': 'menor presença de anemia no grupo',
            'medio': 'presença moderada de anemia no grupo'
        },

        'creatinine_phosphokinase': {
            'alto': 'nível mais alto de creatinina fosfoquinase no sangue',
            'baixo': 'nível mais baixo de creatinina fosfoquinase no sangue',
            'medio': 'nível mediano de creatinina fosfoquinase no sangue'
        },

        'diabetes': {
            'alto': 'maior presença de diabetes no grupo',
            'baixo': 'menor presença de diabetes no grupo',
            'medio': 'presença moderada de diabetes no grupo'
        },

        'ejection_fraction': {
            'alto': 'fração de ejeção mais alta, melhor bombeamento de sangue pelo coração',
            'baixo': 'fração de ejeção mais baixa, maior comprometimento cardíaco',
            'medio': 'fração de ejeção próxima da média geral'
        },

        'high_blood_pressure': {
            'alto': 'maior presença de pressão alta no grupo',
            'baixo': 'menor presença de pressão alta no grupo',
            'medio': 'presença moderada de pressão alta no grupo'
        },

        'platelets': {
            'alto': 'contagem de plaquetas mais alta',
            'baixo': 'contagem de plaquetas mais baixa',
            'medio': 'contagem de plaquetas próxima da média geral'
        },

        'serum_creatinine': {
            'alto': 'creatinina sérica mais alta, podendo indicar maior alteração renal',
            'baixo': 'creatinina sérica mais baixa',
            'medio': 'creatinina sérica próxima da média geral'
        },

        'serum_sodium': {
            'alto': 'sódio sérico mais alto',
            'baixo': 'sódio sérico mais baixo',
            'medio': 'sódio sérico próximo da média geral'
        },

        'sex': {
            'alto': 'maior proporção de pacientes do sexo masculino no grupo',
            'baixo': 'maior proporção de pacientes do sexo feminino no grupo',
            'medio': 'grupo com distribuição mais equilibrada entre os sexos'
        },

        'smoking': {
            'alto': 'maior presença de pacientes fumantes no grupo',
            'baixo': 'menor presença de pacientes fumantes no grupo',
            'medio': 'presença moderada de pacientes fumantes no grupo'
        },

        'time': {
            'alto': 'maior tempo de acompanhamento',
            'baixo': 'menor tempo de acompanhamento',
            'medio': 'tempo de acompanhamento próximo da média geral'
        }
    }

    return descricoes_cluster


def descrever_clusters(dados_norm, modelo, normalizador, media_geral):
    '''
    descrever o que cada cluster representa a partir dos centroides
    '''
    # converter os centroides em dataframe
    centroides = pd.DataFrame(modelo.cluster_centers_, columns=dados_norm.columns)
    centroides_reais = desnormalizar_centroides(centroides, normalizador)

    df_centroides_reais = pd.DataFrame(centroides_reais, columns=dados_norm.columns)

    for i, linha in df_centroides_reais.iterrows():
        interpretar_cluster(linha, media_geral, i)


def analisar_desfecho_por_cluster(dados, dados_atributos, modelo, normalizador):
    '''
    usa DEATH_EVENT e time apenas depois do treinamento para interpretar os clusters
    essas colunas não participam do treinamento.
    '''
    dados_norm = normalizador.transform(dados_atributos)
    clusters = modelo.predict(dados_norm)

    df_analise = dados_atributos.copy()
    df_analise['cluster'] = clusters
    df_analise['DEATH_EVENT'] = dados['DEATH_EVENT'].values
    df_analise['time'] = dados['time'].values

    print('\n======================================')
    print('ANÁLISE DE DESFECHO POR CLUSTER')

    resumo = df_analise.groupby('cluster').agg(
        total_pacientes=('DEATH_EVENT', 'count'),
        total_mortes=('DEATH_EVENT', 'sum'),
        proporcao_mortes=('DEATH_EVENT', 'mean'),
        mediana_time_dias=('time', 'median')
    )

    print(resumo.round(4))

    return resumo
