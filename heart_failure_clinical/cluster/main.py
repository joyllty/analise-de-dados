from treinamento import carregar_dados, normalizar_dados, calcular_distorcoes, calcular_numero_clusters, treinar_kmeans, salvar_modelo

from descrever_centroides import carregar_modelo, carregar_normalizador, descrever_clusters, analisar_desfecho_por_cluster

from inferencia_cluster import normalizar_novo_dado, prever_cluster
import pandas as pd


# ====== carregar os dados ======
dados, dados_atributos = carregar_dados('./heart_failure_clinical_records_dataset.csv')

media = dados_atributos.mean()

#print(dados)
#print(dados_atributos)
#print(dados_desfecho)

# ====== pré-processamento ======
print('\n======================================')
print('PRÉ-PROCESSAMENTO')

#print('>> Frequência das colunas binárias:')
#for coluna in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']:
    #print(dados_atributos[coluna].value_counts())

# ====== normalizar os dados ======
dados_norm = normalizar_dados(dados_atributos)
#print(dados_norm)

# transformar os dados normalizados em dataframe
dados_norm = pd.DataFrame(dados_norm, columns=dados_atributos.columns)

print(dados_norm)


# ====== treinamento do modelo ======
print('\n======================================')
print('TREINAMENTO DO MODELO')


print('>> Justificativa: o problema pede agrupamento/similaridade de pacientes, então KMeans é adequado para separar pacientes em grupos de acordo com proximidade entre atributos clínicos.')


distorcoes, valores_k = calcular_distorcoes(dados_norm)

# escolhi um range de K de 18, pois resultou em 6 clusters, mantendo grupos com quantidade boa de
# pacientes e nenhum com numero de pacientes abaixo de 30 (10% do dataset). eu testei com um numero 
# maior, ou até mesmo com o número maximo de linhas do dataset como range, porém resultava em muitos
# clusters com poucos pacientes, distorcendo muito a proporção de mortes por cluster.

clusters = calcular_numero_clusters(distorcoes, valores_k)

cluster_heart_failure = treinar_kmeans(clusters, dados_norm)

salvar_modelo(cluster_heart_failure)


# ====== descrever clusters ======
modelo_cluster = carregar_modelo('cluster_heart_failure.pkl')

normalizador = carregar_normalizador('scaler_heart_failure.pkl')

descrever_clusters(dados_norm, modelo_cluster, normalizador, media)

# DEATH_EVENT é usado apenas para analisar os grupos depois não para treinar
analisar_desfecho_por_cluster(dados, dados_atributos, modelo_cluster, normalizador)


# ====== inferencia ======
novo_dado = pd.DataFrame([[
        65, # age
        1, # anaemia
        160, # creatinine_phosphokinase
        0, # diabetes
        35, # ejection_fraction
        1, # high_blood_pressure
        250000, # platelets
        1.4, # serum_creatinine
        136, # serum_sodium
        1, # sex
        0, # smoking
    ]], columns=[
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking'
    ])

novo_dado_norm = normalizar_novo_dado(novo_dado, normalizador)

cluster_novo_dado = prever_cluster(modelo_cluster, novo_dado_norm)

print('\n======================================')
print('INFERÊNCIA')
print(f'\n>> Dados do novo paciente:\n{novo_dado}')
print(f'\n>> Novo paciente pertence ao Cluster: {cluster_novo_dado}')
