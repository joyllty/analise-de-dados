import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import  matplotlib.pyplot as plt
import math

# dados estão sem faltantes

df = pd.read_csv('./ObesityDataSet_raw_and_data_sinthetic.csv')
print(len(df))
print(df.columns)
print(df.head(25))

# separando os dados em numericos e categóricos
dados_binary = df[['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']]

dados_num = df.drop(columns=['family_history_with_overweight','FAVC','Gender','CAEC','SMOKE','SCC','CALC','MTRANS', 'NObeyesdad'])

dados_cat = df[['Gender','CAEC','CALC','MTRANS', 'NObeyesdad']]

print(dados_cat)
print(dados_num)
    
# print(df["NObeyesdad"].value_counts())

# ===== Normalização dos dados =====
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)

pickle.dump(normalizador, open('./scaler_dados_num.pkl', 'wb'))

dados_num_norm = normalizador.fit_transform(dados_num)
dados_num_norm = pd.DataFrame(dados_num_norm, columns = dados_num.columns)

dados_cat_norm = pd.get_dummies(dados_cat,dtype=int)

dados_binary = dados_binary.replace({"yes": 1, "no": 0})

#print(dados_num_norm)
#print(dados_cat_norm)
#print(dados_binary)

dados_norm = dados_num_norm.join(dados_cat_norm).join(dados_binary)
print(dados_norm.columns)
# ==================================

# calcular distorções
distorcoes = []
# utilizando uma amostra de 30% do dataset
#base_amostragem = dados_norm.sample(frac=0.25, random_state=42)
# intervalo de pontos da reta da função distorcoes | numero de clusters
K = range(1, 550) 

for i in K:
    # treinando iterativamente e aumentando o numero de clusters
    # testando quantidades de centroides para descobrir a ideal
    modelo = KMeans(n_clusters=i, random_state=42).fit(dados_norm)
    # calcular a distorção
    # media das distancias de cada ponto ao seu centroide
    distorcoes.append(
        sum(
            np.min(cdist(dados_norm, modelo.cluster_centers_, 'euclidean'), axis=1)/dados_norm.shape[0]
            )
        )   

#fig, ax = plt.subplots()
#ax.plot(K, distorcoes)
#ax.set_xlabel('Número de Clusters')
#ax.set_ylabel('Distorção')
#ax.set_title('Método do Cotovelo')
#ax.grid()
#plt.show()

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

# treinar e salvar o modelo
cluster_obesity = KMeans(n_clusters=numero_clusters_otimo, random_state=42).fit(dados_norm)

pickle.dump(cluster_obesity, open('cluster_obesity.pkl', 'wb'))