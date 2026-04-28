import pickle
import pandas as pd

df = pd.read_csv('./ObesityDataSet_raw_and_data_sinthetic.csv')

# abrir o modelo de cluster
cluster_obesity = pickle.load(open('cluster_obesity.pkl', 'rb'))

# colunas do dataset
nomes_colunas = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
        'Gender_Female', 'Gender_Male',
        'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
        'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
        'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
        'MTRANS_Public_Transportation', 'MTRANS_Walking',
        'NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight',
        'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Obesity_Type_II',
        'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I',
        'NObeyesdad_Overweight_Level_II',
        'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC'
]

# transformar centroides em dataframe
centroides = pd.DataFrame(cluster_obesity.cluster_centers_,columns=nomes_colunas)

# segmentar o dataframe em colunas numéricas e colunas categóricas
# separando os dados binários nao treinados dos numericos

dados_num_norm = centroides[['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']]

dados_binary = centroides[['family_history_with_overweight','FAVC','SMOKE','SCC']].round(0)

dados_cat_norm = centroides[['Gender_Female', 'Gender_Male', 'CAEC_Always', 'CAEC_Frequently',
       'CAEC_Sometimes', 'CAEC_no', 'CALC_Always', 'CALC_Frequently',
       'CALC_Sometimes', 'CALC_no', 'MTRANS_Automobile', 'MTRANS_Bike',
       'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking',
       'NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight',
       'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Obesity_Type_II',
       'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I',
       'NObeyesdad_Overweight_Level_II']]

print(centroides)

# carregar o normalizdor de dados
normalizador = pickle.load(open('scaler_dados_num.pkl', 'rb'))

# desnormalizar os dados numericos
dados_num = normalizador.inverse_transform(dados_num_norm)

# transformar os dados desnorm em dataframe 
dados_num = pd.DataFrame(dados_num, columns=dados_num_norm.columns)

# desnormalizar os dados categóricos
#dados_cat = abs(dados_cat_norm.round(0))
dados_cat_desnorm = (dados_cat_norm.round(0).clip(0, 1).astype(int))

# tive problema utilizando o from_dummies, então tive que garantir que cada coluna estava preenchida corretamente
# pra cada coluna categórica
for grupo in [
    ['Gender_Female', 'Gender_Male'],
    ['CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no'],
    ['CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no'],
    [
        'MTRANS_Automobile',
        'MTRANS_Bike',
        'MTRANS_Motorbike',
        'MTRANS_Public_Transportation',
        'MTRANS_Walking'
    ],
    [
        'NObeyesdad_Insufficient_Weight',
        'NObeyesdad_Normal_Weight',
        'NObeyesdad_Obesity_Type_I',
        'NObeyesdad_Obesity_Type_II',
        'NObeyesdad_Obesity_Type_III',
        'NObeyesdad_Overweight_Level_I',
        'NObeyesdad_Overweight_Level_II'
    ]]:
    # qual linha daquele grupo/categorico tem o maior valor 
    idx = dados_cat_desnorm[grupo].idxmax(axis=1)

    # zera todas as colunas do grupo -> reconstruir um dummie perfeito
    dados_cat_desnorm[grupo] = 0

    # pra cada resultado de idxmax
    for i, col in enumerate(idx):
        dados_cat_desnorm.loc[i, col] = 1 # coloca 1 na exata coluna que contem esse valor 

dados_cat = pd.from_dummies(dados_cat_desnorm, sep="_")
print(dados_cat)

# juntar os dois dataframes
cluster_obesity_dados = dados_num.join(dados_binary).join(dados_cat)
print(cluster_obesity_dados)
print(cluster_obesity_dados.columns)

# ===== visualizar cada cluster e interpretar ======
for i in range(len(cluster_obesity_dados)):
    print('\n=================')
    print(f'CLUSTER {i}')
    print('=================')

    linha = cluster_obesity_dados.iloc[i]

    print(f'Idade média: {linha['Age']:.1f} anos')
    print(f'Altura média: {linha['Height']:.2f} m')
    print(f'Peso médio: {linha['Weight']:.2f} kg')
    print(f'Consumo de vegetais (FCVC): {linha['FCVC']:.2f}')
    print(f'Número de refeições principais (NCP): {linha['NCP']:.2f}')
    print(f'Consumo de água (CH2O): {linha['CH2O']:.2f}')
    print(f'Atividade física (FAF): {linha['FAF']:.2f}')
    print(f'Tempo com tecnologia (TUE): {linha['TUE']:.2f}')

    print(f'Histórico familiar de sobrepeso: {linha['family_history_with_overweight']}')
    print(f'Come comida calórica com frequência: {linha['FAVC']}')
    print(f'Fuma: {linha['SMOKE']}')
    print(f'Controla calorias (SCC): {linha['SCC']}')

    print(f'Gênero predominante: {linha['Gender']}')
    print(f'Come entre refeições (CAEC): {linha['CAEC']}')
    print(f'Consumo de álcool (CALC): {linha['CALC']}')
    print(f'Transporte principal: {linha['MTRANS']}')
    print(f'Nível de obesidade: {linha['NObeyesdad']}')