import pickle
import pandas as pd

colunas_normalizadas = pd.DataFrame(columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
    'Gender_Female', 'Gender_Male',
    'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
    'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
    'MTRANS_Public_Transportation', 'MTRANS_Walking',
    'NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight',
    'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Obesity_Type_II',
    'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I',
    'NObeyesdad_Overweight_Level_II',
    'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC'])

# nova entrada 
novo_dado = pd.DataFrame([[
    0.35, 0.55, 0.40, 0.75,
    0.50, 0.60, 0.30, 0.70,
    1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 0, 1, 0,
    0, 1, 0, 0, 0, 1, 0,
    1, 1, 0, 1
]], columns=colunas_normalizadas.columns)

# juntar com dataframe vazio 
novo_dado_normalizado = pd.concat([novo_dado, colunas_normalizadas]).fillna(0)

# carregar modelo treinado
cluster_obesity = pickle.load(open("cluster_obesity.pkl", "rb"))

# prever cluster
cluster_novo_dado = cluster_obesity.predict(novo_dado_normalizado)

print("\n>> Cluster do novo dado:", cluster_novo_dado)

