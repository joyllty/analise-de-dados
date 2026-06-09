import pickle as pkl
import pandas as pd

# carregar o modelo XGBoost treinado
modelo_xgb = pkl.load(open('modelo_xgboost_bank.pkl', 'rb'))

# dados do cliente
# SEX, EDUCATION e MARRIAGE já viram colunas binárias pelo get_dummies no treino
cliente = {
    'LIMIT_BAL': 10000, # limite baixo 
    'AGE': 28,
    'PAY_0': 3, # 3 meses de atraso no mês mais recente
    'PAY_2': 3, # 3 meses de atraso
    'PAY_3': 2, # 2 meses de atraso
    'PAY_4': 2,
    'PAY_5': 1,
    'PAY_6': 1,
    'BILL_AMT1': 9800, # fatura quase no limite
    'BILL_AMT2': 9500,
    'BILL_AMT3': 9100,
    'BILL_AMT4': 8800,
    'BILL_AMT5': 8500,
    'BILL_AMT6': 8000,
    'PAY_AMT1': 0, # não pagou nada
    'PAY_AMT2': 100, # pagou valor mínimo
    'PAY_AMT3': 0,
    'PAY_AMT4': 0,
    'PAY_AMT5': 100,
    'PAY_AMT6': 0,

    'SEX_F': 0, 'SEX_M': 1,

    "EDUCATION_Bachelor's Degree": 0,
    'EDUCATION_Early Childhood Education': 0,
    'EDUCATION_Elementary School ': 0,
    'EDUCATION_High School ': 1, # ensino médio
    'EDUCATION_Middle School ': 0,
    'EDUCATION_Post-Secondary Non-Tertiary Education': 0,
    'EDUCATION_Short-Cycle Tertiary Education': 0,

    'MARRIAGE_Divorced': 0,
    'MARRIAGE_Married': 0,
    'MARRIAGE_Never Married': 1, # solteiro
    'MARRIAGE_Widowed': 0,
}

# converter para DataFrame, o modelo espera o mesmo formato do treino
df_cliente = pd.DataFrame([cliente])

# predict_proba retorna [[P(nao default), P(default)]]
resultado = modelo_xgb.predict_proba(df_cliente)
classificacao = modelo_xgb.predict(df_cliente)[0]

print('Classes do modelo:', modelo_xgb.classes_)
print('Probabilidades:', resultado)
print(f'Probabilidade nao default: {resultado[0][0]:.2f}')
print(f'Probabilidade default: {resultado[0][1]:.2f}')
print(f'Classificação: {'DEFAULT' if classificacao == 1 else 'NAO DEFAULT'}')