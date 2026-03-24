'''
Você deve considerar o seguinte cenário

Após normalizar os dados categóricos nominais, a tabela de dados passou a ter as colunas [cor_azul, cor_verde, cor_vermelho]. Essa estrutura será, então, utilizada para treinamento do modelo de IA;
Quando uma nova instância é recebida, ela terá o atributo cor=Azul.
Ocorre que essa nova instancia precisa ser alterada, de forma a obedecer a estrutura dos dados normalizados: [cor_azul, cor_verde, cor_vermelho]
Pede-se:

Implemente um método que recebe os dados da nova instância e altera sua estrutura de acordo com os dados normalizados com one hot encoder.
'''

import pandas as pd
from transformar_one_hot import transformar_one_hot

dados = pd.DataFrame({
    'cor_azul':     [1, 0, 0],
    'cor_verde':    [0, 1, 0],
    'cor_vermelho': [0, 0, 1]
})

nova_instancia = {'cor': 'Azul'}

df_normalizado = transformar_one_hot(nova_instancia, dados.columns)

print(f'Cor: {nova_instancia['cor']}')
print(df_normalizado)




