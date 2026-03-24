'''
Implementar uma classe capaz de normalizar dados conforme os métodos citados em sala:

MinMax Scaler
Label Encoding
One Hot Encodig
Deve-se implementar os métodos de reversão também

A classe deve ser reaproveitável

Utilizar o arquivo dados_normalizar.csv (na pasta da aula 2) para testar seu código.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class Normalizador():
    def __init__(self, path):
        self.df = pd.read_csv(path, sep=';', decimal=',') # excel
        self.df.columns = self.df.columns.str.lower().str.strip() # normalizando o nome das colunas
        self.scalers = {}
        self.labels = {}
        self.ohe_colunas = {}


    def min_max(self, coluna):
        self.scalers[coluna] = MinMaxScaler()
        dados = self.df[coluna].values.reshape(-1, 1) # converter pra array numpy 2D

        dados_normalizados = self.scalers[coluna].fit_transform(dados)
        
        self.df[coluna] = dados_normalizados # dados 1D

    def reverse_min_max(self, coluna):
        if coluna not in self.scalers:
            print('Chame min_max() antes de reverter!')
            return
        
        dados_normalizados = self.df[coluna].values.reshape(-1, 1) # converter de novo para 2D

        dados_revertidos = self.scalers[coluna].inverse_transform(dados_normalizados)

        self.df[coluna] = dados_revertidos

    def label_encoder(self, coluna):
        self.labels[coluna] = LabelEncoder() # trabalha com dados 1D
        categoricos = self.df[coluna]

        categoricos_norm = self.labels[coluna].fit_transform(categoricos)

        self.df[coluna] = categoricos_norm

    def reverse_label(self, coluna):
        if coluna not in self.labels:
            print('Chame label_encoder() antes de reverter!')
            return
        
        categoricos_norm = self.df[coluna]

        categoricos_reverse = self.labels[coluna].inverse_transform(categoricos_norm) # espera array de inteiros

        self.df[coluna] = categoricos_reverse

    def one_hot_encoder(self, coluna):
        colunas_antes = set(self.df.columns)

        self.df = pd.get_dummies(self.df, columns=[coluna], prefix=coluna, prefix_sep='_', dtype=int) # precisa do df inteiro
        
        colunas_depois = set(self.df.columns)

        colunas_criadas = list(colunas_depois - colunas_antes)

        self.ohe_colunas[coluna] = colunas_criadas

    def reverse_one_hot(self, coluna):
        if coluna not in self.ohe_colunas:
            print('Chame one_hot_encoder() antes de reverter!')
            return
        
        colunas = self.ohe_colunas[coluna]

        valores_originais = self.df[colunas].idxmax(axis=1) # analisa horizontalmente, qual coluna em cada linha 
    

        valores_originais = valores_originais.str.replace(coluna + '_', '') # deixa apenas 'f' ou 'm'

        self.df[coluna] = valores_originais
        self.df = self.df.drop(columns=colunas)

    def mostrar(self):
        # df é sempre sobrescrito com aplicação dos métodos
        print(f'{self.df}\n')


